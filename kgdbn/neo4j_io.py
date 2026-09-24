# Neo4j integration: export KnowledgeGraph and compute GDS embeddings.

from __future__ import annotations

import os
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from kgdbn.ontology import KnowledgeGraph


def _validate_identifier(name: str) -> str:
    # Validate Cypher identifiers (labels and relationship types).
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Invalid Cypher identifier: {name!r}")
    return name


def load_env_file(dotenv_path: str | Path = ".env") -> None:
    # Read key-value pairs from a .env file into os.environ if not already set.
    p = Path(dotenv_path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'\"")
        if key not in os.environ:
            os.environ[key] = val


def get_driver(
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    env_path: str | Path = ".env",
) -> Any:
    # Create a Neo4j driver instance from credentials or environment variables.
    load_env_file(env_path)
    uri = uri or os.getenv("NEO4J_URI")
    user = user or os.getenv("NEO4J_USER", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD")

    if not uri or not password:
        raise ValueError("NEO4J_URI and NEO4J_PASSWORD must be set in environment or provided directly.")

    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError(
            "Paket 'neo4j' belum terpasang. Pasang dengan: pip install neo4j"
        ) from e

    return GraphDatabase.driver(uri, auth=(user, password))


def export_graph(kg: KnowledgeGraph, driver: Any, replace: bool = False) -> None:
    # Export KnowledgeGraph entities and triples to Neo4j using idempotent MERGE.
    # All nodes receive the specific entity type label and a shared :KGEntity label.
    # An index constraint is ensured on :KGEntity(id).
    #
    # MERGE never deletes, so entities/triples removed from the ontology stay in the
    # database. replace=True deletes every :KGEntity node first (nodes without that
    # label are untouched), which is what re-exporting a new ontology version needs.
    try:
        import neo4j  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Paket 'neo4j' belum terpasang. Pasang dengan: pip install neo4j"
        ) from e

    entities_by_type = defaultdict(list)
    for e in kg.entities.values():
        entities_by_type[e.type].append({
            "id": e.id,
            "label": e.label,
            "abstract": e.abstract or "",
        })

    triples_by_rel = defaultdict(list)
    for subj_id, rel, obj_id in kg.triples:
        triples_by_rel[rel].append({"subj_id": subj_id, "obj_id": obj_id})

    with driver.session() as session:
        # Create unique constraint on :KGEntity(id) to ensure fast lookup and data integrity.
        constraint_query = "CREATE CONSTRAINT IF NOT EXISTS FOR (n:KGEntity) REQUIRE n.id IS UNIQUE"
        session.run(constraint_query).consume()

        if replace:
            session.run("MATCH (n:KGEntity) DETACH DELETE n").consume()

        # 1. Export nodes with shared :KGEntity label and specific type label.
        for node_type, items in entities_by_type.items():
            clean_type = _validate_identifier(node_type)
            query = (
                f"UNWIND $batch AS row "
                f"MERGE (n:KGEntity:{clean_type} {{id: row.id}}) "
                f"SET n.label = row.label, n.abstract = row.abstract"
            )
            session.run(query, batch=items).consume()

        # 2. Export relationships matching on :KGEntity label using indexed id.
        for rel_name, items in triples_by_rel.items():
            clean_rel = _validate_identifier(rel_name)
            query = (
                f"UNWIND $batch AS row "
                f"MATCH (s:KGEntity {{id: row.subj_id}}) "
                f"MATCH (o:KGEntity {{id: row.obj_id}}) "
                f"MERGE (s)-[r:{clean_rel}]->(o)"
            )
            session.run(query, batch=items).consume()


def gds_embeddings(
    driver: Any,
    method: str = "fastRP",
    dim: int = 32,
    seed: int = 42,
    relation_types: list[str] | tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    # Compute node embeddings via Neo4j Graph Data Science (GDS).
    #
    # Projects only :KGEntity nodes and relationships as UNDIRECTED to match
    # kgdbn.embedding.node2vec.
    try:
        import neo4j  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Paket 'neo4j' belum terpasang. Pasang dengan: pip install neo4j"
        ) from e

    method_clean = method.strip().lower()
    if method_clean in ("fastrp", "fast_rp"):
        stream_proc = "gds.fastRP.stream"
    elif method_clean == "node2vec":
        stream_proc = "gds.node2vec.stream"
    else:
        raise ValueError(f"Unknown GDS embedding method: {method!r}. Expected 'fastRP' or 'node2vec'.")

    graph_name = f"kgdbn_{uuid.uuid4().hex[:8]}"

    with driver.session() as session:
        # Project only :KGEntity nodes with undirected relationships
        rel_spec = {r: {"type": r, "orientation": "UNDIRECTED"} for r in relation_types} if relation_types else {
            "__ALL__": {"type": "*", "orientation": "UNDIRECTED"}
        }
        project_query = (
            "CALL gds.graph.project($name, 'KGEntity', $rel_spec)"
        )
        session.run(project_query, name=graph_name, rel_spec=rel_spec).consume()

        try:
            query = (
                f"CALL {stream_proc}($name, {{embeddingDimension: $dim, randomSeed: $seed}}) "
                f"YIELD nodeId, embedding "
                f"RETURN gds.util.asNode(nodeId).id AS id, embedding"
            )
            records = session.run(query, name=graph_name, dim=dim, seed=seed)
            embeddings: dict[str, np.ndarray] = {}
            for record in records:
                node_id = record["id"]
                if node_id is not None:
                    embeddings[str(node_id)] = np.asarray(record["embedding"], dtype=np.float32)
            return embeddings
        finally:
            # Clean up temporary GDS projected graph
            session.run("CALL gds.graph.drop($name, false)", name=graph_name).consume()
