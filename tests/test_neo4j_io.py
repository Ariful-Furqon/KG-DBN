import os
import sys
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kgdbn.ontology import Entity, KnowledgeGraph
from kgdbn.neo4j_io import export_graph, gds_embeddings, get_driver, load_env_file


@pytest.fixture
def sample_kg():
    entities = {
        "Blas": Entity(id="Blas", type="PenyakitPadi", label="Penyakit Blas", abstract="Penyakit jamur"),
        "Gejala_Daun": Entity(id="Gejala_Daun", type="Gejala", label="Lesi Daun", abstract=""),
    }
    triples = [("Blas", "memilikiGejala", "Gejala_Daun")]
    return KnowledgeGraph(entities=entities, triples=triples)


def test_import_error_when_neo4j_missing(sample_kg):
    # Hide neo4j module if present
    with patch.dict(sys.modules, {"neo4j": None}):
        with pytest.raises(ImportError, match="Paket 'neo4j' belum terpasang"):
            export_graph(sample_kg, MagicMock())

        with pytest.raises(ImportError, match="Paket 'neo4j' belum terpasang"):
            gds_embeddings(MagicMock())

        with pytest.raises(ImportError, match="Paket 'neo4j' belum terpasang"):
            get_driver(uri="bolt://localhost:7687", user="neo4j", password="secret")


def test_export_graph_mock(sample_kg):
    with patch.dict(sys.modules, {"neo4j": MagicMock()}):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session

        export_graph(sample_kg, mock_driver)

        # 1 constraint + 2 node types + 1 relation type
        assert mock_session.run.call_count == 4
        queries = [call.args[0] for call in mock_session.run.call_args_list]
        assert any("CREATE CONSTRAINT IF NOT EXISTS FOR (n:KGEntity)" in q for q in queries)
        assert any("MERGE (n:KGEntity:PenyakitPadi {id: row.id})" in q for q in queries)
        assert any("MERGE (n:KGEntity:Gejala {id: row.id})" in q for q in queries)
        assert any("MATCH (s:KGEntity {id: row.subj_id})" in q and "MERGE (s)-[r:memilikiGejala]->(o)" in q for q in queries)
        assert mock_result.consume.call_count == 4


@pytest.mark.parametrize("replace", [False, True])
def test_export_graph_replace_deletes_old_entities_first(sample_kg, replace):
    with patch.dict(sys.modules, {"neo4j": MagicMock()}):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        export_graph(sample_kg, mock_driver, replace=replace)

        queries = [call.args[0] for call in mock_session.run.call_args_list]
        deletes = [i for i, q in enumerate(queries) if "DETACH DELETE" in q]
        merges = [i for i, q in enumerate(queries) if "MERGE" in q]
        if replace:
            assert len(deletes) == 1 and "MATCH (n:KGEntity)" in queries[deletes[0]]
            assert deletes[0] < min(merges)
        else:
            assert deletes == []


def test_gds_embeddings_mock():
    with patch.dict(sys.modules, {"neo4j": MagicMock()}):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_project_result = MagicMock()
        mock_drop_result = MagicMock()
        stream_records = [
            {"id": "Blas", "embedding": [0.1] * 8},
            {"id": "Gejala_Daun", "embedding": [0.2] * 8},
        ]

        def fake_run(query, *args, **kwargs):
            if "gds.graph.project" in query:
                return mock_project_result
            if "gds.graph.drop" in query:
                return mock_drop_result
            return stream_records

        mock_session.run.side_effect = fake_run
        mock_driver.session.return_value.__enter__.return_value = mock_session

        emb = gds_embeddings(mock_driver, method="fastRP", dim=8, seed=42)
        assert set(emb.keys()) == {"Blas", "Gejala_Daun"}
        assert emb["Blas"].shape == (8,)
        assert emb["Blas"].dtype == np.float32

        queries = [call.args[0] for call in mock_session.run.call_args_list]
        assert any("CALL gds.graph.project($name, 'KGEntity', $rel_spec)" in q for q in queries)
        assert mock_project_result.consume.called
        assert mock_drop_result.consume.called

        with pytest.raises(ValueError, match="Unknown GDS embedding method"):
            gds_embeddings(mock_driver, method="invalid_method")


def test_load_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_NEO4J_VAR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_NEO4J_VAR=super_secret\n# comment\n", encoding="utf-8")
    load_env_file(env_file)
    assert os.getenv("TEST_NEO4J_VAR") == "super_secret"


# Live integration test: skipped if NEO4J_URI is not set
@pytest.mark.skipif(not os.getenv("NEO4J_URI"), reason="NEO4J_URI not set")
def test_neo4j_live_integration():
    # Unique ids per run, deleted in finally, so the real database is left untouched.
    prefix = f"test_{uuid.uuid4().hex[:8]}_"
    entities = {
        f"{prefix}Blas": Entity(id=f"{prefix}Blas", type="PenyakitPadi", label="Penyakit Blas", abstract="Penyakit jamur"),
        f"{prefix}Gejala_Daun": Entity(id=f"{prefix}Gejala_Daun", type="Gejala", label="Lesi Daun", abstract=""),
    }
    triples = [(f"{prefix}Blas", "memilikiGejala", f"{prefix}Gejala_Daun")]
    test_kg = KnowledgeGraph(entities=entities, triples=triples)

    driver = get_driver()
    try:
        export_graph(test_kg, driver)
        emb = gds_embeddings(driver, method="fastRP", dim=16)
        assert emb[f"{prefix}Blas"].shape == (16,)
    finally:
        with driver.session() as session:
            session.run(
                "MATCH (n:KGEntity) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
                prefix=prefix,
            ).consume()
        driver.close()
