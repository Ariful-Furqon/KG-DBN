# Load the rice pest/disease ontology (OWL, RDF/XML) into a knowledge graph.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import rdflib
from rdflib.namespace import OWL, RDF, RDFS

ABSTRACT = rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#abstract")

# Types allowed as subject/object of each object property. The ontology v1.1
# declares every domain as `Padi`, which is wrong, so validation uses this table
# instead of rdfs:domain/rdfs:range.
TARGET_TYPES = {"PenyakitPadi", "HamaPadi"}
CONTROL_TYPES = {"Fungisida", "Bakterisida", "Pestisida", "Biologis", "Kimia", "KontrolTerintegrasi", "Kultural"}
SCHEMA = {
    "memilikiGejala": (TARGET_TYPES, {"Gejala"}),
    "terkenaPatogen": ({"PenyakitPadi"}, {"PatogenPadi"}),
    "menyebabkanPenyakit": ({"HamaPadi"}, {"PenyakitPadi"}),
    "terserangHama": ({"PenyakitPadi"}, {"HamaPadi"}),
    "diberikanFungisida": (TARGET_TYPES, {"Fungisida"}),
    "diberikanBakterisida": (TARGET_TYPES, {"Bakterisida"}),
    "diberikanPestisida": (TARGET_TYPES, {"Pestisida"}),
    "diberikanAgenBiologi": (TARGET_TYPES, {"Biologis"}),
    "diberikanPengendalianHama": (TARGET_TYPES, CONTROL_TYPES),
}


def local_name(uri) -> str:
    return str(uri).split("#")[-1]


@dataclass
class Entity:
    id: str
    type: str
    label: str
    abstract: str = ""


@dataclass
class KnowledgeGraph:
    entities: dict[str, Entity]
    triples: list[tuple[str, str, str]]
    dropped: list[tuple[str, str, str]] = field(default_factory=list)

    def of_type(self, *types: str) -> list[str]:
        return sorted(e.id for e in self.entities.values() if e.type in types)

    def objects(self, subject: str, relation: str) -> list[str]:
        return sorted(o for s, r, o in self.triples if s == subject and r == relation)

    def symptom_profiles(self, target_types=("PenyakitPadi", "HamaPadi")) -> dict[str, list[str]]:
        # Symptoms of every target entity that has at least one symptom.
        profiles = {t: self.objects(t, "memilikiGejala") for t in self.of_type(*target_types)}
        return {t: s for t, s in profiles.items() if s}

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for e in self.entities.values():
            counts[e.type] = counts.get(e.type, 0) + 1
        lines = [f"{len(self.entities)} entitas, {len(self.triples)} triple valid, {len(self.dropped)} dibuang"]
        lines += [f"  {t}: {n}" for t, n in sorted(counts.items())]
        return "\n".join(lines)


def load_ontology(path: str | Path) -> KnowledgeGraph:
    # Parse individuals and their object-property triples.
    #
    # Triples that violate SCHEMA or are self-loops are moved to `dropped`.
    g = rdflib.Graph()
    g.parse(str(path))

    entities: dict[str, Entity] = {}
    for ind in g.subjects(RDF.type, OWL.NamedIndividual):
        types = [local_name(t) for t in g.objects(ind, RDF.type) if t != OWL.NamedIndividual]
        if len(types) != 1:
            continue
        eid = local_name(ind)
        entities[eid] = Entity(
            id=eid,
            type=types[0],
            label=str(g.value(ind, RDFS.label) or eid.replace("_", " ")),
            abstract=str(g.value(ind, ABSTRACT) or ""),
        )

    triples, dropped = [], []
    for s, p, o in g:
        # Literals (e.g. an rdfs:label equal to an entity id) are not relations.
        if not isinstance(o, rdflib.URIRef):
            continue
        s_id, rel, o_id = local_name(s), local_name(p), local_name(o)
        if s_id not in entities or o_id not in entities:
            continue
        triple = (s_id, rel, o_id)
        allowed = SCHEMA.get(rel)
        valid = (
            allowed is not None
            and s_id != o_id
            and entities[s_id].type in allowed[0]
            and entities[o_id].type in allowed[1]
        )
        (triples if valid else dropped).append(triple)

    return KnowledgeGraph(entities, sorted(triples), sorted(dropped))
