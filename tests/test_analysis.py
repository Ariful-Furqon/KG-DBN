import pytest

from kgdbn import load_ontology, node2vec
from kgdbn.analysis import VARIANTS, embedding_structure, structure_report, symptom_pairs
from kgdbn.experiment import DEFAULT_ONTOLOGY


@pytest.fixture(scope="module")
def kg():
    return load_ontology(DEFAULT_ONTOLOGY)


def test_structure_report_matches_manuscript(kg):
    r = structure_report(kg)
    assert r["diagnosable_targets"] == 41
    assert r["targets_without_symptoms"] == ["JelagaPalsu", "Meloidogyne_spp.", "Tungro", "UlatTandukHijau", "WalangSangit"]
    assert r["identical_pairs"] == [("BelalangSawah", "WerengPunggungPutih"), ("Thrips", "TungauMalaiPadi")]
    assert len(r["nested_targets"]) == 17
    assert r["ceiling_complete_observation"] == pytest.approx(39 / 41)
    assert (r["nodes"], r["edges"], r["components"]) == (144, 247, [143, 1])


def test_symptom_pairs(kg):
    pairs = symptom_pairs(kg)
    assert len(pairs) == 41 * 40 // 2
    # Linked pairs are by definition disjoint in their target sets.
    assert (pairs.loc[pairs["linked"], "jaccard"] == 0).all()
    assert set(pairs.loc[pairs["linked"], "via"].str.split("+").explode()) <= {"pathogen", "control", "pest-disease"}


def test_symptom_only_variant_keeps_only_symptom_edges(kg):
    exclude = VARIANTS["symptom-only"](kg)
    assert "memilikiGejala" not in exclude
    assert {r for _, r, _ in kg.triples} - set(exclude) == {"memilikiGejala"}


def test_embedding_structure_keys(kg):
    emb = node2vec(kg, dim=8, num_walks=2, walk_length=5, epochs=1)
    result = embedding_structure(emb, symptom_pairs(kg))
    assert len(result["cosines"]) == 820
    assert -1 <= result["rho_jaccard"] <= 1
