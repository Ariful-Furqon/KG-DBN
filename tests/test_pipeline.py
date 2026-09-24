import numpy as np
import pytest

from kgdbn import DBN, kg_features, load_cases, load_ontology, multi_hot, node2vec
from kgdbn.experiment import DEFAULT_ONTOLOGY, run


@pytest.fixture(scope="module")
def kg():
    return load_ontology(DEFAULT_ONTOLOGY)


@pytest.fixture
def cases_csv(tmp_path):
    # Small hand-written fixture: two rows per class, symptoms taken from the ontology.
    rows = [
        ("Lesi_pada_Daun;Lesi_pada_batang", "Blas"),
        ("Lesi pada Malai;Pembentukan Sporulasi", "blas"),
        ("Kerapuhan_pelepah;Pembusukan_pelepah", "BusukPelepah"),
        ("Lesi_pada_pelepah;Perubahan_warna_pelepah", "BusukPelepah"),
        ("Batang_berlubang;Tanaman_terdapat_pupa", "PenggerekBatangUngu"),
        ("Batang_berlubang;Layu", "PenggerekBatangUngu"),
    ]
    path = tmp_path / "kasus.csv"
    path.write_text("gejala,label\n" + "\n".join(f"{g},{l}" for g, l in rows), encoding="utf-8")
    return path


def test_ontology_loads_and_drops_invalid_triples(kg):
    assert len(kg.of_type("PenyakitPadi")) == 19
    assert len(kg.of_type("HamaPadi")) == 27
    assert len(kg.of_type("Gejala")) == 41
    assert ("Lesi_pada_bulir", "memilikiGejala", "Lesi_pada_bulir") in kg.dropped
    assert "Lesi_pada_Daun" in kg.symptom_profiles()["Blas"]
    assert "Tungro" not in kg.symptom_profiles()


def test_node2vec_shapes(kg):
    emb = node2vec(kg, dim=8, num_walks=2, walk_length=5, epochs=1)
    assert set(emb) == set(kg.entities)
    assert all(v.shape == (8,) for v in emb.values())


def test_load_cases_resolves_ids_and_labels(kg, cases_csv):
    cases = load_cases(cases_csv, kg)
    assert cases["label"].tolist().count("Blas") == 2
    assert cases.loc[1, "symptoms"] == ["Lesi_pada_Malai", "Pembentukan_Sporulasi"]
    X = multi_hot(cases, kg.of_type("Gejala"))
    assert X.shape == (6, 41) and X.sum() == 12


def test_load_cases_reports_unknown_names(kg, tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("gejala,label\nDaun_ungu,Blas\nLayu,PenyakitX\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(?s)Daun_ungu.*PenyakitX"):
        load_cases(path, kg)


@pytest.mark.parametrize("pretrain", [True, False])
@pytest.mark.parametrize("visible", ["bernoulli", "gaussian"])
def test_dbn_learns_separable_classes(pretrain, visible):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    X = np.eye(3)[y].repeat(4, axis=1) + rng.normal(0, 0.05, (300, 12))
    if visible == "bernoulli":
        X = X.clip(0, 1)
    model = DBN((16, 8), visible=visible, pretrain=pretrain, pretrain_epochs=5, finetune_epochs=100, seed=0)
    model.fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95
    assert model.predict_proba(X).shape == (300, 3)
    assert len(model.history_.get("pretrain", [])) == (2 if pretrain else 0)


def test_run_end_to_end(cases_csv):
    results = run(cases_csv, embedding_dim=8, hidden_layers=(8,), test_size=0.5, verbose=False)
    assert set(results["fitur"]) == {"multi-hot", "KG", "multi-hot + KG"}
    assert results["accuracy"].between(0, 1).all()
