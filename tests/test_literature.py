import pandas as pd
import pytest

from kgdbn import load_cases, load_ontology
from kgdbn.experiment import DEFAULT_ONTOLOGY
from kgdbn.literature import (
    EXTRACTION_COLUMNS,
    SOURCE_COLUMNS,
    export_cases,
    load_source_texts,
    prisma_counts,
    resolve_extraction,
    select_cases,
    symptom_kappa,
)

TEMPLATES = DEFAULT_ONTOLOGY.parent.parent / "templates"


@pytest.fixture(scope="module")
def kg():
    return load_ontology(DEFAULT_ONTOLOGY)


def row(id_kasus, a1, label, a2="", final="", jenis="uji_pakar", sumber="S001", ontologi="tidak", tak=""):
    return {
        **{c: "" for c in EXTRACTION_COLUMNS},
        "id_kasus": id_kasus, "id_sumber": sumber, "jenis_kasus": jenis, "konfirmasi_label": "pakar",
        "dipakai_ontologi": ontologi, "gejala_asli": "teks asli", "gejala_a1": a1, "gejala_a2": a2,
        "gejala_final": final, "gejala_tak_terpetakan": tak, "label": label,
    }


def test_templates_match_columns():
    assert list(pd.read_csv(TEMPLATES / "ekstraksi_kasus.csv").columns) == EXTRACTION_COLUMNS
    assert list(pd.read_csv(TEMPLATES / "sumber.csv").columns) == SOURCE_COLUMNS


def test_resolve_reports_every_bad_value(kg):
    raw = pd.DataFrame([
        row("S001-01", "Lesi_pada_Daun;GejalaPalsu", "Blas"),
        {**row("S001-02", "Layu", "BukanPenyakit"), "jenis_kasus": "entah"},
    ])
    with pytest.raises(ValueError) as err:
        resolve_extraction(raw, kg)
    message = str(err.value)
    assert "GejalaPalsu" in message and "baris 2" in message
    assert "BukanPenyakit" in message and "entah" in message and "baris 3" in message


def test_final_mapping_rules(kg):
    resolved = resolve_extraction(pd.DataFrame([
        row("a", "Lesi_pada_Daun", "Blas"),                                          # single annotator
        row("b", "Lesi_pada_Daun;Layu", "Blas", a2="layu;lesi pada daun"),            # agree, spelling differs
        row("c", "Lesi_pada_Daun", "Blas", a2="Layu"),                                # disagree
        row("d", "Lesi_pada_Daun", "Blas", a2="Layu", final="Layu;Lesi_pada_Daun"),   # adjudicated
    ]), kg)
    finals = list(resolved["final"])
    assert finals[0] == ["Lesi_pada_Daun"]
    assert finals[1] == ["Layu", "Lesi_pada_Daun"]
    assert finals[2] is None
    assert finals[3] == ["Layu", "Lesi_pada_Daun"]


def test_selection_drops_by_rule(kg):
    busuk = kg.symptom_profiles()["BusukPelepah"]
    resolved = resolve_extraction(pd.DataFrame([
        row("1", "Lesi_pada_Daun;Lesi_pada_batang", "Blas"),
        row("2", "Lesi_pada_Daun;Lesi_pada_batang", "Blas"),                          # duplicate within S001
        row("3", "Lesi_pada_Daun;Lesi_pada_batang", "Blas", sumber="S002"),           # same case, other source: kept
        row("4", "Layu", "Blas", jenis="vinyet_pakar", sumber="S003"),              # kept
        row("5", "Layu", "Blas", jenis="uji_acak"),
        row("6", "Layu", "Blas", ontologi="ya"),
        row("7", "Layu", "Blas", a2="Lesi_pada_Daun"),
        row("8", "", "Blas", tak="bau apek"),
        row("9", ";".join(busuk), "BusukPelepah"),
    ]), kg)
    kept, reasons = select_cases(resolved, kg)
    assert list(kept["id_kasus"]) == ["1", "3", "4", "9"]
    assert list(kept["sama_profil_ontologi"]) == [False, False, False, True]
    assert list(kept["duplikat_lintas_sumber"]) == [True, True, False, False]
    assert reasons == {
        "duplikat dalam sumber": 1, "jenis_kasus=uji_acak": 1,
        "sumber dipakai ontologi": 1, "perlu adjudikasi (A1 != A2)": 1, "tanpa gejala terpetakan": 1,
    }


def test_kappa(kg):
    vocab = kg.of_type("Gejala")
    same = resolve_extraction(pd.DataFrame([
        row("1", "Lesi_pada_Daun;Layu", "Blas", a2="Lesi_pada_Daun;Layu"),
        row("2", "Batang_berlubang", "PenggerekBatangUngu", a2="Batang_berlubang"),
    ]), kg)
    assert symptom_kappa(same, vocab) == (pytest.approx(1.0), 2)
    partial = resolve_extraction(pd.DataFrame([
        row("1", "Lesi_pada_Daun;Layu", "Blas", a2="Lesi_pada_Daun"),
        row("2", "Batang_berlubang", "PenggerekBatangUngu"),
    ]), kg)
    kappa, n = symptom_kappa(partial, vocab)
    assert n == 1 and 0 < kappa < 1


def test_export_round_trips_through_load_cases(kg, tmp_path):
    resolved = resolve_extraction(pd.DataFrame([
        row("1", "Lesi pada Daun;Lesi_pada_batang", "blas"),
        row("2", "Batang_berlubang;Layu", "PenggerekBatangUngu"),
    ]), kg)
    kept, _ = select_cases(resolved, kg)
    path = tmp_path / "kasus.csv"
    export_cases(kept, path)
    cases = load_cases(path, kg)
    assert list(cases["label"]) == ["Blas", "PenggerekBatangUngu"]
    assert cases["symptoms"][0] == ["Lesi_pada_Daun", "Lesi_pada_batang"]
    assert list(pd.read_csv(path)["id_sumber"]) == ["S001", "S001"]


def test_prisma_counts():
    sources = pd.DataFrame({
        "id_sumber": ["S1", "S2", "S3", "S4"],
        "tahap": ["dimasukkan", "layak", "disaring", "ditemukan"],
        "alasan_eksklusi": ["", "tanpa kasus", "hanya citra", "duplikat"],
    })
    assert prisma_counts(sources) == {"ditemukan": 4, "disaring": 3, "layak": 2, "dimasukkan": 1}
    with pytest.raises(ValueError, match="S2"):
        prisma_counts(sources.assign(alasan_eksklusi=["", "", "x", "y"]))


def test_verbatim_check_against_source_text(kg, tmp_path):
    (tmp_path / "s1.txt").write_text(
        "Tabel 4. G001 Daun pucuk tanaman layu ringan 0,15\nG013 Pada pangkal batang terdapat bekas\ngerekan larva(ulat)",
        encoding="utf-8",
    )
    sources = pd.DataFrame({"id_sumber": ["S001", "S002"], "berkas": ["s1.txt", "hilang.txt"]})
    texts = load_source_texts(sources, tmp_path)
    assert set(texts) == {"S001"}
    raw = pd.DataFrame([
        # Line break, spacing and punctuation differences are tolerated.
        {**row("1", "Layu", "Blas"), "gejala_asli": "Daun pucuk tanaman layu;Pada pangkal batang terdapat bekas gerekan larva (ulat)"},
        {**row("2", "Layu", "Blas"), "gejala_asli": "Daun pucuk tanaman layu;Malai hampa berwarna putih tegak"},
        {**row("3", "Layu", "Blas", sumber="S002"), "gejala_asli": "Daun pucuk tanaman layu"},
    ])
    kept, reasons = select_cases(resolve_extraction(raw, kg), kg, texts)
    assert list(kept["id_kasus"]) == ["1"]
    assert reasons == {"gejala_asli tidak ada di berkas sumber": 1, "berkas sumber tidak ada": 1}
