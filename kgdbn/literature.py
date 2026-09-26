# Literature-derived cases: extraction sheet -> validated, filtered `kasus.csv`.
#
# python -m kgdbn.literature --ekstraksi data/ekstraksi_kasus.csv --sumber data/sumber.csv --out data/kasus.csv
#
# The extraction sheet columns follow templates/ekstraksi_kasus.csv and
# templates/sumber.csv; the selection rules are in select_cases.

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .cases import _key, _resolver
from .ontology import KnowledgeGraph, load_ontology

EXTRACTION_COLUMNS = [
    "id_kasus", "id_sumber", "halaman", "jenis_kasus", "konfirmasi_label", "dipakai_ontologi",
    "lokasi", "tahun_observasi", "gejala_asli", "label_asli", "gejala_a1", "gejala_a2",
    "gejala_final", "gejala_tak_terpetakan", "label", "catatan",
]
SOURCE_COLUMNS = [
    "id_sumber", "sitasi", "jenis_sumber", "url_doi", "basis_data", "query", "tahap", "alasan_eksklusi", "berkas",
]

# `vinyet_pakar`: a disease profile (rule / knowledge-base row) written by an expert
# independent of the ontology. Kept, but reported and evaluated apart from observed cases.
CASE_TYPES_KEPT = {"lapangan", "uji_pakar", "laporan_penyakit", "vinyet_pakar"}
CASE_TYPES_DROPPED = {"uji_acak"}
ALLOWED = {
    "jenis_kasus": CASE_TYPES_KEPT | CASE_TYPES_DROPPED,
    "konfirmasi_label": {"laboratorium", "pakar", "penulis", "tidak_ada"},
    "dipakai_ontologi": {"ya", "tidak"},
}
STAGES = ["ditemukan", "disaring", "layak", "dimasukkan"]


def _split(text: str, sep: str = ";") -> list[str]:
    return [s.strip() for s in str(text).split(sep) if s.strip()]


def _read(path: str | Path, columns: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in columns if c not in raw.columns]
    if missing:
        raise ValueError(f"Kolom hilang di {path}: {missing}")
    return raw.apply(lambda col: col.str.strip())


def _plain(text: str) -> str:
    # Lowercase alphanumerics only, so PDF line breaks, hyphenation and spacing never matter.
    return re.sub(r"[^0-9a-z]", "", str(text).lower())


def load_source_texts(sources: pd.DataFrame, base_dir: str | Path) -> dict[str, str]:
    # {id_sumber: plain text of `berkas`} for every source whose text file exists.
    #
    # `berkas` is the plain-text extraction of the source (e.g. `pdftotext`),
    # relative to `base_dir` (the folder of sumber.csv).
    texts = {}
    for sid, berkas in zip(sources["id_sumber"], sources["berkas"]):
        path = Path(base_dir) / berkas
        if berkas and path.is_file():
            texts[sid] = _plain(path.read_text(encoding="utf-8", errors="ignore"))
    return texts


def missing_verbatim(row: dict, texts: dict[str, str]) -> list[str] | None:
    # Items of `gejala_asli` not found in the source text; None when the source has no text.
    text = texts.get(row["id_sumber"])
    if text is None:
        return None
    return [g for g in _split(row["gejala_asli"]) if _plain(g) not in text]


def resolve_extraction(raw: pd.DataFrame, kg: KnowledgeGraph, target_types=("PenyakitPadi", "HamaPadi")) -> pd.DataFrame:
    # Validate enum columns and map every symptom/label name to its ontology id.
    #
    # Adds `a1`, `a2`, `final` (sorted id lists; `final` is None when A1 and A2
    # disagree and `gejala_final` is empty) and `label_id`. Raises ValueError
    # listing every bad value with its CSV line.
    symptoms_of, labels_of = _resolver(kg, ["Gejala"]), _resolver(kg, target_types)
    errors = []

    def ids(text, column, line):
        out = set()
        for name in _split(text):
            eid = symptoms_of.get(_key(name))
            if eid is None:
                errors.append(f"{column}: gejala '{name}' tidak ada di ontologi (baris {line})")
            else:
                out.add(eid)
        return sorted(out)

    rows = []
    for line, row in enumerate(raw.to_dict("records"), start=2):
        for column, allowed in ALLOWED.items():
            if row[column] not in allowed:
                errors.append(f"{column}: nilai '{row[column]}' tidak valid, pilih {sorted(allowed)} (baris {line})")
        a1, a2 = ids(row["gejala_a1"], "gejala_a1", line), ids(row["gejala_a2"], "gejala_a2", line)
        manual = ids(row["gejala_final"], "gejala_final", line)
        if manual:
            final = manual
        elif not row["gejala_a2"] or a1 == a2:
            final = a1
        else:
            final = None
        label = labels_of.get(_key(row["label"]))
        if label is None:
            errors.append(f"label: '{row['label']}' tidak ada di ontologi (baris {line})")
        rows.append({**row, "a1": a1, "a2": a2, "final": final, "label_id": label})

    if errors:
        raise ValueError("Lembar ekstraksi tidak valid:\n  " + "\n  ".join(errors))
    return pd.DataFrame(rows, columns=[*raw.columns, "a1", "a2", "final", "label_id"])


def symptom_kappa(resolved: pd.DataFrame, vocabulary: list[str]) -> tuple[float | None, int]:
    # Cohen's kappa of A1 vs A2 over the (case x Gejala) presence matrix.
    #
    # Only cases mapped by both annotators count. Returns (kappa, n_cases).
    both = resolved[(resolved["gejala_a2"] != "") & (resolved["gejala_a1"] != "")]
    if both.empty:
        return None, 0
    index = {s: i for i, s in enumerate(vocabulary)}

    def matrix(column):
        X = np.zeros((len(both), len(vocabulary)), dtype=int)
        for r, symptoms in enumerate(both[column]):
            X[r, [index[s] for s in symptoms]] = 1
        return X.ravel()

    return float(cohen_kappa_score(matrix("a1"), matrix("a2"))), len(both)


def select_cases(
    resolved: pd.DataFrame, kg: KnowledgeGraph, texts: dict[str, str] | None = None
) -> tuple[pd.DataFrame, Counter]:
    # Apply the selection rules: drop cases of a dropped `jenis_kasus` (`uji_acak`),
    # sources used to build the ontology, rows whose `gejala_asli` is not verbatim
    # in the source, rows awaiting adjudication or without mapped symptoms, and
    # duplicates within a source.
    #
    # With `texts` (from load_source_texts) every `gejala_asli` item must appear
    # verbatim in its source text, otherwise the row is dropped.
    #
    # Returns (kept cases, Counter of exclusion reasons). The first matching
    # reason wins, so each dropped case is counted once. Kept cases are flagged
    # `sama_profil_ontologi` (identical to the ontology profile of their label) and
    # `duplikat_lintas_sumber` (same symptoms and label in another source, e.g. a
    # rule table copied between theses); both are reported, not dropped.
    profiles = {t: sorted(s) for t, s in kg.symptom_profiles().items()}
    reasons, kept, seen = Counter(), [], set()
    for row in resolved.to_dict("records"):
        if row["jenis_kasus"] in CASE_TYPES_DROPPED:
            reason = f"jenis_kasus={row['jenis_kasus']}"
        elif row["dipakai_ontologi"] == "ya":
            reason = "sumber dipakai ontologi"
        elif texts is not None and missing_verbatim(row, texts) is None:
            reason = "berkas sumber tidak ada"
        elif texts is not None and missing_verbatim(row, texts):
            reason = "gejala_asli tidak ada di berkas sumber"
        elif row["final"] is None:
            reason = "perlu adjudikasi (A1 != A2)"
        elif not row["final"]:
            reason = "tanpa gejala terpetakan"
        elif (row["id_sumber"], tuple(row["final"]), row["label_id"]) in seen:
            reason = "duplikat dalam sumber"
        else:
            seen.add((row["id_sumber"], tuple(row["final"]), row["label_id"]))
            kept.append({**row, "sama_profil_ontologi": row["final"] == profiles.get(row["label_id"])})
            continue
        reasons[reason] += 1
    kept = pd.DataFrame(kept)
    if len(kept):
        key = kept["final"].map(tuple) + kept["label_id"].map(lambda l: (l,))
        kept["duplikat_lintas_sumber"] = key.map(kept.groupby(key)["id_sumber"].nunique()) > 1
    return kept, reasons


def export_cases(kept: pd.DataFrame, path: str | Path) -> None:
    # Write the `load_cases` format plus provenance columns.
    out = pd.DataFrame({
        "gejala": [";".join(s) for s in kept["final"]],
        "label": kept["label_id"],
        "id_kasus": kept["id_kasus"],
        "id_sumber": kept["id_sumber"],
        "jenis_kasus": kept["jenis_kasus"],
        "konfirmasi_label": kept["konfirmasi_label"],
        "sama_profil_ontologi": kept["sama_profil_ontologi"],
        "duplikat_lintas_sumber": kept["duplikat_lintas_sumber"],
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def prisma_counts(sources: pd.DataFrame) -> dict[str, int]:
    # Number of sources that reached at least each stage.
    bad = sorted(set(sources["tahap"]) - set(STAGES))
    if bad:
        raise ValueError(f"tahap tidak valid: {bad}, pilih {STAGES}")
    missing_reason = sources[(sources["tahap"] != "dimasukkan") & (sources["alasan_eksklusi"] == "")]
    if not missing_reason.empty:
        raise ValueError(f"alasan_eksklusi kosong untuk sumber: {list(missing_reason['id_sumber'])}")
    rank = sources["tahap"].map(STAGES.index)
    return {stage: int((rank >= i).sum()) for i, stage in enumerate(STAGES)}


def report(resolved, kept, reasons, kappa, sources=None, texts=None) -> str:
    lines = [f"Kasus diekstraksi: {len(resolved)}", f"Kasus dipakai: {len(kept)}"]
    lines += [f"  dibuang ({r}): {n}" for r, n in reasons.most_common()]
    if texts is not None:
        unverified = [(row["id_kasus"], missing_verbatim(row, texts)) for row in resolved.to_dict("records")]
        unverified = [(i, m) for i, m in unverified if m]
        if unverified:
            lines.append("Gejala tidak verbatim (maks. 15 kasus):")
            lines += [f"  {i}: {m}" for i, m in unverified[:15]]
    k, n = kappa
    lines.append(f"Kappa A1/A2: {k:.3f} ({n} kasus)" if k is not None else "Kappa A1/A2: - (belum ada anotasi A2)")
    if len(kept):
        lines.append(f"Sumber terwakili: {kept['id_sumber'].nunique()}")
        lines.append("Per jenis_kasus:")
        lines += [f"  {t}: {n}" for t, n in Counter(kept["jenis_kasus"]).most_common()]
        lines.append(f"Sama persis dengan profil ontologi: {int(kept['sama_profil_ontologi'].sum())}")
        lines.append(f"Duplikat lintas sumber: {int(kept['duplikat_lintas_sumber'].sum())}")
        lines.append("Distribusi label:")
        lines += [f"  {label}: {n}" for label, n in Counter(kept["label_id"]).most_common()]
    unmapped = Counter(t.lower() for text in resolved["gejala_tak_terpetakan"] for t in _split(text))
    if unmapped:
        lines.append("Gejala tak terpetakan terbanyak (bahan P1):")
        lines += [f"  {t}: {n}" for t, n in unmapped.most_common(10)]
    if sources is not None:
        lines.append("PRISMA (sumber):")
        lines += [f"  {stage}: {n}" for stage, n in prisma_counts(sources).items()]
    return "\n".join(lines)


def main():
    from .experiment import DEFAULT_ONTOLOGY

    parser = argparse.ArgumentParser(description="Lembar ekstraksi literatur -> kasus.csv tervalidasi.")
    parser.add_argument("--ekstraksi", required=True, help="CSV ekstraksi (templates/ekstraksi_kasus.csv)")
    parser.add_argument("--sumber", help="CSV daftar sumber (templates/sumber.csv), untuk hitungan PRISMA")
    parser.add_argument("--out", default="data/kasus.csv")
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    args = parser.parse_args()

    kg = load_ontology(args.ontology)
    resolved = resolve_extraction(_read(args.ekstraksi, EXTRACTION_COLUMNS), kg)
    sources, texts = None, None
    if args.sumber:
        sources = _read(args.sumber, SOURCE_COLUMNS)
        texts = load_source_texts(sources, Path(args.sumber).parent)
    else:
        print("Peringatan: tanpa --sumber, gejala_asli tidak diperiksa terhadap teks sumber.")
    kept, reasons = select_cases(resolved, kg, texts)
    print(report(resolved, kept, reasons, symptom_kappa(resolved, kg.of_type("Gejala")), sources, texts))
    if len(kept):
        export_cases(kept, args.out)
        print(f"Ditulis: {args.out}")


if __name__ == "__main__":
    main()
