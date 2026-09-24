# Diagnosis cases (observed symptoms -> pest/disease) and their feature encodings.
#
# Case file format (CSV, one row per observed case):
#
#     gejala,label
#     Lesi_pada_Daun;Lesi_pada_batang,Blas
#     Daun menguning;Layu,WerengBatangCoklat
#
# `gejala` is a `;`-separated list of `Gejala` individuals and `label` is a
# `PenyakitPadi` or `HamaPadi` individual. Both may be written as the ontology id
# or its rdfs:label; matching ignores case, spaces and underscores.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ontology import KnowledgeGraph


def _key(name: str) -> str:
    return "".join(str(name).lower().replace("_", " ").split())


def _resolver(kg: KnowledgeGraph, types) -> dict[str, str]:
    lookup = {}
    for eid in kg.of_type(*types):
        lookup[_key(eid)] = eid
        lookup[_key(kg.entities[eid].label)] = eid
    return lookup


def load_cases(
    path: str | Path,
    kg: KnowledgeGraph,
    target_types=("PenyakitPadi", "HamaPadi"),
    symptom_column: str = "gejala",
    label_column: str = "label",
    sep: str = ";",
) -> pd.DataFrame:
    # Read a case CSV and map names to ontology ids.
    #
    # Returns columns `symptoms` (sorted list of ids) and `label` (id). Raises
    # ValueError listing every symptom or label not found in the ontology.
    raw = pd.read_csv(path)
    symptoms_of, labels_of = _resolver(kg, ["Gejala"]), _resolver(kg, target_types)

    rows, unknown_symptoms, unknown_labels = [], set(), set()
    for line, (symptom_text, label_text) in enumerate(zip(raw[symptom_column], raw[label_column]), start=2):
        names = [s.strip() for s in str(symptom_text).split(sep) if s.strip()]
        ids = [symptoms_of.get(_key(s)) for s in names]
        unknown_symptoms.update(f"{s} (baris {line})" for s, i in zip(names, ids) if i is None)
        label = labels_of.get(_key(label_text))
        if label is None:
            unknown_labels.add(f"{label_text} (baris {line})")
        if not ids:
            unknown_symptoms.add(f"<kosong> (baris {line})")
        rows.append({"symptoms": sorted({i for i in ids if i}), "label": label})

    if unknown_symptoms or unknown_labels:
        raise ValueError(
            "Nama tidak ditemukan di ontologi.\n"
            f"  Gejala: {sorted(unknown_symptoms)}\n"
            f"  Label: {sorted(unknown_labels)}"
        )
    return pd.DataFrame(rows)


def multi_hot(cases: pd.DataFrame, vocabulary: list[str]) -> np.ndarray:
    index = {s: i for i, s in enumerate(vocabulary)}
    X = np.zeros((len(cases), len(vocabulary)), dtype=np.float32)
    for row, symptoms in enumerate(cases["symptoms"]):
        X[row, [index[s] for s in symptoms]] = 1.0
    return X


def kg_features(cases: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> np.ndarray:
    # Mean KG embedding of the observed symptoms.
    return np.stack([np.mean([embeddings[s] for s in symptoms], axis=0) for symptoms in cases["symptoms"]]).astype(
        np.float32
    )
