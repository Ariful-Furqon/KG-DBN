"""End-to-end KG-DBN experiment: ontology -> KG embedding -> cases -> models -> metrics.

    python -m kgdbn.experiment --cases data/kasus.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler

from .cases import kg_features, load_cases, multi_hot
from .dbn import DBN
from .embedding import node2vec
from .ontology import load_ontology

DEFAULT_ONTOLOGY = Path(__file__).resolve().parent.parent / "Ontologi" / "OntologiHamaPenyakitPadi_v1.1.rdf"


def accuracy_ceiling(cases: pd.DataFrame) -> float:
    """Best accuracy any classifier can reach on `cases` given only the symptom set."""
    groups: dict[tuple, Counter] = {}
    for symptoms, label in zip(cases["symptoms"], cases["label"]):
        groups.setdefault(tuple(symptoms), Counter())[label] += 1
    return sum(c.most_common(1)[0][1] for c in groups.values()) / len(cases)


def build_features(train, test, vocabulary, embeddings):
    """Return {name: (X_train, X_test, visible_type)}."""
    mh_train, mh_test = multi_hot(train, vocabulary), multi_hot(test, vocabulary)
    scaler = StandardScaler().fit(kg_features(train, embeddings))
    kg_train, kg_test = scaler.transform(kg_features(train, embeddings)), scaler.transform(kg_features(test, embeddings))
    return {
        "multi-hot": (mh_train, mh_test, "bernoulli"),
        "KG": (kg_train, kg_test, "gaussian"),
        "multi-hot + KG": (np.hstack([mh_train, kg_train]), np.hstack([mh_test, kg_test]), "gaussian"),
    }


def run(
    cases_path,
    ontology=DEFAULT_ONTOLOGY,
    target_types=("PenyakitPadi", "HamaPadi"),
    embedding_dim=32,
    hidden_layers=(128, 64),
    test_size=0.2,
    seed=42,
    verbose=True,
) -> pd.DataFrame:
    log = print if verbose else (lambda *a, **k: None)

    kg = load_ontology(ontology)
    log(kg.summary())
    for triple in kg.dropped:
        log("  dibuang:", *triple)

    embeddings = node2vec(kg, dim=embedding_dim, seed=seed)
    cases = load_cases(cases_path, kg, target_types)
    stratify = cases["label"] if cases["label"].value_counts().min() >= 2 else None
    if stratify is None:
        log("Peringatan: ada kelas dengan < 2 kasus, split tidak di-stratify")
    train, test = train_test_split(cases, test_size=test_size, stratify=stratify, random_state=seed)
    log(f"\n{cases['label'].nunique()} kelas, {len(train)} kasus latih, {len(test)} kasus uji")
    log(f"Batas atas akurasi (test set, dari gejala saja): {accuracy_ceiling(test):.3f}\n")

    models = {
        "KG-DBN (RBM pretraining)": lambda visible: DBN(hidden_layers, visible=visible, seed=seed),
        "MLP (tanpa pretraining)": lambda visible: DBN(hidden_layers, visible=visible, pretrain=False, seed=seed),
        "Random Forest": lambda visible: RandomForestClassifier(300, random_state=seed, n_jobs=-1),
    }

    rows = []
    vocabulary = kg.of_type("Gejala")
    for feature, (X_train, X_test, visible) in build_features(train, test, vocabulary, embeddings).items():
        candidates = dict(models)
        if feature == "multi-hot":
            candidates["Bernoulli NB"] = lambda visible: BernoulliNB()
        for name, make in candidates.items():
            pred = make(visible).fit(X_train, train["label"]).predict(X_test)
            rows.append(
                {
                    "fitur": feature,
                    "model": name,
                    "accuracy": accuracy_score(test["label"], pred),
                    "f1_macro": f1_score(test["label"], pred, average="macro"),
                }
            )
            log(f"{feature:15s} | {name:25s} | acc {rows[-1]['accuracy']:.3f} | F1 {rows[-1]['f1_macro']:.3f}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", required=True, help="CSV kasus (kolom: gejala, label)")
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    parser.add_argument("--targets", nargs="+", default=["PenyakitPadi", "HamaPadi"])
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/results.csv")
    args = parser.parse_args()

    results = run(
        args.cases, args.ontology, tuple(args.targets), args.dim, tuple(args.hidden), seed=args.seed,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(f"\nHasil disimpan ke {args.out}")


if __name__ == "__main__":
    main()
