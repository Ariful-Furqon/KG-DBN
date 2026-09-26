# Analysis of the knowledge layer, computed from the ontology alone (no case data).
#
# python -m kgdbn.analysis [--seeds 10] [--out results/analysis]
#
# 1. structure_report: the ontology as a diagnostic resource (symptom profiles,
#    identical and nested profiles, accuracy ceiling under complete observation).
# 2. embedding_structure: what the node2vec symptom embeddings encode. Symptom
#    pairs are compared with (a) the overlap of their target sets and (b) whether
#    their targets are linked through non-symptom relations (shared pathogen or
#    control agent, pest->disease), which a multi-hot vector cannot represent.

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .embedding import node2vec
from .experiment import DEFAULT_ONTOLOGY
from .ontology import KnowledgeGraph, load_ontology

DEFAULT_TARGETS = ("PenyakitPadi", "HamaPadi")

# Graph variants for the embedding ablation. Symptoms are connected to the rest of
# the graph only through `memilikiGejala`, so hiding that relation leaves every
# symptom isolated; the informative ablation hides the other relations instead.
VARIANTS = {
    "full": lambda kg: (),
    "symptom-only": lambda kg: tuple(sorted({r for _, r, _ in kg.triples} - {"memilikiGejala"})),
    "no-memilikiGejala": lambda kg: ("memilikiGejala",),
}


def structure_report(kg: KnowledgeGraph, target_types=DEFAULT_TARGETS) -> dict:
    targets = kg.of_type(*target_types)
    profiles = {t: frozenset(s) for t, s in kg.symptom_profiles(target_types).items()}
    symptoms = kg.of_type("Gejala")
    sizes = np.array([len(p) for p in profiles.values()])
    by_type = {
        ty: float(np.mean([len(p) for t, p in profiles.items() if kg.entities[t].type == ty])) for ty in target_types
    }
    identical = [(a, b) for a, b in itertools.combinations(sorted(profiles), 2) if profiles[a] == profiles[b]]
    nested = sorted(t for t in profiles if any(profiles[t] < profiles[u] for u in profiles if u != t))
    usage = {s: sum(s in p for p in profiles.values()) for s in symptoms}
    unique = {s for s, n in usage.items() if n == 1}
    distinct = len(set(profiles.values()))

    n, degrees = len(kg.entities), _degrees(kg)
    return {
        "targets": len(targets),
        "diagnosable_targets": len(profiles),
        "diagnosable_by_type": {ty: sum(kg.entities[t].type == ty for t in profiles) for ty in target_types},
        "targets_without_symptoms": sorted(set(targets) - set(profiles)),
        "symptoms_per_target_mean": float(sizes.mean()),
        "symptoms_per_target_range": (int(sizes.min()), int(sizes.max())),
        "symptoms_per_target_by_type": by_type,
        "distinct_profiles": distinct,
        "identical_pairs": identical,
        "nested_targets": nested,
        "symptoms_with_one_target": len(unique),
        "targets_with_unique_symptom": sum(bool(p & unique) for p in profiles.values()),
        "most_shared_symptom": max(usage.items(), key=lambda kv: kv[1]),
        # Complete observation, uniform prior: one target per distinct profile is right.
        "ceiling_complete_observation": distinct / len(profiles),
        "nodes": n,
        "edges": len(kg.triples),
        "components": _components(kg),
        "degree_mean": float(np.mean(list(degrees.values()))),
        "degree_max": max(degrees.values()),
        "density": 2 * len(kg.triples) / (n * (n - 1)),
    }


def _neighbors(kg: KnowledgeGraph, exclude=()) -> dict[str, set[str]]:
    nbrs: dict[str, set[str]] = {e: set() for e in kg.entities}
    for s, r, o in kg.triples:
        if r not in exclude:
            nbrs[s].add(o)
            nbrs[o].add(s)
    return nbrs


def _degrees(kg: KnowledgeGraph) -> dict[str, int]:
    return {e: len(n) for e, n in _neighbors(kg).items()}


def _components(kg: KnowledgeGraph) -> list[int]:
    nbrs, seen, sizes = _neighbors(kg), set(), []
    for start in sorted(kg.entities):
        if start in seen:
            continue
        stack, size = [start], 0
        seen.add(start)
        while stack:
            size += 1
            for x in nbrs[stack.pop()] - seen:
                seen.add(x)
                stack.append(x)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def symptom_pairs(kg: KnowledgeGraph, target_types=DEFAULT_TARGETS) -> pd.DataFrame:
    # One row per pair of symptoms that both have at least one target.
    #
    # `jaccard`: overlap of the two target sets (what a multi-hot vector sees).
    # `linked`: the target sets are disjoint but some target of one is linked to
    # some target of the other by a non-symptom relation, directly or through a
    # shared neighbour; `via` names the kind of link.
    targets_of: dict[str, set[str]] = {}
    for t, symptoms in kg.symptom_profiles(target_types).items():
        for s in symptoms:
            targets_of.setdefault(s, set()).add(t)
    other = _neighbors(kg, exclude=("memilikiGejala",))

    def link_kind(a_targets, b_targets):
        kinds = set()
        for a in a_targets:
            for b in b_targets:
                if b in other[a]:
                    kinds.add("pest-disease")
                for x in other[a] & other[b]:
                    kinds.add("pathogen" if kg.entities[x].type == "PatogenPadi" else "control")
        return kinds

    rows = []
    for a, b in itertools.combinations(sorted(targets_of), 2):
        ta, tb = targets_of[a], targets_of[b]
        jaccard = len(ta & tb) / len(ta | tb)
        kinds = link_kind(ta, tb) if jaccard == 0 else set()
        rows.append({"a": a, "b": b, "jaccard": jaccard, "linked": bool(kinds), "via": "+".join(sorted(kinds))})
    return pd.DataFrame(rows)


def _cosine(embeddings, a, b) -> float:
    x, y = embeddings[a], embeddings[b]
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def embedding_structure(embeddings: dict[str, np.ndarray], pairs: pd.DataFrame) -> dict:
    # rho: Spearman correlation between cosine similarity and target-set Jaccard.
    # gap: mean cosine of disjoint-but-linked pairs minus disjoint-unlinked pairs.
    cos = np.array([_cosine(embeddings, a, b) for a, b in zip(pairs["a"], pairs["b"])])
    disjoint = pairs["jaccard"].values == 0
    linked = pairs["linked"].values
    return {
        "rho_jaccard": float(spearmanr(cos, pairs["jaccard"]).statistic),
        "cos_shared": float(cos[~disjoint].mean()),
        "cos_linked": float(cos[linked].mean()),
        "cos_unlinked": float(cos[disjoint & ~linked].mean()),
        "gap_linked": float(cos[linked].mean() - cos[disjoint & ~linked].mean()),
        "cosines": cos,
    }


def run(
    ontology=DEFAULT_ONTOLOGY,
    seeds=range(10),
    variants=tuple(VARIANTS),
    configs=({},),
    verbose=True,
) -> pd.DataFrame:
    # One row per (variant, node2vec config, seed); `stability` is the mean
    # Spearman correlation of pairwise symptom similarities across seeds.
    log = print if verbose else (lambda *a, **k: None)
    kg = load_ontology(ontology)
    pairs = symptom_pairs(kg)
    log(f"{len(pairs)} symptom pairs, {(pairs['jaccard'] > 0).sum()} share a target, "
        f"{pairs['linked'].sum()} disjoint but linked, {((pairs['jaccard'] == 0) & ~pairs['linked']).sum()} unlinked")

    rows = []
    for variant in variants:
        exclude = VARIANTS[variant](kg)
        for config in configs:
            cosines = []
            for seed in seeds:
                result = embedding_structure(node2vec(kg, seed=seed, exclude_relations=exclude, **config), pairs)
                cosines.append(result.pop("cosines"))
                rows.append({"variant": variant, **config, "seed": seed, **result})
            stability = np.mean([spearmanr(x, y).statistic for x, y in itertools.combinations(cosines, 2)])
            for row in rows[-len(cosines):]:
                row["stability"] = float(stability)
            last = pd.DataFrame(rows[-len(cosines):])
            log(f"{variant:18s} {str(config):28s} rho {last['rho_jaccard'].mean():.3f}±{last['rho_jaccard'].std():.3f}"
                f"  gap {last['gap_linked'].mean():+.3f}±{last['gap_linked'].std():.3f}  stability {stability:.3f}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Analisis knowledge layer KG-DBN (tanpa data kasus).")
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--sensitivity", action="store_true", help="juga variasikan dim, p, q pada graf lengkap")
    parser.add_argument("--out", default="results/analysis")
    args = parser.parse_args()

    kg = load_ontology(args.ontology)
    for key, value in structure_report(kg).items():
        print(f"{key}: {value}")
    print()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seeds)
    run(args.ontology, seeds).to_csv(out / "embedding_variants.csv", index=False)
    symptom_pairs(kg).to_csv(out / "symptom_pairs.csv", index=False)
    if args.sensitivity:
        configs = [{"dim": d} for d in (16, 64)] + [{"p": 1.0, "q": q} for q in (0.5, 2.0)] + [{"window": 3}]
        run(args.ontology, seeds, variants=("full",), configs=configs).to_csv(out / "sensitivity.csv", index=False)
    print(f"\nHasil disimpan ke {out}")


if __name__ == "__main__":
    main()
