# node2vec embeddings of the knowledge graph (biased random walks + skip-gram).

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .ontology import KnowledgeGraph


def _adjacency(kg: KnowledgeGraph, exclude_relations=()) -> tuple[list[str], list[set[int]]]:
    nodes = sorted(kg.entities)
    index = {n: i for i, n in enumerate(nodes)}
    adj: list[set[int]] = [set() for _ in nodes]
    for s, r, o in kg.triples:
        if r in exclude_relations:
            continue
        adj[index[s]].add(index[o])
        adj[index[o]].add(index[s])
    return nodes, adj


def random_walks(adj, num_walks, walk_length, p, q, rng) -> list[list[int]]:
    # Second-order node2vec walks on an undirected, unweighted graph.
    neighbors = [np.array(sorted(a)) for a in adj]
    walks = []
    for _ in range(num_walks):
        for start in rng.permutation(len(adj)):
            walk = [int(start)]
            while len(walk) < walk_length:
                cur = walk[-1]
                nbrs = neighbors[cur]
                if len(nbrs) == 0:
                    break
                if len(walk) == 1:
                    walk.append(int(rng.choice(nbrs)))
                    continue
                prev = walk[-2]
                weights = np.where(nbrs == prev, 1 / p, np.where(np.isin(nbrs, neighbors[prev]), 1.0, 1 / q))
                walk.append(int(rng.choice(nbrs, p=weights / weights.sum())))
            walks.append(walk)
    return walks


class _SkipGram(nn.Module):
    def __init__(self, n, dim):
        super().__init__()
        self.inp = nn.Embedding(n, dim)
        self.out = nn.Embedding(n, dim)
        nn.init.uniform_(self.inp.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out.weight)

    def forward(self, center, context, negatives):
        c = self.inp(center)
        pos = (c * self.out(context)).sum(-1)
        neg = torch.bmm(self.out(negatives), c.unsqueeze(-1)).squeeze(-1)
        return -(nn.functional.logsigmoid(pos) + nn.functional.logsigmoid(-neg).sum(-1)).mean()


def node2vec(
    kg: KnowledgeGraph,
    dim: int = 32,
    num_walks: int = 20,
    walk_length: int = 20,
    window: int = 5,
    p: float = 1.0,
    q: float = 1.0,
    negatives: int = 5,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 0.01,
    exclude_relations=(),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    # Return {entity_id: vector}.
    #
    # `exclude_relations` drops edges before the walks, e.g. every relation except
    # "memilikiGejala" for the symptom-only ablation (see analysis.VARIANTS).
    # Dropping "memilikiGejala" itself isolates every symptom, leaving its vector
    # at the random initialisation.
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    nodes, adj = _adjacency(kg, exclude_relations)
    walks = random_walks(adj, num_walks, walk_length, p, q, rng)

    pairs = [
        (walk[i], walk[j])
        for walk in walks
        for i in range(len(walk))
        for j in range(max(0, i - window), min(len(walk), i + window + 1))
        if i != j
    ]
    pairs = torch.tensor(pairs, dtype=torch.long) if pairs else torch.empty((0, 2), dtype=torch.long)

    # Unigram^0.75 noise distribution over walk occurrences.
    freq = np.bincount([n for w in walks for n in w], minlength=len(nodes)).astype(float) ** 0.75
    noise = torch.tensor(freq / freq.sum(), dtype=torch.float)

    model = _SkipGram(len(nodes), dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for batch in pairs[torch.randperm(len(pairs))].split(batch_size):
            neg = torch.multinomial(noise, len(batch) * negatives, replacement=True).view(len(batch), negatives)
            loss = model(batch[:, 0], batch[:, 1], neg)
            opt.zero_grad()
            loss.backward()
            opt.step()

    vectors = model.inp.weight.detach().numpy()
    return {n: vectors[i] for i, n in enumerate(nodes)}
