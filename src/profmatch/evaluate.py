"""Offline evaluation: nDCG@k, MRR, recall@k and coverage on a labelled query set.

Rankers:
- `scorer`: the profmatch query scorer.
- `raw_mean`: the same scorer without shrinkage (prior strength 0).
- `text_only`: the scorer with weights (1, 0, 0).
- `popularity`: candidates by total responses.
- `random`: a seeded shuffle of the candidates.
- `legacy_knn`: the old method. It ranks the neighbours of the FIRST candidate row, not of the query.
Weights are tuned on the dev half of the queries and reported on the test half.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .recommend import Query, Recommender, Weights
from .schema import Catalog

RELEVANT_GRADE = 2


def read_queries(path: str | Path) -> list[dict]:
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        q = json.loads(line)
        for key in ("query_id", "interests", "relevance"):
            if key not in q:
                raise ValueError(f"{path} line {i}: missing {key!r}")
        out.append(q)
    return out


def dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: list[str], rel: dict[str, int], candidates: list[str], k: int) -> float:
    ideal = sorted((rel.get(c, 0) for c in candidates), reverse=True)[:k]
    best = dcg(ideal)
    return dcg([rel.get(p, 0) for p in ranked[:k]]) / best if best > 0 else 0.0


def reciprocal_rank(ranked: list[str], rel: dict[str, int]) -> float:
    for i, p in enumerate(ranked, 1):
        if rel.get(p, 0) >= RELEVANT_GRADE:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked: list[str], rel: dict[str, int], candidates: list[str], k: int) -> float:
    relevant = {c for c in candidates if rel.get(c, 0) >= RELEVANT_GRADE}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


Ranker = Callable[[dict], tuple[list[str], list[str]]]  # query -> (ranking, candidates)


def _query(q: dict, k: int = 5) -> Query:
    return Query(q["interests"], q.get("course"), q.get("challenge"), k).validated()


def candidates_of(cat: Catalog, q: dict) -> list[str]:
    return cat.teachers_of(q["course"]) if q.get("course") else list(cat.professors)


def scorer_ranker(rec: Recommender) -> Ranker:
    def run(q: dict):
        rows = rec.score_all(_query(q))
        return [r[0] for r in rows], candidates_of(rec.cat, q)
    return run


def popularity_ranker(cat: Catalog) -> Ranker:
    totals: dict[str, int] = {}
    for o in cat.offerings:
        totals[o.professor_id] = totals.get(o.professor_id, 0) + o.responses

    def run(q: dict):
        cands = candidates_of(cat, q)
        return sorted(cands, key=lambda p: (-totals.get(p, 0), p)), cands
    return run


def random_ranker(cat: Catalog, seed: int = 0) -> Ranker:
    def run(q: dict):
        cands = candidates_of(cat, q)
        order = list(cands)
        random.Random(f"{seed}-{q['query_id']}").shuffle(order)
        return order, cands
    return run


def legacy_knn_ranker(rec: Recommender) -> Ranker:
    """Re-creates the old logic for comparison: one row for each offering, a KNN fit on all rows,
    and the 5 neighbours of row 0. The query only enters through a feature column."""
    cat = rec.cat

    def run(q: dict):
        cands = set(candidates_of(cat, q))
        rows = [o for o in cat.offerings if o.professor_id in cands]
        if not rows:
            return [], sorted(cands)
        text = rec.matcher.scores(q["interests"])
        feats = np.array([[2 * text.get(o.professor_id, 0.0), o.challenge_index or 0.0, o.overall_rating or 0.0]
                          for o in rows])
        X = StandardScaler().fit_transform(feats)
        nn = NearestNeighbors(n_neighbors=min(5, len(rows))).fit(X)
        _, idx = nn.kneighbors(X)
        picked = [rows[i] for i in idx[0]]
        picked = [o for o in picked if text.get(o.professor_id, 0.0) > 0] or picked
        ranking: list[str] = []
        for o in sorted(picked, key=lambda o: -text.get(o.professor_id, 0.0)):
            if o.professor_id not in ranking:
                ranking.append(o.professor_id)
        return ranking, sorted(cands)
    return run


def evaluate_ranker(ranker: Ranker, queries: list[dict], n_professors: int, k: int = 5) -> dict:
    nd, mrr, rec_k, shown = [], [], [], set()
    for q in queries:
        ranking, cands = ranker(q)
        rel = {p: int(g) for p, g in q["relevance"].items()}
        nd.append(ndcg_at_k(ranking, rel, cands, k))
        mrr.append(reciprocal_rank(ranking, rel))
        rec_k.append(recall_at_k(ranking, rel, cands, k))
        shown.update(ranking[:k])
    return {"queries": len(queries), f"ndcg@{k}": round(float(np.mean(nd)), 4), "mrr": round(float(np.mean(mrr)), 4),
            f"recall@{k}": round(float(np.mean(rec_k)), 4),
            "coverage": round(len(shown) / max(1, n_professors), 4)}


def weight_grid(step: float = 0.1) -> list[Weights]:
    n = int(round(1 / step))
    out = []
    for a, b in itertools.product(range(n + 1), repeat=2):
        if a + b <= n:
            out.append(Weights(round(a * step, 3), round(b * step, 3), round((n - a - b) * step, 3)))
    return [w for w in out if w.text > 0]


def tune_weights(cat: Catalog, dev: list[dict], matcher: str = "tfidf", step: float = 0.1,
                 prior_strength: float = 20.0) -> tuple[Weights, float]:
    """Return the weights with the best dev nDCG@5. Ties keep the larger text weight."""
    base = Recommender(cat, matcher, prior_strength=prior_strength)
    best: tuple[float, Weights] | None = None
    for w in sorted(weight_grid(step), key=lambda w: (-w.text, -w.quality)):
        base.weights = w
        score = evaluate_ranker(scorer_ranker(base), dev, len(cat.professors))["ndcg@5"]
        if best is None or score > best[0]:
            best = (score, w)
    assert best is not None
    return best[1], best[0]


def split_queries(queries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Even positions go to dev, odd positions to test: a fixed, documented split."""
    return queries[0::2], queries[1::2]


def full_report(cat: Catalog, queries: list[dict], matcher: str = "tfidf", weights: Weights | None = None,
                tune: bool = True, prior_strength: float = 20.0, seed: int = 0) -> dict:
    dev, test = split_queries(queries)
    report: dict = {"dev_queries": len(dev), "test_queries": len(test), "matcher": matcher}
    if tune:
        weights, dev_score = tune_weights(cat, dev, matcher, prior_strength=prior_strength)
        report["tuned_on_dev"] = {"weights": weights.__dict__, "dev_ndcg@5": dev_score}
    weights = weights or Weights()
    rec = Recommender(cat, matcher, weights, prior_strength=prior_strength)
    raw = Recommender(cat, matcher, weights, prior_strength=0.0)
    text_only = Recommender(cat, matcher, Weights(1.0, 0.0, 0.0), prior_strength=prior_strength)
    n = len(cat.professors)
    report["test"] = {
        "scorer": evaluate_ranker(scorer_ranker(rec), test, n),
        "raw_mean": evaluate_ranker(scorer_ranker(raw), test, n),
        "text_only": evaluate_ranker(scorer_ranker(text_only), test, n),
        "popularity": evaluate_ranker(popularity_ranker(cat), test, n),
        "random": evaluate_ranker(random_ranker(cat, seed), test, n),
        "legacy_knn": evaluate_ranker(legacy_knn_ranker(rec), test, n),
    }
    return report
