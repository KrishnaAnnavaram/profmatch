"""The query scorer and the offline evaluation (problems 1, 3, 5 and 7)."""
import dataclasses
import random

import pytest

from profmatch.encoders import Bm25Matcher, TfidfMatcher, build_matcher
from profmatch.evaluate import (dcg, evaluate_ranker, full_report, legacy_knn_ranker, ndcg_at_k, random_ranker,
                                read_queries, recall_at_k, reciprocal_rank, scorer_ranker, split_queries,
                                tune_weights, weight_grid)
from profmatch.recommend import Query, QueryError, Recommender, Weights, challenge_fit
from profmatch.schema import Catalog
from profmatch.synthetic import write_queries
from profmatch.textproc import CourseCodeError


def test_the_query_decides_the_ranking(small_rec):
    """Problem 1: the result depends on the query."""
    db = small_rec.recommend(Query("query optimization and indexes", k=3))
    net = small_rec.recommend(Query("wireless routing", k=3))
    assert db.recommendations[0].professor_id == "A"
    assert net.recommendations[0].professor_id == "C"


def test_ranking_does_not_depend_on_row_order(synth):
    """Problem 1: shuffle every table, get the same ranking."""
    cat, _, queries = synth
    shuffled = Catalog(dict(reversed(list(cat.professors.items()))), dict(reversed(list(cat.courses.items()))),
                       list(reversed(cat.offerings)), list(reversed(cat.publications)),
                       {k: v for k, v in reversed(list(cat.keywords.items()))})
    a, b = Recommender(cat), Recommender(shuffled)
    for q in queries[:15]:
        query = Query(q["interests"], q["course"], q["challenge"], 5)
        assert [r[0] for r in a.score_all(query)] == [r[0] for r in b.score_all(query)]


def test_small_candidate_sets_do_not_crash(small_rec):
    """Problem 5: a course with fewer teachers than k returns all of them and a note."""
    res = small_rec.recommend(Query("networks", course="netw-5100", k=5))
    assert [r.professor_id for r in res.recommendations] == ["C"]
    assert res.notes == ["only 1 professor(s) taught NETW 5100"]


def test_course_filter_is_exact(small_rec):
    with pytest.raises(QueryError):
        small_rec.recommend(Query("databases", course="DATA 510"))
    with pytest.raises(CourseCodeError):
        small_rec.recommend(Query("databases", course="DATA.*"))


@pytest.mark.parametrize("q", [Query(""), Query("the and of"), Query("ml", challenge="extreme"), Query("ml", k=0)])
def test_invalid_queries(small_rec, q):
    with pytest.raises(QueryError):
        small_rec.recommend(q)


def test_a_name_does_not_match(small_rec):
    """Problem 2: the professor name is not in the profile."""
    res = small_rec.recommend(Query("Robin Example"))
    assert res.recommendations == [] and res.notes == ["no professor profile matches the interests"]


def test_shrinkage_changes_the_order_inside_a_course(small_catalog):
    q = Query("transaction processing query optimization", course="DATA 5100", k=2)
    smooth = Recommender(small_catalog, weights=Weights(0.2, 0.8, 0.0), prior_strength=20).recommend(q)
    raw = Recommender(small_catalog, weights=Weights(0.2, 0.8, 0.0), prior_strength=0).recommend(q)
    assert smooth.recommendations[0].professor_id == "A"   # 60 responses near 4.5
    assert raw.recommendations[0].professor_id == "B"      # 2 responses of 5.0


def test_many_sections_do_not_inflate_the_text_score(small_catalog):
    """Problem 7: ten copies of one offering give the same profile and the same text score."""
    one = Recommender(small_catalog).matcher.scores("database systems")["A"]
    small_catalog.offerings += [dataclasses.replace(small_catalog.offerings[0], term=f"t{i}") for i in range(10)]
    many = Recommender(small_catalog).matcher.scores("database systems")["A"]
    assert one == pytest.approx(many)


def test_explanations(small_rec):
    rec = small_rec.recommend(Query("query optimization", course="DATA 5100", challenge="higher")).recommendations[0]
    assert rec.professor_id == "A" and "query optimization" in rec.matched_terms
    assert any("responses" in r for r in rec.reasons) and any("taught DATA 5100" in r for r in rec.reasons)
    assert "rating" not in rec.signals  # band display by default


def test_challenge_fit_and_weights():
    assert challenge_fit("higher", "higher") == 1.0 and challenge_fit("balanced", "higher") == 0.5
    assert challenge_fit("lower", "higher") == 0.0 and challenge_fit(None, "lower") == 0.5
    assert challenge_fit("lower", None) == 0.0
    with pytest.raises(ValueError):
        Weights(-0.1, 0.5, 0.6)
    assert all(abs(w.text + w.quality + w.fit - 1) < 1e-9 for w in weight_grid())


def test_matchers(small_catalog):
    texts = {"A": "query optimization databases", "B": "transaction processing", "C": "wireless networks"}
    for m in (TfidfMatcher().fit(texts), Bm25Matcher().fit(texts)):
        s = m.scores("query optimization")
        assert max(s, key=s.get) == "A" and s["C"] == 0
        assert "optimization" in " ".join(m.matched_terms("query optimization", "A"))
    with pytest.raises(ValueError):
        build_matcher("word2vec")


def test_ranking_metrics():
    rel = {"a": 3, "b": 2, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], rel, ["a", "b", "c"], 3) == 1.0
    assert ndcg_at_k(["c", "b", "a"], rel, ["a", "b", "c"], 3) < 1.0
    assert reciprocal_rank(["c", "b", "a"], rel) == 0.5
    assert recall_at_k(["c", "a"], rel, ["a", "b", "c"], 2) == 0.5
    assert dcg([]) == 0 and ndcg_at_k(["x"], {}, ["x"], 5) == 0.0


def test_scorer_beats_baselines_and_the_legacy_method(synth):
    """Problem 3: an offline evaluation with graded labels. Problem 1: the old KNN step is worse."""
    cat, _, queries = synth
    _, test = split_queries(queries)
    rec = Recommender(cat)
    n = len(cat.professors)
    ours = evaluate_ranker(scorer_ranker(rec), test, n)
    assert ours["ndcg@5"] > evaluate_ranker(random_ranker(cat), test, n)["ndcg@5"]
    assert ours["ndcg@5"] > evaluate_ranker(legacy_knn_ranker(rec), test, n)["ndcg@5"]


def test_legacy_method_depends_on_row_order(synth):
    cat, _, queries = synth
    rec = Recommender(cat)
    shuffled = Catalog(cat.professors, cat.courses, random.Random(3).sample(cat.offerings, len(cat.offerings)),
                       cat.publications, cat.keywords)
    rec2 = Recommender(shuffled)
    changed = sum(legacy_knn_ranker(rec)(q)[0] != legacy_knn_ranker(rec2)(q)[0] for q in queries[:30])
    assert changed > 0


def test_tuning_uses_dev_and_report_has_all_rankers(synth):
    cat, _, queries = synth
    dev, test = split_queries(queries)
    assert len(dev) == len(test) == 40 and not {q["query_id"] for q in dev} & {q["query_id"] for q in test}
    w, score = tune_weights(cat, dev[:10], step=0.25)
    assert w in weight_grid(0.25) and 0 <= score <= 1
    rep = full_report(cat, queries[:20], tune=False)
    assert set(rep["test"]) == {"scorer", "raw_mean", "text_only", "popularity", "random", "legacy_knn"}


def test_query_file_round_trip_and_validation(tmp_path, synth):
    _, _, queries = synth
    path = write_queries(queries[:3], tmp_path / "q.jsonl")
    assert read_queries(path) == queries[:3]
    path.write_text('{"query_id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_queries(path)
