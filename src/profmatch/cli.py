"""The `profmatch` command line."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .recommend import Query, QueryError, Recommender
from .schema import SchemaError, read_bundle, write_bundle
from .store import load, save
from .textproc import CourseCodeError


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _recommender(s: Settings, db: Path | None = None) -> Recommender:
    return Recommender(load(db or s.db_path), s.matcher, s.weights, s.prior_strength, s.min_responses, s.display)


def cmd_synth(args, s):
    from .synthetic import make_catalog, make_queries, write_queries

    cat, truth = make_catalog(args.professors, args.seed)
    write_bundle(cat, args.out)
    write_queries(make_queries(cat, truth, args.queries, args.seed + 1), Path(args.out) / "queries.jsonl")
    _print({"out": str(args.out), "professors": len(cat.professors), "courses": len(cat.courses),
            "offerings": len(cat.offerings), "queries": args.queries})


def cmd_load(args, s):
    cat = read_bundle(args.bundle)
    path = save(cat, args.db or s.db_path)
    _print({"db": str(path), "professors": len(cat.professors), "courses": len(cat.courses),
            "offerings": len(cat.offerings), "publications": len(cat.publications),
            "keywords": sum(len(v) for v in cat.keywords.values())})


def cmd_recommend(args, s):
    rec = _recommender(s, args.db)
    res = rec.recommend(Query(args.interests, args.course, args.challenge, args.k))
    if args.json:
        _print(res.to_dict())
        return
    print(f"{res.candidates} candidate(s). " + " ".join(res.notes))
    for r in res.recommendations:
        print(f"{r.rank}. {r.name} ({r.title}) score {r.score:.3f}  text {r.text_match:.3f}  "
              f"quality {r.quality:.3f}  fit {r.challenge_fit:.1f}")
        for reason in r.reasons:
            print(f"     - {reason}")


def cmd_profile(args, s):
    rec = _recommender(s, args.db)
    pid = args.professor_id
    if pid not in rec.cat.professors:
        raise QueryError(f"unknown professor_id {pid}")
    p = rec.profiles[pid]
    _print({"professor": rec.cat.professors[pid].__dict__, "courses": list(p.course_codes), "profile": p.parts,
            "signals": rec.signals_for(None)[pid].to_dict(s.display)})


def cmd_eval(args, s):
    from .evaluate import full_report, read_queries

    cat = load(args.db or s.db_path)
    report = full_report(cat, read_queries(args.queries), s.matcher, None if args.tune else s.weights,
                         tune=args.tune, prior_strength=s.prior_strength)
    _print(report)


def cmd_scrape(args, s):
    from .ingest import HttpFetcher, PortalConfig, scrape

    cfg = PortalConfig.from_toml(args.config)
    fetcher = HttpFetcher(cfg.base_url, s.cache_dir, args.interval, terms_accepted=s.terms_accepted)
    cat, report = scrape(fetcher, cfg)
    write_bundle(cat, args.out)
    _print({"out": str(args.out), "profiles": report.profiles, "offerings": report.offerings,
            "warnings": report.warnings})


def cmd_serve(args, s):
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(settings=s), host=args.host, port=args.port)


def cmd_demo(args, s):
    from .evaluate import full_report
    from .synthetic import make_catalog, make_queries, write_queries

    out = Path(args.out)
    cat, truth = make_catalog(60, args.seed)
    write_bundle(cat, out / "bundle")
    queries = make_queries(cat, truth, 80, args.seed + 1)
    write_queries(queries, out / "bundle" / "queries.jsonl")
    save(read_bundle(out / "bundle"), out / "profmatch.db")
    loaded = load(out / "profmatch.db")
    report = full_report(loaded, queries, "tfidf", tune=True, prior_strength=s.prior_strength)
    rec = Recommender(loaded, "tfidf", s.weights, s.prior_strength, s.min_responses, s.display)
    q = queries[1]
    example = rec.recommend(Query(q["interests"], q["course"], q["challenge"], 3)).to_dict()
    summary = {"catalog": {"professors": len(loaded.professors), "courses": len(loaded.courses),
                           "offerings": len(loaded.offerings)}, "evaluation": report, "example": example}
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print(summary)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="profmatch", description="Explainable professor recommendations.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("synth", help="write a synthetic catalog bundle and a labelled query set")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--professors", type=int, default=60)
    c.add_argument("--queries", type=int, default=80)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=cmd_synth)

    c = sub.add_parser("load", help="validate a CSV bundle and load it into SQLite")
    c.add_argument("bundle", type=Path)
    c.add_argument("--db", type=Path)
    c.set_defaults(func=cmd_load)

    c = sub.add_parser("recommend", help="rank professors for interests, a course and a challenge preference")
    c.add_argument("interests")
    c.add_argument("--course")
    c.add_argument("--challenge", choices=["lower", "balanced", "higher"])
    c.add_argument("-k", type=int, default=5)
    c.add_argument("--db", type=Path)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_recommend)

    c = sub.add_parser("profile", help="show the profile and the teaching signals of one professor")
    c.add_argument("professor_id")
    c.add_argument("--db", type=Path)
    c.set_defaults(func=cmd_profile)

    c = sub.add_parser("eval", help="nDCG@5, MRR, recall@5 and coverage against baselines")
    c.add_argument("queries", type=Path)
    c.add_argument("--db", type=Path)
    c.add_argument("--tune", action="store_true", help="tune the weights on the dev half")
    c.set_defaults(func=cmd_eval)

    c = sub.add_parser("scrape", help="read a faculty portal (extra scrape, terms check, robots.txt)")
    c.add_argument("--config", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--interval", type=float, default=2.0, help="seconds between requests")
    c.set_defaults(func=cmd_scrape)

    c = sub.add_parser("serve", help="HTTP API and form (extra api)")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=8000)
    c.set_defaults(func=cmd_serve)

    c = sub.add_parser("demo", help="synthetic catalog, load, evaluation and one example")
    c.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args, Settings.from_env())
    except (QueryError, CourseCodeError, SchemaError, ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
