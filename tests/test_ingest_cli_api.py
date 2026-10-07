"""Portal parsing, the CLI, the settings and the optional API (problems 8, 9 and 10)."""
import json
from pathlib import Path

import pytest

from profmatch.cli import main
from profmatch.config import ConfigError, Settings
from profmatch.ingest import FetchError, FixtureFetcher, HttpFetcher, PortalConfig, scrape
from profmatch.ingest.html import parse, section_items, table_records
from profmatch.schema import read_bundle, write_bundle

FIX = Path(__file__).parent / "fixtures"


def _pages():
    return {"/directory": (FIX / "directory.html").read_text(encoding="utf-8"),
            **{f"/profile/{p}": (FIX / f"prof_{p}.html").read_text(encoding="utf-8") for p in ("ab12", "cd34", "ef56")}}


def test_scrape_fixture_portal(tmp_path):
    cfg = PortalConfig(base_url="https://portal.example.org", directory_paths=["/directory"],
                       exclude_terms=["2026-1"])
    fetcher = FixtureFetcher(_pages())
    cat, report = scrape(fetcher, cfg)
    assert set(cat.professors) == {"ab12", "cd34", "ef56"}
    assert fetcher.calls.count("/profile/ab12") == 1  # duplicate links are fetched once
    ab = [o for o in cat.offerings if o.professor_id == "ab12"]
    assert [(o.course_code, o.responses, o.overall_rating) for o in ab] == [("DSCI 5100", 20, 4.5), ("DSCI 5100", 0, None)]
    assert cat.courses["DSCI 5100"].course_title == "Machine Learning Foundations"
    assert cat.keywords["ab12"] == ["deep learning", "model calibration", "transfer learning"]
    assert len([p for p in cat.publications if p.professor_id == "ab12"]) == 2
    w = " | ".join(report.warnings)
    assert "bad counts for DSCI 5200" in w and "bad course code 'Seminar'" in w
    assert "cd34: no teaching table" in w and "no fixture for /profile/gone" in w
    write_bundle(cat, tmp_path / "b")
    assert read_bundle(tmp_path / "b").professors.keys() == cat.professors.keys()


def test_tables_are_read_by_header_names_in_any_order():
    """Problem 8: no fixed cell positions. The second table matches by aliases."""
    root = parse((FIX / "prof_ef56.html").read_text(encoding="utf-8"))
    recs = table_records(root, {"course": ["course id"], "term": ["session"], "enrolled": ["headcount"],
                                "responses": ["respondents"]}, ["course", "term", "enrolled", "responses"])
    assert recs[0] == {"course": "HCID 5100", "term": "2024-1", "enrolled": "30", "responses": "12"}
    assert table_records(root, {"course": ["nothing"]}, ["course"]) == []
    assert section_items(root, ["keywords"]) == ["usability studies", "accessibility"]
    assert section_items(root, ["awards"]) == []


def test_broken_html_does_not_crash():
    root = parse("<div><table><tr><td>a<td>b</table></span><p>Title: X")
    assert root.find("table") is not None


def test_portal_config_from_toml(tmp_path):
    p = tmp_path / "portal.toml"
    p.write_text('base_url = "https://portal.example.org"\ndirectory_paths = ["/d"]\n'
                 '[aliases]\ncourse = ["class number"]\n', encoding="utf-8")
    cfg = PortalConfig.from_toml(p)
    assert cfg.aliases["course"] == ["class number"] and "term" in cfg.aliases


def test_http_fetcher_needs_the_terms_check(tmp_path):
    """Problem 9: no scrape without an explicit terms-of-use acknowledgement."""
    with pytest.raises(FetchError):
        HttpFetcher("https://portal.example.org", tmp_path, terms_accepted=False)


def test_settings(tmp_path):
    s = Settings.from_env({"PROFMATCH_WEIGHTS": "0.5,0.3,0.2", "PROFMATCH_CORS_ORIGINS": "https://a.example"},
                          dotenv=None)
    assert s.weights.quality == 0.3 and s.cors_origins == ("https://a.example",) and not s.terms_accepted
    for bad in ({"PROFMATCH_WEIGHTS": "1,2"}, {"PROFMATCH_CORS_ORIGINS": "*"}, {"PROFMATCH_SIGNAL_DISPLAY": "raw"},
                {"PROFMATCH_PRIOR_STRENGTH": "-1"}):
        with pytest.raises(ConfigError):
            Settings.from_env(bad, dotenv=None)


def test_cli_flow(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROFMATCH_DB", str(tmp_path / "pm.db"))
    assert main(["synth", "--out", "bundle", "--professors", "21", "--queries", "10"]) == 0
    assert main(["load", "bundle"]) == 0
    capsys.readouterr()
    assert main(["recommend", "deep learning and neural networks", "--json", "-k", "3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert 1 <= len(out["recommendations"]) <= 3 and out["recommendations"][0]["reasons"]
    assert main(["recommend", "neural networks", "--challenge", "higher"]) == 0
    assert "1." in capsys.readouterr().out
    assert main(["profile", "P000"]) == 0
    assert "signals" in json.loads(capsys.readouterr().out)
    assert main(["eval", "bundle/queries.jsonl"]) == 0
    assert "legacy_knn" in json.loads(capsys.readouterr().out)["test"]
    assert main(["recommend", "x", "--course", "BAD("]) == 2
    assert main(["profile", "nobody"]) == 2
    assert main(["load", "missing_folder"]) == 2


def test_cli_demo(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path / "demo")]) == 0
    summary = json.loads((tmp_path / "demo" / "demo_summary.json").read_text())
    t = summary["evaluation"]["test"]
    assert t["scorer"]["ndcg@5"] > t["legacy_knn"]["ndcg@5"]


def test_api(small_rec):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from profmatch.api import create_app

    s = Settings.from_env({}, dotenv=None)
    client = TestClient(create_app(small_rec, s))
    assert client.get("/health").json()["professors"] == 3
    r = client.post("/recommend", json={"interests": "query optimization", "course": "DATA 5100"})
    assert r.status_code == 200 and r.json()["recommendations"][0]["professor_id"] == "A"
    assert client.post("/recommend", json={"interests": "x", "course": "DATA.*"}).status_code == 422
    assert client.post("/recommend", json={"interests": "databases", "k": 99}).status_code == 422
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers  # problem 10: no CORS by default
    assert "<form" in client.get("/").text

    allowed = Settings.from_env({"PROFMATCH_CORS_ORIGINS": "https://ok.example"}, dotenv=None)
    c2 = TestClient(create_app(small_rec, allowed))
    assert c2.get("/health", headers={"Origin": "https://ok.example"}).headers["access-control-allow-origin"] == \
        "https://ok.example"
