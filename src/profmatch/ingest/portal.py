"""Parse a faculty portal into a catalog, with page rules from a TOML config.

The parser reads tables by header names (with aliases), never by cell positions. A page with a
missing section gives a warning and an empty part. A page that fails gives a warning and the
scrape continues with the next page.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..schema import Catalog, Course, Offering, Professor, Publication
from ..textproc import CourseCodeError, parse_course_code
from .fetch import FetchError, Fetcher
from .html import links, parse, prefixed_value, section_items, table_records

DEFAULT_ALIASES = {
    "course": ["course", "course code", "course id"],
    "course_title": ["course name", "course title"],
    "term": ["term", "session", "period"],
    "enrolled": ["enrolled", "enrollment", "headcount"],
    "responses": ["responses", "respondents", "number of responses"],
    "overall_rating": ["overall rating", "overall summative rating", "rating"],
    "challenge_index": ["challenge index", "challenge and engagement index"],
}


@dataclass
class PortalConfig:
    base_url: str
    directory_paths: list[str]
    profile_link_contains: str = "/profile/"
    name_tag: str = "h1"
    title_prefix: str = "Title:"
    department_prefix: str = "Department:"
    publication_headings: list[str] = field(default_factory=lambda: ["publications"])
    keyword_headings: list[str] = field(default_factory=lambda: ["keywords", "research interests"])
    aliases: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_ALIASES))
    exclude_terms: list[str] = field(default_factory=list)

    @classmethod
    def from_toml(cls, path: str | Path) -> "PortalConfig":
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        aliases = dict(DEFAULT_ALIASES)
        aliases.update(data.pop("aliases", {}))
        return cls(aliases=aliases, **data)


@dataclass
class ScrapeReport:
    profiles: int = 0
    offerings: int = 0
    warnings: list[str] = field(default_factory=list)


def _num(s: str) -> float | None:
    s = (s or "").strip().replace(",", "")
    try:
        return float(s) if s and s.lower() not in ("n/a", "na", "-") else None
    except ValueError:
        return None


def parse_profile(html: str, pid: str, cfg: PortalConfig, cat: Catalog, report: ScrapeReport) -> None:
    root = parse(html)
    name_node = root.find(cfg.name_tag)
    name = name_node.text() if name_node else ""
    if not name:
        report.warnings.append(f"{pid}: no <{cfg.name_tag}> name, page skipped")
        return
    dept = prefixed_value(root, cfg.department_prefix)
    cat.professors[pid] = Professor(pid, name, prefixed_value(root, cfg.title_prefix), dept)
    report.profiles += 1
    records = table_records(root, cfg.aliases, required=["course", "term", "enrolled", "responses"])
    if not records:
        report.warnings.append(f"{pid}: no teaching table")
    for rec in records:
        if rec.get("term", "") in cfg.exclude_terms:
            continue
        try:
            code = parse_course_code(rec["course"])
        except CourseCodeError:
            report.warnings.append(f"{pid}: bad course code {rec.get('course')!r}")
            continue
        enrolled, responses = _num(rec.get("enrolled", "")), _num(rec.get("responses", ""))
        if enrolled is None or responses is None or responses > enrolled:
            report.warnings.append(f"{pid}: bad counts for {code} {rec.get('term')}")
            continue
        rating, chal = _num(rec.get("overall_rating", "")), _num(rec.get("challenge_index", ""))
        if responses == 0:
            rating = chal = None
        if rating is not None and not 1 <= rating <= 5:
            rating = None
        if chal is not None and not 1 <= chal <= 7:
            chal = None
        if code not in cat.courses:
            cat.courses[code] = Course(code, rec.get("course_title", ""), dept)
        cat.offerings.append(Offering(pid, code, rec.get("term", ""), int(enrolled), int(responses), rating, chal))
        report.offerings += 1
    for title in section_items(root, cfg.publication_headings):
        cat.publications.append(Publication(pid, title, None))
    kws = section_items(root, cfg.keyword_headings)
    if kws:
        cat.keywords[pid] = kws


def scrape(fetcher: Fetcher, cfg: PortalConfig) -> tuple[Catalog, ScrapeReport]:
    cat, report = Catalog(), ScrapeReport()
    profile_urls: list[str] = []
    for path in cfg.directory_paths:
        try:
            profile_urls += [u for u in links(parse(fetcher.get(path)), cfg.profile_link_contains)
                             if u not in profile_urls]
        except FetchError as exc:
            report.warnings.append(str(exc))
    for url in profile_urls:
        pid = url.rstrip("/").rsplit("/", 1)[-1] or url
        try:
            parse_profile(fetcher.get(url), pid, cfg, cat, report)
        except FetchError as exc:
            report.warnings.append(str(exc))
    return cat, report
