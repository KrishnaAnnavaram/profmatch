"""The normalised catalog: professors, courses, offerings, publications and keywords.

`read_bundle` reads five CSV files and validates each row (types, ranges, keys). A missing
evaluation score stays missing. It is never filled with the professor's own mean.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

from .textproc import CourseCodeError, parse_course_code

FILES = {
    "professors": ("professor_id", "name", "title", "department"),
    "courses": ("course_code", "course_title", "department"),
    "offerings": ("professor_id", "course_code", "term", "enrolled", "responses", "overall_rating",
                  "challenge_index"),
    "publications": ("professor_id", "title", "year"),
    "keywords": ("professor_id", "keyword"),
}
RATING_RANGE = (1.0, 5.0)
CHALLENGE_RANGE = (1.0, 7.0)


class SchemaError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        more = f" (+{len(problems) - 8} more)" if len(problems) > 8 else ""
        super().__init__(f"{len(problems)} data problem(s): " + "; ".join(problems[:8]) + more)


@dataclass(frozen=True)
class Professor:
    professor_id: str
    name: str
    title: str
    department: str


@dataclass(frozen=True)
class Course:
    course_code: str
    course_title: str
    department: str


@dataclass(frozen=True)
class Offering:
    professor_id: str
    course_code: str
    term: str
    enrolled: int
    responses: int
    overall_rating: float | None
    challenge_index: float | None


@dataclass(frozen=True)
class Publication:
    professor_id: str
    title: str
    year: int | None


@dataclass
class Catalog:
    professors: dict[str, Professor] = field(default_factory=dict)
    courses: dict[str, Course] = field(default_factory=dict)
    offerings: list[Offering] = field(default_factory=list)
    publications: list[Publication] = field(default_factory=list)
    keywords: dict[str, list[str]] = field(default_factory=dict)

    def offerings_of(self, professor_id: str) -> list[Offering]:
        return [o for o in self.offerings if o.professor_id == professor_id]

    def teachers_of(self, course_code: str) -> list[str]:
        code = parse_course_code(course_code)
        return sorted({o.professor_id for o in self.offerings if o.course_code == code})


def _num(raw: str, kind, where: str, problems: list[str], lo=None, hi=None, optional=False):
    raw = (raw or "").strip()
    if raw == "":
        if optional:
            return None
        problems.append(f"{where}: value is missing")
        return None
    try:
        value = kind(float(raw)) if kind is int else float(raw)
    except ValueError:
        problems.append(f"{where}: {raw!r} is not a number")
        return None
    if isinstance(value, float) and not math.isfinite(value):
        problems.append(f"{where}: {raw!r} is not finite")
        return None
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        problems.append(f"{where}: {value} is outside [{lo}, {hi}]")
        return None
    return value


def _rows(path: Path, columns: tuple[str, ...], problems: list[str]) -> list[dict]:
    if not path.is_file():
        problems.append(f"{path.name}: file not found")
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            problems.append(f"{path.name}: missing column(s) {missing}")
            return []
        return [{k: (v or "").strip() for k, v in r.items() if k} for r in reader]


def read_bundle(folder: str | Path) -> Catalog:
    folder = Path(folder)
    problems: list[str] = []
    raw = {name: _rows(folder / f"{name}.csv", cols, problems) for name, cols in FILES.items()}
    cat = Catalog()
    for i, r in enumerate(raw["professors"], 2):
        pid = r["professor_id"]
        if not pid or not r["name"]:
            problems.append(f"professors.csv line {i}: professor_id and name are required")
        elif pid in cat.professors:
            problems.append(f"professors.csv line {i}: duplicate professor_id {pid}")
        else:
            cat.professors[pid] = Professor(pid, r["name"], r["title"], r["department"])
    for i, r in enumerate(raw["courses"], 2):
        try:
            code = parse_course_code(r["course_code"])
        except CourseCodeError as exc:
            problems.append(f"courses.csv line {i}: {exc}")
            continue
        if code in cat.courses:
            problems.append(f"courses.csv line {i}: duplicate course_code {code}")
        cat.courses[code] = Course(code, r["course_title"], r["department"])
    for i, r in enumerate(raw["offerings"], 2):
        where = f"offerings.csv line {i}"
        if r["professor_id"] not in cat.professors:
            problems.append(f"{where}: unknown professor_id {r['professor_id']!r}")
            continue
        try:
            code = parse_course_code(r["course_code"])
        except CourseCodeError as exc:
            problems.append(f"{where}: {exc}")
            continue
        if code not in cat.courses:
            problems.append(f"{where}: unknown course_code {code}")
            continue
        enrolled = _num(r["enrolled"], int, f"{where} enrolled", problems, lo=0)
        responses = _num(r["responses"], int, f"{where} responses", problems, lo=0)
        rating = _num(r["overall_rating"], float, f"{where} overall_rating", problems, *RATING_RANGE, optional=True)
        challenge = _num(r["challenge_index"], float, f"{where} challenge_index", problems, *CHALLENGE_RANGE,
                         optional=True)
        if enrolled is None or responses is None:
            continue
        if responses > enrolled:
            problems.append(f"{where}: responses {responses} > enrolled {enrolled}")
            continue
        if responses == 0 and (rating is not None or challenge is not None):
            problems.append(f"{where}: scores given with 0 responses")
            continue
        cat.offerings.append(Offering(r["professor_id"], code, r["term"], enrolled, responses, rating, challenge))
    for i, r in enumerate(raw["publications"], 2):
        if r["professor_id"] not in cat.professors:
            problems.append(f"publications.csv line {i}: unknown professor_id {r['professor_id']!r}")
            continue
        year = _num(r["year"], int, f"publications.csv line {i} year", problems, 1900, 2100, optional=True)
        if r["title"]:
            cat.publications.append(Publication(r["professor_id"], r["title"], year))
    for i, r in enumerate(raw["keywords"], 2):
        if r["professor_id"] not in cat.professors:
            problems.append(f"keywords.csv line {i}: unknown professor_id {r['professor_id']!r}")
            continue
        if r["keyword"]:
            cat.keywords.setdefault(r["professor_id"], []).append(r["keyword"])
    if not cat.professors:
        problems.append("professors.csv: no professor")
    if problems:
        raise SchemaError(problems)
    return cat


def write_bundle(cat: Catalog, folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    def dump(name, rows):
        with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(FILES[name])
            w.writerows(rows)

    dump("professors", [(p.professor_id, p.name, p.title, p.department) for p in cat.professors.values()])
    dump("courses", [(c.course_code, c.course_title, c.department) for c in cat.courses.values()])
    dump("offerings", [(o.professor_id, o.course_code, o.term, o.enrolled, o.responses,
                        "" if o.overall_rating is None else o.overall_rating,
                        "" if o.challenge_index is None else o.challenge_index) for o in cat.offerings])
    dump("publications", [(p.professor_id, p.title, "" if p.year is None else p.year) for p in cat.publications])
    dump("keywords", [(pid, k) for pid, ks in cat.keywords.items() for k in ks])
    return folder
