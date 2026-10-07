"""One text profile for each professor.

The profile joins the titles of the courses that the professor taught, the publication titles and
the research keywords. The professor's name is never part of the profile, so a query can not match a
name. Each course title counts once, also when the professor taught it in many terms.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import Catalog
from .textproc import normalise


@dataclass(frozen=True)
class Profile:
    professor_id: str
    text: str
    course_codes: tuple[str, ...]
    parts: dict


def build_profiles(cat: Catalog, max_publications: int = 50) -> dict[str, Profile]:
    pubs: dict[str, list[tuple[int, str]]] = {}
    for p in cat.publications:
        pubs.setdefault(p.professor_id, []).append((p.year or 0, p.title))
    out = {}
    for pid in cat.professors:
        codes = sorted({o.course_code for o in cat.offerings if o.professor_id == pid})
        course_titles = [cat.courses[c].course_title for c in codes if cat.courses[c].course_title]
        recent = [t for _, t in sorted(pubs.get(pid, []), reverse=True)[:max_publications]]
        keywords = cat.keywords.get(pid, [])
        parts = {"courses": course_titles, "publications": recent, "keywords": keywords}
        text = " ".join(normalise(x) for x in course_titles + recent + keywords)
        out[pid] = Profile(pid, text, tuple(codes), parts)
    return out
