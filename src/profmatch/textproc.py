"""Text normalisation, a built-in stop-word list and exact course-code parsing."""
from __future__ import annotations

import re

# A short English stop-word list, so that no download runs at import time.
STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but
by can did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not of off on once only or other
our ours ourselves out over own same she so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where which while who whom why will with
you your yours yourself yourselves using based via toward towards new study studies approach approaches paper
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[-+][a-z0-9]+)*")
_COURSE = re.compile(r"^\s*([A-Za-z]{2,5})\s*[- ]?\s*(\d{3,5}[A-Za-z]?)\s*$")


class CourseCodeError(ValueError):
    """A course code does not have the form `ABCD 1234`."""


def tokens(text: str) -> list[str]:
    """Lower-case word tokens without stop words and without one-letter tokens."""
    return [t for t in _TOKEN.findall(str(text or "").lower()) if t not in STOP_WORDS and len(t) > 1]


def normalise(text: str) -> str:
    return " ".join(tokens(text))


def parse_course_code(raw: str) -> str:
    """Return the canonical form `ABCD 1234`. Raise CourseCodeError for anything else.

    The code is compared exactly after this step. It is never used as a regular expression.
    """
    m = _COURSE.match(str(raw or ""))
    if not m:
        raise CourseCodeError(f"course code {raw!r} must look like 'ABCD 1234'")
    return f"{m.group(1).upper()} {m.group(2).upper()}"
