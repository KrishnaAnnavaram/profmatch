"""The query scorer.

    score = w_text * text_match + w_quality * quality + w_fit * challenge_fit

- `text_match`: similarity between the query and the professor profile (0..1).
- `quality`: the smoothed overall rating, scaled to 0..1.
- `challenge_fit`: 1 if the challenge level equals the preference, 0.5 if it is next to it, 0 if it is
  the opposite level. Without a preference, the fit term is 0 for all candidates.
Each candidate is scored against the query itself. No step depends on the row order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .encoders import Matcher, build_matcher
from .profiles import Profile, build_profiles
from .schema import Catalog
from .signals import CHALLENGE_LEVELS, TeachingSignals, compute_signals
from .textproc import normalise, parse_course_code


class QueryError(ValueError):
    pass


@dataclass(frozen=True)
class Weights:
    text: float = 0.7
    quality: float = 0.2
    fit: float = 0.1

    def __post_init__(self):
        if min(self.text, self.quality, self.fit) < 0 or self.text + self.quality + self.fit <= 0:
            raise ValueError("weights must be >= 0 with a positive sum")


@dataclass(frozen=True)
class Query:
    interests: str
    course: str | None = None
    challenge: str | None = None
    k: int = 5

    def validated(self) -> "Query":
        course = parse_course_code(self.course) if self.course else None
        challenge = (self.challenge or "").strip().lower() or None
        if challenge is not None and challenge not in CHALLENGE_LEVELS:
            raise QueryError(f"challenge must be one of {CHALLENGE_LEVELS}, got {self.challenge!r}")
        if not normalise(self.interests) and course is None:
            raise QueryError("give interests with at least one content word, a course code, or both")
        if not 1 <= self.k <= 50:
            raise QueryError("k must be between 1 and 50")
        return Query(self.interests.strip(), course, challenge, self.k)


@dataclass
class Recommendation:
    rank: int
    professor_id: str
    name: str
    title: str
    score: float
    text_match: float
    quality: float
    challenge_fit: float
    matched_terms: list[str]
    courses: list[str]
    signals: dict
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Result:
    query: Query
    recommendations: list[Recommendation]
    candidates: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"query": asdict(self.query), "candidates": self.candidates, "notes": self.notes,
                "recommendations": [r.to_dict() for r in self.recommendations]}


def challenge_fit(level: str | None, preference: str | None) -> float:
    if preference is None:
        return 0.0
    if level is None:
        return 0.5
    gap = abs(CHALLENGE_LEVELS.index(level) - CHALLENGE_LEVELS.index(preference))
    return {0: 1.0, 1: 0.5}.get(gap, 0.0)


class Recommender:
    def __init__(self, cat: Catalog, matcher: str | Matcher = "tfidf", weights: Weights | None = None,
                 prior_strength: float = 20.0, min_responses: int = 10, display: str = "band"):
        self.cat = cat
        self.profiles: dict[str, Profile] = build_profiles(cat)
        self.matcher = build_matcher(matcher) if isinstance(matcher, str) else matcher
        self.matcher.fit({pid: p.text for pid, p in self.profiles.items()})
        self.weights = weights or Weights()
        self.prior_strength = prior_strength
        self.min_responses = min_responses
        self.display = display
        self._overall = compute_signals(cat, prior_strength, min_responses)

    def signals_for(self, course: str | None) -> dict[str, TeachingSignals]:
        if course is None:
            return self._overall
        return compute_signals(self.cat, self.prior_strength, self.min_responses, course_code=course)

    def score_all(self, q: Query) -> list[tuple[str, float, float, float, float]]:
        """(professor_id, score, text, quality, fit) for each candidate, best first."""
        return self._score(q.validated(), [])[0]

    def _score(self, q: Query, notes: list[str]):
        if q.course is not None:
            if q.course not in self.cat.courses:
                raise QueryError(f"unknown course {q.course}")
            candidates = self.cat.teachers_of(q.course)
        else:
            candidates = list(self.cat.professors)
        text = self.matcher.scores(q.interests) if normalise(q.interests) else {pid: 0.0 for pid in candidates}
        if q.course is None:
            candidates = [pid for pid in candidates if text.get(pid, 0.0) > 0]
            if not candidates:
                notes.append("no professor profile matches the interests")
        signals = self.signals_for(q.course)
        w = self.weights
        rows = []
        for pid in candidates:
            s = signals[pid]
            t = text.get(pid, 0.0)
            fit = challenge_fit(s.challenge_level, q.challenge)
            rows.append((pid, w.text * t + w.quality * s.quality_score + w.fit * fit, t, s.quality_score, fit))
        rows.sort(key=lambda r: (-r[1], r[0]))
        return rows, signals

    def recommend(self, query: Query) -> Result:
        q = query.validated()
        notes: list[str] = []
        rows, signals = self._score(q, notes)
        if q.course is not None and len(rows) < q.k:
            notes.append(f"only {len(rows)} professor(s) taught {q.course}")
        recs = []
        for rank, (pid, score, t, quality, fit) in enumerate(rows[: q.k], 1):
            prof = self.cat.professors[pid]
            sig = signals[pid]
            terms = self.matcher.matched_terms(q.interests, pid) if t > 0 else []
            reasons = []
            if terms:
                reasons.append("profile matches: " + ", ".join(terms))
            if q.course:
                reasons.append(f"taught {q.course}")
            if sig.quality_band:
                reasons.append(f"rating {sig.quality_band} ({sig.responses} responses)")
            else:
                reasons.append(f"no rating band ({sig.responses} responses)")
            if not sig.sufficient:
                reasons.append(f"few responses ({sig.responses}): the rating stays near the mean")
            if q.challenge and sig.challenge_level:
                reasons.append(f"challenge level {sig.challenge_level}, preference {q.challenge}")
            recs.append(Recommendation(rank, pid, prof.name, prof.title, round(score, 4), round(t, 4),
                                       round(quality, 4), fit, terms, list(self.profiles[pid].course_codes),
                                       sig.to_dict(self.display), reasons))
        return Result(q, recs, len(rows), notes)
