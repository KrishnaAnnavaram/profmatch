"""Teaching signals with Bayesian shrinkage.

For each professor, the smoothed value of a score is

    smoothed = (sum(score_i * n_i) + m * prior) / (sum(n_i) + m)

where n_i is the number of responses of offering i, `prior` is the response-weighted mean over all
offerings, and m (`prior_strength`) is the weight of the prior in responses. A professor with few
responses stays near the prior, and the result always shows the response count.
The response rate is a separate data-quality value. It is not part of the challenge level.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .schema import Catalog, Offering

CHALLENGE_LEVELS = ("lower", "balanced", "higher")
QUALITY_BANDS = ("below average", "average", "above average")


@dataclass(frozen=True)
class TeachingSignals:
    professor_id: str
    responses: int
    enrolled: int
    response_rate: float | None
    rating: float | None          # smoothed overall rating, 1..5
    challenge: float | None       # smoothed challenge index, 1..7
    challenge_level: str | None   # lower / balanced / higher (tertiles over professors)
    quality_band: str | None      # tertiles of the smoothed rating
    sufficient: bool              # responses >= min_responses

    @property
    def quality_score(self) -> float:
        return 0.5 if self.rating is None else (self.rating - 1.0) / 4.0

    @property
    def challenge_position(self) -> float:
        return 0.5 if self.challenge is None else (self.challenge - 1.0) / 6.0

    def to_dict(self, display: str = "band") -> dict:
        d = asdict(self)
        if display == "band":
            d.pop("rating")
            d.pop("challenge")
        return d


def _weighted_mean(values: list[tuple[float, int]]) -> float | None:
    total = sum(n for _, n in values)
    return sum(v * n for v, n in values) / total if total else None


def _smooth(values: list[tuple[float, int]], prior: float | None, m: float) -> float | None:
    if prior is None:
        return None
    total = sum(n for _, n in values)
    if total + m <= 0:
        return None
    return (sum(v * n for v, n in values) + m * prior) / (total + m)


def _band(value: float | None, cuts: tuple[float, float] | None, names: tuple[str, str, str]) -> str | None:
    if value is None or cuts is None:
        return None
    lo, hi = cuts
    return names[0] if value < lo else names[2] if value > hi else names[1]


def _raw_values(cat: Catalog, offerings: list[Offering], rating_prior, challenge_prior, m: float) -> dict:
    by_prof: dict[str, list[Offering]] = {}
    for o in offerings:
        by_prof.setdefault(o.professor_id, []).append(o)
    raw = {}
    for pid in cat.professors:
        offs = by_prof.get(pid, [])
        rating = _smooth([(o.overall_rating, o.responses) for o in offs if o.overall_rating is not None],
                         rating_prior, m)
        challenge = _smooth([(o.challenge_index, o.responses) for o in offs if o.challenge_index is not None],
                            challenge_prior, m)
        raw[pid] = (sum(o.responses for o in offs), sum(o.enrolled for o in offs), rating, challenge)
    return raw


def compute_signals(cat: Catalog, prior_strength: float = 20.0, min_responses: int = 10,
                    course_code: str | None = None) -> dict[str, TeachingSignals]:
    """Signals for each professor. With `course_code`, only the offerings of that course count.

    The band cuts (tertiles) always come from the overall values of all professors, so a course with
    few teachers still gets bands.
    """
    rating_prior = _weighted_mean([(o.overall_rating, o.responses) for o in cat.offerings
                                   if o.overall_rating is not None])
    challenge_prior = _weighted_mean([(o.challenge_index, o.responses) for o in cat.offerings
                                      if o.challenge_index is not None])
    overall = _raw_values(cat, cat.offerings, rating_prior, challenge_prior, prior_strength)
    if course_code is None:
        raw = overall
    else:
        raw = _raw_values(cat, [o for o in cat.offerings if o.course_code == course_code], rating_prior,
                          challenge_prior, prior_strength)

    def tertiles(idx: int):
        vals = [v[idx] for v in overall.values() if v[idx] is not None and v[0] > 0]
        return tuple(np.percentile(vals, [100 / 3, 200 / 3])) if len(vals) >= 3 else None

    r_cuts, c_cuts = tertiles(2), tertiles(3)
    out = {}
    for pid, (responses, enrolled, rating, challenge) in raw.items():
        out[pid] = TeachingSignals(
            professor_id=pid,
            responses=responses,
            enrolled=enrolled,
            response_rate=round(responses / enrolled, 4) if enrolled else None,
            rating=None if rating is None else round(rating, 4),
            challenge=None if challenge is None else round(challenge, 4),
            challenge_level=_band(challenge, c_cuts, CHALLENGE_LEVELS),
            quality_band=_band(rating, r_cuts, QUALITY_BANDS),
            sufficient=responses >= min_responses,
        )
    return out
