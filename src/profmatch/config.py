"""Settings from environment variables and an optional local .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .recommend import Weights

DISPLAYS = ("band", "value")


class ConfigError(ValueError):
    pass


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    p = Path(path)
    out: dict[str, str] = {}
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip().strip("'\"")
    return out


@dataclass(frozen=True)
class Settings:
    db_path: Path
    matcher: str
    weights: Weights
    prior_strength: float
    min_responses: int
    display: str
    cors_origins: tuple[str, ...]
    terms_accepted: bool
    cache_dir: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, dotenv: str | Path | None = ".env") -> "Settings":
        e: dict[str, str] = {}
        if dotenv is not None:
            e.update(load_dotenv(dotenv))
        e.update(os.environ if env is None else env)
        try:
            parts = [float(x) for x in (e.get("PROFMATCH_WEIGHTS") or "0.7,0.2,0.1").split(",")]
            if len(parts) != 3:
                raise ValueError
            weights = Weights(*parts)
        except ValueError as exc:
            raise ConfigError("PROFMATCH_WEIGHTS must be three numbers >= 0, for example 0.7,0.2,0.1") from exc
        try:
            prior = float(e.get("PROFMATCH_PRIOR_STRENGTH") or 20)
            min_resp = int(e.get("PROFMATCH_MIN_RESPONSES") or 10)
        except ValueError as exc:
            raise ConfigError("PROFMATCH_PRIOR_STRENGTH and PROFMATCH_MIN_RESPONSES must be numbers") from exc
        if prior < 0 or min_resp < 0:
            raise ConfigError("PROFMATCH_PRIOR_STRENGTH and PROFMATCH_MIN_RESPONSES must be >= 0")
        display = (e.get("PROFMATCH_SIGNAL_DISPLAY") or "band").lower()
        if display not in DISPLAYS:
            raise ConfigError(f"PROFMATCH_SIGNAL_DISPLAY must be one of {DISPLAYS}")
        origins = tuple(o.strip() for o in (e.get("PROFMATCH_CORS_ORIGINS") or "").split(",") if o.strip())
        if "*" in origins:
            raise ConfigError("PROFMATCH_CORS_ORIGINS must list exact origins, not '*'")
        return cls(
            db_path=Path(e.get("PROFMATCH_DB") or "artifacts/profmatch.db"),
            matcher=e.get("PROFMATCH_MATCHER") or "tfidf",
            weights=weights,
            prior_strength=prior,
            min_responses=min_resp,
            display=display,
            cors_origins=origins,
            terms_accepted=(e.get("PROFMATCH_PORTAL_TERMS_ACCEPTED") or "").lower() == "yes",
            cache_dir=Path(e.get("PROFMATCH_CACHE_DIR") or "artifacts/portal_cache"),
        )
