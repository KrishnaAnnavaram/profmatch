"""Page fetchers. `HttpFetcher` obeys robots.txt, waits between requests and caches each page.
`FixtureFetcher` serves saved HTML, so that the parser tests need no network.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser


class FetchError(RuntimeError):
    pass


class Fetcher(Protocol):
    def get(self, url: str) -> str: ...


class FixtureFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url: str) -> str:
        self.calls.append(url)
        if url not in self.pages:
            raise FetchError(f"no fixture for {url}")
        return self.pages[url]


class HttpFetcher:  # pragma: no cover - network
    def __init__(self, base_url: str, cache_dir: str | Path, min_interval_s: float = 2.0,
                 user_agent: str = "profmatch-research-bot", terms_accepted: bool = False, timeout_s: float = 20.0):
        if not terms_accepted:
            raise FetchError("read the portal terms of use, then set PROFMATCH_PORTAL_TERMS_ACCEPTED=yes")
        import httpx

        self.client = httpx.Client(timeout=timeout_s, headers={"User-Agent": user_agent}, follow_redirects=True)
        self.base = base_url
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval_s
        self.user_agent = user_agent
        self._last = 0.0
        self.robots = RobotFileParser()
        r = self.client.get(urljoin(base_url, "/robots.txt"))
        self.robots.parse(r.text.splitlines() if r.status_code == 200 else [])

    def get(self, url: str) -> str:
        url = urljoin(self.base, url)
        if urlparse(url).netloc != urlparse(self.base).netloc:
            raise FetchError(f"{url} is outside the configured portal")
        key = self.cache / (hashlib.sha1(url.encode()).hexdigest() + ".html")
        if key.exists():
            return key.read_text(encoding="utf-8")
        if not self.robots.can_fetch(self.user_agent, url):
            raise FetchError(f"robots.txt does not allow {url}")
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = self.client.get(url)
        self._last = time.monotonic()
        if r.status_code != 200:
            raise FetchError(f"{url}: HTTP {r.status_code}")
        key.write_text(r.text, encoding="utf-8")
        return r.text
