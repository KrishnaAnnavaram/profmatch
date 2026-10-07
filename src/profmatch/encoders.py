"""Text matchers behind one interface: `fit(profiles)`, `scores(query)`, `matched_terms(query, pid)`.

- `TfidfMatcher`: word unigrams and bigrams, sublinear tf, cosine similarity (default).
- `Bm25Matcher`: Okapi BM25, scores divided by the best score of the query, so they are in [0, 1].
- `SentenceMatcher`: sentence-transformers embeddings (extra `embed`), imported lazily.
Each matcher is fit once on all professor profiles, not on a filtered subset, so the IDF weights do
not change with the candidate set.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .textproc import normalise, tokens


class Matcher(Protocol):
    name: str

    def fit(self, texts: dict[str, str]) -> "Matcher": ...
    def scores(self, query: str) -> dict[str, float]: ...
    def matched_terms(self, query: str, professor_id: str, top: int = 5) -> list[str]: ...


class TfidfMatcher:
    name = "tfidf"

    def fit(self, texts: dict[str, str]) -> "TfidfMatcher":
        self.ids = list(texts)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, token_pattern=r"(?u)\b\w[\w+-]*\b")
        docs = [texts[i] or "" for i in self.ids]
        if not any(d.strip() for d in docs):
            docs = docs + ["placeholder"]
        self.matrix = self.vec.fit_transform(docs)[: len(self.ids)]
        self.vocab = np.array(self.vec.get_feature_names_out())
        return self

    def _qvec(self, query: str):
        return self.vec.transform([normalise(query)])

    def scores(self, query: str) -> dict[str, float]:
        q = self._qvec(query)
        sims = (self.matrix @ q.T).toarray().ravel()  # rows are L2-normalised: this is the cosine
        return {pid: float(s) for pid, s in zip(self.ids, sims)}

    def matched_terms(self, query: str, professor_id: str, top: int = 5) -> list[str]:
        q = self._qvec(query)
        row = self.matrix[self.ids.index(professor_id)]
        contrib = row.multiply(q).toarray().ravel()
        order = np.argsort(contrib)[::-1]
        return [str(self.vocab[i]) for i in order[:top] if contrib[i] > 0]


class Bm25Matcher:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, texts: dict[str, str]) -> "Bm25Matcher":
        self.ids = list(texts)
        self.docs = [Counter(tokens(texts[i])) for i in self.ids]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        return self

    def _raw(self, q_terms: list[str], i: int) -> dict[str, float]:
        d, dl = self.docs[i], self.lens[i]
        out = {}
        for t in set(q_terms):
            tf = d.get(t, 0)
            if tf:
                norm = 1 - self.b + self.b * (dl / self.avg if self.avg else 1)
                out[t] = self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return out

    def scores(self, query: str) -> dict[str, float]:
        q = tokens(query)
        raw = [sum(self._raw(q, i).values()) for i in range(len(self.ids))]
        best = max(raw) if raw else 0.0
        return {pid: (s / best if best > 0 else 0.0) for pid, s in zip(self.ids, raw)}

    def matched_terms(self, query: str, professor_id: str, top: int = 5) -> list[str]:
        parts = self._raw(tokens(query), self.ids.index(professor_id))
        return [t for t, _ in sorted(parts.items(), key=lambda kv: -kv[1])[:top]]


class SentenceMatcher:  # pragma: no cover - needs the optional extra
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = f"sbert:{model_name}"
        self._terms = TfidfMatcher()

    def fit(self, texts: dict[str, str]) -> "SentenceMatcher":
        self.ids = list(texts)
        self.emb = self.model.encode([texts[i] for i in self.ids], normalize_embeddings=True)
        self._terms.fit(texts)
        return self

    def scores(self, query: str) -> dict[str, float]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return {pid: float(max(0.0, s)) for pid, s in zip(self.ids, self.emb @ q)}

    def matched_terms(self, query: str, professor_id: str, top: int = 5) -> list[str]:
        return self._terms.matched_terms(query, professor_id, top)


def build_matcher(name: str) -> Matcher:
    if name == "tfidf":
        return TfidfMatcher()
    if name == "bm25":
        return Bm25Matcher()
    if name.startswith("sbert"):
        model = name.split(":", 1)[1] if ":" in name else "all-MiniLM-L6-v2"
        return SentenceMatcher(model)
    raise ValueError(f"unknown matcher {name!r}; use tfidf, bm25 or sbert[:model]")
