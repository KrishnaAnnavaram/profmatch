"""A synthetic catalog and a labelled query set, for the offline demo and the tests.

Each synthetic professor has a main topic, a second topic, a latent teaching quality and a latent
challenge. The observed scores are noisy, and some professors have very few responses. The gold
relevance of a professor for a query uses the latent values, so the evaluation can show if the
smoothing and the scorer recover them. No real person or real portal is in this data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import Catalog, Course, Offering, Professor, Publication
from .signals import CHALLENGE_LEVELS

TOPICS: dict[str, dict] = {
    "machine learning": {
        "prefix": "DSCI",
        "phrases": ["neural networks", "deep learning", "gradient boosting", "representation learning",
                    "supervised classification", "model calibration", "transfer learning", "kernel methods"],
        "courses": ["Machine Learning Foundations", "Deep Learning", "Statistical Learning"],
    },
    "databases": {
        "prefix": "DATA",
        "phrases": ["query optimization", "index structures", "transaction processing", "relational databases",
                    "data warehouses", "storage engines", "distributed databases", "schema design"],
        "courses": ["Database Systems", "Data Warehousing", "Distributed Data Management"],
    },
    "networks": {
        "prefix": "NETW",
        "phrases": ["wireless networks", "routing protocols", "congestion control", "packet scheduling",
                    "software defined networking", "network measurement", "edge computing", "mobile networks"],
        "courses": ["Computer Networks", "Wireless Systems", "Network Programming"],
    },
    "human computer interaction": {
        "prefix": "HCID",
        "phrases": ["usability studies", "interaction design", "accessibility", "user experience research",
                    "information visualization", "participatory design", "eye tracking", "conversational interfaces"],
        "courses": ["Interaction Design", "Usability Engineering", "Information Visualization"],
    },
    "information retrieval": {
        "prefix": "INFR",
        "phrases": ["search engines", "ranking models", "query expansion", "relevance feedback",
                    "recommender systems", "text retrieval", "learning to rank", "digital libraries"],
        "courses": ["Information Retrieval", "Search Engine Design", "Recommender Systems"],
    },
    "security": {
        "prefix": "SECU",
        "phrases": ["applied cryptography", "intrusion detection", "malware analysis", "privacy engineering",
                    "authentication protocols", "access control", "threat modeling", "secure software"],
        "courses": ["Computer Security", "Applied Cryptography", "Privacy Engineering"],
    },
    "health informatics": {
        "prefix": "HINF",
        "phrases": ["electronic health records", "clinical decision support", "patient outcomes",
                    "medical imaging", "health data standards", "population health", "clinical text mining",
                    "telehealth"],
        "courses": ["Health Informatics", "Clinical Data Analytics", "Health Information Systems"],
    },
}
FIRST = ["Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Jordan", "Kendall", "Logan", "Morgan",
         "Parker", "Quinn", "Reese", "Rowan", "Sawyer", "Skyler", "Taylor"]
LAST = ["Alder", "Birch", "Cedar", "Dune", "Elm", "Fern", "Grove", "Heath", "Isle", "Juniper", "Knoll", "Lark",
        "Marsh", "North", "Oak", "Pine", "Quarry", "Ridge", "Stone", "Vale"]
TITLES = ["Assistant Professor", "Associate Professor", "Professor", "Lecturer"]
TERMS = ["2023-1", "2023-2", "2024-1", "2024-2", "2025-1", "2025-2"]


def _tertile(values: dict[str, float]) -> dict[str, str]:
    lo, hi = np.percentile(list(values.values()), [100 / 3, 200 / 3])
    return {k: CHALLENGE_LEVELS[0] if v < lo else CHALLENGE_LEVELS[2] if v > hi else CHALLENGE_LEVELS[1]
            for k, v in values.items()}


def make_catalog(n_professors: int = 60, seed: int = 0) -> tuple[Catalog, dict]:
    """Return the catalog and the latent truth (topics, quality, challenge) of each professor."""
    rng = np.random.default_rng(seed)
    topics = list(TOPICS)
    cat = Catalog()
    for t, spec in TOPICS.items():
        for j, title in enumerate(spec["courses"]):
            code = f"{spec['prefix']} {5100 + 100 * j + int(rng.integers(0, 10))}"
            cat.courses[code] = Course(code, title, t)
    by_topic = {t: [c for c in cat.courses.values() if c.department == t] for t in topics}
    truth: dict[str, dict] = {}
    for i in range(n_professors):
        pid = f"P{i:03d}"
        main = topics[i % len(topics)]
        second = topics[(i % len(topics) + 1 + int(rng.integers(0, len(topics) - 1))) % len(topics)]
        name = f"{FIRST[int(rng.integers(len(FIRST)))]} {LAST[int(rng.integers(len(LAST)))]}"
        cat.professors[pid] = Professor(pid, name, TITLES[int(rng.integers(len(TITLES)))], main)
        quality = float(np.clip(rng.normal(3.9, 0.5), 1.5, 5.0))
        challenge = float(np.clip(rng.normal(4.5, 1.0), 1.5, 7.0))
        few = rng.random() < 0.2  # a professor with very few responses
        truth[pid] = {"main": main, "second": second, "quality": quality, "challenge": challenge}
        main_phr, sec_phr = TOPICS[main]["phrases"], TOPICS[second]["phrases"]
        kws = list(rng.choice(main_phr, 4, replace=False)) + list(rng.choice(sec_phr, 2, replace=False))
        cat.keywords[pid] = [str(k) for k in kws]
        for _ in range(int(rng.integers(5, 12))):
            pool = main_phr if rng.random() < 0.7 else sec_phr
            a, b = rng.choice(pool, 2, replace=False)
            cat.publications.append(Publication(pid, f"{str(a).title()} for {b}", int(rng.integers(2012, 2026))))
        courses = list(rng.choice(by_topic[main], int(rng.integers(1, 3)), replace=False))
        if rng.random() < 0.4:
            courses.append(by_topic[second][int(rng.integers(len(by_topic[second])))])
        for course in courses:
            for term in rng.choice(TERMS, int(rng.integers(1, 4)), replace=False):
                enrolled = int(rng.integers(15, 80))
                responses = int(rng.integers(1, 4)) if few else int(enrolled * rng.uniform(0.1, 0.8))
                if rng.random() < 0.05:
                    responses = 0
                if responses == 0:
                    rating = chal = None
                else:
                    noise = 1.2 / np.sqrt(responses)
                    rating = round(float(np.clip(quality + rng.normal(0, noise), 1, 5)), 2)
                    chal = round(float(np.clip(challenge + rng.normal(0, noise * 1.5), 1, 7)), 2)
                    if few and rng.random() < 0.5:
                        rating = 5.0  # a small, enthusiastic sample
                cat.offerings.append(Offering(pid, course.course_code, str(term), enrolled, responses, rating, chal))
    levels = _tertile({p: t["challenge"] for p, t in truth.items()})
    q_hi = np.percentile([t["quality"] for t in truth.values()], 200 / 3)
    for pid, t in truth.items():
        t["challenge_level"] = levels[pid]
        t["top_quality"] = bool(t["quality"] > q_hi)
    return cat, truth


def relevance(truth: dict, topic: str, challenge: str | None) -> dict[str, int]:
    """Graded relevance: topic (2 main, 1 second) + 1 for top quality + 1 for the preferred challenge."""
    out = {}
    for pid, t in truth.items():
        grade = 2 if t["main"] == topic else 1 if t["second"] == topic else 0
        if grade == 0:
            continue
        grade += int(t["top_quality"])
        grade += int(challenge is not None and t["challenge_level"] == challenge)
        out[pid] = grade
    return out


def make_queries(cat: Catalog, truth: dict, n: int = 80, seed: int = 1) -> list[dict]:
    rng = np.random.default_rng(seed)
    topics = list(TOPICS)
    out = []
    for i in range(n):
        topic = topics[int(rng.integers(len(topics)))]
        phrases = [str(p) for p in rng.choice(TOPICS[topic]["phrases"], int(rng.integers(1, 3)), replace=False)]
        interests = "I am interested in " + " and ".join(phrases)
        course = None
        if rng.random() < 0.5:
            codes = sorted({o.course_code for o in cat.offerings if cat.courses[o.course_code].department == topic})
            course = codes[int(rng.integers(len(codes)))] if codes else None
        challenge = None if rng.random() < 0.3 else CHALLENGE_LEVELS[int(rng.integers(3))]
        out.append({"query_id": f"Q{i:03d}", "interests": interests, "course": course, "challenge": challenge,
                    "relevance": relevance(truth, topic, challenge)})
    return out


def write_queries(queries: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8")
    return path
