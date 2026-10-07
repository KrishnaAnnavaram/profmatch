import os

import pytest

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

from profmatch.recommend import Recommender  # noqa: E402
from profmatch.schema import Catalog, Course, Offering, Professor, Publication  # noqa: E402
from profmatch.synthetic import make_catalog, make_queries  # noqa: E402


@pytest.fixture(scope="session")
def synth():
    cat, truth = make_catalog(60, seed=0)
    return cat, truth, make_queries(cat, truth, 80, seed=1)


@pytest.fixture
def small_catalog():
    """Three professors, hand-made, for exact checks."""
    cat = Catalog()
    cat.professors = {
        "A": Professor("A", "Robin Example", "Professor", "Data"),
        "B": Professor("B", "Sam Sample", "Lecturer", "Data"),
        "C": Professor("C", "Alex Placeholder", "Associate Professor", "Networks"),
    }
    cat.courses = {
        "DATA 5100": Course("DATA 5100", "Database Systems", "Data"),
        "NETW 5100": Course("NETW 5100", "Computer Networks", "Networks"),
    }
    cat.offerings = [
        Offering("A", "DATA 5100", "2024-1", 40, 30, 4.6, 5.5),
        Offering("A", "DATA 5100", "2024-2", 40, 30, 4.4, 5.3),
        Offering("B", "DATA 5100", "2024-1", 40, 2, 5.0, 2.0),
        Offering("C", "NETW 5100", "2024-1", 50, 25, 3.5, 4.0),
    ]
    cat.publications = [
        Publication("A", "Query Optimization for Relational Databases", 2023),
        Publication("B", "Transaction Processing at Scale", 2022),
        Publication("C", "Congestion Control in Wireless Networks", 2021),
    ]
    cat.keywords = {"A": ["query optimization", "index structures"], "B": ["transaction processing"],
                    "C": ["wireless networks", "routing protocols"]}
    return cat


@pytest.fixture
def small_rec(small_catalog):
    return Recommender(small_catalog, "tfidf", prior_strength=20, min_responses=10)
