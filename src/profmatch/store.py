"""SQLite storage of the catalog (stdlib `sqlite3`, no server)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .schema import Catalog, Course, Offering, Professor, Publication

DDL = """
CREATE TABLE IF NOT EXISTS professors (
    professor_id TEXT PRIMARY KEY, name TEXT NOT NULL, title TEXT, department TEXT);
CREATE TABLE IF NOT EXISTS courses (
    course_code TEXT PRIMARY KEY, course_title TEXT, department TEXT);
CREATE TABLE IF NOT EXISTS offerings (
    professor_id TEXT NOT NULL REFERENCES professors(professor_id),
    course_code TEXT NOT NULL REFERENCES courses(course_code),
    term TEXT, enrolled INTEGER NOT NULL CHECK (enrolled >= 0),
    responses INTEGER NOT NULL CHECK (responses >= 0 AND responses <= enrolled),
    overall_rating REAL CHECK (overall_rating IS NULL OR overall_rating BETWEEN 1 AND 5),
    challenge_index REAL CHECK (challenge_index IS NULL OR challenge_index BETWEEN 1 AND 7));
CREATE TABLE IF NOT EXISTS publications (
    professor_id TEXT NOT NULL REFERENCES professors(professor_id), title TEXT NOT NULL, year INTEGER);
CREATE TABLE IF NOT EXISTS keywords (
    professor_id TEXT NOT NULL REFERENCES professors(professor_id), keyword TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_offerings_course ON offerings(course_code);
"""


def save(cat: Catalog, db_path: str | Path) -> Path:
    """Replace the content of the database with the catalog, in one transaction."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript(DDL)
        with con:
            for table in ("keywords", "publications", "offerings", "courses", "professors"):
                con.execute(f"DELETE FROM {table}")
            con.executemany("INSERT INTO professors VALUES (?,?,?,?)",
                            [(p.professor_id, p.name, p.title, p.department) for p in cat.professors.values()])
            con.executemany("INSERT INTO courses VALUES (?,?,?)",
                            [(c.course_code, c.course_title, c.department) for c in cat.courses.values()])
            con.executemany("INSERT INTO offerings VALUES (?,?,?,?,?,?,?)",
                            [(o.professor_id, o.course_code, o.term, o.enrolled, o.responses, o.overall_rating,
                              o.challenge_index) for o in cat.offerings])
            con.executemany("INSERT INTO publications VALUES (?,?,?)",
                            [(p.professor_id, p.title, p.year) for p in cat.publications])
            con.executemany("INSERT INTO keywords VALUES (?,?)",
                            [(pid, k) for pid, ks in cat.keywords.items() for k in ks])
    finally:
        con.close()
    return db_path


def load(db_path: str | Path) -> Catalog:
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"database {db_path} not found; run `profmatch load` first")
    con = sqlite3.connect(db_path)
    try:
        cat = Catalog()
        for row in con.execute("SELECT professor_id, name, title, department FROM professors ORDER BY professor_id"):
            cat.professors[row[0]] = Professor(*row)
        for row in con.execute("SELECT course_code, course_title, department FROM courses ORDER BY course_code"):
            cat.courses[row[0]] = Course(*row)
        for row in con.execute("SELECT * FROM offerings ORDER BY professor_id, course_code, term"):
            cat.offerings.append(Offering(*row))
        for row in con.execute("SELECT professor_id, title, year FROM publications ORDER BY rowid"):
            cat.publications.append(Publication(*row))
        for pid, kw in con.execute("SELECT professor_id, keyword FROM keywords ORDER BY rowid"):
            cat.keywords.setdefault(pid, []).append(kw)
        return cat
    finally:
        con.close()
