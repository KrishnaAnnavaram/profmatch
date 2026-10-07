"""Course codes, the catalog schema, SQLite storage, signals and profiles (problems 2, 4, 6, 7 and 9)."""
import csv
import dataclasses

import pytest

from profmatch.profiles import build_profiles
from profmatch.schema import SchemaError, read_bundle, write_bundle
from profmatch.signals import compute_signals
from profmatch.store import load, save
from profmatch.textproc import CourseCodeError, normalise, parse_course_code, tokens


@pytest.mark.parametrize("raw, code", [("ABCD 5000", "ABCD 5000"), ("abcd5000", "ABCD 5000"),
                                        ("abcd-5000", "ABCD 5000"), ("  Data 6101a ", "DATA 6101A")])
def test_course_codes_are_parsed_exactly(raw, code):
    assert parse_course_code(raw) == code


@pytest.mark.parametrize("raw", ["", "ABCD", "5000", "ABCD.*", "(ABCD 5000", "ABCD 50", "Hard", "A 5000"])
def test_bad_course_codes_raise_a_clear_error(raw):
    """Problem 6: input is never used as a regular expression."""
    with pytest.raises(CourseCodeError):
        parse_course_code(raw)


def test_tokens_drop_stop_words_without_a_download():
    assert tokens("I am interested in the Deep-Learning of graphs") == ["interested", "deep-learning", "graphs"]
    assert normalise("The and of") == ""


def test_bundle_round_trip_and_sqlite_round_trip(tmp_path, small_catalog):
    write_bundle(small_catalog, tmp_path / "b")
    cat = read_bundle(tmp_path / "b")
    assert cat.professors == small_catalog.professors and cat.offerings == small_catalog.offerings
    save(cat, tmp_path / "x.db")
    again = load(tmp_path / "x.db")
    assert again.offerings == cat.offerings and again.keywords == cat.keywords
    save(cat, tmp_path / "x.db")  # a second load replaces, it does not append
    assert len(load(tmp_path / "x.db").offerings) == len(cat.offerings)


def _edit(folder, name, mutate):
    path = folder / f"{name}.csv"
    rows = list(csv.reader(path.open(encoding="utf-8")))
    mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)


def test_schema_errors_are_collected(tmp_path, small_catalog):
    folder = write_bundle(small_catalog, tmp_path / "b")

    def bad(rows):
        rows[1][4] = "99"        # responses > enrolled
        rows[2][5] = "7.5"       # rating out of range
        rows.append(["Z", "DATA 5100", "t", "10", "5", "", ""])     # unknown professor
        rows.append(["A", "DATA.*", "t", "10", "5", "", ""])        # bad course code
        rows.append(["A", "DATA 5100", "t", "10", "0", "4.0", ""])  # scores with 0 responses
    _edit(folder, "offerings", bad)
    with pytest.raises(SchemaError) as err:
        read_bundle(folder)
    text = str(err.value)
    for part in ("responses 99 > enrolled", "outside [1.0, 5.0]", "unknown professor_id", "must look like",
                 "scores given with 0 responses"):
        assert part in text


def test_missing_file_or_column(tmp_path, small_catalog):
    folder = write_bundle(small_catalog, tmp_path / "b")
    (folder / "keywords.csv").unlink()
    _edit(folder, "courses", lambda rows: rows.__setitem__(0, ["code", "course_title", "department"]))
    with pytest.raises(SchemaError) as err:
        read_bundle(folder)
    assert "keywords.csv: file not found" in str(err.value) and "missing column" in str(err.value)


def test_missing_scores_stay_missing(tmp_path, small_catalog):
    """Problem 4: no imputation with the professor's own mean, no zero fill."""
    small_catalog.offerings.append(dataclasses.replace(small_catalog.offerings[0], term="2025-1", responses=0,
                                                       overall_rating=None, challenge_index=None))
    folder = write_bundle(small_catalog, tmp_path / "b")
    cat = read_bundle(folder)
    assert cat.offerings[-1].overall_rating is None and cat.offerings[-1].challenge_index is None


def test_shrinkage_keeps_small_samples_near_the_prior(small_catalog):
    sig = compute_signals(small_catalog, prior_strength=20)
    # B has 2 responses with 5.0; A has 60 responses near 4.5. A raw mean ranks B first.
    assert sig["B"].rating < sig["A"].rating
    prior = (4.6 * 30 + 4.4 * 30 + 5.0 * 2 + 3.5 * 25) / 87
    assert abs(sig["B"].rating - (5.0 * 2 + 20 * prior) / 22) < 1e-3
    assert not sig["B"].sufficient and sig["A"].sufficient
    raw = compute_signals(small_catalog, prior_strength=0)
    assert raw["B"].rating == 5.0


def test_response_rate_is_not_part_of_the_challenge(small_catalog):
    before = compute_signals(small_catalog)["A"]
    small_catalog.offerings[0] = dataclasses.replace(small_catalog.offerings[0], enrolled=300)
    after = compute_signals(small_catalog)["A"]
    assert after.response_rate < before.response_rate
    assert after.challenge == before.challenge and after.rating == before.rating


def test_course_signals_use_only_that_course(small_catalog):
    sig = compute_signals(small_catalog, course_code="NETW 5100")
    assert sig["C"].responses == 25 and sig["A"].responses == 0


def test_band_display_hides_raw_scores(small_catalog):
    """Problem 9: by default the output shows bands and counts, not raw numbers."""
    s = compute_signals(small_catalog)["A"]
    assert "rating" not in s.to_dict("band") and "quality_band" in s.to_dict("band")
    assert s.to_dict("value")["rating"] == s.rating


def test_one_profile_per_professor_without_the_name(small_catalog):
    """Problems 2 and 7: the name is not profile text, and one course title counts once."""
    profiles = build_profiles(small_catalog)
    assert set(profiles) == {"A", "B", "C"}
    a = profiles["A"]
    assert "robin" not in a.text and "example" not in a.text
    assert a.text.count("database systems") == 1 and a.course_codes == ("DATA 5100",)
    assert "query optimization" in a.text
