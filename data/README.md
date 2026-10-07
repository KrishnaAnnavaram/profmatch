# Data

No catalog data is in this repository. A real catalog holds the names of real people and their
teaching evaluation scores. It is personal data. Do not commit it and do not publish it.

## Sources

| Source | Terms | How profmatch reads it |
|---|---|---|
| A faculty portal that you are allowed to read | The terms of use of that portal. Read them first | `profmatch scrape --config <portal.toml> --out <bundle>` (extra `scrape`) |
| An export that your organisation gives you | Your data agreement | Write the five CSV files below by hand or with a script |
| The synthetic generator | MIT, no real person | `profmatch synth --out <bundle>` |

The scraper refuses to start until `PROFMATCH_PORTAL_TERMS_ACCEPTED=yes`. It obeys `robots.txt`,
waits 2 seconds between requests (`--interval`) and keeps each page in `PROFMATCH_CACHE_DIR`.

## The CSV bundle

A bundle is a folder with five UTF-8 CSV files. `profmatch load <bundle>` validates each row and
writes the SQLite database (`PROFMATCH_DB`).

| File | Columns | Rules |
|---|---|---|
| `professors.csv` | `professor_id`, `name`, `title`, `department` | `professor_id` and `name` required, `professor_id` unique |
| `courses.csv` | `course_code`, `course_title`, `department` | Code like `ABCD 1234` (2 to 5 letters, 3 to 5 digits, one optional letter) |
| `offerings.csv` | `professor_id`, `course_code`, `term`, `enrolled`, `responses`, `overall_rating`, `challenge_index` | `0 <= responses <= enrolled`. Rating 1 to 5, challenge 1 to 7, both empty when unknown. No score with 0 responses |
| `publications.csv` | `professor_id`, `title`, `year` | `year` empty or 1900 to 2100 |
| `keywords.csv` | `professor_id`, `keyword` | One keyword in each row |

## Labelled queries

`profmatch eval <queries.jsonl>` reads one JSON object in each line:

```json
{"query_id": "Q001", "interests": "search engines and ranking", "course": "INFR 5103", "challenge": "higher",
 "relevance": {"P012": 3, "P040": 1}}
```

`relevance` maps a `professor_id` to a grade (0 to 4). A professor that is not in the map has grade 0.
The synthetic query set has graded labels from the latent truth of the generator. For a real
evaluation, advisors must label the queries.
