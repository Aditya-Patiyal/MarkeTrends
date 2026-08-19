# Job Market & Skills Trend Analysis

An interactive dashboard that analyzes job postings to surface which skills
are trending, broken down by role category, location, and time.

**Stack:** Python 3.11, pandas, regex/keyword-based skill extraction, Streamlit + Plotly.

> **Note on the data:** `data/job_postings.csv` is a **synthetic dataset**
> (~2,000 postings, fixed random seed). It is not scraped from a real job
> board, but it is deliberately structured to mirror real-world job-postings
> data (e.g. the schema and skill-taxonomy approach used by public Kaggle
> job-postings datasets), so the pipeline and dashboard generalize to a real
> dataset with the same columns. This project is intended as a portfolio /
> demo piece. In addition to the columns in the original brief, each posting
> also carries a synthetic `salary_usd` estimate (role base pay x a location
> cost-of-living multiplier), used for the Salary Insights tab.

## Pipeline overview

```
data/job_postings.csv          data/skills_taxonomy.csv
        │                                │
        ▼                                ▼
src/generate_data.py    src/extract_skills.py (regex/keyword matching)
        │                                │
        └──────────────┬─────────────────┘
                        ▼
          data/processed_skills.csv (long format:
          job_id, skill, category, role_category,
          location, posted_date)
                        │
                        ▼
                    app.py (Streamlit + Plotly dashboard)
```

1. **Data ingestion** (`src/extract_skills.py::load_job_postings`) — loads
   `data/job_postings.csv`, strips whitespace, parses dates, drops rows with
   missing required fields, and removes duplicate `job_id`s.
2. **Skill extraction** (`src/extract_skills.py`) — for every skill in
   `data/skills_taxonomy.csv` (skill name + category + pipe-separated
   aliases), compiles a case-insensitive word-boundary regex and scans each
   job description for a match. No ML/NLP model is used — this is
   intentionally a lightweight, fully deterministic keyword-matching
   approach, which is appropriate given the taxonomy is a fixed, known list
   of ~40 skills. The result is a long-format table: one row per
   `(job_id, skill)` match, with `role_category`, `location`, and
   `posted_date` carried along for easy filtering/aggregation.
3. **Dashboard** (`app.py`) — loads and processes data through
   `st.cache_data`-wrapped functions (so re-running filters doesn't re-parse
   the CSV or re-run the regex scan), then renders six tabs:
   - **Overview** — top 15 most in-demand skills (bar chart), monthly demand
     trend for a user-selected skill (line chart), top skills compared
     across role categories (heatmap)
   - **Salary Insights** — median salary by role category and by top skill,
     plus a salary distribution box plot per role (synthetic `salary_usd`
     column: role base pay x a location cost-of-living multiplier)
   - **Skill Relationships** — a co-occurrence heatmap and ranked table
     showing which skills are most often requested together in the same
     posting
   - **Emerging Skills** — a month-over-month % change leaderboard
     (fastest-growing / fastest-declining skills) with an adjustable
     comparison window
   - **Locations** — a bubble map of postings by city (`Remote` excluded,
     no fixed coordinates)
   - **Raw Data** — the filtered postings table with a CSV download button
   - A filter panel (role category, location, posted-date range) in the
     sidebar applies to every tab
   - Sidebar summary stats: postings analyzed, date range covered, most
     in-demand skill overall — all computed on the *filtered* data

## Project structure

```
job-market-trends/
  app.py                    # Streamlit dashboard entrypoint
  src/generate_data.py      # builds the synthetic dataset (run once)
  src/extract_skills.py     # data loading + skill extraction logic
  data/job_postings.csv     # synthetic postings (generated)
  data/skills_taxonomy.csv  # ~40-skill taxonomy with aliases
  data/processed_skills.csv # long-format extraction output (cached)
  requirements.txt
  README.md
```

## Setup

Requires Python 3.11.

```bash
cd job-market-trends
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

The dataset and processed-skills files are already included in `data/`, so
you can go straight to launching the dashboard:

```bash
streamlit run app.py
```

This opens the dashboard at `http://localhost:8501`.

### Regenerating the data from scratch

If you want to rebuild the dataset (e.g. after changing the generator or the
taxonomy):

```bash
python src/generate_data.py     # rebuilds data/job_postings.csv (fixed seed = 42, always reproducible)
python src/extract_skills.py    # rebuilds data/processed_skills.csv
```

The app also runs skill extraction on the fly (cached via `st.cache_data`),
so `processed_skills.csv` is a convenience artifact for offline inspection,
not a hard dependency of the dashboard.

## Extending this project

Implemented so far: salary insights, skill co-occurrence, an emerging/declining
skills leaderboard, a location bubble map, and CSV export. Further directions
worth exploring:

- **Real data**: swap `data/job_postings.csv` for a real Kaggle job-postings
  dataset with the same column names and the pipeline should work unchanged.
- **Better extraction**: swap the regex matcher for spaCy `PhraseMatcher` or
  a small embeddings-based matcher to catch skill mentions the fixed alias
  list misses (e.g. typos, unusual phrasing).
- **Skill co-occurrence as a network graph**: the current implementation is a
  heatmap + ranked table; a force-directed graph (e.g. via `pyvis` or
  `networkx` + Plotly) could make dense clusters easier to read.
- **Real salary data**: the current `salary_usd` is a synthetic estimate
  (role base pay x location multiplier); a real dataset would let this
  reflect actual market rates.
