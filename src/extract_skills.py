"""
Skill extraction for job-market-trends.

Loads the cleaned job postings and the skills taxonomy, matches each job
description against the taxonomy using case-insensitive regex/keyword
matching (no heavy NLP model needed), and produces a long-format table:

    job_id, skill, category, role_category, location, posted_date

Run directly to (re)build the cached output:
    python src/extract_skills.py

Output: data/processed_skills.csv
"""

import re
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
JOB_POSTINGS_PATH = DATA_DIR / "job_postings.csv"
TAXONOMY_PATH = DATA_DIR / "skills_taxonomy.csv"
PROCESSED_PATH = DATA_DIR / "processed_skills.csv"


def load_job_postings(path=JOB_POSTINGS_PATH):
    """Load and clean the raw job postings CSV."""
    df = pd.read_csv(path)

    df = df.dropna(subset=["job_id", "description", "role_category"])
    df["job_title"] = df["job_title"].astype(str).str.strip()
    df["company"] = df["company"].astype(str).str.strip()
    df["location"] = df["location"].astype(str).str.strip()
    df["role_category"] = df["role_category"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()

    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    df = df.dropna(subset=["posted_date"])

    if "salary_usd" in df.columns:
        df["salary_usd"] = pd.to_numeric(df["salary_usd"], errors="coerce")

    df = df.drop_duplicates(subset=["job_id"])
    return df.reset_index(drop=True)


def load_skills_taxonomy(path=TAXONOMY_PATH):
    """Load the skills taxonomy and compile a matcher regex per skill."""
    taxonomy = pd.read_csv(path)

    compiled = []
    for _, row in taxonomy.iterrows():
        alias_terms = [row["skill"]] + str(row["aliases"]).split("|")
        # Longer terms first so multi-word aliases aren't shadowed by short ones.
        alias_terms = sorted(set(t.strip() for t in alias_terms if t.strip()), key=len, reverse=True)
        pattern = r"(?<!\w)(?:" + "|".join(alias_terms) + r")(?!\w)"
        compiled.append({
            "skill": row["skill"],
            "category": row["category"],
            "regex": re.compile(pattern, re.IGNORECASE),
        })
    return compiled


def extract_skills_from_text(text, compiled_taxonomy):
    """Return the set of skill names found in a single description."""
    found = []
    for entry in compiled_taxonomy:
        if entry["regex"].search(text):
            found.append(entry["skill"])
    return found


def build_processed_skills(postings_df, compiled_taxonomy):
    """Produce the long-format (job_id, skill, role_category, location, posted_date) table."""
    skill_lookup = {e["skill"]: e["category"] for e in compiled_taxonomy}
    records = []

    for row in postings_df.itertuples(index=False):
        matched = extract_skills_from_text(row.description, compiled_taxonomy)
        for skill in matched:
            records.append({
                "job_id": row.job_id,
                "skill": skill,
                "category": skill_lookup[skill],
                "role_category": row.role_category,
                "location": row.location,
                "posted_date": row.posted_date,
            })

    long_df = pd.DataFrame(records, columns=[
        "job_id", "skill", "category", "role_category", "location", "posted_date"
    ])
    return long_df


def build_skill_cooccurrence(long_df, top_skills=None):
    """
    Count how often pairs of skills appear together in the same posting.

    Returns a long-format DataFrame: skill_a, skill_b, count (skill_a < skill_b,
    each unordered pair counted once).
    """
    if top_skills is not None:
        long_df = long_df[long_df["skill"].isin(top_skills)]

    pair_counts = {}
    for _, skills in long_df.groupby("job_id")["skill"]:
        unique_skills = sorted(set(skills))
        for a, b in combinations(unique_skills, 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    if not pair_counts:
        return pd.DataFrame(columns=["skill_a", "skill_b", "count"])

    records = [{"skill_a": a, "skill_b": b, "count": c} for (a, b), c in pair_counts.items()]
    return pd.DataFrame(records).sort_values("count", ascending=False).reset_index(drop=True)


def run_pipeline(postings_path=JOB_POSTINGS_PATH, taxonomy_path=TAXONOMY_PATH):
    postings_df = load_job_postings(postings_path)
    compiled_taxonomy = load_skills_taxonomy(taxonomy_path)
    long_df = build_processed_skills(postings_df, compiled_taxonomy)
    return postings_df, long_df


def main():
    postings_df, long_df = run_pipeline()
    long_df.to_csv(PROCESSED_PATH, index=False)
    print(f"Loaded {len(postings_df)} postings")
    print(f"Extracted {len(long_df)} (job_id, skill) matches "
          f"({long_df['skill'].nunique()} distinct skills)")
    print(f"Wrote long-format table to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
