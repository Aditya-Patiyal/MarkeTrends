"""
Generates a synthetic job postings dataset that mirrors the structure of
real-world Kaggle job-postings datasets, for the job-market-trends project.

Run once from the project root:
    python src/generate_data.py

Output: data/job_postings.csv (~2000 rows)

The random seed is fixed so the dataset (and therefore every downstream
chart) is reproducible across runs and machines.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_POSTINGS = 2000

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "job_postings.csv"

# ---------------------------------------------------------------------------
# Reference lists
# ---------------------------------------------------------------------------

ROLE_TITLES = {
    "Data Analyst": [
        "Data Analyst",
        "Business Data Analyst",
        "Junior Data Analyst",
        "Senior Data Analyst",
        "Reporting Analyst",
        "BI Analyst",
        "Marketing Data Analyst",
    ],
    "Data Scientist": [
        "Data Scientist",
        "Senior Data Scientist",
        "Applied Data Scientist",
        "Data Scientist II",
        "Research Data Scientist",
        "Lead Data Scientist",
    ],
    "ML Engineer": [
        "Machine Learning Engineer",
        "Senior ML Engineer",
        "AI Engineer",
        "ML Ops Engineer",
        "Applied ML Engineer",
        "Deep Learning Engineer",
    ],
    "Data Engineer": [
        "Data Engineer",
        "Senior Data Engineer",
        "Big Data Engineer",
        "Analytics Engineer",
        "ETL Developer",
        "Cloud Data Engineer",
    ],
    "Software Engineer": [
        "Software Engineer",
        "Backend Engineer",
        "Full Stack Developer",
        "Senior Software Engineer",
        "Platform Engineer",
        "Software Development Engineer",
    ],
}

ROLE_CATEGORIES = list(ROLE_TITLES.keys())

# Per-role skill pools, weighted so each role has a realistic skill profile.
# Each entry is (skill_phrase_as_used_in_text, weight).
ROLE_SKILL_POOLS = {
    "Data Analyst": [
        ("SQL", 12), ("Excel", 11), ("Power BI", 9), ("Tableau", 8),
        ("Python", 7), ("Statistics", 6), ("A/B testing", 4),
        ("Looker", 4), ("R", 3), ("Data warehousing", 3), ("ETL", 2),
        ("Google Cloud Platform", 2), ("MongoDB", 1), ("Git", 3),
    ],
    "Data Scientist": [
        ("Python", 12), ("Machine Learning", 11), ("SQL", 9),
        ("Statistics", 9), ("Scikit-learn", 7), ("Deep Learning", 6),
        ("A/B testing", 6), ("TensorFlow", 5), ("PyTorch", 5),
        ("NLP", 5), ("R", 4), ("Computer Vision", 3), ("Spark", 3),
        ("AWS", 3), ("Tableau", 2), ("Power BI", 1),
    ],
    "ML Engineer": [
        ("Python", 12), ("Machine Learning", 10), ("TensorFlow", 9),
        ("PyTorch", 9), ("Docker", 8), ("Kubernetes", 7), ("AWS", 6),
        ("Deep Learning", 6), ("Scikit-learn", 5), ("Spark", 4),
        ("NLP", 4), ("Computer Vision", 4), ("Git", 4), ("Azure", 2),
        ("Airflow", 2),
    ],
    "Data Engineer": [
        ("SQL", 11), ("Python", 10), ("Spark", 9), ("Airflow", 8),
        ("ETL", 8), ("AWS", 7), ("Snowflake", 6), ("Redshift", 5),
        ("BigQuery", 5), ("Kafka", 5), ("dbt", 4), ("Hadoop", 4),
        ("Data warehousing", 4), ("Azure", 3), ("Docker", 3),
        ("NoSQL", 2), ("Google Cloud Platform", 2),
    ],
    "Software Engineer": [
        ("Java", 9), ("JavaScript", 9), ("Python", 8), ("Git", 8),
        ("REST API", 7), ("SQL", 6), ("Docker", 6), ("AWS", 5),
        ("Kubernetes", 4), ("C++", 4), ("Linux", 4), ("Agile", 5),
        ("MongoDB", 3), ("PostgreSQL", 3), ("NoSQL", 2),
    ],
}

COMPANIES = [
    "Nimbus Analytics", "Bluepeak Technologies", "Cortex Systems", "DataForge Inc",
    "Northwind Digital", "Vertex Insight Labs", "Skyline Software", "Quantum Retail Group",
    "Meridian Health Analytics", "Orbit Financial Services", "Riverstone Tech",
    "Clearwater Consulting", "Summit Cloud Solutions", "Pixel & Co", "Granite Data Partners",
    "Lighthouse AI", "Evergreen Logistics", "Falcon FinTech", "Harbor Robotics",
    "Ironclad Security", "Junction Mobility", "Kestrel Media Group", "Lumina Commerce",
    "Nova Insurance", "Onyx Manufacturing", "Prism Media Analytics", "Redwood Bank",
    "Sable Energy", "Tundra Telecom", "Vantage Point Retail", "Willow Health Systems",
    "Zenith Aerospace", "Anchor Payments", "Brightline Education", "Coral Reef Gaming",
    "Delta Freight Systems", "Echo Streaming", "Fenwick Pharma", "Gridline Utilities",
    "Hilltop Ventures", "Ivory Real Estate Tech", "Juniper Cybersecurity", "Keystone Auto Group",
    "Lattice Biotech", "Marlin Shipping", "Nectar Foods", "Opaline Cosmetics Analytics",
    "Pathfinder Travel Tech", "Quicksilver Payments", "Rustic Agritech",
]

LOCATIONS = [
    "New York, NY", "San Francisco, CA", "Austin, TX", "Seattle, WA", "Chicago, IL",
    "Boston, MA", "Denver, CO", "Atlanta, GA", "Los Angeles, CA", "Washington, DC",
    "Bengaluru, India", "Hyderabad, India", "Pune, India", "Mumbai, India", "Delhi, India",
    "London, UK", "Berlin, Germany", "Toronto, Canada", "Singapore", "Dublin, Ireland",
    "Remote",
]
# Weights so a handful of hubs dominate, like real postings data.
LOCATION_WEIGHTS = [7, 8, 5, 5, 4, 4, 3, 3, 5, 3, 8, 5, 4, 4, 3, 4, 3, 3, 3, 2, 10]

# Approximate lat/lon per location, used only for the dashboard's map view.
# "Remote" has no fixed coordinates so it's omitted from the map (still
# included in every other chart/filter).
LOCATION_COORDS = {
    "New York, NY": (40.7128, -74.0060),
    "San Francisco, CA": (37.7749, -122.4194),
    "Austin, TX": (30.2672, -97.7431),
    "Seattle, WA": (47.6062, -122.3321),
    "Chicago, IL": (41.8781, -87.6298),
    "Boston, MA": (42.3601, -71.0589),
    "Denver, CO": (39.7392, -104.9903),
    "Atlanta, GA": (33.7490, -84.3880),
    "Los Angeles, CA": (34.0522, -118.2437),
    "Washington, DC": (38.9072, -77.0369),
    "Bengaluru, India": (12.9716, 77.5946),
    "Hyderabad, India": (17.3850, 78.4867),
    "Pune, India": (18.5204, 73.8567),
    "Mumbai, India": (19.0760, 72.8777),
    "Delhi, India": (28.7041, 77.1025),
    "London, UK": (51.5072, -0.1276),
    "Berlin, Germany": (52.5200, 13.4050),
    "Toronto, Canada": (43.6532, -79.3832),
    "Singapore": (1.3521, 103.8198),
    "Dublin, Ireland": (53.3498, -6.2603),
}

# Rough annual base-salary distribution per role (USD, mean/std), before the
# location cost-of-living multiplier below is applied. These are illustrative
# midpoints for a synthetic dataset, not a real market survey.
ROLE_SALARY_USD = {
    "Data Analyst": (75_000, 12_000),
    "Data Scientist": (125_000, 20_000),
    "ML Engineer": (145_000, 22_000),
    "Data Engineer": (120_000, 18_000),
    "Software Engineer": (115_000, 20_000),
}

# Cost-of-living / market-rate multiplier applied on top of the role's base
# USD salary, so postings in different regions land at realistic relative
# levels (still USD-denominated, for easy cross-location comparison).
LOCATION_SALARY_MULTIPLIER = {
    "New York, NY": 1.30, "San Francisco, CA": 1.40, "Austin, TX": 1.05,
    "Seattle, WA": 1.25, "Chicago, IL": 1.05, "Boston, MA": 1.20,
    "Denver, CO": 1.05, "Atlanta, GA": 1.00, "Los Angeles, CA": 1.20,
    "Washington, DC": 1.15, "Bengaluru, India": 0.40, "Hyderabad, India": 0.38,
    "Pune, India": 0.36, "Mumbai, India": 0.42, "Delhi, India": 0.38,
    "London, UK": 0.95, "Berlin, Germany": 0.85, "Toronto, Canada": 0.90,
    "Singapore": 0.95, "Dublin, Ireland": 0.90, "Remote": 1.00,
}

DESCRIPTION_TEMPLATES = [
    "We are looking for a {title} to join our growing team at {company}. "
    "You will work closely with cross-functional stakeholders to turn data into "
    "actionable insight. Key requirements include hands-on experience with {skills}. "
    "The ideal candidate is comfortable working in a fast-paced, collaborative environment "
    "and has a strong track record of delivering measurable business impact.",

    "{company} is hiring a {title} to help scale our data capabilities. "
    "In this role you'll design, build, and maintain solutions using {skills}. "
    "We're looking for someone with strong problem-solving skills, excellent communication, "
    "and a passion for data-driven decision making.",

    "As a {title} at {company}, you will partner with engineering and product teams to "
    "deliver high-quality, reliable solutions. Required skills: {skills}. "
    "Bonus points if you have experience mentoring junior team members and contributing "
    "to a strong engineering culture.",

    "{company} seeks an experienced {title} to support our expanding analytics function. "
    "Day-to-day responsibilities include working with {skills} to build scalable, "
    "production-grade systems. We value curiosity, ownership, and clear communication.",

    "Join {company} as a {title}! You'll be responsible for building and maintaining "
    "systems that rely on {skills}. This is a great opportunity for someone who enjoys "
    "solving complex problems and working with a talented, supportive team.",
]

EXPERIENCE_BLURBS = [
    "3+ years of relevant experience preferred.",
    "Entry-level candidates with strong internship experience are encouraged to apply.",
    "5+ years of professional experience required.",
    "Minimum of 2 years in a similar role.",
    "Ideal for candidates with 4-6 years of industry experience.",
]


def weighted_choice(pool, k):
    """Pick k distinct items from a (item, weight) pool without replacement."""
    items = [p[0] for p in pool]
    weights = np.array([p[1] for p in pool], dtype=float)
    weights = weights / weights.sum()
    k = min(k, len(items))
    chosen_idx = np.random.choice(len(items), size=k, replace=False, p=weights)
    return [items[i] for i in chosen_idx]


def random_posted_date(reference_date, days_back=365):
    offset = random.randint(0, days_back)
    return reference_date - timedelta(days=offset)


def random_salary(role_category, location):
    mean, std = ROLE_SALARY_USD[role_category]
    multiplier = LOCATION_SALARY_MULTIPLIER.get(location, 1.0)
    salary = np.random.normal(mean, std) * multiplier
    salary = max(salary, 20_000)  # floor to avoid unrealistic low outliers
    return int(round(salary / 500) * 500)  # round to nearest $500


def build_description(title, company, skills):
    template = random.choice(DESCRIPTION_TEMPLATES)
    if len(skills) > 1:
        skills_text = ", ".join(skills[:-1]) + f", and {skills[-1]}"
    else:
        skills_text = skills[0]
    body = template.format(title=title, company=company, skills=skills_text)
    return f"{body} {random.choice(EXPERIENCE_BLURBS)}"


def generate_dataset(n=N_POSTINGS, reference_date=None):
    if reference_date is None:
        reference_date = datetime.today()

    rows = []
    for job_id in range(1, n + 1):
        role_category = random.choice(ROLE_CATEGORIES)
        job_title = random.choice(ROLE_TITLES[role_category])
        company = random.choice(COMPANIES)
        location = random.choices(LOCATIONS, weights=LOCATION_WEIGHTS, k=1)[0]
        posted_date = random_posted_date(reference_date)

        n_skills = random.randint(4, 7)
        skills = weighted_choice(ROLE_SKILL_POOLS[role_category], n_skills)
        description = build_description(job_title, company, skills)
        salary_usd = random_salary(role_category, location)

        rows.append({
            "job_id": job_id,
            "job_title": job_title,
            "company": company,
            "location": location,
            "posted_date": posted_date.strftime("%Y-%m-%d"),
            "role_category": role_category,
            "salary_usd": salary_usd,
            "description": description,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("posted_date").reset_index(drop=True)
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reference_date = datetime(2026, 8, 19)  # fixed so output is fully reproducible
    df = generate_dataset(N_POSTINGS, reference_date=reference_date)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    print(df["role_category"].value_counts())


if __name__ == "__main__":
    main()
