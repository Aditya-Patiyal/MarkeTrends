"""
job-market-trends dashboard.

Interactive Streamlit + Plotly app that surfaces which skills are trending
in job postings, by role category and over time.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from extract_skills import (  # noqa: E402
    JOB_POSTINGS_PATH,
    TAXONOMY_PATH,
    load_job_postings,
    load_skills_taxonomy,
    build_processed_skills,
    build_skill_cooccurrence,
)
from generate_data import LOCATION_COORDS  # noqa: E402

st.set_page_config(
    page_title="MarkeTrends",
    layout="wide",
)

TOP_N_SKILLS = 15


# ---------------------------------------------------------------------------
# Cached data loading / pipeline
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and cleaning job postings...")
def get_postings():
    return load_job_postings(JOB_POSTINGS_PATH)


@st.cache_data(show_spinner="Extracting skills from job descriptions...")
def get_processed_skills(_postings_df):
    compiled_taxonomy = load_skills_taxonomy(TAXONOMY_PATH)
    long_df = build_processed_skills(_postings_df, compiled_taxonomy)
    long_df["posted_date"] = pd.to_datetime(long_df["posted_date"])
    return long_df


def load_data():
    if not JOB_POSTINGS_PATH.exists():
        st.error(
            f"Missing `{JOB_POSTINGS_PATH.relative_to(ROOT)}`. "
            "Run `python src/generate_data.py` first to create the synthetic dataset."
        )
        st.stop()
    postings_df = get_postings()
    skills_df = get_processed_skills(postings_df)
    return postings_df, skills_df


postings_df, skills_df = load_data()


# ---------------------------------------------------------------------------
# Sidebar: filters + summary stats
# ---------------------------------------------------------------------------

st.sidebar.title("Filters")

role_options = sorted(postings_df["role_category"].unique())
selected_roles = st.sidebar.multiselect(
    "Role category", role_options, default=role_options
)

location_options = sorted(postings_df["location"].unique())
selected_locations = st.sidebar.multiselect(
    "Location", location_options, default=location_options
)

min_date = postings_df["posted_date"].min().date()
max_date = postings_df["posted_date"].max().date()
date_range = st.sidebar.date_input(
    "Posted date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Apply filters to both tables
postings_filtered = postings_df[
    postings_df["role_category"].isin(selected_roles)
    & postings_df["location"].isin(selected_locations)
    & (postings_df["posted_date"].dt.date >= start_date)
    & (postings_df["posted_date"].dt.date <= end_date)
]

skills_filtered = skills_df[
    skills_df["job_id"].isin(postings_filtered["job_id"])
]

st.sidebar.markdown("---")
st.sidebar.subheader("Summary")
st.sidebar.metric("Postings analyzed", f"{len(postings_filtered):,}")
if len(postings_filtered):
    st.sidebar.metric(
        "Date range covered",
        f"{postings_filtered['posted_date'].min().date()} → "
        f"{postings_filtered['posted_date'].max().date()}",
    )
    if len(skills_filtered):
        top_skill = skills_filtered["skill"].value_counts().idxmax()
        st.sidebar.metric("Most in-demand skill", top_skill)
    else:
        st.sidebar.metric("Most in-demand skill", "—")
else:
    st.sidebar.info("No postings match the current filters.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset is synthetic, generated with a fixed random seed for "
    "reproducibility. Structured to mirror real job-posting data."
)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# --- Hero -------------------------------------------------------------------
st.title("MarkeTrends")
st.markdown("#### Job Market & Skills Trend Analysis")
st.caption(
    "Explore which skills are trending across roles, locations, and time, "
    "based on a synthetic dataset of job postings."
)

# --- What MarkeTrends offers -------------------------------------------------
st.subheader("What MarkeTrends Offers")
st.markdown(
    "MarkeTrends runs a small data pipeline over a set of job postings: it "
    "reads each posting's description, scans it against a taxonomy of about "
    "40 tech and data skills using case-insensitive keyword/regex matching "
    "(no heavy NLP model — every match is a literal skill name or a known "
    "alias, so results stay fully explainable), and turns that into a "
    "long-format table of one row per `(posting, skill)` match. Every tab "
    "below is a different way of slicing that table. The dataset itself is "
    "synthetic (2,000 generated postings with a fixed random seed), built to "
    "mirror the shape of a real job-postings dataset, so the same pipeline "
    "and charts work unchanged if you swap in real data with the same "
    "columns."
)

with st.expander("Overview — which skills are in demand, and when"):
    st.markdown(
        "- **Top skills bar chart**: counts, within your current filters, how many "
        "postings mention each skill at least once, then ranks the top 15. A skill "
        "at the top of this list is asked for broadly, across many postings — not "
        "necessarily tied to one role or company.\n"
        "- **Skill demand over time (line chart)**: pick any skill from the dropdown "
        "and see how many postings mentioned it in each calendar month, based on "
        "`posted_date`. Use this to spot whether a skill's demand is flat, rising, "
        "seasonal, or fading — the same underlying logic as the Emerging Skills tab, "
        "but for one skill you choose, viewed as a full time series instead of a "
        "single percentage.\n"
        "- **Role x skill heatmap**: takes the same top-15 skills and cross-tabulates "
        "them against role category, so you can see at a glance that, for example, "
        "SQL is heavily requested across every role while TensorFlow clusters almost "
        "entirely inside ML Engineer and Data Scientist postings."
    )

with st.expander("Salary Insights — how pay lines up with roles and skills"):
    st.markdown(
        "Every posting carries a synthetic `salary_usd` estimate: a role-specific "
        "base salary (e.g. ML Engineer pays more on average than Data Analyst) "
        "multiplied by a location cost-of-living factor (a San Francisco posting is "
        "scaled up, a Pune posting is scaled down), plus random noise. This is an "
        "illustrative estimate for demo purposes, not a real market survey, so treat "
        "absolute numbers as directional rather than authoritative.\n\n"
        "- **Median salary by role category**: the middle salary value for each role, "
        "which is more robust to outliers than a mean.\n"
        "- **Median salary by top skill**: for each of the top 15 skills, the median "
        "salary across every posting that mentions it — this is *not* saying the "
        "skill causes the pay level, just that postings mentioning it tend to cluster "
        "around that figure (a skill common in senior ML roles will show a higher "
        "number than one common in entry-level analyst roles, for instance).\n"
        "- **Salary distribution box plot**: shows the full spread (quartiles and "
        "outliers) per role category, not just the median, so you can see how wide "
        "or narrow the pay range is within a role."
    )

with st.expander("Skill Relationships — which skills are asked for together"):
    st.markdown(
        "For every posting, MarkeTrends looks at the *set* of skills it mentions and "
        "counts every pair that co-occurs (e.g. a posting mentioning Python, SQL, and "
        "AWS contributes one count each to the Python-SQL, Python-AWS, and SQL-AWS "
        "pairs). Do this across all postings and you get a picture of which skills "
        "travel together in real requirements — useful for spotting a natural skill "
        "bundle to learn as a set (like Docker + Kubernetes + AWS) rather than in "
        "isolation.\n\n"
        "The **'Number of top skills to compare' slider** controls how many of the "
        "most in-demand skills (by count, within your filters) are included in the "
        "comparison — a smaller number keeps the heatmap focused on the biggest "
        "skills, a larger number surfaces more, rarer combinations. The heatmap is "
        "symmetric (skill A x skill B is the same as skill B x skill A); the table "
        "below it lists the single strongest pairs by raw co-occurrence count."
    )

with st.expander("Emerging Skills — what's trending up or down right now"):
    st.markdown(
        "This tab splits your filtered date range into two equal, adjacent windows — "
        "a **recent** window and the **prior** window immediately before it — and "
        "compares how many postings mentioned each skill in each window. The percent "
        "change is `(recent - prior) / prior x 100`, so a skill that went from 10 "
        "mentions to 15 shows +50%.\n\n"
        "The **'Comparison window (months)' slider** sets how many months make up "
        "each of the two windows (e.g. a window of 3 compares the last 3 months "
        "against the 3 months before that). To avoid noisy, meaningless swings from "
        "skills with only a handful of mentions, any skill with fewer than 5 "
        "mentions in the prior window is excluded from the leaderboard entirely. "
        "Growing skills are shown in green, declining skills in red — both ranked by "
        "the size of the swing, not the absolute mention count."
    )

with st.expander("Locations — where the postings are"):
    st.markdown(
        "A bubble map of postings by city, using fixed latitude/longitude "
        "coordinates for each location in the dataset. Bubble size and color both "
        "scale with the number of postings at that location within your current "
        "filters, so it doubles as both a geographic and a magnitude view. `Remote` "
        "postings have no physical coordinates and are therefore left off the map "
        "entirely — they're still counted in every other tab, filter, and the "
        "sidebar summary, just not plotted here."
    )

with st.expander("Raw Data — the underlying postings, and exporting them"):
    st.markdown(
        "A row-level table of every posting that currently matches your filters "
        "(job title, company, location, role category, posted date, and salary), "
        "sorted most-recent first. Use the **download button** below the table to "
        "export exactly what you're looking at — the filtered subset, not the full "
        "dataset — as a CSV file."
    )

with st.expander("How to Use"):
    st.markdown(
        "**1. Start with the sidebar filters.** Role category, location, and posted-"
        "date range all apply globally — every chart, table, and the sidebar summary "
        "stats recompute against whatever subset you've selected. There's no need to "
        "re-apply filters per tab; set them once on the left and switch tabs freely.\n\n"
        "**2. Pick a starting tab based on the question you have.** "
        "*'What's hot right now?'* → Overview's top-skills bar chart. "
        "*'Is X trending up or down?'* → either the Overview trend line for skill X "
        "specifically, or the Emerging Skills leaderboard for a ranked view across "
        "many skills at once. *'What does role Y pay, and what does it need?'* → "
        "Salary Insights plus the Overview heatmap. *'What should I learn alongside "
        "skill X?'* → Skill Relationships. *'Where are these jobs?'* → Locations.\n\n"
        "**3. Use each tab's own controls to go deeper.** The Overview tab's skill "
        "dropdown, the Skill Relationships slider, and the Emerging Skills window "
        "slider all reshape their chart in place without touching your sidebar "
        "filters — they're a second, finer layer of control local to that tab.\n\n"
        "**4. Watch the sidebar summary as a sanity check.** If a chart looks sparse "
        "or empty, check 'Postings analyzed' in the sidebar first — an overly narrow "
        "filter combination (e.g. one role, one city, one month) can leave too few "
        "postings for a meaningful chart, and some views (like Emerging Skills) "
        "intentionally suppress low-volume, noisy results.\n\n"
        "**5. Export when you're done exploring.** The Raw Data tab's download "
        "button always reflects your current filters, so you can take the exact "
        "slice you were just looking at and continue the analysis elsewhere."
    )

st.divider()

if postings_filtered.empty:
    st.warning("No data matches the selected filters. Try widening your selection.")
    st.stop()

top_overall_skills = (
    skills_filtered["skill"].value_counts().head(TOP_N_SKILLS).index.tolist()
)

tab_overview, tab_salary, tab_relationships, tab_emerging, tab_map, tab_raw = st.tabs(
    ["Overview", "Salary Insights", "Skill Relationships", "Emerging Skills", "Locations", "Raw Data"]
)

# ---------------------------------------------------------------------------
# Overview: top skills, monthly trend for a chosen skill, role x skill heatmap
# ---------------------------------------------------------------------------
with tab_overview:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Top {TOP_N_SKILLS} In-Demand Skills")
        top_skills_counts = (
            skills_filtered["skill"].value_counts().head(TOP_N_SKILLS).sort_values()
        )
        fig_top_skills = px.bar(
            top_skills_counts,
            x=top_skills_counts.values,
            y=top_skills_counts.index,
            orientation="h",
            labels={"x": "Number of postings mentioning skill", "y": "Skill"},
            color=top_skills_counts.values,
            color_continuous_scale="Blues",
        )
        fig_top_skills.update_layout(showlegend=False, coloraxis_showscale=False, height=520)
        st.plotly_chart(fig_top_skills, width="stretch")

    with col2:
        st.subheader("Skill Demand Over Time")
        all_skills_sorted = sorted(skills_filtered["skill"].unique())
        default_skill = (
            skills_filtered["skill"].value_counts().idxmax() if all_skills_sorted else None
        )
        default_index = (
            all_skills_sorted.index(default_skill) if default_skill in all_skills_sorted else 0
        )
        selected_skill = st.selectbox(
            "Select a skill", all_skills_sorted, index=default_index if all_skills_sorted else 0
        )

        skill_trend = skills_filtered[skills_filtered["skill"] == selected_skill].copy()
        skill_trend["month"] = skill_trend["posted_date"].dt.to_period("M").dt.to_timestamp()
        monthly_counts = (
            skill_trend.groupby("month").size().reset_index(name="postings")
        )
        fig_trend = px.line(
            monthly_counts,
            x="month",
            y="postings",
            markers=True,
            labels={"month": "Month", "postings": "Postings mentioning skill"},
        )
        fig_trend.update_layout(height=520)
        st.plotly_chart(fig_trend, width="stretch")

    st.subheader("Top Skills by Role Category")
    role_skill_matrix = (
        skills_filtered[skills_filtered["skill"].isin(top_overall_skills)]
        .groupby(["role_category", "skill"])
        .size()
        .reset_index(name="count")
    )
    pivot = role_skill_matrix.pivot(index="skill", columns="role_category", values="count").fillna(0)
    pivot = pivot.reindex(top_overall_skills)

    fig_heatmap = px.imshow(
        pivot,
        labels=dict(x="Role Category", y="Skill", color="Postings"),
        aspect="auto",
        color_continuous_scale="Blues",
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, width="stretch")

# ---------------------------------------------------------------------------
# Salary Insights
# ---------------------------------------------------------------------------
with tab_salary:
    if "salary_usd" not in postings_filtered.columns:
        st.info("No `salary_usd` column found in the postings data.")
    else:
        st.caption(
            "Salaries are synthetic, USD-normalized estimates (role base pay x a "
            "location cost-of-living multiplier) meant to illustrate relative "
            "trends, not real market data."
        )
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Median Salary by Role Category")
            role_salary = (
                postings_filtered.groupby("role_category")["salary_usd"]
                .median()
                .sort_values()
            )
            fig_role_salary = px.bar(
                role_salary,
                x=role_salary.values,
                y=role_salary.index,
                orientation="h",
                labels={"x": "Median annual salary (USD)", "y": "Role category"},
                color=role_salary.values,
                color_continuous_scale="Greens",
            )
            fig_role_salary.update_layout(showlegend=False, coloraxis_showscale=False, height=460)
            st.plotly_chart(fig_role_salary, width="stretch")

        with col2:
            st.subheader("Median Salary by Top Skill")
            skill_salary_df = skills_filtered[skills_filtered["skill"].isin(top_overall_skills)].merge(
                postings_filtered[["job_id", "salary_usd"]], on="job_id", how="left"
            )
            skill_salary = (
                skill_salary_df.groupby("skill")["salary_usd"].median().sort_values()
            )
            fig_skill_salary = px.bar(
                skill_salary,
                x=skill_salary.values,
                y=skill_salary.index,
                orientation="h",
                labels={"x": "Median annual salary (USD)", "y": "Skill"},
                color=skill_salary.values,
                color_continuous_scale="Greens",
            )
            fig_skill_salary.update_layout(showlegend=False, coloraxis_showscale=False, height=460)
            st.plotly_chart(fig_skill_salary, width="stretch")

        st.subheader("Salary Distribution by Role Category")
        fig_box = px.box(
            postings_filtered,
            x="role_category",
            y="salary_usd",
            color="role_category",
            labels={"role_category": "Role category", "salary_usd": "Annual salary (USD)"},
        )
        fig_box.update_layout(showlegend=False, height=480)
        st.plotly_chart(fig_box, width="stretch")

# ---------------------------------------------------------------------------
# Skill Relationships: co-occurrence heatmap
# ---------------------------------------------------------------------------
with tab_relationships:
    st.subheader("Which Skills Appear Together?")
    st.caption(
        "How often pairs of top skills are mentioned in the same posting — "
        "useful for spotting common skill combinations employers ask for."
    )
    cooc_top_n = st.slider("Number of top skills to compare", 5, 20, 10)
    cooc_skills = skills_filtered["skill"].value_counts().head(cooc_top_n).index.tolist()

    cooc_df = build_skill_cooccurrence(skills_filtered, top_skills=cooc_skills)

    if cooc_df.empty:
        st.info("Not enough overlapping data to compute skill co-occurrence for this selection.")
    else:
        matrix = pd.DataFrame(0, index=cooc_skills, columns=cooc_skills, dtype=int)
        for row in cooc_df.itertuples(index=False):
            matrix.loc[row.skill_a, row.skill_b] = row.count
            matrix.loc[row.skill_b, row.skill_a] = row.count

        fig_cooc = px.imshow(
            matrix,
            labels=dict(x="Skill", y="Skill", color="Postings together"),
            aspect="auto",
            color_continuous_scale="Purples",
        )
        fig_cooc.update_layout(height=600)
        st.plotly_chart(fig_cooc, width="stretch")

        st.subheader("Top Skill Pairs")
        st.dataframe(
            cooc_df.rename(columns={"skill_a": "Skill A", "skill_b": "Skill B", "count": "Postings together"}).head(15),
            width="stretch",
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# Emerging vs. declining skills
# ---------------------------------------------------------------------------
with tab_emerging:
    st.subheader("Fastest-Growing & Fastest-Declining Skills")

    skills_with_month = skills_filtered.copy()
    skills_with_month["month"] = skills_with_month["posted_date"].dt.to_period("M").dt.to_timestamp()
    available_months = sorted(skills_with_month["month"].unique())

    if len(available_months) < 2:
        st.info("Need at least two distinct months in the filtered range to compute a trend.")
    else:
        window = st.slider(
            "Comparison window (months)", 1, max(1, len(available_months) // 2),
            min(3, max(1, len(available_months) // 2)),
        )
        recent_months = available_months[-window:]
        prior_months = available_months[-2 * window:-window]

        if not prior_months:
            st.info("Not enough history for the selected window. Try a smaller window or a wider date range.")
        else:
            recent_counts = (
                skills_with_month[skills_with_month["month"].isin(recent_months)]
                .groupby("skill").size()
            )
            prior_counts = (
                skills_with_month[skills_with_month["month"].isin(prior_months)]
                .groupby("skill").size()
            )

            change_df = pd.DataFrame({"recent": recent_counts, "prior": prior_counts}).fillna(0)
            change_df["pct_change"] = (
                (change_df["recent"] - change_df["prior"]) / change_df["prior"].replace(0, pd.NA)
            ) * 100
            change_df = change_df.dropna(subset=["pct_change"])
            change_df = change_df[change_df["prior"] >= 5]  # avoid noisy % change on tiny bases
            change_df = change_df.sort_values("pct_change", ascending=False)

            if change_df.empty:
                st.info("Not enough postings per skill in this range to compute a reliable trend.")
            else:
                top_growth = change_df.head(10).sort_values("pct_change")
                top_decline = change_df.tail(10).sort_values("pct_change")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Growing** (last {window} mo. vs. prior {window} mo.)")
                    fig_growth = px.bar(
                        top_growth, x="pct_change", y=top_growth.index, orientation="h",
                        labels={"pct_change": "% change", "y": "Skill"},
                        color_discrete_sequence=["#2e7d32"],
                    )
                    fig_growth.update_layout(height=420, showlegend=False)
                    st.plotly_chart(fig_growth, width="stretch")

                with col2:
                    st.markdown(f"**Declining** (last {window} mo. vs. prior {window} mo.)")
                    fig_decline = px.bar(
                        top_decline, x="pct_change", y=top_decline.index, orientation="h",
                        labels={"pct_change": "% change", "y": "Skill"},
                        color_discrete_sequence=["#c62828"],
                    )
                    fig_decline.update_layout(height=420, showlegend=False)
                    st.plotly_chart(fig_decline, width="stretch")

# ---------------------------------------------------------------------------
# Locations: postings by geography
# ---------------------------------------------------------------------------
with tab_map:
    st.subheader("Postings by Location")
    st.caption("'Remote' postings are excluded from the map (no fixed coordinates) but are still counted elsewhere.")

    location_counts = postings_filtered["location"].value_counts().reset_index()
    location_counts.columns = ["location", "postings"]
    location_counts["lat"] = location_counts["location"].map(lambda loc: LOCATION_COORDS.get(loc, (None, None))[0])
    location_counts["lon"] = location_counts["location"].map(lambda loc: LOCATION_COORDS.get(loc, (None, None))[1])
    location_counts = location_counts.dropna(subset=["lat", "lon"])

    if location_counts.empty:
        st.info("No mappable locations in the current filter selection.")
    else:
        fig_map = go.Figure(
            go.Scattergeo(
                lon=location_counts["lon"],
                lat=location_counts["lat"],
                text=location_counts["location"] + ": " + location_counts["postings"].astype(str) + " postings",
                marker=dict(
                    size=location_counts["postings"],
                    sizemode="area",
                    sizeref=2.0 * location_counts["postings"].max() / (45.0 ** 2),
                    sizemin=4,
                    color=location_counts["postings"],
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(title="Postings"),
                    line=dict(width=0.5, color="#444"),
                ),
                hoverinfo="text",
            )
        )
        fig_map.update_geos(showcountries=True, showcoastlines=True, showland=True, landcolor="#f0f0f0")
        fig_map.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_map, width="stretch")

# ---------------------------------------------------------------------------
# Raw data + export
# ---------------------------------------------------------------------------
with tab_raw:
    st.subheader("Filtered Postings")
    display_cols = ["job_id", "job_title", "company", "location", "role_category", "posted_date"]
    if "salary_usd" in postings_filtered.columns:
        display_cols.append("salary_usd")

    display_df = postings_filtered[display_cols].sort_values("posted_date", ascending=False)
    st.dataframe(display_df, width="stretch")

    st.download_button(
        "Download filtered postings as CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="job_postings_filtered.csv",
        mime="text/csv",
    )
