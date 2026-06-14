"""Streamlit dashboard.

Launched by `skilltrend web`. Shows product-facing trend tables, with internal
pipeline metrics separated into an Ops tab.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from skilltrend.categories import all_categories
from skilltrend.storage import load_postings, load_run_summaries
from skilltrend.trends import compute_trend


st.set_page_config(page_title="skilltrend", layout="wide")
st.title("skilltrend — agentic skill-demand analyzer")

postings = load_postings()
summaries = load_run_summaries()

all_companies = (sorted(postings["company"].dropna().unique().tolist())
                 if not postings.empty else [])

filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    selected_companies = st.multiselect(
        "Filter by company",
        options=all_companies,
        default=[],
        placeholder="All companies — select to focus the trend",
        help=("Restricts the CORPUS — totals, rising/declining tables, and Δ "
              "share denominators all recompute against the selected subset."),
    )
with filter_col2:
    selected_categories = st.multiselect(
        "Filter by skill category",
        options=all_categories(),
        default=[],
        placeholder="All categories — select to focus on a skill family",
        help=("Restricts the OUTPUT — only skills in the selected categories "
              "show up in rising/declining tables. Δ share denominators stay "
              "at the corpus-wide value so cross-category comparison stays fair."),
    )

filter_active = bool(selected_companies)
filtered_postings = (postings[postings["company"].isin(selected_companies)]
                     if filter_active else postings)

trend_tab, ops_tab = st.tabs(["Trends", "Ops"])

with trend_tab:
    c1, c2 = st.columns(2)
    c1.metric(
        "Postings" + (" (filtered)" if filter_active else ""),
        len(filtered_postings),
        delta=(f"-{len(postings) - len(filtered_postings)} vs all"
               if filter_active else None),
        delta_color="off",
    )
    c2.metric(
        "Companies" + (" (filtered)" if filter_active else ""),
        filtered_postings["company"].nunique() if not filtered_postings.empty else 0,
    )

    if filter_active and filtered_postings.empty:
        st.warning("No postings match the selected companies.")
        st.stop()

    st.divider()

    trend_companies = selected_companies if filter_active else None
    trend_categories = selected_categories if selected_categories else None

    company_scope = (", ".join(selected_companies) if filter_active
                     else f"all {len(all_companies)} companies")
    category_scope = (", ".join(selected_categories) if selected_categories
                      else "all categories")
    st.subheader(f"Skill trends — {company_scope} · {category_scope}")
    windows = {
        "30d / vs prior 60d": (30, 90),
        "90d / vs prior 90d": (90, 180),
        "180d / vs prior 180d": (180, 360),
        "365d / vs prior 365d": (365, 730),
    }
    tabs = st.tabs(list(windows.keys()))
    for tab, (label, (window, baseline)) in zip(tabs, windows.items()):
        with tab:
            report = compute_trend(window_days=window, baseline_days=baseline,
                                   top_n=25, companies=trend_companies,
                                   categories=trend_categories)
            baseline_span = baseline - window
            n_cur = report.window_total_postings
            n_base = report.baseline_total_postings

            # Prominent sample-size metric cards so the share denominator is obvious.
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric(
                    f"Current window — last {window}d",
                    f"{n_cur} postings",
                    help=f"All postings whose posted_at is within the last {window} days.",
                )
            with mc2:
                st.metric(
                    f"Baseline window — {window}–{baseline}d ago ({baseline_span}d span)",
                    f"{n_base} postings",
                    help=(f"Postings whose posted_at is between {window} and "
                          f"{baseline} days ago. Used as the comparison baseline."),
                )

            # Spell out the math so 'Current=98 / Baseline=63' isn't ambiguous.
            st.caption(
                f"**Δ share (pp)** = (skill_mentions_current / **{n_cur}**) − "
                f"(skill_mentions_baseline / **{n_base}**), in percentage points. "
                f"A skill can have *higher* count and *negative* Δ if the total "
                f"posting mix changed."
            )

            col_rising, col_declining = st.columns(2)
            cur_col = f"Mentions (of {n_cur})"
            base_col = f"Mentions (of {n_base})"
            with col_rising:
                st.markdown("**Rising**")
                st.dataframe(pd.DataFrame([{
                    "Skill": t.skill,
                    "Category": t.category,
                    cur_col: t.current_count,
                    base_col: t.baseline_count,
                    "Δ share (pp)": round(t.delta_pct, 2),
                } for t in report.rising]), hide_index=True, width="stretch")
            with col_declining:
                st.markdown("**Declining**")
                st.dataframe(pd.DataFrame([{
                    "Skill": t.skill,
                    "Category": t.category,
                    cur_col: t.current_count,
                    base_col: t.baseline_count,
                    "Δ share (pp)": round(t.delta_pct, 2),
                } for t in report.declining]), hide_index=True, width="stretch")

with ops_tab:
    st.subheader("Pipeline ops")
    st.metric("Extraction runs", len(summaries))

    st.markdown("**Extraction runs (systems metrics)**")
    if not summaries:
        st.info("Run `skilltrend extract` to populate this panel.")
    else:
        runs_df = pd.DataFrame([{
            "run_id": s.run_id, "mode": s.mode, "workers": s.workers,
            "postings": s.total_postings, "wall_s": round(s.wall_clock_s, 2),
            "throughput/s": round(s.throughput_postings_per_s, 2),
            "p50 ms": round(s.p50_latency_ms, 0),
            "p95 ms": round(s.p95_latency_ms, 0),
            "prompt_tok": s.total_prompt_tokens,
            "completion_tok": s.total_completion_tokens,
            "model": s.model,
        } for s in summaries])
        st.dataframe(runs_df, hide_index=True, width="stretch")
