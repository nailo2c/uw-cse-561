"""Time-window trend computation.

For a window of N days, count how many postings (active during that window)
mention each canonical skill. Compare against a baseline window to surface
rising and declining skills. Counts are by posting, not by total mentions —
a skill listed twice in one posting counts once."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from .blocklist import is_blocked
from .categories import category_for
from .storage import load_extractions, load_postings


@dataclass
class SkillTrend:
    skill: str
    current_count: int
    baseline_count: int
    delta_abs: int
    delta_pct: float  # ((current/total_curr) - (baseline/total_base)) * 100, points
    evidence_posting_ids: list[str]
    category: str = "Uncategorized"


@dataclass
class TrendReport:
    window_days: int
    baseline_days: int
    window_total_postings: int
    baseline_total_postings: int
    rising: list[SkillTrend]
    declining: list[SkillTrend]
    generated_at: str


def _parse_ts(value) -> datetime | None:
    """Robust timestamp parser. Handles:
       - ISO 8601 strings (Greenhouse / Ashby): "2025-03-12T16:38:15.322+00:00"
       - Unix epoch milliseconds as string (Lever): "1700000000000"
       - Empty / NaN / "nan"
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    if s.isdigit() and len(s) >= 12:
        try:
            return datetime.fromtimestamp(int(s) / 1000.0, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            pass
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Some providers (Microsoft JSON-LD) emit ISO without an offset. Treat
    # those as UTC so comparisons against window_start (always UTC-aware)
    # don't raise "offset-naive vs offset-aware".
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _effective_ts(row) -> datetime | None:
    """Prefer the ATS-reported posted_at (real history). Fall back to
    first_seen — the date we first observed the posting — when the provider
    doesn't expose a posted date.

    Why first_seen instead of last_seen: last_seen advances on every re-scan
    (the posting is still on the ATS), so a posting with empty posted_at
    would forever appear in the "current window" and never age into the
    baseline. first_seen anchors the posting to when we first saw it, which
    is the closest proxy to posted_at we have."""
    ts = _parse_ts(row.get("posted_at"))
    if ts is not None:
        return ts
    return _parse_ts(row.get("first_seen"))


def _join(companies: list[str] | None = None) -> pd.DataFrame:
    p = load_postings()
    e = load_extractions()
    if p.empty or e.empty:
        return pd.DataFrame()
    if companies:
        p = p[p["company"].isin(companies)]
        if p.empty:
            return pd.DataFrame()
    return e.merge(
        p[["posting_id", "first_seen", "last_seen", "posted_at", "company", "title"]],
        on="posting_id", how="inner",
    )


def _canonicalize_case(df: pd.DataFrame) -> pd.DataFrame:
    """Group skills by lowercased name and rewrite each row to the most
    common original casing in that group. Preserves intentional casing like
    "PyTorch" / "BigQuery" while merging accidental splits like
    "fintech" / "Fintech"."""
    if df.empty:
        return df
    df = df.copy()
    lower = df["skill"].str.lower()
    # For each lowercase form, pick the casing that appears most often in
    # the corpus. Ties resolved by first-seen.
    casing_map = (df.assign(_lower=lower)
                    .groupby("_lower")["skill"]
                    .agg(lambda s: s.value_counts().idxmax())
                    .to_dict())
    df["skill"] = lower.map(casing_map)
    return df


def _explode_skills(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in df.iterrows():
        try:
            req = json.loads(r["required_skills"] or "[]")
        except Exception:
            req = []
        try:
            pref = json.loads(r["preferred_skills"] or "[]")
        except Exception:
            pref = []
        for s in (req + pref):
            name = (s.get("name") or "").strip()
            if not name or is_blocked(name):
                continue
            rows.append({
                "posting_id": r["posting_id"],
                "skill": name,
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
                "posted_at": r.get("posted_at", ""),
            })
    return pd.DataFrame(rows)


def compute_trend(window_days: int = 30, baseline_days: int = 90,
                  top_n: int = 20,
                  companies: list[str] | None = None,
                  categories: list[str] | None = None) -> TrendReport:
    """Compute rising/declining trends.

    Filters:
    - `companies`: restrict corpus to these company names (None = all).
    - `categories`: only return skills in these categories (None = all).
      Note: companies filters the corpus (changes share denominators);
      categories filters the output (denominators stay corpus-wide so a
      Languages-only view still says 'Python is in 40% of postings').
    """
    joined = _join(companies=companies)
    if joined.empty:
        return TrendReport(window_days, baseline_days, 0, 0, [], [],
                           datetime.now(timezone.utc).isoformat(timespec="seconds"))

    exploded = _explode_skills(joined)
    if exploded.empty:
        return TrendReport(window_days, baseline_days, 0, 0, [], [],
                           datetime.now(timezone.utc).isoformat(timespec="seconds"))

    exploded = _canonicalize_case(exploded)

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=window_days)
    baseline_start = now - timedelta(days=baseline_days)

    exploded["effective_ts"] = exploded.apply(_effective_ts, axis=1)
    in_window = exploded[exploded["effective_ts"].apply(
        lambda d: d is not None and d >= window_start
    )]
    in_baseline = exploded[exploded["effective_ts"].apply(
        lambda d: d is not None and baseline_start <= d < window_start
    )]

    window_postings = in_window["posting_id"].nunique()
    baseline_postings = in_baseline["posting_id"].nunique()

    def _per_skill_counts(frame: pd.DataFrame) -> pd.Series:
        return (frame.drop_duplicates(["posting_id", "skill"])
                .groupby("skill")["posting_id"].nunique())

    cur = _per_skill_counts(in_window)
    base = _per_skill_counts(in_baseline)
    all_skills = set(cur.index) | set(base.index)

    cat_filter = set(categories) if categories else None
    trends: list[SkillTrend] = []
    for skill in all_skills:
        cat = category_for(skill)
        if cat_filter is not None and cat not in cat_filter:
            continue
        c = int(cur.get(skill, 0))
        b = int(base.get(skill, 0))
        cur_share = (c / window_postings) if window_postings else 0.0
        base_share = (b / baseline_postings) if baseline_postings else 0.0
        delta_pct = (cur_share - base_share) * 100.0
        evidence_ids = (in_window[in_window["skill"] == skill]["posting_id"]
                        .drop_duplicates().head(5).tolist())
        trends.append(SkillTrend(
            skill=skill,
            current_count=c,
            baseline_count=b,
            delta_abs=c - b,
            delta_pct=delta_pct,
            evidence_posting_ids=evidence_ids,
            category=cat,
        ))

    rising = sorted(trends, key=lambda t: t.delta_pct, reverse=True)[:top_n]
    declining = sorted(trends, key=lambda t: t.delta_pct)[:top_n]
    return TrendReport(
        window_days=window_days,
        baseline_days=baseline_days,
        window_total_postings=window_postings,
        baseline_total_postings=baseline_postings,
        rising=rising,
        declining=declining,
        generated_at=now.isoformat(timespec="seconds"),
    )


def write_report(report: TrendReport, out_dir) -> str:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.replace(":", "").replace("+0000", "Z")
    target = out / f"trend-{report.window_days}d-{stamp}.md"
    lines = [
        f"# Skill Trend Report ({report.window_days}d vs prior {report.baseline_days - report.window_days}d)",
        f"_Generated at {report.generated_at}_",
        "",
        f"- Postings active in current window: **{report.window_total_postings}**",
        f"- Postings active in baseline window: **{report.baseline_total_postings}**",
        "",
        "## Rising skills",
        "",
        "| Skill | Current | Baseline | Δ share (pp) |",
        "|---|---:|---:|---:|",
    ]
    for t in report.rising:
        lines.append(f"| {t.skill} | {t.current_count} | {t.baseline_count} | {t.delta_pct:+.2f} |")
    lines += ["", "## Declining skills", "",
              "| Skill | Current | Baseline | Δ share (pp) |",
              "|---|---:|---:|---:|"]
    for t in report.declining:
        lines.append(f"| {t.skill} | {t.current_count} | {t.baseline_count} | {t.delta_pct:+.2f} |")
    target.write_text("\n".join(lines))
    return str(target)
