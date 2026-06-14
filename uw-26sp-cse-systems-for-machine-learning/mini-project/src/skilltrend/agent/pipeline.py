"""Sequential vs concurrent extraction pipelines.

Both modes share the same per-posting worker; the difference is purely how
many run at once. We keep them in one file so the paper's comparison ("does
worker concurrency improve end-to-end latency, and how does it interact with
backend queueing?") is a single readable diff."""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
import uuid
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ..blocklist import is_blocked
from ..models import Extraction, Posting, RunMetric, RunSummary, utcnow_iso
from ..settings import settings
from ..storage import append_extractions, append_metric, write_run_summary
from .extractor import extract_one
from .normalizer import normalize

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    mode: str  # "sequential" | "concurrent"
    workers: int
    run_id: str


def _s(row: pd.Series, key: str, default: str = "") -> str:
    """Pandas reads empty CSV cells as NaN (float). `nan or ""` is `nan`
    because nan is truthy — so we need an explicit isna check before
    handing the value to a Pydantic str field."""
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value)


def _row_to_posting(row: pd.Series) -> Posting:
    return Posting(
        posting_id=_s(row, "posting_id"),
        url=_s(row, "url"),
        company=_s(row, "company"),
        title=_s(row, "title"),
        location=_s(row, "location"),
        source=_s(row, "source"),
        first_seen=_s(row, "first_seen"),
        last_seen=_s(row, "last_seen"),
        active=bool(row.get("active", True)),
        posted_at=_s(row, "posted_at"),
        description=_s(row, "description"),
    )


def _normalize_extraction(e: Extraction) -> Extraction:
    # Drop blocklisted (generic / soft / work-activity) names, then canonicalise.
    e.required_skills = [s for s in e.required_skills if not is_blocked(s.name)]
    e.preferred_skills = [s for s in e.preferred_skills if not is_blocked(s.name)]
    for s in e.required_skills:
        s.name = normalize(s.name)
    for s in e.preferred_skills:
        s.name = normalize(s.name)
    return e


async def _process(posting: Posting, cfg: PipelineConfig) -> tuple[Extraction | None, RunMetric]:
    start = time.perf_counter()
    start_ts = time.time()
    try:
        extraction = await extract_one(posting, run_id=cfg.run_id)
        extraction = _normalize_extraction(extraction)
        end = time.perf_counter()
        metric = RunMetric(
            run_id=cfg.run_id,
            posting_id=posting.posting_id,
            mode=cfg.mode,
            workers=cfg.workers,
            model=extraction.model,
            start_ts=start_ts,
            end_ts=time.time(),
            latency_ms=(end - start) * 1000.0,
            prompt_tokens=extraction.prompt_tokens,
            completion_tokens=extraction.completion_tokens,
            total_tokens=extraction.prompt_tokens + extraction.completion_tokens,
            ok=True,
        )
        return extraction, metric
    except Exception as exc:  # noqa: BLE001
        end = time.perf_counter()
        metric = RunMetric(
            run_id=cfg.run_id,
            posting_id=posting.posting_id,
            mode=cfg.mode,
            workers=cfg.workers,
            model=settings.model,
            start_ts=start_ts,
            end_ts=time.time(),
            latency_ms=(end - start) * 1000.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            ok=False,
            error=str(exc)[:240],
        )
        log.warning("extraction failed for %s: %s", posting.posting_id, exc)
        return None, metric


ProgressCb = Callable[[int, int, RunMetric], None]


async def run_pipeline(postings_df: pd.DataFrame, *, mode: str, workers: int,
                       on_progress: ProgressCb | None = None) -> RunSummary:
    if mode not in {"sequential", "concurrent"}:
        raise ValueError(f"unknown mode: {mode}")
    if mode == "sequential":
        workers = 1
    cfg = PipelineConfig(mode=mode, workers=workers, run_id=str(uuid.uuid4())[:12])

    postings = [_row_to_posting(row) for _, row in postings_df.iterrows()]
    total = len(postings)
    started_at = utcnow_iso()
    wall_start = time.perf_counter()

    extractions: list[Extraction] = []
    metrics: list[RunMetric] = []
    done = 0

    if mode == "sequential":
        for p in postings:
            ext, metric = await _process(p, cfg)
            append_metric(metric)
            metrics.append(metric)
            if ext is not None:
                extractions.append(ext)
            done += 1
            if on_progress is not None:
                on_progress(done, total, metric)
    else:
        sem = asyncio.Semaphore(workers)

        async def _bounded(p: Posting):
            async with sem:
                return await _process(p, cfg)

        tasks = [asyncio.create_task(_bounded(p)) for p in postings]
        for fut in asyncio.as_completed(tasks):
            ext, metric = await fut
            append_metric(metric)
            metrics.append(metric)
            if ext is not None:
                extractions.append(ext)
            done += 1
            if on_progress is not None:
                on_progress(done, total, metric)

    wall = time.perf_counter() - wall_start
    finished_at = utcnow_iso()

    append_extractions(extractions)

    latencies = [m.latency_ms for m in metrics if m.ok]
    summary = RunSummary(
        run_id=cfg.run_id,
        mode=mode,
        workers=workers,
        model=settings.model,
        started_at=started_at,
        finished_at=finished_at,
        total_postings=len(postings),
        successful=sum(1 for m in metrics if m.ok),
        failed=sum(1 for m in metrics if not m.ok),
        total_latency_s=sum(m.latency_ms for m in metrics) / 1000.0,
        wall_clock_s=wall,
        total_prompt_tokens=sum(m.prompt_tokens for m in metrics),
        total_completion_tokens=sum(m.completion_tokens for m in metrics),
        throughput_postings_per_s=(len(postings) / wall) if wall > 0 else 0.0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        p50_latency_ms=statistics.median(latencies) if latencies else 0.0,
        p95_latency_ms=_pct(latencies, 95) if latencies else 0.0,
    )
    write_run_summary(summary)
    return summary


def _pct(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    k = max(0, min(len(vs) - 1, int(round((p / 100.0) * (len(vs) - 1)))))
    return vs[k]
