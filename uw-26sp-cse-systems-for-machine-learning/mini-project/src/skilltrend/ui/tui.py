"""Textual TUI.

Three panes:
- left:   tabs for trend window (30d / 90d / 180d / 365d), each with a sortable
          table of rising skills.
- right:  details for the selected skill — sample evidence postings.
- bottom: latest extraction-run metrics, so the systems analysis is one
          keystroke away.
"""
from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (DataTable, Footer, Header, Static, TabbedContent,
                             TabPane)

from ..storage import load_postings, load_run_summaries
from ..trends import compute_trend


WINDOWS = [
    ("30d / vs prior 60d", 30, 90),
    ("90d / vs prior 90d", 90, 180),
    ("180d / vs prior 180d", 180, 360),
    ("365d / vs prior 365d", 365, 730),
]


class SkillTrendApp(App):
    CSS = """
    Screen { layout: vertical; }
    #top { height: 1fr; }
    #left { width: 2fr; }
    #right { width: 1fr; border-left: solid grey; padding: 0 1; }
    #bottom { height: 12; border-top: solid grey; padding: 0 1; }
    DataTable { height: 1fr; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reload", "Reload"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top"):
            with Vertical(id="left"):
                with TabbedContent():
                    for label, window, baseline in WINDOWS:
                        with TabPane(label, id=f"tab-{window}"):
                            table = DataTable(id=f"table-{window}", cursor_type="row")
                            table.zebra_stripes = True
                            yield table
            with Vertical(id="right"):
                yield Static("Select a skill to see evidence postings.",
                             id="detail", expand=True)
        yield Static(id="bottom")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "skilltrend"
        self.sub_title = "agentic skill-demand TUI"
        self.action_reload()

    # ---------------------------------------------------------------- actions

    def action_reload(self) -> None:
        postings = load_postings()
        self._postings_df = postings
        self._reports = {}
        for label, window, baseline in WINDOWS:
            report = compute_trend(window_days=window, baseline_days=baseline, top_n=25)
            self._reports[window] = report
            table = self.query_one(f"#table-{window}", DataTable)
            table.clear(columns=True)
            table.add_columns("Skill", "Δ pp", "Current", "Baseline")
            for t in report.rising:
                table.add_row(t.skill, f"{t.delta_pct:+.2f}",
                              str(t.current_count), str(t.baseline_count),
                              key=t.skill)
        self._update_bottom()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        skill = event.row_key.value if event.row_key else None
        if not skill:
            return
        window = self._active_window()
        report = self._reports.get(window)
        if report is None:
            return
        match = next((t for t in report.rising + report.declining if t.skill == skill), None)
        if match is None or self._postings_df.empty:
            self.query_one("#detail", Static).update(f"No postings found for {skill}.")
            return
        sample = self._postings_df[self._postings_df["posting_id"].isin(match.evidence_posting_ids)]
        body = [f"[b]{skill}[/]", f"current: {match.current_count}  baseline: {match.baseline_count}",
                f"Δ share: {match.delta_pct:+.2f} pp", ""]
        for _, row in sample.iterrows():
            body.append(f"• {row['company']} — {row['title']}")
            if row.get("location"):
                body.append(f"  {row['location']}")
            body.append(f"  {row['url']}")
            body.append("")
        self.query_one("#detail", Static).update("\n".join(body))

    def _active_window(self) -> int:
        tabs = self.query_one(TabbedContent)
        active = tabs.active
        if not active:
            return WINDOWS[0][1]
        return int(active.split("-")[-1])

    def _update_bottom(self) -> None:
        summaries = load_run_summaries()[-5:]
        if not summaries:
            self.query_one("#bottom", Static).update(
                "no extraction runs yet — run `skilltrend extract`")
            return
        lines = ["[b]recent extraction runs[/]"]
        for s in summaries:
            lines.append(
                f"  {s.run_id}  mode={s.mode:<10}  workers={s.workers}  "
                f"n={s.total_postings}  wall={s.wall_clock_s:.2f}s  "
                f"thr={s.throughput_postings_per_s:.2f}/s  "
                f"p50={s.p50_latency_ms:.0f}ms  p95={s.p95_latency_ms:.0f}ms"
            )
        self.query_one("#bottom", Static).update("\n".join(lines))


if __name__ == "__main__":
    SkillTrendApp().run()
