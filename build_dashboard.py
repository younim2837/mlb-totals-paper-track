"""
Build a static share-friendly dashboard from paper-tracking outputs.

Usage:
    python build_dashboard.py
    python build_dashboard.py --season 2026
"""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = PROJECT_DIR / "predictions"
PAPER_TRACKING_DIR = PROJECT_DIR / "paper_tracking"
DASHBOARD_DIR = PROJECT_DIR / "docs"
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


@dataclass
class DashboardSummary:
    tracked: int
    settled: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    roi_pct: float
    profit: float
    average_edge_pct: float
    total_recommended_bet: float


def current_season() -> int:
    return int(datetime.now(PACIFIC_TZ).strftime("%Y"))


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def load_tracker(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return _empty_df(columns)
    try:
        df = pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return _empty_df(columns)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def summarize_kalshi(df: pd.DataFrame) -> DashboardSummary:
    if df.empty:
        return DashboardSummary(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    settled = df[df["settled"].astype(str).str.lower().eq("true")].copy()
    wins = int((settled["result"] == "win").sum())
    losses = int((settled["result"] == "loss").sum())
    pushes = int((settled["result"] == "push").sum())
    non_push = wins + losses
    total_cost = float(pd.to_numeric(settled["kalshi_side_market_price"], errors="coerce").fillna(0).sum())
    profit = float(pd.to_numeric(settled["profit_per_contract"], errors="coerce").fillna(0).sum())
    roi_pct = profit / total_cost * 100.0 if total_cost > 0 else 0.0
    avg_edge = float(pd.to_numeric(df["kalshi_edge_pct"], errors="coerce").dropna().mean()) if not df.empty else 0.0
    total_bet = float(pd.to_numeric(df["kalshi_recommended_bet"], errors="coerce").fillna(0).sum())
    return DashboardSummary(
        tracked=int(len(df)),
        settled=int(len(settled)),
        wins=wins,
        losses=losses,
        pushes=pushes,
        win_rate=(wins / non_push) if non_push else 0.0,
        roi_pct=roi_pct,
        profit=profit,
        average_edge_pct=avg_edge,
        total_recommended_bet=total_bet,
    )


def _data_row_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return -1


def _choose_best_daily_file(paths: list[Path]) -> list[Path]:
    chosen_by_date: dict[str, Path] = {}
    row_counts: dict[Path, int] = {}
    for path in sorted(paths):
        target_date = path.name[:10]
        current = chosen_by_date.get(target_date)
        if current is None:
            chosen_by_date[target_date] = path
            continue
        current_rows = row_counts.setdefault(current, _data_row_count(current))
        candidate_rows = row_counts.setdefault(path, _data_row_count(path))
        current_is_all_games = current.name.endswith("-all-games-picks.tsv") or current.name.endswith("-all-games-board.tsv")
        candidate_is_all_games = path.name.endswith("-all-games-picks.tsv") or path.name.endswith("-all-games-board.tsv")
        if candidate_rows > current_rows:
            chosen_by_date[target_date] = path
        elif candidate_rows == current_rows and candidate_is_all_games and not current_is_all_games:
            chosen_by_date[target_date] = path
    return [chosen_by_date[key] for key in sorted(chosen_by_date)]


def latest_picks_file(season: int) -> Path | None:
    pattern = f"{season}-*-picks.tsv"
    candidates = _choose_best_daily_file(list(PREDICTIONS_DIR.glob(pattern)))
    return candidates[-1] if candidates else None


def load_latest_picks(season: int) -> tuple[str | None, pd.DataFrame]:
    picks_path = latest_picks_file(season)
    if picks_path is None or not picks_path.exists() or picks_path.stat().st_size == 0:
        return None, pd.DataFrame()
    picks = pd.read_csv(picks_path, sep="\t")
    if picks.empty:
        return picks_path.name[:10], picks
    return picks_path.name[:10], picks


def recent_results(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    settled = df[df["settled"].astype(str).str.lower().eq("true")].copy()
    if settled.empty:
        return pd.DataFrame()
    sort_cols = [col for col in ["target_date", "game_id"] if col in settled.columns]
    settled = settled.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return settled.head(limit).reset_index(drop=True)


def monthly_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    settled = df[df["settled"].astype(str).str.lower().eq("true")].copy()
    if settled.empty:
        return []

    rows = []
    for month, month_df in sorted(settled.groupby("month")):
        wins = int((month_df["result"] == "win").sum())
        losses = int((month_df["result"] == "loss").sum())
        pushes = int((month_df["result"] == "push").sum())
        cost = float(pd.to_numeric(month_df["kalshi_side_market_price"], errors="coerce").fillna(0).sum())
        profit = float(pd.to_numeric(month_df["profit_per_contract"], errors="coerce").fillna(0).sum())
        non_push = wins + losses
        rows.append(
            {
                "month": month,
                "trades": int(len(month_df)),
                "record": f"{wins}-{losses}-{pushes}",
                "win_rate": (wins / non_push) if non_push else 0.0,
                "roi_pct": (profit / cost * 100.0) if cost > 0 else 0.0,
            }
        )
    return rows


def _fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def _fmt_money(value: float) -> str:
    return f"${value:,.0f}" if abs(value) >= 1 else f"${value:,.2f}"


def _fmt_number(value) -> str:
    if pd.isna(value):
        return "—"
    return str(value)


def _table(headers: list[str], rows: list[list[str]], empty_message: str, compact: bool = False) -> str:
    if not rows:
        return f'<div class="empty-state">{html.escape(empty_message)}</div>'
    class_name = "data-table compact" if compact else "data-table"
    head_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<table class="{class_name}"><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>'


def render_dashboard(
    season: int,
    updated_at: str,
    summary: DashboardSummary,
    latest_date: str | None,
    latest_picks: pd.DataFrame,
    recent_settled: pd.DataFrame,
    monthly: list[dict],
) -> str:
    latest_pick_rows = []
    if not latest_picks.empty:
        latest_picks = latest_picks.copy()
        if "kalshi_recommended_bet" in latest_picks.columns:
            latest_picks = latest_picks.sort_values(
                by=["kalshi_recommended_bet", "kalshi_edge_pct"],
                ascending=[False, False],
            )
        for _, row in latest_picks.iterrows():
            matchup = f"{html.escape(str(row.get('away_team', '')))} @ {html.escape(str(row.get('home_team', '')))}"
            latest_pick_rows.append(
                [
                    matchup,
                    html.escape(_fmt_number(row.get("kalshi_side"))),
                    html.escape(_fmt_number(row.get("kalshi_line"))),
                    html.escape(_fmt_number(row.get("predicted_total"))),
                    html.escape(f"{float(row.get('kalshi_side_market_prob', 0) or 0):.1f}%") if pd.notna(row.get("kalshi_side_market_prob")) else "—",
                    html.escape(f"{float(row.get('kalshi_fair_price_pct', 0) or 0):.1f}%") if pd.notna(row.get("kalshi_fair_price_pct")) else "—",
                    html.escape(_fmt_pct(float(row.get("kalshi_edge_pct", 0) or 0))),
                    html.escape(_fmt_money(float(row.get("kalshi_recommended_bet", 0) or 0))),
                ]
            )

    recent_rows = []
    for _, row in recent_settled.iterrows():
        result = str(row.get("result", "")).upper()
        result_class = f"pill {'win' if result == 'WIN' else 'loss' if result == 'LOSS' else 'push'}"
        recent_rows.append(
            [
                html.escape(str(row.get("target_date", ""))),
                html.escape(f"{row.get('away_team', '')} @ {row.get('home_team', '')}"),
                html.escape(_fmt_number(row.get("kalshi_side"))),
                html.escape(_fmt_number(row.get("kalshi_line"))),
                f'<span class="{result_class}">{html.escape(result)}</span>',
                html.escape(_fmt_number(row.get("total_runs"))),
                html.escape(_fmt_pct(float(row.get("roi_pct", 0) or 0))),
            ]
        )

    monthly_rows_html = _table(
        ["Month", "Trades", "Record", "Win Rate", "ROI"],
        [
            [
                html.escape(item["month"]),
                str(item["trades"]),
                html.escape(item["record"]),
                html.escape(f"{item['win_rate']:.1%}"),
                html.escape(_fmt_pct(item["roi_pct"])),
            ]
            for item in monthly
        ],
        "Monthly performance will appear after settled paper trades exist.",
        compact=True,
    )

    latest_table_html = _table(
        ["Matchup", "Side", "Line", "Model", "Market", "Fair", "Edge", "Bet"],
        latest_pick_rows,
        "No daily Kalshi bets logged yet. Once the paper tracker runs, today's bets will show here.",
    )

    recent_table_html = _table(
        ["Date", "Matchup", "Side", "Line", "Result", "Runs", "Contract ROI"],
        recent_rows,
        "Settled paper trades will show up here once games have final scores.",
    )

    latest_heading = latest_date or "No daily run yet"
    generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MLB Totals Paper Tracker</title>
  <style>
    :root {{
      --bg: #f2efe7;
      --ink: #14213d;
      --ink-soft: #46506a;
      --card: rgba(255, 253, 247, 0.82);
      --line: rgba(20, 33, 61, 0.14);
      --accent: #b65c3a;
      --accent-2: #2f7f6d;
      --accent-3: #d6a32e;
      --shadow: 0 20px 45px rgba(20, 33, 61, 0.10);
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(182, 92, 58, 0.20), transparent 34%),
        radial-gradient(circle at top right, rgba(47, 127, 109, 0.18), transparent 28%),
        linear-gradient(180deg, #f9f4e7 0%, var(--bg) 55%, #efe8d8 100%);
      min-height: 100vh;
    }}

    .shell {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 56px;
    }}

    .hero {{
      display: grid;
      gap: 18px;
      padding: 30px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background:
        linear-gradient(145deg, rgba(255,255,255,0.72), rgba(255,250,238,0.88)),
        linear-gradient(115deg, rgba(182, 92, 58, 0.08), rgba(47, 127, 109, 0.08));
      box-shadow: var(--shadow);
    }}

    h1, h2, h3 {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      letter-spacing: -0.03em;
    }}

    h1 {{
      font-size: clamp(2.3rem, 4vw, 4rem);
      line-height: 0.95;
      max-width: 12ch;
    }}

    .hero-copy {{
      color: var(--ink-soft);
      font-size: 1rem;
      max-width: 70ch;
      line-height: 1.6;
    }}

    .meta-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .tag {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(20, 33, 61, 0.06);
      color: var(--ink);
      font-size: 0.92rem;
    }}

    .section {{
      margin-top: 22px;
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--card);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}

    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 18px;
    }}

    .section-subtitle {{
      color: var(--ink-soft);
      font-size: 0.98rem;
      line-height: 1.5;
      max-width: 62ch;
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
    }}

    .metric {{
      padding: 18px;
      border-radius: 20px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(20, 33, 61, 0.08);
    }}

    .metric-label {{
      color: var(--ink-soft);
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.09em;
    }}

    .metric-value {{
      margin-top: 10px;
      font-size: clamp(1.4rem, 3vw, 2.4rem);
      font-weight: 700;
      line-height: 1;
    }}

    .metric-note {{
      margin-top: 8px;
      color: var(--ink-soft);
      font-size: 0.92rem;
    }}

    .grid-two {{
      display: grid;
      grid-template-columns: 1.25fr 0.95fr;
      gap: 20px;
      align-items: start;
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 18px;
      background: rgba(255,255,255,0.78);
    }}

    .data-table th,
    .data-table td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(20, 33, 61, 0.08);
      text-align: left;
      font-size: 0.95rem;
    }}

    .data-table th {{
      background: rgba(20, 33, 61, 0.05);
      color: var(--ink-soft);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .data-table.compact td,
    .data-table.compact th {{
      padding: 10px 12px;
      font-size: 0.9rem;
    }}

    .empty-state {{
      padding: 18px;
      border-radius: 18px;
      background: rgba(255,255,255,0.7);
      border: 1px dashed rgba(20, 33, 61, 0.18);
      color: var(--ink-soft);
      line-height: 1.6;
    }}

    .pill {{
      display: inline-flex;
      min-width: 66px;
      justify-content: center;
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}

    .pill.win {{ background: rgba(47, 127, 109, 0.14); color: #1f6557; }}
    .pill.loss {{ background: rgba(182, 92, 58, 0.14); color: #8f4429; }}
    .pill.push {{ background: rgba(214, 163, 46, 0.18); color: #8f6921; }}

    .footer-note {{
      margin-top: 18px;
      color: var(--ink-soft);
      font-size: 0.9rem;
      line-height: 1.6;
    }}

    @media (max-width: 920px) {{
      .grid-two {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 700px) {{
      .shell {{ width: min(100% - 18px, 1180px); padding-top: 18px; }}
      .hero, .section {{ padding: 18px; border-radius: 20px; }}
      .data-table {{ display: block; overflow-x: auto; }}
      .section-head {{ display: grid; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="meta-strip">
        <span class="tag">Paper Trading Dashboard</span>
        <span class="tag">Season {season}</span>
        <span class="tag">Updated {html.escape(updated_at)}</span>
      </div>
      <h1>MLB totals paper-tracking, built to be shared.</h1>
      <div class="hero-copy">
        This page tracks the live Kalshi-style paper results for the MLB totals model and shows the bets the bot would have made on the latest run day. It is rebuilt automatically from the same files your GitHub Actions jobs already maintain.
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Season Snapshot</h2>
          <div class="section-subtitle">High-level scorecard for the current paper-trading sample. This is the first place to send someone who just wants to know how the model is doing.</div>
        </div>
      </div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Tracked Trades</div>
          <div class="metric-value">{summary.tracked}</div>
          <div class="metric-note">All logged Kalshi paper trades so far.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Settled Trades</div>
          <div class="metric-value">{summary.settled}</div>
          <div class="metric-note">Games with final scores and graded outcomes.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Record</div>
          <div class="metric-value">{summary.wins}-{summary.losses}-{summary.pushes}</div>
          <div class="metric-note">Wins, losses, pushes.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Win Rate</div>
          <div class="metric-value">{summary.win_rate:.1%}</div>
          <div class="metric-note">Pushes excluded from the denominator.</div>
        </div>
        <div class="metric">
          <div class="metric-label">ROI</div>
          <div class="metric-value">{_fmt_pct(summary.roi_pct)}</div>
          <div class="metric-note">Measured per Kalshi contract cost.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Profit / Contract Units</div>
          <div class="metric-value">{summary.profit:+.2f}</div>
          <div class="metric-note">Aggregate paper profit across settled contracts.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Average Edge</div>
          <div class="metric-value">{_fmt_pct(summary.average_edge_pct)}</div>
          <div class="metric-note">Mean model edge across tracked paper trades.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Recommended Stakes</div>
          <div class="metric-value">{_fmt_money(summary.total_recommended_bet)}</div>
          <div class="metric-note">Sum of recommended Kalshi bet sizes logged.</div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Latest Daily Bets</h2>
          <div class="section-subtitle">The bets the bot would have made on the latest saved run date. This is the easiest section to show when someone asks what the model liked today.</div>
        </div>
        <div class="tag">{html.escape(latest_heading)}</div>
      </div>
      {latest_table_html}
    </section>

    <section class="section grid-two">
      <div>
        <div class="section-head">
          <div>
            <h2>Recent Settled Trades</h2>
            <div class="section-subtitle">Latest graded paper results, including outcome and per-contract ROI.</div>
          </div>
        </div>
        {recent_table_html}
      </div>
      <div>
        <div class="section-head">
          <div>
            <h2>Monthly Performance</h2>
            <div class="section-subtitle">A compact rolling view of how the forward test is behaving over time.</div>
          </div>
        </div>
        {monthly_rows_html}
      </div>
    </section>

    <div class="footer-note">
      Generated at {html.escape(generated_at)}. If you want to share this publicly with friends, the cleanest next step is to expose this `docs/index.html` page through a public repo or GitHub Pages.
    </div>
  </main>
</body>
</html>
"""


def build_dashboard(season: int) -> Path:
    kalshi_columns = [
        "target_date",
        "month",
        "game_id",
        "away_team",
        "home_team",
        "predicted_total",
        "kalshi_line",
        "kalshi_side",
        "kalshi_edge_pct",
        "kalshi_fair_price_pct",
        "kalshi_side_model_prob",
        "kalshi_side_market_prob",
        "kalshi_side_market_price",
        "kalshi_recommended_bet",
        "away_score",
        "home_score",
        "total_runs",
        "result",
        "profit_per_contract",
        "roi_pct",
        "settled",
    ]
    kalshi_df = load_tracker(PAPER_TRACKING_DIR / f"kalshi_tracker_{season}.tsv", kalshi_columns)
    summary = summarize_kalshi(kalshi_df)
    latest_date, latest_picks = load_latest_picks(season)
    recent = recent_results(kalshi_df)
    monthly = monthly_rows(kalshi_df)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    dashboard_path = DASHBOARD_DIR / "index.html"
    updated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    dashboard_path.write_text(
        render_dashboard(
            season=season,
            updated_at=updated_at,
            summary=summary,
            latest_date=latest_date,
            latest_picks=latest_picks,
            recent_settled=recent,
            monthly=monthly,
        ),
        encoding="utf-8",
    )
    return dashboard_path


def main():
    parser = argparse.ArgumentParser(description="Build a shareable static dashboard from paper-tracking outputs.")
    parser.add_argument("--season", type=int, default=current_season(), help="Season year to render.")
    args = parser.parse_args()
    path = build_dashboard(args.season)
    print(f"Dashboard written to {path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
