"""
Build a static share-friendly dashboard from paper-tracking outputs.

Usage:
    python build_dashboard.py
    python build_dashboard.py --season 2026
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from paper_bankroll import load_starting_bankroll

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
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
    starting_bankroll: float
    current_bankroll: float
    over_bets: int = 0
    under_bets: int = 0
    over_win_rate: float = 0.0
    under_win_rate: float = 0.0


@dataclass
class HistoricalSummary:
    total_games: int
    games_with_kalshi: int
    bets_placed: int
    wins: int
    losses: int
    win_rate: float
    roi_pct: float
    total_pnl: float
    total_wagered: float
    avg_edge_pct: float
    kelly_fraction: float
    max_bet_pct: float
    min_bet: float
    min_kalshi_edge_pct: float
    min_kalshi_confidence_pct: float
    over_bets: int = 0
    under_bets: int = 0
    over_win_rate: float = 0.0
    under_win_rate: float = 0.0


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


def load_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summarize_kalshi(df: pd.DataFrame) -> DashboardSummary:
    starting_bankroll = load_starting_bankroll()
    if df.empty:
        return DashboardSummary(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, starting_bankroll, starting_bankroll)

    settled = df[df["settled"].astype(str).str.lower().eq("true")].copy()
    wins = int((settled["result"] == "win").sum())
    losses = int((settled["result"] == "loss").sum())
    pushes = int((settled["result"] == "push").sum())
    non_push = wins + losses
    total_cost = float(pd.to_numeric(settled["kalshi_side_market_price"], errors="coerce").fillna(0).sum())
    profit = float(pd.to_numeric(settled["profit_per_contract"], errors="coerce").fillna(0).sum())
    roi_pct = profit / total_cost * 100.0 if total_cost > 0 else 0.0
    side_edges = df.apply(lambda row: _kalshi_side_display(row)[1], axis=1).dropna() if not df.empty else pd.Series(dtype=float)
    avg_edge = float(side_edges.mean()) if not side_edges.empty else 0.0
    bankroll_after_series = pd.to_numeric(df.get("paper_bankroll_after_day"), errors="coerce").dropna()
    current_bankroll = float(bankroll_after_series.iloc[-1]) if not bankroll_after_series.empty else starting_bankroll

    # Only count rows where a bet was actually placed (recommended_bet > 0)
    bet_placed = pd.to_numeric(df.get("kalshi_recommended_bet"), errors="coerce").fillna(0) > 0
    bets_df = df[bet_placed]
    side_col = bets_df.get("kalshi_side") if not bets_df.empty else None
    over_bets = int((side_col == "OVER").sum()) if side_col is not None else 0
    under_bets = int((side_col == "UNDER").sum()) if side_col is not None else 0
    over_settled = settled[settled.get("kalshi_side") == "OVER"] if "kalshi_side" in settled.columns else settled.iloc[:0]
    under_settled = settled[settled.get("kalshi_side") == "UNDER"] if "kalshi_side" in settled.columns else settled.iloc[:0]
    over_wins = int((over_settled["result"] == "win").sum()) if not over_settled.empty else 0
    under_wins = int((under_settled["result"] == "win").sum()) if not under_settled.empty else 0
    over_non_push = int(((over_settled["result"] == "win") | (over_settled["result"] == "loss")).sum()) if not over_settled.empty else 0
    under_non_push = int(((under_settled["result"] == "win") | (under_settled["result"] == "loss")).sum()) if not under_settled.empty else 0

    return DashboardSummary(
        tracked=int(bet_placed.sum()),
        settled=int(len(settled)),
        wins=wins,
        losses=losses,
        pushes=pushes,
        win_rate=(wins / non_push) if non_push else 0.0,
        roi_pct=roi_pct,
        profit=profit,
        average_edge_pct=avg_edge,
        starting_bankroll=starting_bankroll,
        current_bankroll=current_bankroll,
        over_bets=over_bets,
        under_bets=under_bets,
        over_win_rate=(over_wins / over_non_push) if over_non_push else 0.0,
        under_win_rate=(under_wins / under_non_push) if under_non_push else 0.0,
    )


def summarize_historical(summary: dict, sim_df: pd.DataFrame | None = None) -> HistoricalSummary:
    over_bets = under_bets = 0
    over_win_rate = under_win_rate = 0.0
    avg_edge_pct = float(summary.get("avg_edge_pct", 0.0) or 0.0)
    if sim_df is not None and not sim_df.empty and "kalshi_side" in sim_df.columns:
        bets_df = sim_df[pd.to_numeric(sim_df.get("bet_amount"), errors="coerce").fillna(0) > 0]
        over = bets_df[bets_df["kalshi_side"] == "OVER"]
        under = bets_df[bets_df["kalshi_side"] == "UNDER"]
        over_bets = len(over)
        under_bets = len(under)
        over_won = over["result"].eq("won") | over["result"].eq("win") | over["won"].astype(str).eq("True") if not over.empty else pd.Series(dtype=bool)
        under_won = under["result"].eq("won") | under["result"].eq("win") | under["won"].astype(str).eq("True") if not under.empty else pd.Series(dtype=bool)
        over_win_rate = float(over_won.mean()) if len(over) > 0 else 0.0
        under_win_rate = float(under_won.mean()) if len(under) > 0 else 0.0
        side_edges = bets_df.apply(lambda row: _kalshi_side_display(row)[1], axis=1).dropna()
        if not side_edges.empty:
            avg_edge_pct = float(side_edges.mean())

    return HistoricalSummary(
        total_games=int(summary.get("total_games", 0) or 0),
        games_with_kalshi=int(summary.get("games_with_kalshi", 0) or 0),
        bets_placed=int(summary.get("bets_placed", 0) or 0),
        wins=int(summary.get("wins", 0) or 0),
        losses=int(summary.get("losses", 0) or 0),
        win_rate=float(summary.get("win_rate", 0.0) or 0.0) / 100.0,
        roi_pct=float(summary.get("roi_pct", 0.0) or 0.0),
        total_pnl=float(summary.get("total_pnl", 0.0) or 0.0),
        total_wagered=float(summary.get("total_wagered", 0.0) or 0.0),
        avg_edge_pct=avg_edge_pct,
        kelly_fraction=float(summary.get("kelly_fraction", 0.0) or 0.0),
        max_bet_pct=float(summary.get("max_bet_pct", 0.0) or 0.0),
        min_bet=float(summary.get("min_bet", 0.0) or 0.0),
        min_kalshi_edge_pct=float(summary.get("min_kalshi_edge_pct", 0.0) or 0.0),
        min_kalshi_confidence_pct=float(summary.get("min_kalshi_confidence_pct", 0.0) or 0.0),
        over_bets=over_bets,
        under_bets=under_bets,
        over_win_rate=over_win_rate,
        under_win_rate=under_win_rate,
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


def load_historical_sim(season: int) -> pd.DataFrame:
    columns = [
        "date",
        "game_id",
        "away_team",
        "home_team",
        "predicted_total",
        "kalshi_line",
        "kalshi_side",
        "kalshi_side_market_prob",
        "kalshi_fair_price_pct",
        "kalshi_edge_pct",
        "bet_amount",
        "bet_pct_bankroll",
        "actual_total",
        "result",
        "pnl_dollars",
        "roi_pct",
        "settled",
    ]
    return load_tracker(DATA_DIR / f"season_sim_{season}.tsv", columns)


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


def _fmt_money(value: float, signed: bool = False) -> str:
    prefix = "+" if signed and value > 0 else ""
    if abs(value) >= 1_000_000:
        return f"{prefix}${value / 1_000_000:.1f}m"
    if abs(value) >= 1_000:
        return f"{prefix}${value / 1_000:.1f}k"
    if abs(value) >= 1:
        return f"{prefix}${value:,.0f}"
    return f"{prefix}${value:,.2f}"


def _fmt_plain_pct(value: float) -> str:
    return f"{value:.1f}%"


def _fmt_number(value) -> str:
    if pd.isna(value):
        return "—"
    return str(value)


def _num(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _kalshi_side_display(row: pd.Series) -> tuple[float | None, float | None]:
    side = str(row.get("kalshi_side") or "").upper()
    side_market_prob = row.get("kalshi_side_market_prob")
    side_model_prob = row.get("kalshi_side_model_prob")
    over_fair_prob = row.get("kalshi_fair_price_pct")
    over_edge_pct = row.get("kalshi_edge_pct")

    if pd.notna(side_model_prob):
        fair_prob = _num(side_model_prob)
    elif pd.notna(over_fair_prob):
        fair_prob = _num(over_fair_prob)
        if side == "UNDER":
            fair_prob = 100.0 - fair_prob
    else:
        fair_prob = None

    if pd.notna(side_market_prob) and fair_prob is not None:
        edge_pct = fair_prob - _num(side_market_prob)
    elif pd.notna(over_edge_pct):
        edge_pct = _num(over_edge_pct)
        if side == "UNDER":
            edge_pct = -edge_pct
    else:
        edge_pct = None

    return fair_prob, edge_pct


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
    historical_summary: HistoricalSummary,
    latest_date: str | None,
    latest_picks: pd.DataFrame,
    recent_settled: pd.DataFrame,
    monthly: list[dict],
    historical_recent: pd.DataFrame,
) -> str:
    latest_pick_rows = []
    if not latest_picks.empty:
        latest_picks = latest_picks.copy()
        if "kalshi_recommended_bet" in latest_picks.columns:
            latest_picks = latest_picks[
                pd.to_numeric(latest_picks["kalshi_recommended_bet"], errors="coerce").fillna(0) > 0
            ].sort_values(
                by=["kalshi_recommended_bet", "kalshi_edge_pct"],
                ascending=[False, False],
            )
        for _, row in latest_picks.iterrows():
            matchup = f"{html.escape(str(row.get('away_team', '')))} @ {html.escape(str(row.get('home_team', '')))}"
            side_fair_prob, side_edge_pct = _kalshi_side_display(row)
            latest_pick_rows.append(
                [
                    matchup,
                    html.escape(_fmt_number(row.get("kalshi_side"))),
                    html.escape(_fmt_number(row.get("kalshi_line"))),
                    html.escape(_fmt_number(row.get("predicted_total"))),
                    html.escape(f"{_num(row.get('kalshi_side_market_prob')):.1f}%") if pd.notna(row.get("kalshi_side_market_prob")) else "—",
                    html.escape(f"{side_fair_prob:.1f}%") if side_fair_prob is not None else "—",
                    html.escape(_fmt_pct(side_edge_pct)) if side_edge_pct is not None else "—",
                    html.escape(_fmt_money(_num(row.get("kalshi_recommended_bet")))),
                    html.escape(_fmt_plain_pct(_num(row.get("kalshi_bet_pct_bankroll")))),
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

    historical_rows = []
    for _, row in historical_recent.iterrows():
        _, side_edge_pct = _kalshi_side_display(row)
        historical_rows.append(
            [
                html.escape(str(row.get("date", ""))),
                html.escape(f"{row.get('away_team', '')} @ {row.get('home_team', '')}"),
                html.escape(_fmt_number(row.get("kalshi_side"))),
                html.escape(_fmt_number(row.get("kalshi_line"))),
                html.escape(_fmt_pct(side_edge_pct)) if side_edge_pct is not None else "—",
                html.escape(_fmt_money(_num(row.get("bet_amount")))),
                html.escape(_fmt_plain_pct(_num(row.get("bet_pct_bankroll")))),
                html.escape(str(row.get("result", "")).upper()),
                html.escape(_fmt_pct(_num(row.get("roi_pct")))),
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
        "Monthly results will appear once trades have settled.",
        compact=True,
    )

    latest_table_html = _table(
        ["Matchup", "Side", "Line", "Model", "Market", "Fair", "Edge", "Bet", "% Roll"],
        latest_pick_rows,
        "No daily bets logged yet.",
    )

    recent_table_html = _table(
        ["Date", "Matchup", "Side", "Line", "Result", "Runs", "Contract ROI"],
        recent_rows,
        "Settled trades will appear here once games are final.",
    )

    historical_table_html = _table(
        ["Date", "Matchup", "Side", "Line", "Edge", "Bet", "% Roll", "Result", "ROI"],
        historical_rows,
        "Historical replay results will appear here once the simulator has been run.",
    )

    kelly_label = _fmt_plain_pct(historical_summary.kelly_fraction * 100.0)
    max_bet_label = "No hard cap" if historical_summary.max_bet_pct <= 0 else _fmt_plain_pct(historical_summary.max_bet_pct)
    min_bet_label = _fmt_money(historical_summary.min_bet)

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
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
      gap: 14px;
    }}

    .metric {{
      display: flex;
      flex-direction: column;
      padding: 18px;
      border-radius: 20px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(20, 33, 61, 0.08);
      overflow: hidden;
    }}

    .metric-label {{
      color: var(--ink-soft);
      font-size: clamp(0.72rem, 0.68rem + 0.2vw, 0.84rem);
      text-transform: uppercase;
      letter-spacing: 0.07em;
      white-space: nowrap;
    }}

    .metric-value {{
      margin-top: 10px;
      font-size: clamp(1.15rem, 0.95rem + 1vw, 2.15rem);
      font-weight: 700;
      line-height: 1;
      white-space: nowrap;
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
        <span class="tag">Paper Trading</span>
        <span class="tag">Season {season}</span>
        <span class="tag">Updated {html.escape(updated_at)}</span>
      </div>
      <h1>MLB Totals Model Dashboard</h1>
      <div class="hero-copy">
        Live paper-trading results and latest model picks.
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Season Snapshot</h2>
          <div class="section-subtitle">Current season results at a glance.</div>
        </div>
      </div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Tracked Trades</div>
          <div class="metric-value">{summary.tracked}</div>
          <div class="metric-note">All logged paper trades.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Settled Trades</div>
          <div class="metric-value">{summary.settled}</div>
          <div class="metric-note">Trades with final results.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Record</div>
          <div class="metric-value">{summary.wins}-{summary.losses}</div>
          <div class="metric-note">Wins - Losses (settled bets).</div>
        </div>
        <div class="metric">
          <div class="metric-label">Win Rate</div>
          <div class="metric-value">{summary.win_rate:.1%}</div>
          <div class="metric-note">Pushes excluded.</div>
        </div>
        <div class="metric">
          <div class="metric-label">ROI</div>
          <div class="metric-value">{_fmt_pct(summary.roi_pct)}</div>
          <div class="metric-note">Per contract cost.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Profit / Contract Units</div>
          <div class="metric-value">{summary.profit:+.2f}</div>
          <div class="metric-note">Settled trade total.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Average Edge</div>
          <div class="metric-value">{_fmt_pct(summary.average_edge_pct)}</div>
          <div class="metric-note">Mean model edge.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Current Bankroll</div>
          <div class="metric-value">{_fmt_money(summary.current_bankroll)}</div>
          <div class="metric-note">Started at {_fmt_money(summary.starting_bankroll)}.</div>
        </div>
        <div class="metric">
          <div class="metric-label">OVER Bets</div>
          <div class="metric-value">{summary.over_bets}</div>
          <div class="metric-note">Win rate: {summary.over_win_rate:.1%}</div>
        </div>
        <div class="metric">
          <div class="metric-label">UNDER Bets</div>
          <div class="metric-value">{summary.under_bets}</div>
          <div class="metric-note">Win rate: {summary.under_win_rate:.1%}</div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Latest Daily Bets</h2>
          <div class="section-subtitle">Latest model picks.</div>
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
            <div class="section-subtitle">Most recent graded results.</div>
          </div>
        </div>
        {recent_table_html}
      </div>
      <div>
        <div class="section-head">
          <div>
            <h2>Monthly Performance</h2>
            <div class="section-subtitle">Monthly results.</div>
          </div>
        </div>
        {monthly_rows_html}
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Kalshi Rules</h2>
          <div class="section-subtitle">Current thresholds and sizing used for the Kalshi replay and live paper-tracked bets.</div>
        </div>
      </div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Minimum Edge</div>
          <div class="metric-value">{_fmt_plain_pct(historical_summary.min_kalshi_edge_pct)}</div>
          <div class="metric-note">Absolute edge vs market price required to bet.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Minimum Confidence</div>
          <div class="metric-value">{_fmt_plain_pct(historical_summary.min_kalshi_confidence_pct)}</div>
          <div class="metric-note">Model win probability required on the chosen side.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Kelly Fraction</div>
          <div class="metric-value">{kelly_label}</div>
          <div class="metric-note">Reduced from 25% through May 2026 (early-season edge calibration).</div>
        </div>
        <div class="metric">
          <div class="metric-label">Max Bet Cap</div>
          <div class="metric-value">{html.escape(max_bet_label)}</div>
          <div class="metric-note">Hard cap on bankroll risk per trade.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Minimum Bet</div>
          <div class="metric-value">{html.escape(min_bet_label)}</div>
          <div class="metric-note">Smallest rounded Kalshi bet the bot will place.</div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Historical Replay</h2>
          <div class="section-subtitle">Backfilled 2026 Kalshi results using historical 10 AM PT prices.</div>
        </div>
      </div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Replay Bets</div>
          <div class="metric-value">{historical_summary.bets_placed}</div>
          <div class="metric-note">Historical bets identified.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Replay Record</div>
          <div class="metric-value">{historical_summary.wins}-{historical_summary.losses}</div>
          <div class="metric-note">Settled historical wins and losses.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Replay ROI</div>
          <div class="metric-value">{_fmt_pct(historical_summary.roi_pct)}</div>
          <div class="metric-note">Historical Kalshi replay ROI.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Replay P&amp;L</div>
          <div class="metric-value">{_fmt_money(historical_summary.total_pnl, signed=True)}</div>
          <div class="metric-note">Historical replay profit.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Kalshi Games</div>
          <div class="metric-value">{historical_summary.games_with_kalshi}</div>
          <div class="metric-note">Games with historical Kalshi coverage.</div>
        </div>
        <div class="metric">
          <div class="metric-label">Average Edge</div>
          <div class="metric-value">{_fmt_pct(historical_summary.avg_edge_pct)}</div>
          <div class="metric-note">Mean replay edge.</div>
        </div>
        <div class="metric">
          <div class="metric-label">OVER Bets</div>
          <div class="metric-value">{historical_summary.over_bets}</div>
          <div class="metric-note">Win rate: {historical_summary.over_win_rate:.1%}</div>
        </div>
        <div class="metric">
          <div class="metric-label">UNDER Bets</div>
          <div class="metric-value">{historical_summary.under_bets}</div>
          <div class="metric-note">Win rate: {historical_summary.under_win_rate:.1%}</div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Recent Historical Replay Bets</h2>
          <div class="section-subtitle">Most recent settled bets from the historical 2026 Kalshi backfill.</div>
        </div>
      </div>
      {historical_table_html}
    </section>

    <div class="footer-note">
      Generated at {html.escape(generated_at)}.
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
        "kalshi_bet_pct_bankroll",
        "kalshi_recommended_bet",
        "paper_bankroll_after_day",
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
    historical_df = load_historical_sim(season)
    historical_summary = summarize_historical(load_json(DATA_DIR / "season_sim_summary.json"), sim_df=historical_df)
    latest_date, latest_picks = load_latest_picks(season)
    recent = recent_results(kalshi_df)
    historical_recent = historical_df[historical_df["bet_amount"].fillna(0) > 0].sort_values(
        by=["date", "bet_amount"], ascending=[False, False]
    ).head(12).reset_index(drop=True) if not historical_df.empty else pd.DataFrame()
    monthly = monthly_rows(kalshi_df)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    dashboard_path = DASHBOARD_DIR / "index.html"
    updated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    dashboard_path.write_text(
        render_dashboard(
            season=season,
            updated_at=updated_at,
            summary=summary,
            historical_summary=historical_summary,
            latest_date=latest_date,
            latest_picks=latest_picks,
            recent_settled=recent,
            monthly=monthly,
            historical_recent=historical_recent,
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
