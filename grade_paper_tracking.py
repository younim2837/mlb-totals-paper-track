"""
Rebuild season-long paper-tracking logs from saved daily board exports.

Usage:
    python grade_paper_tracking.py
    python grade_paper_tracking.py --season 2026
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from paper_bankroll import load_starting_bankroll


PROJECT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = PROJECT_DIR / "predictions"
PAPER_TRACKING_DIR = PROJECT_DIR / "paper_tracking"
DATA_DIR = PROJECT_DIR / "data"
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
SPORTSBOOK_WIN_UNITS = 100.0 / 110.0


@dataclass
class Summary:
    total: int
    settled: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    roi_pct: float
    profit_units: float


def current_season() -> int:
    return int(datetime.now(PACIFIC_TZ).strftime("%Y"))


def _board_data_row_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return -1


def list_board_files(season: int) -> list[Path]:
    pattern = f"{season}-*-board.tsv"
    chosen_by_date: dict[str, Path] = {}
    row_counts: dict[Path, int] = {}

    for path in sorted(PREDICTIONS_DIR.glob(pattern)):
        target_date = path.name[:10]
        current = chosen_by_date.get(target_date)
        if current is None:
            chosen_by_date[target_date] = path
            continue

        current_rows = row_counts.setdefault(current, _board_data_row_count(current))
        candidate_rows = row_counts.setdefault(path, _board_data_row_count(path))
        current_is_all_games = current.name.endswith("-all-games-board.tsv")
        candidate_is_all_games = path.name.endswith("-all-games-board.tsv")

        if candidate_rows > current_rows:
            chosen_by_date[target_date] = path
        elif candidate_rows == current_rows and candidate_is_all_games and not current_is_all_games:
            chosen_by_date[target_date] = path

    return [chosen_by_date[date_key] for date_key in sorted(chosen_by_date)]


def load_results_lookup() -> pd.DataFrame:
    games = pd.read_csv(DATA_DIR / "mlb_games_raw.tsv", sep="\t")
    if "total_runs" not in games.columns:
        games["total_runs"] = games["away_score"] + games["home_score"]
    games["date"] = pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d")
    return games[[
        "game_id",
        "date",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "total_runs",
    ]].drop_duplicates(subset=["game_id"])


def attach_actuals(board: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    attached = board.copy()
    if "target_date" not in attached.columns:
        attached["target_date"] = None

    def _normalize_text(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip().lower()

    results_by_game_id = {}
    results_by_matchup = {}
    for _, row in results.iterrows():
        actual_payload = {
            "away_score": row.get("away_score"),
            "home_score": row.get("home_score"),
            "total_runs": row.get("total_runs"),
        }
        game_id = _normalize_text(row.get("game_id"))
        if game_id:
            results_by_game_id[game_id] = actual_payload
        matchup_key = (
            _normalize_text(row.get("date")),
            _normalize_text(row.get("away_team")),
            _normalize_text(row.get("home_team")),
        )
        results_by_matchup[matchup_key] = actual_payload

    actual_rows = []
    for _, row in attached.iterrows():
        actual = None
        game_id = _normalize_text(row.get("game_id"))
        if game_id:
            actual = results_by_game_id.get(game_id)
        if actual is None:
            matchup_key = (
                _normalize_text(row.get("target_date")),
                _normalize_text(row.get("away_team")),
                _normalize_text(row.get("home_team")),
            )
            actual = results_by_matchup.get(matchup_key)
        actual_rows.append(actual or {"away_score": pd.NA, "home_score": pd.NA, "total_runs": pd.NA})

    actual_df = pd.DataFrame(actual_rows)
    for col in ["away_score", "home_score", "total_runs"]:
        attached[col] = actual_df[col] if col in actual_df.columns else pd.NA

    for col in [
        "bet_signal",
        "bet_confidence",
        "edge",
        "market_line_source",
        "market_adjustment_method",
        "kalshi_line",
        "kalshi_over_pct",
        "kalshi_side",
        "kalshi_edge_pct",
        "kalshi_fair_price_pct",
        "kalshi_side_model_prob",
        "kalshi_side_market_prob",
        "kalshi_recommended_bet",
        "kalshi_bankroll_used",
        "kalshi_bet_pct_bankroll",
    ]:
        if col not in attached.columns:
            attached[col] = pd.NA
    return attached


def sportsbook_outcome(side: str, actual_total: float, line: float) -> tuple[str, float]:
    if pd.isna(actual_total) or pd.isna(line):
        return "pending", 0.0
    if actual_total == line:
        return "push", 0.0

    won = (side == "OVER" and actual_total > line) or (side == "UNDER" and actual_total < line)
    return ("win", SPORTSBOOK_WIN_UNITS) if won else ("loss", -1.0)


def kalshi_outcome(side: str, actual_total: float, line: float, market_price: float) -> tuple[str, float]:
    if pd.isna(actual_total) or pd.isna(line) or pd.isna(market_price):
        return "pending", 0.0
    if actual_total == line:
        return "push", 0.0

    won = (side == "OVER" and actual_total > line) or (side == "UNDER" and actual_total < line)
    profit = (1.0 - market_price) if won else (-market_price)
    return ("win", profit) if won else ("loss", profit)


def build_sportsbook_tracker(board_rows: pd.DataFrame) -> pd.DataFrame:
    picks = board_rows.loc[board_rows["bet_signal"].isin(["OVER", "UNDER"])].copy()
    if picks.empty:
        return pd.DataFrame()

    graded = picks.apply(
        lambda row: sportsbook_outcome(
            side=str(row["bet_signal"]),
            actual_total=float(row["total_runs"]) if pd.notna(row["total_runs"]) else float("nan"),
            line=float(row["posted_line"]) if pd.notna(row["posted_line"]) else float("nan"),
        ),
        axis=1,
        result_type="expand",
    )
    picks[["result", "profit_units"]] = graded
    picks["settled"] = picks["result"].isin(["win", "loss", "push"])
    picks["month"] = picks["target_date"].astype(str).str[:7]
    return picks[[
        "target_date",
        "month",
        "game_id",
        "away_team",
        "home_team",
        "predicted_total",
        "posted_line",
        "posted_odds",
        "bet_signal",
        "bet_confidence",
        "edge",
        "market_line_source",
        "market_adjustment_method",
        "away_score",
        "home_score",
        "total_runs",
        "result",
        "profit_units",
        "settled",
    ]].sort_values(["target_date", "game_id"]).reset_index(drop=True)


def build_kalshi_tracker(board_rows: pd.DataFrame) -> pd.DataFrame:
    kalshi = board_rows.loc[board_rows["kalshi_side"].isin(["OVER", "UNDER"])].copy()
    if kalshi.empty:
        return pd.DataFrame()
    for col in ["away_score", "home_score", "total_runs"]:
        if col not in kalshi.columns:
            kalshi[col] = pd.NA

    market_price = []
    for _, row in kalshi.iterrows():
        over_price = float(row["kalshi_over_pct"]) / 100.0 if pd.notna(row["kalshi_over_pct"]) else float("nan")
        market_price.append(over_price if row["kalshi_side"] == "OVER" else 1.0 - over_price)
    kalshi["kalshi_side_market_price"] = market_price

    graded = kalshi.apply(
        lambda row: kalshi_outcome(
            side=str(row["kalshi_side"]),
            actual_total=float(row["total_runs"]) if pd.notna(row["total_runs"]) else float("nan"),
            line=float(row["kalshi_line"]) if pd.notna(row["kalshi_line"]) else float("nan"),
            market_price=float(row["kalshi_side_market_price"]) if pd.notna(row["kalshi_side_market_price"]) else float("nan"),
        ),
        axis=1,
        result_type="expand",
    )
    kalshi[["result", "profit_per_contract"]] = graded
    kalshi["settled"] = kalshi["result"].isin(["win", "loss", "push"])
    kalshi["month"] = kalshi["target_date"].astype(str).str[:7]
    kalshi["stake_dollars"] = kalshi["kalshi_side_market_price"]
    kalshi["kalshi_recommended_bet"] = pd.to_numeric(kalshi["kalshi_recommended_bet"], errors="coerce").fillna(0.0)
    kalshi["kalshi_bankroll_used"] = pd.to_numeric(kalshi.get("kalshi_bankroll_used"), errors="coerce").fillna(0.0)
    kalshi["kalshi_bet_pct_bankroll"] = pd.to_numeric(kalshi.get("kalshi_bet_pct_bankroll"), errors="coerce").fillna(0.0)
    kalshi["kalshi_contracts"] = kalshi.apply(
        lambda row: (row["kalshi_recommended_bet"] / row["kalshi_side_market_price"])
        if row["kalshi_recommended_bet"] > 0 and row["kalshi_side_market_price"] > 0
        else 0.0,
        axis=1,
    )
    kalshi["pnl_dollars"] = kalshi.apply(
        lambda row: (row["kalshi_contracts"] * row["profit_per_contract"])
        if row["result"] in {"win", "loss", "push"}
        else 0.0,
        axis=1,
    )
    kalshi["roi_pct"] = kalshi.apply(
        lambda row: (row["profit_per_contract"] / row["stake_dollars"] * 100.0)
        if row["result"] in {"win", "loss"} and row["stake_dollars"] > 0
        else 0.0,
        axis=1,
    )
    kalshi = kalshi.sort_values(["target_date", "game_id"]).reset_index(drop=True)

    starting_bankroll = load_starting_bankroll()
    running_bankroll = float(starting_bankroll)
    bankroll_before_day = []
    bankroll_after_day = []
    for target_date, day_df in kalshi.groupby("target_date", sort=True):
        opening_roll = float(running_bankroll)
        settled_mask = day_df["settled"].astype(bool)
        date_pnl = float(pd.to_numeric(day_df.loc[settled_mask, "pnl_dollars"], errors="coerce").fillna(0.0).sum())
        all_day_bets_settled = bool(settled_mask.all())
        closing_roll = opening_roll + date_pnl if all_day_bets_settled else opening_roll
        bankroll_before_day.extend([round(opening_roll, 2)] * len(day_df))
        bankroll_after_day.extend([round(closing_roll, 2)] * len(day_df))
        if all_day_bets_settled:
            running_bankroll = closing_roll

    kalshi["paper_bankroll_at_bet"] = bankroll_before_day
    kalshi["paper_bankroll_after_day"] = bankroll_after_day
    zero_bankroll_mask = kalshi["kalshi_bankroll_used"] <= 0
    kalshi.loc[zero_bankroll_mask, "kalshi_bankroll_used"] = kalshi.loc[zero_bankroll_mask, "paper_bankroll_at_bet"]
    zero_pct_mask = kalshi["kalshi_bet_pct_bankroll"] <= 0
    kalshi.loc[zero_pct_mask, "kalshi_bet_pct_bankroll"] = kalshi.apply(
        lambda row: (row["kalshi_recommended_bet"] / row["paper_bankroll_at_bet"] * 100.0)
        if row["paper_bankroll_at_bet"] > 0
        else 0.0,
        axis=1,
    )
    return kalshi[[
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
        "kalshi_bankroll_used",
        "kalshi_bet_pct_bankroll",
        "kalshi_recommended_bet",
        "kalshi_contracts",
        "away_score",
        "home_score",
        "total_runs",
        "result",
        "profit_per_contract",
        "pnl_dollars",
        "roi_pct",
        "paper_bankroll_at_bet",
        "paper_bankroll_after_day",
        "settled",
    ]].sort_values(["target_date", "game_id"]).reset_index(drop=True)


def summarize_sportsbook(df: pd.DataFrame) -> Summary:
    if df.empty:
        return Summary(0, 0, 0, 0, 0, 0.0, 0.0, 0.0)
    settled = df[df["settled"]].copy()
    wins = int((settled["result"] == "win").sum())
    losses = int((settled["result"] == "loss").sum())
    pushes = int((settled["result"] == "push").sum())
    non_push = max(wins + losses, 1)
    total_count = int(len(settled))
    profit_units = float(settled["profit_units"].sum())
    win_rate = wins / non_push if non_push else 0.0
    roi_pct = (profit_units / total_count * 100.0) if total_count else 0.0
    return Summary(
        total=int(len(df)),
        settled=total_count,
        wins=wins,
        losses=losses,
        pushes=pushes,
        win_rate=win_rate,
        roi_pct=roi_pct,
        profit_units=profit_units,
    )


def summarize_kalshi(df: pd.DataFrame) -> Summary:
    if df.empty:
        return Summary(0, 0, 0, 0, 0, 0.0, 0.0, 0.0)
    settled = df[df["settled"]].copy()
    wins = int((settled["result"] == "win").sum())
    losses = int((settled["result"] == "loss").sum())
    pushes = int((settled["result"] == "push").sum())
    non_push = max(wins + losses, 1)
    total_cost = float(settled["kalshi_side_market_price"].sum())
    profit_units = float(settled["profit_per_contract"].sum())
    win_rate = wins / non_push if non_push else 0.0
    roi_pct = (profit_units / total_cost * 100.0) if total_cost > 0 else 0.0
    return Summary(
        total=int(len(df)),
        settled=int(len(settled)),
        wins=wins,
        losses=losses,
        pushes=pushes,
        win_rate=win_rate,
        roi_pct=roi_pct,
        profit_units=profit_units,
    )


def monthly_summary(df: pd.DataFrame, profit_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "bets", "win_rate", "roi_pct"])

    settled = df[df["settled"]].copy()
    if settled.empty:
        return pd.DataFrame(columns=["month", "bets", "win_rate", "roi_pct"])

    rows = []
    for month, month_df in sorted(settled.groupby("month")):
        wins = int((month_df["result"] == "win").sum())
        losses = int((month_df["result"] == "loss").sum())
        pushes = int((month_df["result"] == "push").sum())
        non_push = max(wins + losses, 1)
        profit = float(month_df[profit_col].sum())
        if profit_col == "profit_units":
            roi_pct = profit / len(month_df) * 100.0
        else:
            cost = float(month_df["kalshi_side_market_price"].sum())
            roi_pct = profit / cost * 100.0 if cost > 0 else 0.0
        rows.append({
            "month": month,
            "bets": int(len(month_df)),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": wins / non_push if non_push else 0.0,
            "roi_pct": roi_pct,
        })
    return pd.DataFrame(rows)


def format_summary_block(title: str, summary: Summary) -> list[str]:
    return [
        f"## {title}",
        f"- Tracked: {summary.total}",
        f"- Settled: {summary.settled}",
        f"- Record: {summary.wins}-{summary.losses}-{summary.pushes} (W-L-P)",
        f"- Win rate: {summary.win_rate:.1%}",
        f"- ROI: {summary.roi_pct:+.1f}%",
        f"- Profit: {summary.profit_units:+.2f}",
        "",
    ]


def write_latest_status_markdown(
    season: int,
    board_files: list[Path],
    sportsbook_df: pd.DataFrame,
    kalshi_df: pd.DataFrame,
) -> Path:
    kalshi_summary = summarize_kalshi(kalshi_df)
    sportsbook_summary = summarize_sportsbook(sportsbook_df)

    lines = [
        f"# Latest Status ({season})",
        "",
        f"- Updated: {datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')}",
        f"- Daily board files found: {len(board_files)}",
        "",
        "## Kalshi",
        f"- Tracked: {kalshi_summary.total}",
        f"- Settled: {kalshi_summary.settled}",
        f"- Record: {kalshi_summary.wins}-{kalshi_summary.losses}-{kalshi_summary.pushes}",
        f"- Win rate: {kalshi_summary.win_rate:.1%}",
        f"- ROI: {kalshi_summary.roi_pct:+.1f}%",
        "",
        "## Sportsbook (Optional)",
        f"- Tracked: {sportsbook_summary.total}",
        f"- Settled: {sportsbook_summary.settled}",
        f"- Record: {sportsbook_summary.wins}-{sportsbook_summary.losses}-{sportsbook_summary.pushes}",
        f"- Win rate: {sportsbook_summary.win_rate:.1%}",
        f"- ROI: {sportsbook_summary.roi_pct:+.1f}%",
        "",
        f"- Full summary: `paper_summary_{season}.md`",
        f"- Kalshi detail: `kalshi_tracker_{season}.tsv`",
        f"- Sportsbook detail: `sportsbook_tracker_{season}.tsv`",
    ]

    latest_path = PAPER_TRACKING_DIR / "LATEST_STATUS.md"
    latest_path.write_text("\n".join(lines), encoding="utf-8")
    return latest_path


def write_summary_markdown(
    season: int,
    board_files: list[Path],
    sportsbook_df: pd.DataFrame,
    kalshi_df: pd.DataFrame,
) -> Path:
    sportsbook_summary = summarize_sportsbook(sportsbook_df)
    kalshi_summary = summarize_kalshi(kalshi_df)
    sportsbook_monthly = monthly_summary(sportsbook_df, "profit_units")
    kalshi_monthly = monthly_summary(kalshi_df, "profit_per_contract")

    lines = [
        f"# {season} Paper Tracking Summary",
        "",
        f"- Updated: {datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')}",
        f"- Daily board files found: {len(board_files)}",
        "- Primary scorecard: Kalshi forward tracking",
        "- Sportsbook section is optional and may stay empty if no Odds API key is configured",
        "",
    ]
    lines.extend(format_summary_block("Kalshi Forward Tracking", kalshi_summary))
    if not kalshi_monthly.empty:
        lines.append("### Kalshi Monthly")
        lines.append("")
        lines.append("| Month | Trades | Record | Win Rate | ROI |")
        lines.append("| --- | ---: | --- | ---: | ---: |")
        for _, row in kalshi_monthly.iterrows():
            record = f"{int(row['wins'])}-{int(row['losses'])}-{int(row['pushes'])}"
            lines.append(
                f"| {row['month']} | {int(row['bets'])} | {record} | {row['win_rate']:.1%} | {row['roi_pct']:+.1f}% |"
            )
        lines.append("")

    lines.extend(format_summary_block("Sportsbook Picks (Optional)", sportsbook_summary))
    if not sportsbook_monthly.empty:
        lines.append("### Sportsbook Monthly")
        lines.append("")
        lines.append("| Month | Bets | Record | Win Rate | ROI |")
        lines.append("| --- | ---: | --- | ---: | ---: |")
        for _, row in sportsbook_monthly.iterrows():
            record = f"{int(row['wins'])}-{int(row['losses'])}-{int(row['pushes'])}"
            lines.append(
                f"| {row['month']} | {int(row['bets'])} | {record} | {row['win_rate']:.1%} | {row['roi_pct']:+.1f}% |"
            )
        lines.append("")

    PAPER_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = PAPER_TRACKING_DIR / f"paper_summary_{season}.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Rebuild season-long paper tracking from saved board exports.")
    parser.add_argument("--season", type=int, default=current_season(), help="Season year to rebuild.")
    args = parser.parse_args()

    board_files = list_board_files(args.season)
    results = load_results_lookup()

    board_frames = []
    for board_path in board_files:
        board = pd.read_csv(board_path, sep="\t")
        if "target_date" not in board.columns:
            board["target_date"] = board_path.name[:10]
        board_frames.append(attach_actuals(board, results))

    combined = pd.concat(board_frames, ignore_index=True) if board_frames else pd.DataFrame()
    sportsbook_df = build_sportsbook_tracker(combined) if not combined.empty else pd.DataFrame()
    kalshi_df = build_kalshi_tracker(combined) if not combined.empty else pd.DataFrame()

    PAPER_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    sportsbook_path = PAPER_TRACKING_DIR / f"sportsbook_tracker_{args.season}.tsv"
    kalshi_path = PAPER_TRACKING_DIR / f"kalshi_tracker_{args.season}.tsv"
    summary_path = write_summary_markdown(args.season, board_files, sportsbook_df, kalshi_df)
    latest_path = write_latest_status_markdown(args.season, board_files, sportsbook_df, kalshi_df)

    sportsbook_df.to_csv(sportsbook_path, sep="\t", index=False)
    kalshi_df.to_csv(kalshi_path, sep="\t", index=False)

    print(f"Rebuilt {args.season} paper tracking from {len(board_files)} board files.")
    print(f"Sportsbook tracker: {sportsbook_path.relative_to(PROJECT_DIR)}")
    print(f"Kalshi tracker: {kalshi_path.relative_to(PROJECT_DIR)}")
    print(f"Summary: {summary_path.relative_to(PROJECT_DIR)}")
    print(f"Latest status: {latest_path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
