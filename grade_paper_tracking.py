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


def list_board_files(season: int) -> list[Path]:
    pattern = f"{season}-*-board.tsv"
    files = []
    for path in sorted(PREDICTIONS_DIR.glob(pattern)):
        if path.name.endswith("-all-games-board.tsv"):
            continue
        files.append(path)
    return files


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
    merged = board.merge(
        results,
        on="game_id",
        how="left",
        suffixes=("", "_actual"),
    )
    if "target_date" not in merged.columns:
        merged["target_date"] = None
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
    ]:
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged


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
    kalshi["roi_pct"] = kalshi.apply(
        lambda row: (row["profit_per_contract"] / row["stake_dollars"] * 100.0)
        if row["result"] in {"win", "loss"} and row["stake_dollars"] > 0
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
        "kalshi_recommended_bet",
        "away_score",
        "home_score",
        "total_runs",
        "result",
        "profit_per_contract",
        "roi_pct",
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
        "",
    ]
    lines.extend(format_summary_block("Sportsbook Picks", sportsbook_summary))
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

    lines.extend(format_summary_block("Kalshi Proxy Trades", kalshi_summary))
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

    sportsbook_df.to_csv(sportsbook_path, sep="\t", index=False)
    kalshi_df.to_csv(kalshi_path, sep="\t", index=False)

    print(f"Rebuilt {args.season} paper tracking from {len(board_files)} board files.")
    print(f"Sportsbook tracker: {sportsbook_path.relative_to(PROJECT_DIR)}")
    print(f"Kalshi tracker: {kalshi_path.relative_to(PROJECT_DIR)}")
    print(f"Summary: {summary_path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
