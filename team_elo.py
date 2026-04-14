"""
Team Offensive/Defensive Elo Rating Engine
===========================================
Computes schedule-adjusted team strength ratings that update after every game.
Unlike rolling averages, Elo accounts for opponent quality — beating a strong
defense improves your offensive rating more than beating a weak one.

Each team carries two ratings:
  off_elo  — run-scoring ability (higher = scores more)
  def_elo  — run-prevention ability (higher = allows fewer runs)

Expected runs scored by team A against team B:
  expected = LEAGUE_AVG + (off_elo_A - 1500) * SCALE - (def_elo_B - 1500) * SCALE

After each game, all four ratings update proportional to the surprise
(actual - expected). No leakage: pre-game ratings are recorded before update.

Output: data/team_elo_ratings.tsv
  One row per game, pre-game ratings for both teams.
  These become features in build_features.py.

Standalone usage:
    python team_elo.py           # compute ratings for all historical games
    python team_elo.py --summary # print current team ratings (most recent)
"""

import pandas as pd
import numpy as np
import os
import sys
import json

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Elo hyperparameters ────────────────────────────────────────────────────────
INIT_ELO        = 1500    # starting rating for every team, every season
K_FACTOR        = 8       # learning rate — how much one game moves the needle
                          # (8 is appropriate for baseball's high variance)
ELO_SCALE       = 0.003   # converts Elo points → expected runs
                          # 100 Elo pts above avg ≈ +0.3 runs/game
HOME_ADVANTAGE  = 0.15    # runs added to expected home team score
OFFSEASON_CARRY = 0.75    # how much of prior-season rating carries over
                          # (0.75 → 25% regression to mean each new season)
LEAGUE_AVG_RUNS = 4.5     # per-team per-game baseline


def expected_runs(off_elo: float, def_elo: float, is_home: bool = False) -> float:
    """
    Expected runs scored by a team with off_elo against a defense with def_elo.
    is_home adds a small home-field bonus.
    """
    base = LEAGUE_AVG_RUNS
    if is_home:
        base += HOME_ADVANTAGE
    return base + (off_elo - INIT_ELO) * ELO_SCALE - (def_elo - INIT_ELO) * ELO_SCALE


def update_four_ratings(
    home_off: float, home_def: float,
    away_off: float, away_def: float,
    actual_home: float, actual_away: float,
    k: float = K_FACTOR,
) -> tuple[float, float, float, float]:
    """
    Update all four ratings from one game result.

    Logic:
      - Offense improves when team scores more than expected (vs opponent defense)
      - Defense improves when team allows fewer than expected (vs opponent offense)

    Returns (home_off, home_def, away_off, away_def) — updated values.
    """
    exp_home = expected_runs(home_off, away_def, is_home=True)
    exp_away = expected_runs(away_off, home_def, is_home=False)

    new_home_off = home_off + k * (actual_home - exp_home)
    new_away_off = away_off + k * (actual_away - exp_away)

    # Defense: if allowed fewer than expected, def_elo goes up
    new_home_def = home_def + k * (exp_away - actual_away)
    new_away_def = away_def + k * (exp_home - actual_home)

    return new_home_off, new_home_def, new_away_off, new_away_def


def regress_to_mean(ratings: dict, carry: float = OFFSEASON_CARRY) -> dict:
    """
    Apply offseason regression to the mean. Called at the start of each new season.
    Shrinks every rating toward INIT_ELO, reflecting offseason roster changes.
    """
    return {
        team_id: {
            "off_elo": INIT_ELO + carry * (r["off_elo"] - INIT_ELO),
            "def_elo": INIT_ELO + carry * (r["def_elo"] - INIT_ELO),
        }
        for team_id, r in ratings.items()
    }


def compute_elo_ratings(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process all games chronologically and compute pre-game Elo ratings.

    Returns a DataFrame with one row per game:
      game_id, date,
      home_off_elo, home_def_elo,
      away_off_elo, away_def_elo,
      (+ derived: elo_offense_sum, elo_matchup_score, elo_home_advantage)
    """
    games = games_df.sort_values(["date", "game_id"]).reset_index(drop=True)

    # Current ratings: {team_id: {"off_elo": float, "def_elo": float}}
    ratings: dict = {}
    current_season = None

    records = []

    for _, row in games.iterrows():
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        season  = int(pd.Timestamp(row["date"]).year)

        # ── Season rollover: regress to mean at start of each new season ─────
        if current_season is None:
            current_season = season
        elif season != current_season:
            ratings = regress_to_mean(ratings)
            current_season = season

        # ── Initialize any new team ───────────────────────────────────────────
        for tid in (home_id, away_id):
            if tid not in ratings:
                ratings[tid] = {"off_elo": INIT_ELO, "def_elo": INIT_ELO}

        h_off = ratings[home_id]["off_elo"]
        h_def = ratings[home_id]["def_elo"]
        a_off = ratings[away_id]["off_elo"]
        a_def = ratings[away_id]["def_elo"]

        # ── Record PRE-game ratings (no leakage) ─────────────────────────────
        exp_home = expected_runs(h_off, a_def, is_home=True)
        exp_away = expected_runs(a_off, h_def, is_home=False)

        records.append({
            "game_id":          int(row["game_id"]),
            "date":             row["date"],
            "home_team_id":     home_id,
            "away_team_id":     away_id,
            # Raw ratings
            "home_off_elo":     round(h_off, 2),
            "home_def_elo":     round(h_def, 2),
            "away_off_elo":     round(a_off, 2),
            "away_def_elo":     round(a_def, 2),
            # Derived: expected total runs this game
            "elo_expected_total": round(exp_home + exp_away, 3),
            # Sum of offensive ratings → how much both teams want to score
            "elo_offense_sum":  round(h_off + a_off, 2),
            # Net balance: high offense - high defense → more runs; opposite → fewer
            "elo_matchup_score": round((h_off + a_off) - (h_def + a_def), 2),
            # Home team's structural edge in this matchup
            "elo_home_edge":    round((h_off - a_def) - (a_off - h_def), 2),
        })

        # ── Update ratings only if we have the actual score ──────────────────
        if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
            h_off, h_def, a_off, a_def = update_four_ratings(
                h_off, h_def, a_off, a_def,
                float(row["home_score"]), float(row["away_score"]),
            )
            ratings[home_id]["off_elo"] = h_off
            ratings[home_id]["def_elo"] = h_def
            ratings[away_id]["off_elo"] = a_off
            ratings[away_id]["def_elo"] = a_def

    return pd.DataFrame(records), ratings


def get_current_ratings(games_df: pd.DataFrame) -> dict:
    """
    Run the Elo engine through all historical games and return the most
    recent ratings for every team. Used by predict_today.py.

    Returns {team_id: {"off_elo": float, "def_elo": float}}.
    """
    _, ratings = compute_elo_ratings(games_df)
    return ratings


def main():
    games_path = os.path.join(DATA_DIR, "mlb_games_raw.tsv")
    out_path   = os.path.join(DATA_DIR, "team_elo_ratings.tsv")

    print("Loading game data...")
    games = pd.read_csv(games_path, sep="\t", parse_dates=["date"])
    print(f"  {len(games)} games loaded")

    print("Computing Elo ratings...")
    elo_df, final_ratings = compute_elo_ratings(games)
    print(f"  {len(elo_df)} game-rating rows computed")

    elo_df.to_csv(out_path, sep="\t", index=False)
    print(f"  Saved to {out_path}")

    # Save current ratings snapshot for quick loading in predict_today.py
    snapshot_path = os.path.join(DATA_DIR, "team_elo_current.json")
    with open(snapshot_path, "w") as f:
        json.dump(final_ratings, f)
    print(f"  Current ratings snapshot saved to {snapshot_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if "--summary" in sys.argv:
        # Load team name lookup from games
        name_lookup = {}
        for _, row in games.iterrows():
            name_lookup[int(row["home_team_id"])] = row["home_team"]
            name_lookup[int(row["away_team_id"])] = row["away_team"]

        rows = []
        for tid, r in final_ratings.items():
            rows.append({
                "team":    name_lookup.get(tid, str(tid)),
                "off_elo": round(r["off_elo"], 1),
                "def_elo": round(r["def_elo"], 1),
                "net_elo": round(r["off_elo"] + r["def_elo"] - 2 * INIT_ELO, 1),
            })
        summary = pd.DataFrame(rows).sort_values("net_elo", ascending=False)
        print(f"\n{'Team':<30} {'Off Elo':>8} {'Def Elo':>8} {'Net':>6}")
        print("-" * 56)
        for _, r in summary.iterrows():
            print(f"  {r['team']:<28} {r['off_elo']:>8.1f} {r['def_elo']:>8.1f} "
                  f"{r['net_elo']:>+6.1f}")

    # ── Validation ───────────────────────────────────────────────────────────
    print("\n--- Validation (Elo feature correlation with actual total runs) ---")
    elo_df["date"] = pd.to_datetime(elo_df["date"])
    merged = elo_df.merge(
        games[["game_id", "total_runs", "home_score", "away_score"]],
        on="game_id", how="left"
    ).dropna(subset=["total_runs"])

    corr_expected = merged["elo_expected_total"].corr(merged["total_runs"])
    corr_offense  = merged["elo_offense_sum"].corr(merged["total_runs"])
    corr_matchup  = merged["elo_matchup_score"].corr(merged["total_runs"])
    print(f"  elo_expected_total  vs total_runs: r = {corr_expected:.4f}")
    print(f"  elo_offense_sum     vs total_runs: r = {corr_offense:.4f}")
    print(f"  elo_matchup_score   vs total_runs: r = {corr_matchup:.4f}")

    # Check 2025 (out-of-sample) separately
    oos = merged[merged["date"].dt.year == 2025]
    if not oos.empty:
        corr_oos = oos["elo_expected_total"].corr(oos["total_runs"])
        mae = (oos["elo_expected_total"] - oos["total_runs"]).abs().mean()
        print(f"\n  2025 out-of-sample:")
        print(f"    elo_expected_total correlation: r = {corr_oos:.4f}")
        print(f"    MAE vs actual total runs: {mae:.2f}")


if __name__ == "__main__":
    main()
