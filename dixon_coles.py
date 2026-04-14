"""
Dixon-Coles team-prior engine for MLB totals.

This module serves two purposes:
1. Fit current team attack/defense priors for live prediction.
2. Generate pregame historical priors for every game so XGBoost can train on
   leakage-safe Dixon-Coles features instead of Elo.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

CURRENT_CACHE_PATH = os.path.join(DATA_DIR, "dc_params_current.json")
HISTORY_PATH = os.path.join(DATA_DIR, "dc_ratings_history.tsv")

TIME_DECAY_HALFLIFE = 60
TRAINING_WINDOW = 365
MIN_GAMES_FOR_FIT = 100
L2_PENALTY = 0.05

COMMON_TOTAL_LINES = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]


def load_games(path: str | None = None) -> pd.DataFrame:
    games_path = path or os.path.join(DATA_DIR, "mlb_games_raw.tsv")
    games = pd.read_csv(games_path, sep="\t", parse_dates=["date"])
    games["date"] = pd.to_datetime(games["date"]).dt.normalize()
    return games.sort_values(["date", "game_id"]).reset_index(drop=True)


def time_weights(
    dates: np.ndarray,
    reference_date: pd.Timestamp,
    halflife: float = TIME_DECAY_HALFLIFE,
) -> np.ndarray:
    """Exponential time decay so recent games matter more."""
    days_ago = (reference_date - pd.to_datetime(dates)).days.astype(float)
    return np.exp(-np.log(2) * days_ago / halflife)


def build_team_index(games: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    team_idx = {team: i for i, team in enumerate(teams)}
    return teams, team_idx


def _prepare_fit_arrays(games: pd.DataFrame, team_idx: dict[str, int]) -> dict[str, np.ndarray]:
    return {
        "home_idx": games["home_team"].map(team_idx).to_numpy(dtype=np.int32),
        "away_idx": games["away_team"].map(team_idx).to_numpy(dtype=np.int32),
        "home_score": games["home_score"].to_numpy(dtype=np.float64),
        "away_score": games["away_score"].to_numpy(dtype=np.float64),
    }


def _build_initial_params(
    teams: list[str],
    games: pd.DataFrame,
    previous_params: dict | None = None,
) -> np.ndarray:
    n_teams = len(teams)
    home_mean = max(float(games["home_score"].mean()), 0.1)
    away_mean = max(float(games["away_score"].mean()), 0.1)
    mu_init = np.log((home_mean + away_mean) / 2.0)
    home_adv_init = np.log(home_mean / away_mean)

    attack_init = np.zeros(n_teams, dtype=np.float64)
    defense_init = np.zeros(n_teams, dtype=np.float64)

    if previous_params:
        mu_init = float(previous_params.get("mu", mu_init))
        home_adv_init = float(previous_params.get("home_adv", home_adv_init))
        prev_attack = previous_params.get("attack", {})
        prev_defense = previous_params.get("defense", {})
        for i, team in enumerate(teams):
            attack_init[i] = float(prev_attack.get(team, 0.0))
            defense_init[i] = float(prev_defense.get(team, 0.0))

    # attack[0] is omitted for identifiability
    return np.concatenate([[mu_init, home_adv_init], attack_init[1:], defense_init])


def neg_log_likelihood(
    params: np.ndarray,
    fit_data: dict[str, np.ndarray],
    n_teams: int,
    weights: np.ndarray,
    l2_penalty: float = L2_PENALTY,
) -> float:
    """
    Negative weighted log-likelihood for the independent-Poisson DC model.

    Parameter layout:
      [mu, home_adv, attack_1..attack_n-1, defense_0..defense_n-1]
    attack_0 is fixed to zero for identifiability.
    """
    mu = params[0]
    home_adv = params[1]

    attack = np.empty(n_teams, dtype=np.float64)
    attack[0] = 0.0
    attack[1:] = params[2:2 + n_teams - 1]
    defense = params[2 + n_teams - 1:]

    lam_home = np.exp(mu + attack[fit_data["home_idx"]] - defense[fit_data["away_idx"]] + home_adv)
    lam_away = np.exp(mu + attack[fit_data["away_idx"]] - defense[fit_data["home_idx"]])

    home_score = fit_data["home_score"]
    away_score = fit_data["away_score"]

    log_p_home = home_score * np.log(lam_home) - lam_home - gammaln(home_score + 1.0)
    log_p_away = away_score * np.log(lam_away) - lam_away - gammaln(away_score + 1.0)
    weighted_log_lik = np.sum(weights * (log_p_home + log_p_away))

    penalty = l2_penalty * (
        np.sum(attack[1:] ** 2) +
        np.sum(defense ** 2) +
        home_adv ** 2
    )

    return float(-weighted_log_lik + penalty)


def fit_dixon_coles(
    games: pd.DataFrame,
    reference_date=None,
    halflife: float = TIME_DECAY_HALFLIFE,
    window_days: int = TRAINING_WINDOW,
    min_games: int = MIN_GAMES_FOR_FIT,
    previous_params: dict | None = None,
    l2_penalty: float = L2_PENALTY,
) -> dict | None:
    """
    Fit a time-decayed Dixon-Coles model using games strictly before reference_date.
    """
    reference_date = (
        pd.Timestamp.today().normalize()
        if reference_date is None
        else pd.Timestamp(reference_date).normalize()
    )
    cutoff_start = reference_date - timedelta(days=window_days)

    subset = games[
        (games["date"] < reference_date) &
        (games["date"] >= cutoff_start) &
        games["home_score"].notna() &
        games["away_score"].notna()
    ].copy()

    if len(subset) < min_games:
        return None

    teams, team_idx = build_team_index(subset)
    fit_data = _prepare_fit_arrays(subset, team_idx)
    weights = time_weights(subset["date"].to_numpy(), reference_date, halflife=halflife)
    x0 = _build_initial_params(teams, subset, previous_params=previous_params)

    result = minimize(
        neg_log_likelihood,
        x0,
        args=(fit_data, len(teams), weights, l2_penalty),
        method="L-BFGS-B",
        options={"maxiter": 250, "ftol": 1e-8},
    )

    if not result.success and not np.isfinite(result.fun):
        return None

    params = result.x
    attack = np.empty(len(teams), dtype=np.float64)
    attack[0] = 0.0
    attack[1:] = params[2:2 + len(teams) - 1]
    defense = params[2 + len(teams) - 1:]

    return {
        "teams": teams,
        "team_idx": team_idx,
        "attack": {team: float(attack[i]) for i, team in enumerate(teams)},
        "defense": {team: float(defense[i]) for i, team in enumerate(teams)},
        "mu": float(params[0]),
        "home_adv": float(params[1]),
        "fit_date": str(reference_date.date()),
        "n_games": int(len(subset)),
        "halflife": float(halflife),
        "window_days": int(window_days),
        "min_games": int(min_games),
        "l2_penalty": float(l2_penalty),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iters": int(getattr(result, "nit", 0)),
    }


def predict_game(dc_params: dict, home_team: str, away_team: str) -> dict | None:
    """Predict matchup scoring from fitted DC team parameters."""
    if not dc_params:
        return None

    attack = dc_params["attack"]
    defense = dc_params["defense"]
    mu = float(dc_params["mu"])
    home_adv = float(dc_params["home_adv"])

    home_attack = float(attack.get(home_team, 0.0))
    home_defense = float(defense.get(home_team, 0.0))
    away_attack = float(attack.get(away_team, 0.0))
    away_defense = float(defense.get(away_team, 0.0))

    lambda_home = float(np.exp(mu + home_attack - away_defense + home_adv))
    lambda_away = float(np.exp(mu + away_attack - home_defense))
    expected_total = lambda_home + lambda_away

    p_over = {}
    for line in COMMON_TOTAL_LINES:
        p_over[line] = round(float(1 - poisson.cdf(int(line), expected_total)), 4)

    return {
        "home_attack": home_attack,
        "home_defense": home_defense,
        "away_attack": away_attack,
        "away_defense": away_defense,
        "lambda_home": round(lambda_home, 3),
        "lambda_away": round(lambda_away, 3),
        "expected_total": round(expected_total, 3),
        "home_edge": round(lambda_home - lambda_away, 3),
        "p_over": p_over,
    }


def _history_is_fresh(
    history_df: pd.DataFrame,
    games_df: pd.DataFrame,
    halflife: float = TIME_DECAY_HALFLIFE,
    window_days: int = TRAINING_WINDOW,
    min_games: int = MIN_GAMES_FOR_FIT,
    l2_penalty: float = L2_PENALTY,
) -> bool:
    if history_df.empty:
        return False
    if len(history_df) != len(games_df):
        return False
    required_cols = {"dc_halflife", "dc_window_days", "dc_min_games", "dc_l2_penalty"}
    if not required_cols.issubset(history_df.columns):
        return False
    same_cfg = (
        history_df["dc_halflife"].nunique() == 1 and
        history_df["dc_window_days"].nunique() == 1 and
        history_df["dc_min_games"].nunique() == 1 and
        history_df["dc_l2_penalty"].nunique() == 1 and
        float(history_df["dc_halflife"].iloc[0]) == float(halflife) and
        int(history_df["dc_window_days"].iloc[0]) == int(window_days) and
        int(history_df["dc_min_games"].iloc[0]) == int(min_games) and
        float(history_df["dc_l2_penalty"].iloc[0]) == float(l2_penalty)
    )
    if not same_cfg:
        return False
    return (
        pd.to_datetime(history_df["date"]).max() ==
        pd.to_datetime(games_df["date"]).max()
    )


def build_pregame_feature_history(
    games: pd.DataFrame,
    output_path: str | None = HISTORY_PATH,
    halflife: float = TIME_DECAY_HALFLIFE,
    window_days: int = TRAINING_WINDOW,
    min_games: int = MIN_GAMES_FOR_FIT,
    l2_penalty: float = L2_PENALTY,
) -> pd.DataFrame:
    """
    Fit a pregame DC model once per unique date and emit leakage-safe features for
    every historical game.
    """
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True).copy()
    unique_dates = list(pd.to_datetime(games["date"]).dt.normalize().drop_duplicates())
    rows: list[dict] = []
    previous_params = None

    print(f"Building Dixon-Coles history for {len(unique_dates)} dates...")
    for i, game_date in enumerate(unique_dates, start=1):
        day_games = games[games["date"] == game_date]
        params = fit_dixon_coles(
            games,
            reference_date=game_date,
            halflife=halflife,
            window_days=window_days,
            min_games=min_games,
            previous_params=previous_params,
            l2_penalty=l2_penalty,
        )
        if params is not None:
            previous_params = params

        for _, row in day_games.iterrows():
            base = {
                "game_id": row["game_id"],
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
            }
            if params is None:
                rows.append(
                    base | {
                        "home_dc_attack": np.nan,
                        "home_dc_defense": np.nan,
                        "away_dc_attack": np.nan,
                        "away_dc_defense": np.nan,
                        "dc_lambda_home": np.nan,
                        "dc_lambda_away": np.nan,
                        "dc_expected_total": np.nan,
                    "dc_home_edge": np.nan,
                    "dc_fit_n_games": 0,
                    "dc_halflife": float(halflife),
                    "dc_window_days": int(window_days),
                    "dc_min_games": int(min_games),
                    "dc_l2_penalty": float(l2_penalty),
                    }
                )
                continue

            pred = predict_game(params, row["home_team"], row["away_team"])
            rows.append(
                base | {
                    "home_dc_attack": pred["home_attack"],
                    "home_dc_defense": pred["home_defense"],
                    "away_dc_attack": pred["away_attack"],
                    "away_dc_defense": pred["away_defense"],
                    "dc_lambda_home": pred["lambda_home"],
                    "dc_lambda_away": pred["lambda_away"],
                    "dc_expected_total": pred["expected_total"],
                    "dc_home_edge": pred["home_edge"],
                    "dc_fit_n_games": params["n_games"],
                    "dc_halflife": float(halflife),
                    "dc_window_days": int(window_days),
                    "dc_min_games": int(min_games),
                    "dc_l2_penalty": float(l2_penalty),
                }
            )

        if i == 1 or i % 25 == 0 or i == len(unique_dates):
            last_n = 0 if params is None else params["n_games"]
            print(f"  {i}/{len(unique_dates)} dates complete (fit window games: {last_n})")

    history = pd.DataFrame(rows)
    if output_path:
        history.to_csv(output_path, sep="\t", index=False)
        print(f"Saved DC history to {output_path}")
    return history


def load_or_build_history(
    games_df: pd.DataFrame,
    history_path: str | None = HISTORY_PATH,
    halflife: float = TIME_DECAY_HALFLIFE,
    window_days: int = TRAINING_WINDOW,
    min_games: int = MIN_GAMES_FOR_FIT,
    l2_penalty: float = L2_PENALTY,
) -> pd.DataFrame:
    if history_path and os.path.exists(history_path):
        try:
            history = pd.read_csv(history_path, sep="\t", parse_dates=["date"])
            if _history_is_fresh(
                history,
                games_df,
                halflife=halflife,
                window_days=window_days,
                min_games=min_games,
                l2_penalty=l2_penalty,
            ):
                return history
        except Exception:
            pass

    return build_pregame_feature_history(
        games_df,
        output_path=history_path,
        halflife=halflife,
        window_days=window_days,
        min_games=min_games,
        l2_penalty=l2_penalty,
    )


def load_or_fit(
    games_df: pd.DataFrame,
    reference_date: str,
    cache_path: str | None = CURRENT_CACHE_PATH,
    halflife: float = TIME_DECAY_HALFLIFE,
    window_days: int = TRAINING_WINDOW,
    min_games: int = MIN_GAMES_FOR_FIT,
    l2_penalty: float = L2_PENALTY,
) -> dict | None:
    """
    Load cached current params if compatible; otherwise fit and save.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            if (
                cached.get("fit_date") == reference_date and
                float(cached.get("halflife", halflife)) == float(halflife) and
                int(cached.get("window_days", window_days)) == int(window_days) and
                int(cached.get("min_games", min_games)) == int(min_games) and
                float(cached.get("l2_penalty", l2_penalty)) == float(l2_penalty)
            ):
                return cached
        except Exception:
            pass

    params = fit_dixon_coles(
        games_df,
        reference_date=reference_date,
        halflife=halflife,
        window_days=window_days,
        min_games=min_games,
        previous_params=None,
        l2_penalty=l2_penalty,
    )

    if params and cache_path:
        saveable = {k: v for k, v in params.items() if k != "team_idx"}
        saveable["team_idx"] = {str(team): int(idx) for team, idx in params["team_idx"].items()}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(saveable, f, indent=2)

    return params


def summarize_params(params: dict) -> pd.DataFrame:
    rows = []
    for team in params["teams"]:
        attack = float(params["attack"][team])
        defense = float(params["defense"][team])
        rows.append(
            {
                "team": team,
                "attack": round(attack, 3),
                "defense": round(defense, 3),
                "net": round(attack - defense, 3),
            }
        )
    return pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)


def validate_2025(games: pd.DataFrame) -> None:
    print("\n--- 2025 out-of-sample validation ---")
    history = load_or_build_history(games, history_path=None)
    merged = history.merge(
        games[["game_id", "date", "total_runs"]],
        on=["game_id", "date"],
        how="left",
    )
    merged = merged[
        (merged["date"].dt.year == 2025) &
        merged["dc_expected_total"].notna() &
        merged["total_runs"].notna()
    ].copy()

    if merged.empty:
        print("  No 2025 games with DC priors available.")
        return

    mae = (merged["dc_expected_total"] - merged["total_runs"]).abs().mean()
    print(f"  MAE: {mae:.3f} runs on {len(merged)} games")

    print("\n  Calibration by P(over 8.5):")
    over_probs = 1 - poisson.cdf(8, merged["dc_expected_total"])
    actual = (merged["total_runs"] > 8.5).astype(int)
    for lo, hi in [(0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.00)]:
        mask = (over_probs >= lo) & (over_probs < hi)
        if mask.sum() == 0:
            continue
        print(
            f"    {lo:.0%}-{hi:.0%}: {actual[mask].mean():.1%} actual "
            f"(n={int(mask.sum())})"
        )


def evaluate_season(
    games: pd.DataFrame,
    season: int = 2025,
    halflife: float = TIME_DECAY_HALFLIFE,
    window_days: int = TRAINING_WINDOW,
    min_games: int = MIN_GAMES_FOR_FIT,
    l2_penalty: float = L2_PENALTY,
) -> dict:
    """
    Evaluate one DC configuration on a target season using only information
    available before each game date.
    """
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True).copy()
    season_games = games[games["date"].dt.year == season].copy()
    if season_games.empty:
        return {
            "season": season,
            "halflife": halflife,
            "window_days": window_days,
            "min_games": min_games,
            "l2_penalty": l2_penalty,
            "games": 0,
            "coverage": 0.0,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_total_error": np.nan,
            "calibration_abs_error_8_5": np.nan,
        }

    rows = []
    previous_params = None
    season_dates = season_games["date"].drop_duplicates().tolist()

    for i, game_date in enumerate(season_dates, start=1):
        day_games = season_games[season_games["date"] == game_date]
        params = fit_dixon_coles(
            games,
            reference_date=game_date,
            halflife=halflife,
            window_days=window_days,
            min_games=min_games,
            previous_params=previous_params,
            l2_penalty=l2_penalty,
        )
        if params is not None:
            previous_params = params

        for _, row in day_games.iterrows():
            pred = predict_game(params, row["home_team"], row["away_team"]) if params else None
            rows.append({
                "game_id": row["game_id"],
                "date": row["date"],
                "actual_total": row["total_runs"],
                "pred_total": np.nan if pred is None else pred["expected_total"],
                "p_over_8_5": np.nan if pred is None else pred["p_over"].get(8.5, np.nan),
            })

        if i == 1 or i % 25 == 0 or i == len(season_dates):
            fitted_games = 0 if params is None else params["n_games"]
            print(f"  Eval {season} {i}/{len(season_dates)} dates complete (fit window games: {fitted_games})")

    results = pd.DataFrame(rows)
    covered = results["pred_total"].notna()
    covered_results = results[covered].copy()

    if covered_results.empty:
        return {
            "season": season,
            "halflife": halflife,
            "window_days": window_days,
            "min_games": min_games,
            "l2_penalty": l2_penalty,
            "games": int(len(results)),
            "coverage": 0.0,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_total_error": np.nan,
            "calibration_abs_error_8_5": np.nan,
        }

    errors = covered_results["pred_total"] - covered_results["actual_total"]
    p_over = covered_results["p_over_8_5"].clip(0, 1)
    actual_over = (covered_results["actual_total"] > 8.5).astype(float)

    calib_bucket_errors = []
    for lo, hi in [(0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.80)]:
        mask = (p_over >= lo) & (p_over < hi)
        if mask.sum() >= 25:
            calib_bucket_errors.append(abs(p_over[mask].mean() - actual_over[mask].mean()))

    return {
        "season": season,
        "halflife": float(halflife),
        "window_days": int(window_days),
        "min_games": int(min_games),
        "l2_penalty": float(l2_penalty),
        "games": int(len(results)),
        "coverage": float(covered.mean()),
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_total_error": float(errors.mean()),
        "calibration_abs_error_8_5": float(np.mean(calib_bucket_errors)) if calib_bucket_errors else np.nan,
    }


def tune_parameter_grid(
    games: pd.DataFrame,
    season: int,
    halflifes: list[float],
    windows: list[int],
    min_games_list: list[int],
    l2_penalties: list[float],
) -> pd.DataFrame:
    """
    Run a compact grid search across DC hyperparameters.
    """
    combos = list(itertools.product(halflifes, windows, min_games_list, l2_penalties))
    rows = []
    print(f"Tuning Dixon-Coles on {len(combos)} configurations for {season}...")

    for idx, (halflife, window_days, min_games, l2_penalty) in enumerate(combos, start=1):
        print(
            f"\n[{idx}/{len(combos)}] halflife={halflife} "
            f"window={window_days} min_games={min_games} l2={l2_penalty}"
        )
        score = evaluate_season(
            games,
            season=season,
            halflife=halflife,
            window_days=window_days,
            min_games=min_games,
            l2_penalty=l2_penalty,
        )
        rows.append(score)
        print(
            f"  -> MAE={score['mae']:.3f} RMSE={score['rmse']:.3f} "
            f"coverage={score['coverage']:.1%} calib_err={score['calibration_abs_error_8_5']:.3f}"
        )

    results = pd.DataFrame(rows).sort_values(
        ["mae", "rmse", "calibration_abs_error_8_5"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Dixon-Coles MLB prior engine")
    parser.add_argument("--date", help="Reference date YYYY-MM-DD (default: today)")
    parser.add_argument("--history", action="store_true", help="Build full pregame DC history")
    parser.add_argument("--cache-only", action="store_true", help="Fit current params and save cache only")
    parser.add_argument("--validate", action="store_true", help="Run 2025 validation report")
    parser.add_argument("--tune", action="store_true", help="Run a compact hyperparameter search")
    parser.add_argument("--season", type=int, default=2025, help="Season used for validation/tuning")
    args = parser.parse_args()

    games = load_games()
    print(f"Loading game data...\n  {len(games)} games loaded")

    if args.history:
        load_or_build_history(games, history_path=HISTORY_PATH)
        return

    if args.tune:
        results = tune_parameter_grid(
            games,
            season=args.season,
            halflifes=[45, 60, 90],
            windows=[240, 365, 540],
            min_games_list=[50],
            l2_penalties=[0.01, 0.05],
        )
        out_path = os.path.join(DATA_DIR, f"dc_tuning_results_{args.season}.tsv")
        results.to_csv(out_path, sep="\t", index=False)
        print(f"\nTop results:\n{results.head(10).to_string(index=False)}")
        print(f"\nSaved tuning results to {out_path}")
        return

    reference_date = args.date or datetime.now().strftime("%Y-%m-%d")
    print(f"\nFitting Dixon-Coles model as of {reference_date}...")
    params = load_or_fit(games, reference_date, cache_path=CURRENT_CACHE_PATH)

    if params is None:
        print("  Failed to fit: not enough data.")
        return

    print(
        f"  Fit on {params['n_games']} games | mu={params['mu']:.3f} "
        f"home_adv={params['home_adv']:.3f} | iters={params.get('optimizer_iters', 0)}"
    )
    print(f"  Saved to {CURRENT_CACHE_PATH}")

    if args.cache_only:
        return

    summary = summarize_params(params)
    print(f"\n{'Team':<32} {'Attack':>8} {'Defense':>8} {'Net':>8}")
    print("-" * 62)
    for _, row in summary.iterrows():
        print(
            f"  {row['team']:<30} {row['attack']:>8.3f} "
            f"{row['defense']:>8.3f} {row['net']:>+8.3f}"
        )

    if args.validate:
        if args.season == 2025:
            validate_2025(games)
        else:
            result = evaluate_season(games, season=args.season)
            print(f"\n--- {args.season} out-of-sample validation ---")
            print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
