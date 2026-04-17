# MLB Totals Betting Model

A production MLB game-total prediction system built on Dixon-Coles Bayesian priors, XGBoost team-split models, and Kalshi binary market integration. The model paper-trades on Kalshi's run-total markets with fractional Kelly sizing.

**Live dashboard:** https://younim2837.github.io/mlb-totals-paper-track/

---

## Architecture overview

The pipeline has four layers, each building on the one before:

```
Raw game data → Feature engineering → XGBoost models → Calibration/Betting
```

### Layer 1: Bayesian prior (Dixon-Coles)

[dixon_coles.py](dixon_coles.py) fits a bivariate Poisson model to historical scores:

- **Parameters:** team attack strength, team defense strength, global scoring level, home-field advantage
- **Time decay:** exponential weighting so recent games dominate
- **Regularization:** L2 penalty prevents unstable team ratings with small samples
- **Leakage safety:** priors are fit using only data available before each game

Key outputs:

| Feature | Meaning |
|---|---|
| `home_dc_attack` / `away_dc_attack` | Offensive strength parameter |
| `home_dc_defense` / `away_dc_defense` | Defensive strength parameter |
| `dc_lambda_home` / `dc_lambda_away` | Expected runs per team |
| `dc_expected_total` | Sum of Poisson means |

Artifacts: `data/dc_params_current.json`, `data/dc_ratings_history.tsv`

### Layer 2: Feature engineering

[build_features.py](build_features.py) assembles the full training matrix into `data/mlb_model_data.tsv`.

**Feature families:**

| Family | Source script | Key features |
|---|---|---|
| Dixon-Coles priors | `dixon_coles.py` | `dc_expected_total`, `dc_lambda_*` |
| Team rolling form | `collect_games.py` | Runs scored/allowed over 10- and 30-game windows |
| Team batting quality | `collect_team_batting.py` | OPS, wOBA, ISO (7-/30-/90-game rolling) |
| Park factors | `venue_metadata.py` | `park_factor` (run environment scalar) |
| Weather | `collect_weather.py` | Temperature, wind speed/direction, precipitation |
| Umpire tendency | `collect_umpires.py` | `ump_era_adj`, `ump_runs_per_game_above_avg` |
| Pitching quality | `collect_pitcher_stats.py` | Starter ERA, K/9, BB/9, IP/start (rolling) |
| Bullpen availability | `collect_bullpen_usage.py` | Reliever usage tiers, rest, quality score |
| Lineup strength | `collect_team_lineups.py` | Actual starting-nine OPS, wRC+, handedness splits |
| Head-to-head | `collect_games.py` | Recent series total history |

All rolling features are shifted one period forward — no future leakage into training.

### Layer 3: XGBoost team-split model

[train_model.py](train_model.py) trains two separate regressors:

- **Home model** predicts `home_score` as a residual on top of `dc_lambda_home`
- **Away model** predicts `away_score` as a residual on top of `dc_lambda_away`
- **Total** = home prediction + away prediction

Two auxiliary models:

- **Uncertainty model**: second XGBoost predicts per-game absolute error → `prediction_std`
- **Market edge model**: XGBoost classifier trained on historical Kalshi outcomes to filter for true edge

**Training data:** 2021–2025 seasons (~11,700 games)

**Walk-forward fold validation** (no leakage):

| Fold | Train | Test | MAE |
|---|---|---|---|
| Fold 1 | 2021–2023 | 2024 | 3.364 |
| Fold 2 | 2021–2024 | 2025 | 3.552 |

### Layer 4: Calibration, tail classifiers, and betting

After the raw point prediction:

1. **Isotonic calibration** — blended mapping fit on out-of-fold training predictions with tail expansion
2. **High-tail classifier** — `P(total ≥ 9.5)` from a separate XGBoost binary model
3. **Low-tail classifier** — `P(total ≤ 7.5)` from a separate XGBoost binary model
4. **Heteroscedastic sigma** — per-game uncertainty estimate (not one global spread)

Betting:

- **Kelly criterion**: fractional Kelly (15% of full Kelly) applied to `P(over)` vs Kalshi ask price
- **Edge filter**: market edge model must confirm the bet before sizing
- **Bankroll**: configurable in `model_config.yaml`

---

## Repository structure

### Data collection

| Script | Purpose | Update mode |
|---|---|---|
| [collect_games.py](collect_games.py) | Historical MLB game results | `--update` |
| [collect_pitcher_stats.py](collect_pitcher_stats.py) | Per-start pitcher game logs | `--update-current` |
| [collect_team_batting.py](collect_team_batting.py) | Team batting logs by game | `--update-current` |
| [collect_team_lineups.py](collect_team_lineups.py) | Starting lineup features | `--update-current` |
| [collect_bullpen_usage.py](collect_bullpen_usage.py) | Reliever appearance logs | `--update-current` |
| [collect_umpires.py](collect_umpires.py) | Home-plate umpire history | `--update` |
| [collect_weather.py](collect_weather.py) | Historical venue weather | (batch) |
| [collect_kalshi_lines.py](collect_kalshi_lines.py) | Live Kalshi run-total market | (live) |
| [collect_kalshi_historical.py](collect_kalshi_historical.py) | Historical Kalshi snapshots | (backfill) |

### Modeling

| Script | Purpose |
|---|---|
| [dixon_coles.py](dixon_coles.py) | Fit Bayesian team attack/defense priors |
| [build_features.py](build_features.py) | Assemble full training matrix |
| [train_model.py](train_model.py) | Train XGBoost suite + calibration + edge model |

### Prediction

| Script | Purpose |
|---|---|
| [predict_pregame.py](predict_pregame.py) | Pregame per-game predictions with Kalshi sizing |
| [predict_today.py](predict_today.py) | Full-day prediction report |
| [run_today.py](run_today.py) | Daily entry point (collects, then predicts) |

### Evaluation

| Script | Purpose |
|---|---|
| [backtest.py](backtest.py) | Holdout and line-based evaluation |
| [walk_forward_backtest.py](walk_forward_backtest.py) | Month-by-month walk-forward evaluation |
| [walk_forward_betting_backtest.py](walk_forward_betting_backtest.py) | Walk-forward with Kelly sizing |
| [sweep_kalshi_2026.py](sweep_kalshi_2026.py) | Kalshi market simulation for 2026 |

### Paper trading

| Script | Purpose |
|---|---|
| [paper_track_daily.py](paper_track_daily.py) | Grade each day's predictions against results |
| [grade_paper_tracking.py](grade_paper_tracking.py) | Aggregate paper P&L across all graded days |
| [paper_bankroll.py](paper_bankroll.py) | Track running bankroll history |
| [build_dashboard.py](build_dashboard.py) | Regenerate GitHub Pages dashboard |

---

## Automated workflows

Two GitHub Actions workflows keep the pipeline current:

### 1. Daily grader (`.github/workflows/daily-paper-grade.yml`)

Runs at **12:00 UTC** every day. Sequence:

1. Refresh completed games (`collect_games.py`)
2. Refresh team batting stats (`collect_team_batting.py --update-current`)
3. Refresh pitcher stats (`collect_pitcher_stats.py --update-current`)
4. Refresh bullpen usage (`collect_bullpen_usage.py --update-current`)
5. Refresh umpire log (`collect_umpires.py --update`)
6. Refresh lineup features (`collect_team_lineups.py --update-current`)
7. Replay historical Kalshi snapshots (`collect_kalshi_historical.py`)
8. Grade yesterday's predictions (`paper_track_daily.py`)
9. Rebuild dashboard (`build_dashboard.py`)
10. Commit and push updated data + dashboard

### 2. Pregame predictor (`.github/workflows/predict-pregame.yml`)

Runs **every 20 minutes** during game hours. For each game starting 5–35 minutes from now:

1. Fetch current Kalshi market data
2. Run `predict_pregame.py` for that game
3. Append to today's picks file
4. Rebuild dashboard
5. Commit and push

---

## Data artifacts

| File | Description |
|---|---|
| `data/mlb_games_raw.tsv` | Completed game results (2021–present) |
| `data/mlb_model_data.tsv` | Full training matrix with all features |
| `data/dc_params_current.json` | Current live Dixon-Coles fit |
| `data/dc_ratings_history.tsv` | Historical Dixon-Coles priors per game |
| `data/pitcher_game_logs.tsv` | Per-start stats for all tracked starters |
| `data/team_batting_logs.tsv` | Team batting stats per game |
| `data/team_lineup_features.tsv` | Daily starting lineup features |
| `data/bullpen_appearance_logs.tsv` | Reliever usage history |
| `data/umpire_game_log.tsv` | Home-plate umpire per game |
| `models/home_runs_xgb.json` | Home score regressor |
| `models/away_runs_xgb.json` | Away score regressor |
| `models/total_runs_uncertainty_xgb.json` | Heteroscedastic sigma model |
| `models/total_runs_high_tail_xgb.json` | P(total ≥ 9.5) classifier |
| `models/total_runs_low_tail_xgb.json` | P(total ≤ 7.5) classifier |
| `models/model_meta.json` | Feature list, calibration config, fold metrics |

---

## Dashboard

The GitHub Pages dashboard at https://younim2837.github.io/mlb-totals-paper-track/ shows:

- **Latest Daily Bets**: today's Kalshi picks with confidence and sizing
- **Recent Performance**: last 7 days of paper trade outcomes
- **Season Totals**: cumulative P&L, win rate, ROI

Rebuilt automatically by `build_dashboard.py` after each grading run.

---

## Rebuilding from scratch

```bash
# 1. Collect all historical data
python collect_games.py
python collect_team_batting.py
python collect_pitcher_stats.py
python collect_team_lineups.py --update-current
python collect_bullpen_usage.py --update-current
python collect_umpires.py --update
python collect_weather.py

# 2. Fit priors and build features
python dixon_coles.py --cache-only
python build_features.py

# 3. Train
python train_model.py

# 4. Evaluate
python backtest.py
python walk_forward_backtest.py

# 5. Predict
python predict_today.py
```

---

## Configuration

[model_config.yaml](model_config.yaml) controls:

- Bankroll and Kelly fraction
- Edge and confidence thresholds
- Dixon-Coles decay and regularization settings
- Display behavior and manual overrides

---

## Design decisions

- **No ELO.** Dixon-Coles replaced ELO entirely. All model features are DC-derived or rolling empirical stats.
- **Team-split residuals.** The model learns home-score and away-score residuals on top of DC expected runs, not raw totals. This keeps the model grounded in the Bayesian prior.
- **No 2026 model leakage.** Model weights are trained on 2021–2025. The 2026 season contributes only live feature values (DC params, rolling stats, lineup, bullpen, umpire).
- **Monthly median imputation.** Feature medians are computed per-month to avoid early-season bias from global medians.
- **Pregame only.** The production scheduler targets games 5–35 minutes before first pitch, after lineups and umpires are posted.
- **Paper trading until June 2026.** The model paper-trades to accumulate real out-of-sample signal before any live capital is risked.
