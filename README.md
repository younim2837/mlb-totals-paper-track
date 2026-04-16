# MLB Totals Betting Model

This project builds an MLB game-total model for betting workflows.

The current production pipeline is:

1. Collect historical game, pitcher, weather, bullpen, lineup, umpire, and market data
2. Fit Dixon-Coles team priors from historical scores
3. Build leakage-safe pregame features for every historical game
4. Train XGBoost models for home runs and away runs
5. Calibrate the total prediction and its uncertainty
6. Compare model probabilities to sportsbook and Kalshi lines
7. Size bets with quarter-Kelly on a configurable bankroll

The model is built around totals, not moneylines or sides.

## Current methodology

### 1. Bayesian prior layer

The project no longer uses Elo as the main prior.

The active prior engine is [dixon_coles.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/dixon_coles.py), which fits:

- team attack strength
- team defense strength
- global scoring level
- home-field effect

Important details:

- fits are time-decayed, so recent games matter more than old games
- fits use a rolling historical window
- fits are regularized with an L2 penalty to prevent unstable team parameters
- historical priors are generated pregame only, so the training set does not leak future information

Outputs:

- `data/dc_params_current.json`
- `data/dc_ratings_history.tsv`

Those priors become model features such as:

- `home_dc_attack`
- `home_dc_defense`
- `away_dc_attack`
- `away_dc_defense`
- `dc_lambda_home`
- `dc_lambda_away`
- `dc_expected_total`

### 2. Feature engineering

[build_features.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/build_features.py) converts raw historical game data into `data/mlb_model_data.tsv`.

The feature set currently includes:

- recent team scoring and prevention form over 10- and 30-game windows
- season-to-date team scoring and prevention
- park factor
- weather for outdoor games
- umpire historical scoring tendency
- head-to-head recent total history
- Dixon-Coles priors
- rolling team batting quality
- same-day lineup strength features
- bullpen quality-tier and availability features
- starting pitcher recent form
- starter workload / leash / opener-style context

Key design choices:

- all rolling features are shifted to remain pregame and leakage-safe
- indoor/retractable-roof venues suppress weather effects
- same-day lineup features are built from actual starting nines when available
- lineup stats are shrunk toward league-average priors so tiny early-season samples do not dominate

### 3. Point model

[train_model.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/train_model.py) trains a team-split XGBoost model family:

- one model predicts `home_score`
- one model predicts `away_score`
- both are trained as residuals on top of Dixon-Coles expected runs
- the two team predictions are summed into the final total

This means the model is not learning totals from scratch. It starts from the Bayesian prior and learns where that prior is wrong.

### 4. Calibration and uncertainty

The raw point prediction is not used directly as the final betting probability layer.

The project adds three post-point-model layers:

1. Total calibration
- a blended isotonic mapping is fit on out-of-fold training predictions
- an additional tail-expansion step helps keep the distribution from collapsing too tightly into the middle

2. Dynamic uncertainty model
- a second XGBoost model predicts game-specific absolute error
- that becomes `prediction_std`, a per-game sigma instead of one global spread for every matchup

3. Tail models
- a high-tail classifier estimates `P(total > 9.5)`
- a low-tail classifier estimates `P(total < 7.5)`
- these do not replace the point total
- instead, they improve probability estimates in tail scenarios without forcing every point prediction wider

This is important because MLB totals are noisy. A model can have a reasonable expected total while still needing help describing shootout and dud-game risk.

### 5. Market comparison

The model compares its probabilities to:

- sportsbook totals from The Odds API when an API key is configured
- Kalshi total-run markets via the public Kalshi API

Kalshi support lives in [collect_kalshi_lines.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/collect_kalshi_lines.py).

Important note:

- the displayed Kalshi line is the actual tradable strike
- the code also estimates a crossover level internally, but the report shows the real half-run contract line

### 6. Bet sizing

Bet sizing is Kelly-based and configured in [model_config.yaml](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/model_config.yaml).

Current behavior:

- bankroll is configurable
- quarter-Kelly is the default
- hard caps can be disabled
- both sportsbook-style and Kalshi-style Kelly sizing are supported

Bet sizing uses the final model probabilities, so it does incorporate the Bayesian prior layer indirectly through the full model stack.

## Project structure

### Data collection

- [collect_games.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/collect_games.py): historical MLB game results
- [collect_pitcher_stats.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/collect_pitcher_stats.py): per-start pitcher game logs
- [collect_weather.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/collect_weather.py): historical venue weather
- [collect_team_lineups.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/collect_team_lineups.py): historical starting lineup features
- [collect_bullpen_usage.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/collect_bullpen_usage.py): reliever appearance logs
- [collect_umpires.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/collect_umpires.py): historical home-plate umpires
- [collect_lines.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/collect_lines.py): historical / live sportsbook totals
- [collect_lines_historical_oddsapi.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/collect_lines_historical_oddsapi.py): historical Odds API backfill for featured MLB markets (`h2h`, `spreads`, `totals`)

### Shared feature helpers

- [lineup_features.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/lineup_features.py)
- [bullpen_usage.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/bullpen_usage.py)

### Priors and modeling

- [dixon_coles.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/dixon_coles.py)
- [build_features.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/build_features.py)
- [train_model.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/train_model.py)

### Prediction and evaluation

- [predict_today.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/predict_today.py): live prediction report
- [run_today.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/run_today.py): daily entry point
- [backtest.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/backtest.py): holdout and line-based evaluation
- [walk_forward_backtest.py](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/walk_forward_backtest.py): month-by-month walk-forward evaluation

## Main artifacts

- `data/mlb_games_raw.tsv`: completed game history
- `data/mlb_model_data.tsv`: training matrix
- `data/dc_ratings_history.tsv`: historical Dixon-Coles priors
- `data/dc_params_current.json`: current live Dixon-Coles fit
- `data/team_lineup_features.tsv`: historical lineup features
- `data/bullpen_appearance_logs.tsv`: reliever usage history
- `data/umpire_game_log.tsv`: umpire history
- `data/lines_historical_oddsapi_snapshots.tsv`: raw historical sportsbook snapshots from The Odds API
- `data/lines_historical_oddsapi.tsv`: deduped historical sportsbook featured markets from The Odds API
- `data/lines_historical_oddsapi_requests.tsv`: request-level success / failure log for resumable backfills
- `models/home_runs_xgb.json`
- `models/away_runs_xgb.json`
- `models/total_runs_uncertainty_xgb.json`
- `models/total_runs_high_tail_xgb.json`
- `models/total_runs_low_tail_xgb.json`
- `models/model_meta.json`

## Daily workflow

The normal entry point is:

```powershell
python run_today.py
```

Useful variants:

```powershell
python run_today.py 2026-04-11
python run_today.py --no-update
python run_today.py --quick
python run_today.py 2026-04-11 --no-update --all-games
```

Default behavior is pregame-only. `--all-games` includes live and final games for inspection and writes a separate output file like `predictions/YYYY-MM-DD-all-games.txt`.

## Historical odds prep

The repo is prepared for a future historical sportsbook backfill via The Odds API.

Before buying a plan or adding a key, you can estimate the size of the job:

```powershell
python collect_lines_historical_oddsapi.py --years 2021-2024 --estimate-only
```

Once an `ODDS_API_KEY` is available, the same script can fetch historical totals
snapshots for moneyline, spread, and totals markets, then build a deduped
`lines_historical_oddsapi.tsv` file for later betting backtests and
market-aware calibration work. The request log lets `--resume` retry only the
timestamps that still failed.

More setup detail lives in [HISTORICAL_ODDS_SETUP.txt](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/HISTORICAL_ODDS_SETUP.txt).

## Model evaluation philosophy

This repo uses more than one success metric.

### Point prediction quality

- MAE
- RMSE
- R²

### Probability quality

- Brier score
- log loss
- 1-sigma / 2-sigma coverage
- line-specific probability checks at common totals such as `7.5`, `8.5`, and `9.5`

### Betting quality

- win rate versus historical lines
- ROI by edge bucket
- side-by-side comparison with sportsbook and Kalshi prices

This matters because a betting model can be useful even if raw total-run MAE is still noisy, as long as the probability layer is calibrated well at the actual betting line.

## Current design choices and caveats

- Dixon-Coles is the active prior engine; Elo remains in the repo as a legacy component and fallback for older model variants.
- Kalshi is currently a useful market-context feed, but sportsbook consensus is still preferable when available.
- Point totals are intentionally less volatile than realized MLB scores. The tail classifiers are there to improve betting probabilities without forcing unrealistic point estimates.
- Same-day lineups, bullpens, weather, and umpires can materially change a live projection, so late refreshes are more informative than early-morning runs.
- The project is optimized for MLB totals only.

## Rebuilding from scratch

If you want to rebuild the whole pipeline manually, the rough order is:

```powershell
python collect_games.py
python collect_pitcher_stats.py --update-current
python collect_weather.py
python collect_team_batting.py
python collect_team_lineups.py --update-current
python collect_bullpen_usage.py --update-current
python collect_umpires.py --update
python dixon_coles.py --date 2026-04-11 --cache-only
python build_features.py
python train_model.py
python backtest.py
python walk_forward_backtest.py
python predict_today.py 2026-04-11
```

## Configuration

[model_config.yaml](/c:/Users/Dakota/Sports%20Betting%20Project%20Folder/model_config.yaml) controls:

- bankroll and Kelly sizing
- manual day-level overrides
- Dixon-Coles fit settings
- betting signal thresholds
- display behavior
- prepared future market-line settings under `market_lines`

The comments in that file are meant to be edited directly.
