# Codebase Map

This repo now follows a clearer split between entry scripts, shared runtime
modules, and data collection utilities.

## Entry scripts

- `predict_today.py`: orchestrates the daily prediction pipeline and CLI flow.
- `backtest.py`: historical closing-line backtest and accuracy reporting.
- `backtest_raw_probability.py`: proxy-strike raw probability backtest.
- `walk_forward_betting_backtest.py`: month-by-month walk-forward betting test.
- `walk_forward_snapshot_backtest.py`: snapshot-entry walk-forward test.

## Shared runtime modules

- `modeling_utils.py`: shared calibration and probability math.
- `model_runtime.py`: shared model loading and point-inference helpers.
- `market_adjustment.py`: market-aware shrinkage and edge-model logic.
- `prediction_betting.py`: betting, Kalshi proxy, and market-adjustment helpers.
- `prediction_reporting.py`: daily board export and terminal display helpers.

## Feature and data builders

- `build_features.py`: historical feature engineering for training data.
- `train_model.py`: training pipeline for point, uncertainty, and tail models.
- `lineup_features.py`, `bullpen_usage.py`, `team_elo.py`, `dixon_coles.py`,
  `league_environment.py`, `venue_metadata.py`: reusable domain feature logic.

## Data collectors

- `collect_*.py`: external data ingestion scripts for games, lines, weather,
  team batting, bullpen usage, umpires, pitchers, and Kalshi market context.

## Safety checks

- `smoke_check.py`: low-cost import/load/inference/export smoke test.

## Naming conventions

- Top-level `*_today.py`, `backtest*.py`, and `walk_forward*.py` files are
  script entry points.
- Shared logic belongs in modules with noun-style names such as
  `model_runtime.py` or `prediction_reporting.py`.
- Collectors remain action-oriented as `collect_*.py`.
