# Frozen 2026 Betting Rule

**Frozen date:** 2026-04-13
**Status:** Paper trade only. No real money until exit criteria are met.

## The Rule

Place a paper bet when ALL of the following are true:

1. `market_adjustment_method == edge_model`
2. `max(p_over, p_under) >= 0.55`
3. `|edge| >= 0.5` runs
4. Entry snapshot is within 60 minutes before first pitch
5. At least 2 books contributing to the consensus line

Bet the side indicated by the edge model (OVER if `p_over > p_under`,
UNDER otherwise).

### Why These Thresholds

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Method | `edge_model` only | Legacy shrinkage has no track record worth trusting |
| Confidence | >= 0.55 | 0.53 includes too many marginal bets (see August 2025). 0.55 cuts volume ~15% but filters the weakest signals. Backtest 2025 at 0.55: 1,319 bets, 55.5% win, +5.9% ROI vs +4.6% at 0.53. |
| Edge | >= 0.5 runs | Below 0.5, ROI collapses toward zero in every year |
| Entry time | 60 min pre-game | Matches the historical snapshot backtest rule |
| Min books | 2 | Already in `model_config.yaml`; thin lines are noise |

### What Changed From The Previous Candidate

The framework doc listed `conf >= 0.53`. We raise it to `0.55` because:

- August 2025 had avg confidence 0.535 -- those marginal bets were the worst
  performers (-11.5% ROI, 318 bets, 0 wins above breakeven).
- At 0.55, August 2025 still hurts but the damage is diluted by stronger
  months keeping more of their volume.
- The 0.53 threshold was never validated on pre-2025 data as a final rule.

## Bankroll Rules

- **Paper trade only** through at least 2026-06-30 (minimum ~500 paper bets).
- Log every qualifying signal, including ones you don't act on.
- No threshold changes during the paper-trade window.
- If transitioning to real money:
  - Start at quarter-Kelly on a bankroll you can lose entirely.
  - Cap daily exposure at 3 bets maximum.
  - No bet larger than 3% of bankroll.

## Known Risks

### 1. UNDER Skew (Critical)

The edge model has a severe, worsening directional bias:

| Year | % UNDER bets | UNDER win rate | OVER win rate |
|------|-------------|----------------|---------------|
| 2022 | 58% | 55.6% | 47.7% |
| 2023 | 79% | 53.8% | 51.4% |
| 2024 | 90% | 53.5% | 49.8% |
| 2025 | 96% | 52.9% | 51.6% |

In 2025, the raw model had 54.4% UNDER signals (balanced). The edge model
turned this into 92.9% UNDER signals. The edge model's directional accuracy
(51.8%) was actually worse than the raw model (53.2%).

**This means: the model is not diversified.** A hot-scoring month will hit
harder than it should because nearly every bet is an UNDER. This is
concentrated exposure, not a diversified portfolio.

**Monitoring action:** Track monthly OVER/UNDER ratio. If >90% of bets are
one direction for 2+ consecutive months, flag for review.

### 2. Near-Zero CLV

| Year | Beat-close rate | Mean CLV (runs) |
|------|----------------|-----------------|
| 2022 | 7.4% | +0.008 |
| 2023 | 7.0% | +0.008 |
| 2024 | 9.4% | +0.018 |
| 2025 | 6.9% | +0.006 |

The model almost never gets a better price than the eventual close. Profit
comes from directional accuracy, not from finding mispriced lines early.
This is fragile: you're competing on the same information as the close,
not exploiting an information advantage.

### 3. August-Style Drawdowns

August 2025: 343 bets, 100% UNDER, 46.4% win rate, -11.5% ROI. This one
month wiped out 2-3 good months. The model took zero OVER bets despite the
raw model being roughly balanced. The edge model's UNDER bias amplified the
loss.

**Monitoring action:** If any rolling 30-day window drops below -8% ROI,
pause paper trading and run a post-mortem before resuming.

## Backtest Evidence

### Snapshot Entry (edge_model, conf >= 0.53, edge >= 0.5, 60min)

| Year | Bets | Win Rate | ROI | Brier |
|------|------|----------|-----|-------|
| 2022 | 941 | 52.0% | -0.8% | 0.263 |
| 2023 | 1,364 | 55.1% | +5.3% | 0.264 |
| 2024 | 1,294 | 53.5% | +2.1% | 0.255 |
| 2025 | 1,497 | 54.8% | +4.6% | 0.251 |

3 of 4 years positive. Average ~+2.8% ROI. The losing year (2022) was
nearly flat at -0.8%.

### Closing-Line Walk-Forward (all bets, no filter)

| Year | Bets | Win Rate | ROI |
|------|------|----------|-----|
| 2021 | 1,643 | 50.8% | -3.1% |
| 2022 | 2,010 | 50.1% | -4.3% |
| 2023 | 2,020 | 52.9% | +0.9% |
| 2024 | 2,029 | 51.8% | -1.1% |
| 2025 | 2,699 | 52.8% | +0.9% |

Unfiltered, the model is a coin flip against the close. The bet selection
filter is doing all the work.

## Exit Criteria For Deployment

All of the following must be true before placing real money:

- [ ] 500+ paper bets logged under this exact frozen rule
- [ ] Paper-trade ROI is positive after 500 bets
- [ ] No month worse than -15% ROI
- [ ] UNDER ratio stays below 95% (if it exceeds this consistently, the
      edge model needs recalibration before proceeding)
- [ ] You can explain every bet with the rule -- no judgment calls

## What This Rule Cannot Tell You

- Whether the edge model's UNDER bias is a feature or a bug. It's been
  profitable, but it's getting more extreme every year.
- Whether CLV will ever be meaningfully positive. Right now the model wins
  on direction, not on price timing.
- Whether 2023 (+5.3%) was skill or the best year in a distribution that
  includes -0.8% just as often.

## Commands

Run the frozen rule on historical data:

```powershell
python walk_forward_snapshot_backtest.py --year 2025 --entry-rule closest_before_minutes --minutes-before 60 --min-confidence 0.55 --min-edge 0.5
```

Daily prediction (requires Odds API key for live lines):

```powershell
python run_today.py
```

Score against the goal framework:

```powershell
python betting_goal_framework.py --year 2025
```
