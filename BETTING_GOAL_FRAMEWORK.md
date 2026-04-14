# Betting Goal Framework

## Objective

The model's job is not just to predict total runs. Its job is to generate
betting decisions that beat market prices out of sample and hold up over a
full season.

## North-Star Metric

`Walk-forward ROI vs closing lines`

- Why: this is the cleanest season-level answer to "would this have made money
  against the market?"
- How to evaluate: train only on games available before each evaluation month,
  then bet into that month's historical closing totals.
- Current deployment mindset:
  - `> +2% ROI`: strong, scalable candidate
  - `0% to +2% ROI`: positive but still fragile
  - `-2% to 0% ROI`: watch list
  - `< -2% ROI`: not deployable

## Core Supporting Metrics

`Closing-line Brier score`

- Why: profit can be noisy; Brier tells us whether the probabilities at the
  line are actually calibrated.
- Practical target: keep it around `<= 0.252` on honest walk-forward tests.

`Matched bets`

- Why: a tiny sample can create fake confidence.
- Practical target: at least `1,500` matched bets for a full-season read.

`Monthly stability`

- Why: a profitable season with one catastrophic month is hard to trust or
  bankroll.
- Practical target: investigate any month worse than about `-8% ROI`.

`Forecast health`

- Why: MAE still matters as a model-health metric even though it is not the
  north star.
- Track: `MAE`, `bias`, `sigma coverage`, and tail-probability calibration.

## Bet Selection Metrics

Use walk-forward betting detail to learn which bets are worth taking, not just
whether the model was profitable overall.

Track:

- ROI by `|edge|`
- ROI by confidence, using `max(p_over, p_under)`
- ROI by month
- ROI by direction (`OVER` vs `UNDER`)

This helps us define the real production rule, for example:

- "bet only when confidence is at least 53%"
- "do not assume the biggest raw edge is the best edge"

## Next Metric To Add

`CLV / beat-the-close rate`

- Now that we have historical snapshots, the next major evaluation metric
  should be whether our bet timestamp beats the eventual closing number.
- That requires a defined entry rule first, such as "bet the first snapshot
  after lineups confirm" or "bet 60 minutes before first pitch."

## Commands

Run honest 2025 walk-forward betting:

```powershell
python walk_forward_betting_backtest.py --year 2025
```

Score it against this framework:

```powershell
python betting_goal_framework.py --year 2025
```

## Deployment Gameplan

### P0: Define A Safe 2026 Deployment Rule

Goal: stop changing the betting logic ad hoc and freeze one rule that can be
tested honestly.

Action items:

- Choose one production bet trigger using only pre-2025 evidence.
- Candidate format:
  - bet only when `max(p_over, p_under) >= threshold`
  - optionally require `|edge| >= threshold`
  - optionally require confirmed lineups
- Document bankroll rules:
  - paper trade or minimum size only
  - cap daily exposure
  - no threshold changes mid-season unless a full re-evaluation is done

Deliverable:

- A single frozen 2026 betting policy written down in the repo.

Current candidate:

- only place live bets when `market_adjustment_method == edge_model`
- treat fallback methods like `shrinkage_guarded` as informational only
- keep the current entry approximation at `60` minutes before first pitch
- keep threshold tuning separate from the method-eligibility rule

Exit criteria:

- We can answer "why did we place this bet?" with a fixed rule, not judgment.

### P1: Add Snapshot Entry + CLV Backtesting

Goal: prove we can beat the market before close, not just look decent at close.

Action items:

- Build a backtest using historical odds snapshots instead of closing lines only.
- Define 2-3 entry rules, for example:
  - first snapshot after lineups confirm
  - 60 minutes before first pitch
  - first snapshot where model confidence crosses threshold
- Track:
  - CLV in runs
  - beat-the-close rate
  - ROI by entry rule
  - time-to-first-pitch performance

Deliverable:

- A repeatable snapshot-entry backtest and CLV report.

Exit criteria:

- At least one frozen entry rule shows positive CLV and positive walk-forward ROI.

Command:

```powershell
python walk_forward_snapshot_backtest.py --year 2025 --entry-rule closest_before_minutes --minutes-before 60 --min-confidence 0.53 --min-edge 0.5
```

### P2: Validate The Frozen Rule Across Seasons

Goal: make sure 2025 was not a one-off.

Action items:

- Run the same frozen rule on 2022, 2023, 2024, and 2025.
- Generate one scorecard per season.
- Compare:
  - ROI
  - Brier
  - monthly drawdowns
  - bet volume

Deliverable:

- Multi-season walk-forward scorecards with one unchanged betting rule.

Exit criteria:

- Most seasons are positive or near-flat, with no obvious structural collapse.

### P3: Improve Bet Ranking And Calibration

Goal: make the model better at deciding which bets are truly worth taking.

Action items:

- Refit and evaluate probability calibration at the market line.
- Review ROI by confidence band and by edge band.
- Diagnose over/under asymmetry.
- Add monitoring for:
  - monthly bias
  - calibration drift
  - sigma coverage drift

Deliverable:

- A better-ranked bet list and a clearer confidence threshold.

Exit criteria:

- Closing-line Brier is consistently at or below target and high-confidence
  bets outperform the full set.

### P4: Diagnose Regime Failures

Goal: understand why weak months happen and reduce drawdowns.

Action items:

- Break down 2025 performance by:
  - month
  - total-line range
  - park / roof / weather regime
  - lineup-confirmed vs not
  - over vs under
  - line movement agreement vs disagreement
- Specifically audit August 2025 and any other bad clusters.

Deliverable:

- A short failure analysis with 2-3 concrete hypotheses to test.

Exit criteria:

- We can explain the major drawdown months with evidence instead of guesswork.

### P5: Add The Highest-Leverage Model Improvements

Goal: improve the model only where it clearly supports betting outcomes.

Action items:

- Add interaction features for weather and game timing where useful.
- Improve the market layer using:
  - time to first pitch
  - line movement
  - book disagreement
  - lineup confirmation state
- Continue tail and sigma calibration maintenance.

Deliverable:

- A revised betting model that improves the scorecard, not just MAE.

Exit criteria:

- Changes improve walk-forward betting metrics under the same frozen rule.

## Priority Order

If we do this in the right order, it should be:

1. Freeze the 2026 betting rule.
2. Build snapshot-entry and CLV backtesting.
3. Validate the frozen rule across 2022-2025.
4. Improve calibration and bet ranking.
5. Investigate regime failures and drawdowns.
6. Add model improvements only after the evaluation stack is solid.

## What "Deployable" Means

We should call this deployable only when all of the following are true:

- Frozen betting rule
- Positive multi-season walk-forward ROI
- Positive CLV under a defined entry rule
- Acceptable closing-line calibration
- No major unresolved failure month or regime issue
- Bankroll rules are documented and followed
