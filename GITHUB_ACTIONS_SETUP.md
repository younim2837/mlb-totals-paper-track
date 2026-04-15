# GitHub Cloud Setup

## What this gives you

- daily prediction runs in the cloud
- next-morning grading against completed MLB results
- rolling `paper_tracking/` files you can review in June
- a Kalshi-first forward test without needing your computer on

## One-time setup

1. Create a private GitHub repository.
2. Push this project into that repo.
3. Enable GitHub Actions for the repo.
4. Optional only if you also want sportsbook tracking:
   - add the Actions secret `ODDS_API_KEY`

## Workflows

- `Daily Paper Track`
  - scheduled for `17:07 UTC` every day
  - runs `paper_track_daily.py`
  - refreshes season trackers afterward

- `Daily Paper Grade`
  - scheduled for `13:15 UTC` every day
  - refreshes completed MLB games
  - rebuilds `paper_tracking/` summaries

## Important notes

- GitHub Actions schedules use UTC in these workflow files.
- `Daily Paper Track` is set for a pregame morning run in Pacific time.
- The current `Daily Paper Track` schedule is `10:07 AM Pacific` during daylight saving time and `9:07 AM Pacific` during standard time.
- This is intentionally earlier than first pitch so the run is less likely to miss unusual early slates.
- It is offset from the top of the hour because GitHub Actions is more likely to delay or drop scheduled runs right at `:00`.
- If you want a different run time, edit the cron line in `.github/workflows/daily-paper-track.yml`.
- The workflows commit generated `data/`, `predictions/`, and `paper_tracking/` updates back into the repo so state persists between runs.
- Kalshi tracking works without `ODDS_API_KEY`.
- If no Odds API key is configured, the sportsbook section will stay optional/mostly empty while Kalshi tracking still runs.

## Manual runs

You can also launch both workflows manually from the GitHub Actions tab:

- `Daily Paper Track`: optional date / quick / no-update / all-games inputs
- `Daily Paper Grade`: optional season input

## What to read

- `paper_tracking/LATEST_STATUS.md`
  This is the fastest health/performance snapshot.
- `paper_tracking/paper_summary_YYYY.md`
  This is the fuller monthly summary.
- `paper_tracking/kalshi_tracker_YYYY.tsv`
  This is the detailed Kalshi forward-tracking ledger.
- `Actions`
  Use this tab to confirm the daily jobs are still succeeding.
