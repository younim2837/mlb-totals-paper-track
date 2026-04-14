# GitHub Cloud Setup

## What this gives you

- daily prediction runs in the cloud
- next-morning grading against completed MLB results
- rolling `paper_tracking/` files you can review in June

## One-time setup

1. Create a private GitHub repository.
2. Push this project into that repo.
3. In the GitHub repo, add this Actions secret:
   - `ODDS_API_KEY`
4. Enable GitHub Actions for the repo.

## Workflows

- `Daily Paper Track`
  - scheduled for `15:30 UTC` every day
  - runs `paper_track_daily.py`
  - refreshes season trackers afterward

- `Daily Paper Grade`
  - scheduled for `13:15 UTC` every day
  - refreshes completed MLB games
  - rebuilds `paper_tracking/` summaries

## Important notes

- GitHub Actions schedules use UTC in these workflow files.
- `Daily Paper Track` is set for a pregame morning run in Pacific time.
- If you want a different run time, edit the cron line in `.github/workflows/daily-paper-track.yml`.
- The workflows commit generated `data/`, `predictions/`, and `paper_tracking/` updates back into the repo so state persists between runs.

## Manual runs

You can also launch both workflows manually from the GitHub Actions tab:

- `Daily Paper Track`: optional date / quick / no-update / all-games inputs
- `Daily Paper Grade`: optional season input
