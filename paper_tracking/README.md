# Paper Tracking

This directory is rebuilt by the local/cloud paper-testing pipeline.

Generated files:

- `LATEST_STATUS.md`: quickest at-a-glance status page
- `paper_summary_YYYY.md`: season summary snapshot, Kalshi section first
- `kalshi_tracker_YYYY.tsv`: season-long graded Kalshi forward-tracking ledger
- `runs/YYYY-MM-DD.json`: one manifest per scheduled prediction run
- `sportsbook_tracker_YYYY.tsv`: optional sportsbook ledger if Odds API lines are available

The GitHub Actions workflows update these files automatically.
