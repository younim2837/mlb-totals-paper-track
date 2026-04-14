# Paper Tracking

This directory is rebuilt by the local/cloud paper-testing pipeline.

Generated files:

- `runs/YYYY-MM-DD.json`: one manifest per scheduled prediction run
- `sportsbook_tracker_YYYY.tsv`: season-long graded sportsbook picks
- `kalshi_tracker_YYYY.tsv`: season-long graded Kalshi proxy trades
- `paper_summary_YYYY.md`: season summary snapshot for quick review

The GitHub Actions workflows update these files automatically.
