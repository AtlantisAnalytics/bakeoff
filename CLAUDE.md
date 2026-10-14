# Working on bakeoff

- Read PLAN.md first. It holds the decisions, the public/private boundary and the milestones.
- This repo will be public. Follow the scrub rules in PLAN.md for every commit:
  - no internal hostnames, IPs, ports, usernames or local paths;
  - no secrets.
- Keep raw run output in `runs/` (git-ignored). Only scrubbed bundles go in `results/` (CC BY 4.0).
- The site is plain HTML/CSS in `site/`, with no framework and no third-party scripts or trackers.
- Python: uv, ruff and pytest.
- Bots work on `athena/*` branches and open draft PRs only. Never push to `main` from automation.
