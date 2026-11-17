# Submission checklist

## Current artifacts

- Possession CSV: `evaluation/results/layered_pipeline.csv`
- Technical document: `TECHNICAL_DOCUMENT.md`
- Reproduction dependencies: `requirements-submission.txt`
- Reproduction entry point: `starter/run_possession.py --model layered`
- Public repository remote: `https://github.com/zaourid1/football-possession-detection`

The generated CSV and reports are deliberately ignored by Git. The public code
repository should contain source, tests, documentation, and dependencies, but
must not contain the gated match assets or a generated full-match label file.

## Before submitting

- [ ] Replace `<team_name>` below with the exact registered team name.
- [ ] Copy or rename the exact validated CSV to `<team_name>.csv`.
- [ ] Copy or rename `TECHNICAL_DOCUMENT.md` to `<team_name>.md`, or convert it
      to `<team_name>.pdf` without changing its technical claims.
- [ ] Regenerate the final CSV from a clean environment using the command in
      `TECHNICAL_DOCUMENT.md`.
- [ ] Confirm its SHA-256 is
      `0ed4dbf6b17280e3ce86fbe7a7145bc209ac7b06a9459702cc132027596a8902`.
- [ ] Run `python3 evaluation/validate_submission.py <team_name>.csv` on the
      exact renamed upload file.
- [ ] Confirm `git status --short` shows no match assets, generated report, or
      generated prediction staged for the public repository.
- [ ] Commit the source, tests, documentation, and pinned dependencies.
- [ ] Push the reviewed commit to the public repository and verify it is visible.
- [ ] Check the repository README gives the layered reproduction command.
- [ ] Upload the exact validated CSV, public repository URL, and technical
      document through the form before the deadline.
- [ ] Verify every registered team member used the same team name and that the
      team has 3–5 members as required by RULES.md.

Do not tune or justify this submission using external human-produced statistics,
events, or possession summaries from the provided match.
