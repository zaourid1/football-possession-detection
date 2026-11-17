# Layer earlier predictions with physical corrections

The strict paper pipeline combined the newer feature modules, but required
physical evidence in place of some earlier decisions. Its 59.70% coverage was
not an accuracy improvement. The layered variant now uses the earlier robust
stillness model as its reference and the strict model as a correction layer.

No external match statistics are inputs or calibration/validation targets.
RULES.md prohibits human-produced data about this match. An aggregate possession
percentage also cannot establish frame or possession-interval accuracy.

## Decision policy

- Keep the reference prediction when the strict model abstains or has no
  confirmed physical anchor. Existing reference answers cannot become X.
- Begin an H/A correction at an observed confirmed contact or completed-pass
  reception. Held carriers and uncertain pass candidates cannot start it.
- Retain the correction through the strict model's bounded ownership memory,
  rather than flipping back to the reference whenever contact disappears.
- Return to the reference when strict ownership expires. Do not arbitrarily
  revise frames before an anchor just because they share a later label run.
- Reset corrections at a new half or stoppage. Preserve reference D
  conservatively; this policy can also preserve an incorrect older stoppage.
- Track provenance separately for reference predictions, measured contact,
  completed passes, physical memory, bridges, and stoppages.

Implementation: possession_layered.py. Earlier predictors and the runner's
default `--model paper` remain reproducible. Select layering explicitly:

```sh
python3 -m unittest discover -s starter -p 'test_possession_*.py'
python3 starter/run_possession.py --model layered --out evaluation/results/layered_pipeline.csv --report evaluation/reports/layered_pipeline_summary.json
python3 evaluation/validate_submission.py evaluation/results/layered_pipeline.csv
```

## Verification

All 60 synthetic tests pass, including nine new tests for layering. They cover
reference preservation, measured overrides, bounded continuation, late anchors,
held contacts, possible versus completed passes, D/half resets, and determinism.

Full-match graded-frame diagnostics:

| Measurement | Earlier robust stillness | Strict paper | Layered |
|---|---:|---:|---:|
| Non-X coverage | 75.88% | 59.70% | 76.08% |
| H frames | 69,022 | 54,537 | 68,950 |
| A frames | 47,513 | 36,289 | 47,899 |
| D frames | 4,009 | 4,009 | 4,009 |
| X frames | 38,313 | 64,022 | 37,999 |
| Raw label changes within halves | 690 | 657 | 604 |

Zero reference answers were erased; 6,949 graded labels differ from the
reference. Pass-candidate counts are unchanged because layering changes the
final decision policy rather than the underlying pass detector.

Fewer transitions may indicate less flicker OR missed genuine transitions.
Higher coverage can also introduce wrong answers. This run demonstrates the
intended layering and output invariants, not improved competition accuracy.
The public five-minute labels were not used to select a variant or parameters.
Short one-touch interactions and uncertain stoppage boundaries remain limitations.
