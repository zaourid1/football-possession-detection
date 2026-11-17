# Integrated control and pass pipeline

This implements the proposals in PAPER_REVIEW.md as an opt-in offline pipeline.
The later reference-preserving variant is documented in LAYERED_REVIEW.md.
The supplied baselines and existing prediction entry points remain available.
No ground truth or human-produced events from this match are used. Automatic
tactical tags are used as supporting context. No parameters were selected by
scoring the public five minutes.

## Run and inspect

From the repository root:

```sh
python3 -m unittest discover -s starter -p 'test_possession_*.py'
python3 starter/run_possession.py --out evaluation/results/paper_pipeline.csv --report evaluation/reports/paper_pipeline_summary.json
python3 evaluation/validate_submission.py evaluation/results/paper_pipeline.csv
```

The same NumPy/pandas dependencies already used by the repository suffice.
`--data-dir` can override the match bundle location. Without `--out` the runner
prints aggregate diagnostics only. Reports contain configuration and counts,
not tracking exports. The validator checks format, not label accuracy.

## Components to review

1. **possession_contacts.py:** strict entry/looser retention hysteresis, velocity
   vectors, and local controlled-contact episodes. Both teams must have usable
   distance evidence. Reliable player–ball relative movement distinguishes a
   plausible carrier from incidental proximity. Velocity windows reject missing
   coordinates, implausible steps, and half crossings. Nearest slot indices are
   local hypotheses; a plausible spatial handoff between slots retains the same
   episode. Carrier memory expires after 0.2 seconds by default and explicitly
   distinguishes measured contact from a held hypothesis.
2. **possession_passes.py:** candidate passes require distinct controlled episodes,
   release/travel evidence, sufficient ball movement, and a receiver separated
   from the projected sender. A continuous dribble, duplicate slot, opponent
   control, blocked interval, or half change rejects a completed pass. Partly
   occluded flights can produce uncertain candidates, but never increment the
   completed-pass chain. Chains carry decaying confidence.
3. **possession_pipeline.py:** one explicit path combining geometry, robust
   stillness, physical stoppage confirmation, contacts, passes, team evidence
   accumulation, and bounded offline gap bridging. Strong opposing control
   can establish possession without any passes. Same-team passes retain team
   ownership. Phase tags cannot independently create or switch ownership.
4. **possession_refinement.py:** the new optional `controlled_team` argument lets
   the integrated pipeline veto a bridge using actual opposing control rather
   than a single nearest-opponent observation. Old callers keep their original
   behaviour. Half, restart, D, opposing-label and maximum-gap safeguards remain.
5. **run_possession.py:** deterministic runner, output-domain checks, CSV export,
   and configuration/aggregate diagnostics. The comparison script now includes
   this integrated variant as well as the earlier models.

The physical carrier and team owner use independent memories. Credible observed
contacts refresh team support; completed-pass confidence can extend it, bounded
by eight seconds. A separate decaying team-support snapshot survives a temporary
reset of pass counting during missing observations. Stoppages, new halves, and
opposing control reset stale chain support. An uncertain pass cannot extend
team ownership. No number-of-passes prerequisite delays a genuine turnover.

This is offline: contact onsets can be backdated over the observed confirmation
run; robust stillness uses centred windows; gap bridging uses later observations.
Pass confirmation frames identify offline reception onsets, not real-time alerts.

## Verification and diagnostics

All **51 synthetic tests** pass, including the original model tests. New cases
cover distance jitter, local slot switches, duplicate tracks, invalid velocities,
carrier expiry, pass chains, partly missing flights, a pass near an opponent,
an interception and carry without passes, phase-only evidence, memory expiry,
uncertain-pass rejection, and stoppage/half resets. Repeated synthetic pipeline
calls produce identical labels and leave input positions unchanged.

Full-match run over 158,857 graded frames:

| Output | Frames |
|---|---:|
| H | 54,537 |
| A | 36,289 |
| D | 4,009 |
| X | 64,022 |

Non-X coverage is **59.70%**. The detector reports **1,038 control episodes**,
**317 home and 120 away completed-pass candidates**, four uncertain candidates,
and 21 bridged frames. These pass counts are algorithmic candidates, not verified
match statistics. The full-length CSV contains X on all 40,462 ungraded frames.

The earlier robust-stillness model covered 75.88% of graded frames. Coverage
dropped here because physical relative-motion/contact confirmation is stricter
and phase tags alone no longer assign ownership. This is a tradeoff, not evidence
of better accuracy. Synthetic successes demonstrate intended behaviour only.

## Remaining limitations

- Contacts shorter than the default 0.16-second confirmation remain uncertain.
  Full one-touch pass detection is **not solved**: promoting every trajectory
  change would confuse intentional touches with deflections. Longer observed
  receptions and controlled carries are supported.
- Passes shorter than four metres and unobserved flights may not count. Tracks
  can still swap identity plausibly; projected sender separation is a heuristic.
- Relative motion can be noisy even after validity checks. Contact confidence
  currently distinguishes observed versus held evidence, rather than providing
  a statistically calibrated probability.
- Restart-context/location stillness can miss midfield free kicks or mistake
  a live pause for a stoppage. Missing ball alone remains neutral.
- Memory can miss unseen turnovers, and accumulation delays some true changes.
- Confirmation and stillness windows are engineering defaults, not demonstrated
  optima for this match. Better competition accuracy remains unverified.

Review this variant before choosing it for submission. No official submission,
publication, or external upload is performed by these scripts.
