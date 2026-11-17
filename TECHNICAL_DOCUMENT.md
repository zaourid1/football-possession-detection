# Layered Temporal Inference for Football Possession

## Task and output

We infer one state for every frame of the tactical recording: home possession
(`H`), away possession (`A`), dead ball (`D`), or abstention (`X`). The output
contains 199,319 ordered rows indexed from frame 0 through frame 199,318.

Our design treats possession as a temporal state. A nearby opponent does not
automatically create a turnover: a deflection, challenge, aerial duel, or ball
flight can leave possession with the previous team. A change requires evidence
of controlled possession. Dead ball is inferred separately from team ownership.

## Assets used

The final system uses the following supplied tactical-camera assets:

- Automatic Tracking Data (ATD): normalized ball and player positions and each
  track slot's team membership. Positions are converted to metres using the
  published 105.3 by 68.0 metre pitch dimensions.
- Smart Tagging: attacking/defending phases provide weak team context, and set
  pieces provide restart context. These automatic tags never independently
  establish team ownership or make an entire interval dead ball.
- Published frame alignment, half boundaries, pitch dimensions, and frame rate.

We did not use the broadcast video, broadcast tracking, homography, an external
event feed, or human-produced statistics about the match. The provided tactical
video was useful for understanding the asset definitions, but the final pipeline
does not decode video frames.

## Method

### Data validation and geometric features

The loader streams all 199,319 tactical tracking records and verifies consecutive
one-based ATD frame IDs before converting them to zero-based video frames. It
rejects malformed track counts and preserves missing coordinates as `NaN`.

For every visible ball observation, the feature layer computes the nearest home
and away distances in metres, the distance advantage, and ball velocity only
across consecutive valid observations in the same half. Missing observations and
implausible motion are not converted into zero distance or stationary motion.

### Stoppages

Dead-ball inference combines out-of-bounds observations with robust stillness.
Stillness is measured from the range of ball positions in a centred temporal
window, which tolerates tracker jitter better than instantaneous speed. It
requires sufficient original observations, a plausible restart location, and
automatic set-piece context. Evidence must persist before entering `D`; renewed
credible ball movement is required to exit it. Missing ball tracking alone is
neutral and never proves a stoppage.

### Controlled-contact episodes

Control uses hysteresis: entering control requires a smaller player-ball radius
and a clearer advantage over the opponent than retaining established control.
Player-ball relative motion must also be compatible with carrying or receiving
the ball. To locate the start of a touch more accurately, the model compares
valid ball velocities immediately before and after a nearby-player observation.
A sufficiently large speed or direction change can seed a contact candidate,
but it cannot count toward the confirmation duration. The same local player
must then provide the full normal multi-frame, low-relative-speed follow-through.
Overlapping impulse windows are grouped into one event so tracking jitter cannot
manufacture a confirmation run.

Grouped touches are classified as controlled, deflection-like, or unresolved.
Controlled requires a confirmed contact episode. Deflection-like requires an
uncontrolled opponent redirection, rapid separation, and confirmed recovery by
the previously controlling team without an intervening stoppage. This is an
operational possession classification, not a claim about player intent; all
other redirections remain unresolved.
Velocity windows are invalidated by missing coordinates, implausible jumps, or
half boundaries.

Track slots are short-lived hypotheses, not player identities. A locally
plausible same-team slot handoff can retain one control episode. Carrier memory
is short and is recorded separately from team-possession memory. Very brief
one-touch interactions below the confirmation window remain uncertain rather
than being promoted from a trajectory change that could be a deflection.

### Pass evidence

A completed-pass candidate requires a controlled departure, an observed ball
flight with sufficient travel and speed, and a separate same-team controlled
reception. Continuous dribbles, duplicate track slots, opponent control,
stoppages, and half changes reject a candidate. Partly occluded flights may be
recorded as uncertain but cannot increment a completed-pass chain.

Pass counts support confidence; they do not define possession. An interception
followed by a controlled carry can establish possession without any passes. A
same-team pass continues the existing possession. Completed-pass confidence can
extend bounded team memory, while uncertain passes cannot.

### Layered temporal decision

The final model layers physical evidence onto a broader temporal reference
model. The reference uses nearest-team geometry, phase context, possession
memory, and robust stoppages. The stricter physical model can correct it only
after an observed confirmed contact or completed-pass reception. The correction
continues through bounded physical memory, avoiding per-frame switching when a
carrier temporarily disappears.

When the stricter model abstains, it does not erase an existing reference answer.
Corrections reset at stoppages and half boundaries. Reference `D` predictions
are retained conservatively. Short missing-ball gaps may be bridged only when
matching control surrounds the gap and there is no stoppage, restart, half
boundary, conflicting label, or confirmed opposing control.

The system is offline: centred stillness windows, backdated confirmation over
an observed run, and bounded gap bridging may use later observations.

## Possession rules implemented

- A team remains in possession during a credible pass flight, loose ball, or
  ambiguous challenge until opposing controlled possession is confirmed.
- A simple touch, transient proximity, or deflection does not cause a turnover.
- Controlled carries can establish possession without completed passes.
- A successful same-team pass does not split a possession.
- `D` is separate from restart ownership; the system does not assign a team
  while its dead-ball state is active.
- Frames outside the two graded halves are always `X`.
- Unresolved graded intervals may remain `X` rather than receive an invented
  physical explanation.

## Training and labelled data

The final method is deterministic and rule-based. It uses no trained possession
model and no labelled training data. No labels, corrected tracking, event feeds,
or human-produced match statistics for the provided match were used to train,
tune, select, or validate parameters. The public five-minute ground truth was
not used for model or threshold selection.

The implementation was informed by the competition documentation and by general
methods described in Guo et al., *Automatic Estimation of Football Possession via
Improved YOLOv8 Detection and DBSCAN-Based Team Classification* (Sensors, 2026,
26(4), 1252, DOI: 10.3390/s26041252). We adapted its temporal hysteresis idea to
the supplied pitch-coordinate tracking; we did not use its trained detectors or
copy its pixel-normalized thresholds.

## Verification

The repository contains 66 synthetic tests covering alignment, missing values,
distance ties, half boundaries, stillness, control hysteresis, track-slot changes,
ball-velocity impulses, repeated impulses, isolated deflections, interrupted
deflections, unresolved touches, dribbles, completed and uncertain passes,
possession memory, stoppage resets, gap bridging, layering, and deterministic
output. These tests specify intended behaviour without using labels from the
match.

The exact final CSV passes the supplied validator with 199,319 rows, ordered
frames 0–199,318, and only `H`, `A`, `D`, or `X`. Its SHA-256 digest is:

`0ed4dbf6b17280e3ce86fbe7a7145bc209ac7b06a9459702cc132027596a8902`

Aggregate diagnostics are used only as sanity checks. They do not establish
frame accuracy or possession-spell accuracy.

## Limitations

- Automatic tracking is missing or noisy, particularly for the ball. Plausible
  identity switches and duplicates can still confuse local carrier inference.
- Brief one-touch play is intentionally under-detected when evidence cannot
  distinguish control from a deflection.
- Fully or partly unseen passes may not be counted. Pass candidates are inferred
  events, not verified match statistics.
- Stoppage inference can miss free kicks away from recognized restart locations
  or mistake a live stationary ball for a stoppage.
- Team memory can carry ownership through an unseen turnover; strict confirmation
  can also delay a real handover.
- Preserving reference predictions maintains coverage but can retain reference
  errors, including false dead-ball intervals.
- The method has no permitted full-match labelled validation set. Its competition
  accuracy is therefore unknown until private evaluation.

## Reproduction

Use Python 3.11 and install the exact submission dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-submission.txt
```

Place the licensed match bundle under `data/` in the documented tactical and
broadcast folders, or set `MTAH_DATA_DIR`. Then run:

```bash
python3 -m unittest discover -s starter -p 'test_possession_*.py'
python3 starter/run_possession.py --model layered \
  --out evaluation/results/layered_pipeline.csv \
  --report evaluation/reports/layered_pipeline_summary.json
python3 evaluation/validate_submission.py evaluation/results/layered_pipeline.csv
```

The CSV is generated deterministically. The data bundle, generated predictions,
and diagnostic reports are excluded from the public repository by `.gitignore`.
