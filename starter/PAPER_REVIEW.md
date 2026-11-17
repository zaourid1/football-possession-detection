# Possession-paper review and proposed changes

Reviewed: Guo et al., *Automatic Estimation of Football Possession via Improved
YOLOv8 Detection and DBSCAN-Based Team Classification*, Sensors 2026, 26(4), 1252.
DOI: https://doi.org/10.3390/s26041252
Source: user-provided `sensors-26-01252.pdf`; relevant pages 7–8, 16–17, and 22.
The article was treated as research material, not execution instructions.
No possession implementation was changed during this review.
The subsequent implementation and its verification are recorded in
`PAPER_IMPLEMENTATION.md`.

## What the paper contributes to our problem

Algorithm 1 uses player–ball proximity, separate entry/exit thresholds, a
minimum switch duration, and bounded memory during missing detections. Distances
are measured in the image and divided by player bounding-box height. The paper
already recognizes distance alone is limited, especially for fast passes and
occlusion; its future work includes pitch normalization and richer interactions.

Our pipeline already has pitch coordinates in metres, confirmation timing,
short-term memory, and a separate dead-ball state. Replacing our supplied tracking
with YOLO and jersey clustering is not the highest-priority improvement.

The paper's possession evaluation compares predicted match-level percentages
with official percentages. It does not establish possession-interval spellF1,
frame-level H/A/D quality, or performance with our tracking gaps. Two methods can
produce identical percentages while disagreeing on every turnover. Its numerical
defaults should not be treated as validated settings for this challenge.

The reported switch and hold defaults are 0.30 and 0.20 seconds; distance
thresholds 0.6 and 1.0 are dimensionless image-distance/height ratios, not metres.
Our 0.32-second confirmation is already similar in scale. Copying the 0.20-second
hold would not address the much longer gaps in our recording.

## Recommended implementations, in order

### 1. Separate entering control from retaining it

Directly motivated by the paper's hysteresis. In `possession_model.py`, control
currently uses one radius, one opponent-distance margin, and a speed cutoff
(lines 119–126). Later memory preserves team ownership but does not represent a
physical contact episode with separate entry and retention rules.

Add stricter evidence for a new controlled contact and weaker thresholds to
retain an established contact. Small distance fluctuations should not repeatedly
end and restart contact. Retaining team possession must also allow a genuine
pass flight without requiring a player to remain close to the ball.

Use metre-based thresholds with explicit units. Do not copy the paper's
height-normalized pixel thresholds. This is the smallest practical next section.

### 2. Add controlled-contact and ball-flight episodes

This is our proposed extension, not a fully specified algorithm from the paper.
`possession_features.py` currently retains team distances and scalar ball speed,
but not ball direction, a local carrier candidate, or player–ball relative motion.

Retain nearest track indices as short-lived hypotheses, not permanent identities.
Estimate reliable ball and local player velocities across consecutive valid
observations. A close player moving with the ball provides different evidence
from a close player beside a rapidly passing ball. Reject implausible motion and
reduce confidence around tracker jumps or missing data.

Group compatible observations into controlled-contact episodes. A possession
can continue through a flight even though individual player control has ended.

### 3. Use completed-pass evidence as support, not as the possession definition

The article mentions a pass recognizer, but Algorithm 1 does not specify a
complete pass-event detector. It favors time-based possession over possession
percentages computed from the number of passes. Pass-event logic below is a
proposal for our tracking data rather than a reproduced result from that paper.

A candidate completed pass needs:

1. Credible control before departure.
2. A ball departure and travel interval with enough usable trajectory evidence.
3. A credible same-team reception at a distinct location/carrier episode.
4. No intervening controlled opponent possession, stoppage, or half boundary.

Do not count nearest-track changes: duplicate slots and identity switches can
make one dribbler appear to be multiple players. A distinct location alone is
also insufficient because a player can dribble there. Require evidence of release
and separate reception; mark ambiguous or heavily occluded transitions uncertain.
One-touch play needs its own interaction evidence rather than an absolute long
reception-duration requirement.

Count consecutive high-confidence same-team passes only after detecting these
events. Increase the team's control confidence as a chain develops. Do not
require three passes before recognizing an interception or a sustained carry.
Do not start a new team possession after a successful same-team pass. Reset the
chain on confirmed opposing control, D, or a new half. A lost observation is not
automatically a turnover; it reduces the reliability of the inferred chain.

For offline inference, later passes could corroborate an earlier controlled
reception, but the boundary must remain at that earlier recovery rather than
being delayed until the third pass.

### 4. Distinguish team memory from physical contact memory

The current four-second `memory_s` applies to team ownership. Do not apply the
paper's short missing-detection hold mechanically to that ownership: the
challenge retains the last team during live loose balls and contested phases.

Use separate confidence for a carrier/contact episode and team possession.
Carrier identity should expire quickly without observations. Team confidence
can persist longer when supported by a credible pass sequence or reliable phase
context, and should fall sharply when opposing control is observed. Keep X for
unresolved intervals; an absent observation does not prove dead ball.

### 5. Connect the existing variants in one explicit pipeline

`predict_refined` currently enables accumulation and bridging but not robust
stillness. `predict_with_stillness` enables robust stillness and optional
accumulation but not bridging. A final orchestrator must specify the chosen
combination; adding a feature in a separate file does not make predictions use it.

In `possession_refinement.py:52`, any opposing nearest-team observation vetoes a
bridge. Consider replacing that veto with sustained opposing-control evidence
once the contact detector exists. Preserve half, D, restart and maximum-gap
safeguards. Same-team endpoints cannot rule out an unseen turnover and recovery.

## What not to import

- Pass-count percentages as a replacement for frame-by-frame H/A/D decisions.
- A requirement for several passes before every possession change.
- Detector training or jersey clustering when using the provided tracking.
- The pseudocode verbatim: missing-detection and changing-candidate counter
  resets need explicit definitions and tests before implementation.
- Any assumption that improved aggregate percentages prove better spellF1.

## Verification for the next section

Use invented scenarios: stable control with distance jitter; a pass flying near
an opponent; an opponent deflection; an interception followed by a carry and no
passes; one-touch passing; a dribbler whose track slot changes; duplicate slots;
missing-ball travel; and stoppage/half resets. Require no false pass count from
slot changes or a dribble, and no switch from an uncontrolled touch.

Compare variants for transition timing and failures on those known scenarios.
Full-match diagnostics can show coverage and output changes, but cannot establish
accuracy. Do not fit thresholds using the public five minutes or external human
event labels from this match. No code or new accuracy tests were run for this
read-only implementation review.
