# Accumulated evidence and gap bridging review

Run the checks from the repository root:

```sh
python3 -m unittest discover -s starter -p 'test_possession_*.py'
python3 starter/compare_possession_models.py
```

## Changes

The original `predict_possession` behaviour remains the default. Passing
`accumulated=True` enables a bounded 0.8-second evidence window. Geometric
control contributes one vote; phase tags contribute 0.15. A decision needs both
enough evidence and an advantage over the opposing team. Missing observations
do not vote. Weak phase evidence alone cannot establish ownership with these
defaults, unlike the original model.

`predict_refined` adds an offline bridging pass. Only X frames within a
missing-ball run up to six seconds can change. Both ends need matching confirmed
geometric ownership, with up to half a second allowed for reacquisition. Restart
context, dead-ball predictions, opposing evidence, and half boundaries block a
bridge. A same-team bridge could still hide an unobserved turnover and recovery;
the conditions reduce that risk without eliminating it.

## Verification

All 15 synthetic tests passed. They cover the original features and model,
intermittent evidence, brief opponent touches, sustained turnovers, memory
expiry, and accepted/rejected bridges. A synthetic intermittent-home scenario
recovers more known home frames than the original consecutive-frame approach.
A clean abrupt turnover scenario demonstrates the opposite tradeoff: accumulated
votes delay recognition relative to the original. Passing these checks verifies
specified behaviour, not real-match accuracy.

Full-match comparison, restricted to 158,857 graded frames:

| Measurement | Original | Accumulated | Accumulated + bridge |
|---|---:|---:|---:|
| Non-X coverage | 75.88% | 68.49% | 68.49% |
| X frames | 38,315 | 50,048 | 50,048 |
| Raw label changes within halves | 677 | 613 | 613 |
| D frames | 3,796 | 3,796 | 3,796 |

The bridge filled **zero** real-match frames under these conservative conditions.
All output shape, label-domain, ungraded-X, and preservation checks passed.
Fewer changes are not proof of better possession segmentation; they include
transitions to/from X and can also result from missed real transitions.

## Accuracy conclusion

**Real-match accuracy improvement is unproven.** No ground truth was read and no
thresholds were selected against the public five minutes. Coverage decreased,
and bridging provided no additional answers on this run. These results do not
justify replacing the original as the final submission model on accuracy grounds.
An accessible, permitted labelled dataset from other matches is needed for
meaningful model selection; the private match score remains unavailable.
