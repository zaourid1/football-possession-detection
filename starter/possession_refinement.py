"""Section 3 refinement: accumulated evidence and bounded offline gap bridging.

The original predict_possession default remains available for comparison.
No match ground truth is read here.
"""

import numpy as np

from possession_model import ModelConfig, predict_possession, tag_evidence


def bridge_missing_gaps(labels, reasons, data, features, events, config=None, *,
                        controlled_team=None):
    """Fill only X inside short missing-ball runs with agreeing control anchors.

    Reject runs crossing halves, restart context, dead-ball labels, conflicting
    team labels, or opposing tactical evidence. Never overwrite H/A/D. Leading
    and trailing gaps remain unresolved. Anchors must be confirmed geometric
    control, not ownership inferred only from tags or carried by memory. When
    controlled_team is supplied, only observed opposing control vetoes a bridge;
    the legacy nearest-team veto remains the default for old callers.
    """
    config = config or ModelConfig()
    result, why = labels.copy(), reasons.copy()
    phase_team, restart = tag_evidence(events, len(labels))
    opposition = features.nearest_team if controlled_team is None else np.asarray(controlled_team)
    if opposition.shape != (len(labels),):
        raise ValueError("controlled_team must have one value per frame")
    missing = ~data.ball_available
    max_gap = max(1, round(config.bridge_max_s * data.fps))
    anchor_span = max(1, round(config.bridge_anchor_s * data.fps))
    start = 0
    while start < len(labels):
        if not missing[start] or data.half[start] == 0:
            start += 1
            continue
        end = start + 1
        half = data.half[start]
        while end < len(labels) and missing[end] and data.half[end] == half:
            end += 1
        if end - start <= max_gap and start > 0 and end < len(labels):
            left = start - 1
            # Reacquisition may need several frames to confirm control.
            right = end
            limit = min(len(labels), end + anchor_span)
            while right < limit and reasons[right] != "confirmed_control":
                right += 1
            if (right < limit and reasons[left] == "confirmed_control"
                    and labels[left] in ("H", "A") and labels[left] == labels[right]):
                owner = 0 if labels[left] == "H" else 1
                span = slice(left, right + 1)
                safe = (
                    np.all(data.half[span] == half)
                    and not np.any(restart[span])
                    and not np.any(labels[span] == "D")
                    and not np.any(labels[span] == ("A" if owner == 0 else "H"))
                    and not np.any(phase_team[span] == 1 - owner)
                    and not np.any(opposition[end:right + 1] == 1 - owner)
                )
                if safe:
                    indices = np.arange(start, end)
                    indices = indices[labels[indices] == "X"]
                    result[indices] = labels[left]
                    why[indices] = "bridged_same_team"
        start = end
    return result, why


def predict_refined(data, features, events, pitch, config=None):
    config = config or ModelConfig()
    labels, reasons = predict_possession(
        data, features, events, pitch, config, accumulated=True)
    return bridge_missing_gaps(labels, reasons, data, features, events, config)
