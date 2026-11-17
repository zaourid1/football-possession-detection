"""Retain earlier predictions and apply temporally consistent physical corrections.

No match-level possession targets or external match statistics are inputs.
Coverage preservation is an output invariant, not an accuracy claim.
"""

from dataclasses import dataclass

import numpy as np

from possession_features import extract_features
from possession_model import predict_possession
from possession_pipeline import PipelineConfig, PipelineResult, predict_pipeline


@dataclass
class LayeredResult(PipelineResult):
    reference_labels: np.ndarray
    physical_labels: np.ndarray
    source: np.ndarray


def combine_layers(data, reference_labels, reference_reasons, physical):
    """Apply corrections from confirmed observations, retaining bounded memory.

    A stricter abstention never erases a reference answer. An override begins
    at a measured confirmed contact or completed-pass reception, never at a
    held hypothesis or an arbitrary earlier point in the same label run. It
    continues only while the strict model retains that team. Reference D is
    preserved conservatively; this can also retain an incorrect old stoppage.
    """
    n = len(data.half)
    reference_labels = np.asarray(reference_labels)
    reference_reasons = np.asarray(reference_reasons)
    if (reference_labels.shape != (n,) or reference_reasons.shape != (n,)
            or physical.labels.shape != (n,) or physical.reasons.shape != (n,)):
        raise ValueError("Each prediction/reason array must have one value per frame")
    if (not np.isin(reference_labels, ["H", "A", "D", "X"]).all()
            or not np.isin(physical.labels, ["H", "A", "D", "X"]).all()):
        raise ValueError("Unknown possession label")
    labels = reference_labels.copy()
    reasons = reference_reasons.astype("<U40", copy=True)
    source = np.full(n, "reference", dtype="<U24")
    pass_anchors = {}
    for event in physical.passes.events:
        if event.completed and event.confidence > 0:
            pass_anchors[event.confirmation_frame] = "H" if event.team == 0 else "A"

    owner = "X"
    previous_half = 0
    for frame, half in enumerate(data.half):
        if half != previous_half:
            owner = "X"
        previous_half = half
        if half == 0:
            labels[frame], reasons[frame], source[frame] = "X", "ungraded", "ungraded"
            continue
        strict = physical.labels[frame]
        if reference_labels[frame] == "D":
            owner = "X"
            source[frame] = "reference_stoppage"
            continue
        if strict == "D":
            owner = "X"
            if physical.reasons[frame] == "stoppage_evidence":
                labels[frame], reasons[frame] = "D", "physical_stoppage"
                source[frame] = "physical_stoppage"
            continue
        if strict not in ("H", "A") or strict != owner:
            owner = "X"
        measured = (
            strict in ("H", "A")
            and physical.reasons[frame] == "confirmed_control"
            and physical.contacts.observed[frame]
            and physical.contacts.confidence[frame] > 0
            and physical.contacts.team[frame] == (0 if strict == "H" else 1)
        )
        completed_pass = strict in ("H", "A") and pass_anchors.get(frame) == strict
        if measured or completed_pass:
            owner = strict
        if owner in ("H", "A"):
            labels[frame] = owner
            bridged = physical.reasons[frame] == "bridged_same_team"
            source[frame] = ("physical_contact" if measured else
                             "completed_pass" if completed_pass else
                             "physical_bridge" if bridged else "physical_memory")
            reasons[frame] = ("layered_control" if measured else
                              "layered_pass" if completed_pass else
                              "layered_bridged" if bridged else "layered_memory")

    graded = data.half != 0
    if np.any(graded & (reference_labels != "X") & (labels == "X")):
        raise AssertionError("A physical abstention erased a reference answer")
    return labels, reasons, source


def predict_layered(data, events, pitch, config=None, features=None):
    """Use the robust-stillness model as a base, then contact/pass corrections."""
    config = config or PipelineConfig()
    events = list(events)
    features = features if features is not None else extract_features(data)
    physical = predict_pipeline(data, events, pitch, config, features)
    # Reuse exactly the same robust stillness evidence instead of recalculating
    # it or treating a new module's uncertainty as a reason to discard the base.
    reference, reference_reasons = predict_possession(
        data, features, events, pitch, config.stoppage,
        stationary_evidence=physical.stoppages.stationary_restart)
    labels, reasons, source = combine_layers(data, reference, reference_reasons, physical)
    return LayeredResult(labels, reasons, physical.contacts, physical.passes,
                         physical.stoppages, reference, physical.labels, source)
