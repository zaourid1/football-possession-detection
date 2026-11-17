"""Displacement-based stillness for noisy ball tracking.

Unlike instantaneous speed, a position range tolerates tracking jitter. Unlike
the reference implementation, observation coverage is measured BEFORE smoothing,
windows are confined to each half, and evidence is never dilated into gaps.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from possession_model import predict_possession, tag_evidence


@dataclass(frozen=True)
class StillnessConfig:
    window_s: float = 1.5
    median_s: float = 0.2
    max_axis_range_m: float = 1.2
    min_observed_fraction: float = 0.6
    line_margin_m: float = 4.0
    centre_radius_m: float = 2.5


@dataclass
class StoppageEvidence:
    stationary: np.ndarray
    observed_fraction: np.ndarray
    restart_location: np.ndarray
    stationary_restart: np.ndarray


def detect_stillness(data, events, pitch, config=None):
    """Return offline stillness evidence, not dead-ball labels.

    Require a full window, sufficient actual observations, a currently visible
    ball, and bounded x/y range. Dead-ball support additionally needs set-piece
    context AND a plausible restart location. Free kicks away from these spots
    can be missed; a live stationary ball can still produce a false positive.
    """
    config = config or StillnessConfig()
    if (config.window_s <= 0 or config.median_s <= 0
            or config.max_axis_range_m <= 0
            or not 0 < config.min_observed_fraction <= 1
            or config.line_margin_m < 0 or config.centre_radius_m < 0):
        raise ValueError("Invalid stillness configuration")
    n = len(data.half)
    available = data.ball_available
    stationary = np.zeros(n, dtype=bool)
    observed_fraction = np.zeros(n, dtype=float)
    window = max(1, round(config.window_s * data.fps))
    median_window = max(1, round(config.median_s * data.fps))
    for half in (1, 2):
        indices = np.flatnonzero(data.half == half)
        if not len(indices):
            continue
        # Defensive validation: no rolling window may jump a discontinuity.
        if np.any(np.diff(indices) != 1):
            raise ValueError("Each half must form one contiguous block")
        seen = available[indices]
        observed = pd.Series(seen.astype(float))
        fraction = observed.rolling(window, center=True, min_periods=window).mean().to_numpy()
        observed_fraction[indices] = np.nan_to_num(fraction, nan=0.0)
        coords = data.ball_xy[indices].copy()
        coords[~seen] = np.nan
        smoothed = pd.DataFrame(coords).rolling(
            median_window, center=True, min_periods=1).median()
        # Smoothing cannot turn missing observations into actual evidence.
        smoothed.loc[~seen, :] = np.nan
        rolling = smoothed.rolling(window, center=True, min_periods=1)
        ranges = (rolling.max() - rolling.min()).to_numpy()
        stationary[indices] = (
            seen & (fraction >= config.min_observed_fraction)
            & np.isfinite(ranges).all(axis=1)
            & (ranges <= config.max_axis_range_m).all(axis=1)
        )

    x, y = data.ball_xy[:, 0], data.ball_xy[:, 1]
    length, width = pitch
    margin = config.line_margin_m
    near_line = ((x <= margin) | (x >= length - margin)
                 | (y <= margin) | (y >= width - margin))
    centre = np.hypot(x - length / 2, y - width / 2) <= config.centre_radius_m
    goal_area = ((x <= 6.5) | (x >= length - 6.5)) & (np.abs(y - width / 2) <= 11)
    penalty = ((np.hypot(x - 11, y - width / 2) <= 2)
               | (np.hypot(x - (length - 11), y - width / 2) <= 2))
    location = available & (data.half != 0) & (near_line | centre | goal_area | penalty)
    _, restart = tag_evidence(events, n)
    return StoppageEvidence(stationary, observed_fraction, location,
                            stationary & location & restart)


def predict_with_stillness(data, features, events, pitch, model_config=None,
                           stillness_config=None, *, accumulated=False):
    """Opt-in variant; existing model defaults remain unchanged."""
    evidence = detect_stillness(data, events, pitch, stillness_config)
    labels, reasons = predict_possession(
        data, features, events, pitch, model_config, accumulated=accumulated,
        stationary_evidence=evidence.stationary_restart)
    return labels, reasons, evidence
