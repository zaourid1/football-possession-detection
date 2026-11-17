"""Section 2: geometric evidence, not final possession or dead-ball labels.

Run: python3 starter/possession_features.py
Distances and speeds are based on uncorrected tracking and may contain outliers.
"""

from dataclasses import dataclass

import numpy as np

from possession_data import TrackingData, load_tracking


@dataclass
class PossessionFeatures:
    """One row per tactical frame.

    team_distance_m columns are home (0), away (1); NaN means no observation.
    nearest_team is 0/1, or -1 if unavailable or tied. It is proximity only,
    not evidence that a player actually controlled the ball.
    distance_margin_m is away distance minus home distance: positive favours
    home. It is NaN unless both teams have usable observations.
    ball_speed_mps uses consecutive visible frames in the same graded half.
    missing_run_frames counts consecutive missing-ball frames within a half.
    """

    team_distance_m: np.ndarray
    nearest_team: np.ndarray
    distance_margin_m: np.ndarray
    ball_speed_mps: np.ndarray
    missing_run_frames: np.ndarray


def extract_features(data: TrackingData):
    """Compute evidence without interpolation, smoothing, or label decisions."""
    n_frames = len(data.half)
    ball_available = data.ball_available
    distances = np.full((n_frames, 2), np.nan, dtype=np.float32)
    for team in (0, 1):
        players = data.player_xy[:, data.track_team == team, :]
        if players.shape[1] == 0:
            continue
        valid = np.isfinite(players).all(axis=2) & ball_available[:, None]
        delta = players - data.ball_xy[:, None, :]
        squared = np.sum(delta * delta, axis=2)
        # Infinity excludes missing positions from the minimum without treating
        # them as zero distance or emitting all-NaN reduction warnings.
        minimum = np.sqrt(np.min(np.where(valid, squared, np.inf), axis=1))
        distances[:, team] = np.where(np.isfinite(minimum), minimum, np.nan)

    nearest = np.full(n_frames, -1, dtype=np.int8)
    home, away = distances[:, 0], distances[:, 1]
    home_seen, away_seen = np.isfinite(home), np.isfinite(away)
    nearest[home_seen & (~away_seen | (home < away))] = 0
    nearest[away_seen & (~home_seen | (away < home))] = 1
    margin = away - home

    speed = np.full(n_frames, np.nan, dtype=np.float32)
    adjacent = (
        ball_available[1:] & ball_available[:-1]
        & (data.half[1:] == data.half[:-1]) & (data.half[1:] != 0)
    )
    displacement = np.linalg.norm(data.ball_xy[1:] - data.ball_xy[:-1], axis=1)
    speed[1:] = np.where(adjacent, displacement * data.fps, np.nan)

    missing_run = np.zeros(n_frames, dtype=np.int32)
    run = 0
    previous_half = 0
    for frame, half in enumerate(data.half):
        if half != previous_half:
            run = 0
        if half != 0 and not ball_available[frame]:
            run += 1
        else:
            run = 0
        missing_run[frame] = run
        previous_half = half

    return PossessionFeatures(distances, nearest, margin, speed, missing_run)


if __name__ == "__main__":
    tracking = load_tracking()
    features = extract_features(tracking)
    graded = tracking.half != 0
    print("Graded frames: {}".format(np.count_nonzero(graded)))
    for team, name in ((0, "home"), (1, "away")):
        known = graded & np.isfinite(features.team_distance_m[:, team])
        print("Frames with ball-to-{} distance: {}".format(name, np.count_nonzero(known)))
    print("Frames with measurable ball speed: {}".format(
        np.count_nonzero(graded & np.isfinite(features.ball_speed_mps))))
    longest = int(features.missing_run_frames.max())
    print("Longest graded missing-ball run: {} frames ({:.2f} seconds)".format(
        longest, longest / tracking.fps))
