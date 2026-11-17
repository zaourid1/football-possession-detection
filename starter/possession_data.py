"""Section 1: load tactical observations, without assigning possession labels.

Run from the repository root with: python3 starter/possession_data.py
Only aggregate diagnostics are printed; tracking stays local.
"""

from dataclasses import dataclass

import numpy as np

from baseline import team_of_each_track
from load_atd import atd_paths
from games import GAME, GAMES, half_video_blocks, source_frames


@dataclass
class TrackingData:
    """Arrays indexed by zero-based tactical video frame.

    Positions are metres, missing observations remain NaN, and track teams
    use the baseline convention: home=0, away=1. Track slots are not identities.
    half is 0 outside graded intervals, otherwise 1 or 2.
    """

    ball_xy: np.ndarray
    player_xy: np.ndarray
    track_team: np.ndarray
    half: np.ndarray
    fps: float

    @property
    def ball_available(self):
        return np.isfinite(self.ball_xy).all(axis=1)


def load_tracking(game=GAME, data_dir=None):
    """Read every record once and reject malformed or misaligned inputs."""
    meta_path, raw_path = atd_paths(game=game, data_dir=data_dir)
    teams = team_of_each_track(meta_path)
    n_frames = source_frames("tactical", game)
    pitch = np.asarray(GAMES[game]["pitch"], dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    player_xy = np.full((n_frames, len(teams), 2), np.nan, dtype=np.float32)

    count = 0
    with open(raw_path, encoding="utf-8") as handle:
        for frame, line in enumerate(handle):
            parts = line.strip().split(":")
            if len(parts) != 3 or frame >= n_frames:
                raise ValueError("Unexpected record structure at line {}".format(frame + 1))
            if int(parts[0]) != frame + 1:
                raise ValueError("Nonconsecutive tracking frame at line {}".format(frame + 1))

            players = parts[1].split(";")
            if len(players) != len(teams):
                raise ValueError("Unexpected track count at frame {}".format(frame))
            # Parse all three fields, but retain only positions for this section.
            values = np.asarray([group.split(",") for group in players], dtype=np.float32)
            ball = np.asarray(parts[2].split(","), dtype=np.float32)
            if values.shape != (len(teams), 3) or ball.shape != (3,):
                raise ValueError("Unexpected coordinate fields at frame {}".format(frame))
            player_xy[frame] = values[:, :2] * pitch
            ball_xy[frame] = ball[:2] * pitch
            count += 1

    if count != n_frames:
        raise ValueError("Expected {} records, found {}".format(n_frames, count))

    half = np.zeros(n_frames, dtype=np.int8)
    for number, start, end in half_video_blocks(game):
        half[start:end + 1] = number

    return TrackingData(ball_xy, player_xy, teams, half, GAMES[game]["fps"])


if __name__ == "__main__":
    data = load_tracking()
    available = data.ball_available
    print("Loaded {} tactical frames; {} track slots".format(len(data.half), len(data.track_team)))
    for number in (1, 2):
        mask = data.half == number
        visible = int(np.count_nonzero(available & mask))
        total = int(np.count_nonzero(mask))
        print("Half {}: {} graded frames; ball available {}, missing {}".format(
            number, total, visible, total - visible))
