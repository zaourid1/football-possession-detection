"""Stillness checks on invented observations; no match labels are used."""

import unittest

import numpy as np

from possession_data import TrackingData
from possession_features import extract_features
from possession_model import predict_possession
from possession_stoppages import detect_stillness, predict_with_stillness


class StillnessTests(unittest.TestCase):
    def scene(self, n=100, location=(1., 20.), jitter=True):
        ball = np.tile(location, (n, 1))
        if jitter:
            ball[:, 0] += np.where(np.arange(n) % 2, 0.2, -0.2)
        players = np.zeros((n, 2, 2))
        players[:, 0] = ball + [1, 0]
        players[:, 1] = ball + [6, 0]
        data = TrackingData(ball, players, np.array([0, 1]), np.ones(n, dtype=int), 25)
        event = {"code": "SET PIECES", "start_frame": 0, "end_frame": n,
                 "tags": {"Team": "N/A"}}
        return data, [event]

    def test_jittery_stationary_restart_is_found(self):
        data, events = self.scene()
        features = extract_features(data)
        old, _ = predict_possession(data, features, events, (105.3, 68))
        new, _, evidence = predict_with_stillness(data, features, events, (105.3, 68))
        self.assertNotIn("D", old)
        self.assertTrue(evidence.stationary_restart[50])
        # Known invented stoppage: displacement evidence improves detection.
        self.assertGreater(np.count_nonzero(new == "D"), 0)

    def test_stationary_ball_without_restart_context_is_not_dead(self):
        data, _ = self.scene()
        features = extract_features(data)
        labels, _, evidence = predict_with_stillness(data, features, [], (105.3, 68))
        self.assertTrue(evidence.stationary[50])
        self.assertNotIn("D", labels)

    def test_moving_ball_near_line_is_not_stationary(self):
        data, events = self.scene(jitter=False)
        data.ball_xy[:, 1] = np.arange(len(data.half)) * 0.2
        evidence = detect_stillness(data, events, (105.3, 68))
        self.assertFalse(evidence.stationary.any())

    def test_sparse_observations_cannot_become_dense_after_smoothing(self):
        data, events = self.scene()
        data.ball_xy[np.arange(len(data.half)) % 5 != 0] = np.nan
        evidence = detect_stillness(data, events, (105.3, 68))
        self.assertFalse(evidence.stationary.any())
        self.assertFalse(evidence.stationary_restart.any())

    def test_short_half_cannot_borrow_observations_from_other_half(self):
        data, events = self.scene(n=40)
        data.half[:20] = 1
        data.half[20:] = 2
        evidence = detect_stillness(data, events, (105.3, 68))
        self.assertFalse(evidence.stationary.any())

    def test_midfield_stationary_ball_needs_restart_location(self):
        data, events = self.scene(location=(35., 25.))
        evidence = detect_stillness(data, events, (105.3, 68))
        self.assertTrue(evidence.stationary[50])
        self.assertFalse(evidence.stationary_restart.any())


if __name__ == "__main__":
    unittest.main()
