"""Synthetic checks; no match labels or restricted assets are used."""

import unittest

import numpy as np

from possession_data import TrackingData
from possession_features import extract_features


class FeatureTests(unittest.TestCase):
    def make_data(self, ball, players, half):
        return TrackingData(
            np.asarray(ball, dtype=np.float32),
            np.asarray(players, dtype=np.float32),
            np.array([0, 1]), np.asarray(half), 25,
        )

    def test_distances_ties_and_missing_players(self):
        data = self.make_data(
            [[0, 0]] * 4,
            [[[3, 4], [6, 8]], [[1, 0], [-1, 0]],
             [[np.nan, 0], [2, 0]], [[np.nan, 0], [0, np.nan]]],
            [1] * 4,
        )
        features = extract_features(data)
        np.testing.assert_allclose(features.team_distance_m[0], [5, 10])
        np.testing.assert_array_equal(features.nearest_team, [0, -1, 1, -1])
        self.assertEqual(features.distance_margin_m[0], 5)
        self.assertTrue(np.isnan(features.distance_margin_m[2]))

    def test_speed_does_not_bridge_missing_frames_or_halves(self):
        data = self.make_data(
            [[0, 0], [3, 4], [np.nan, 0], [6, 8], [9, 12]],
            [[[0, 0], [1, 1]]] * 5, [1, 1, 1, 1, 2],
        )
        features = extract_features(data)
        self.assertEqual(features.ball_speed_mps[1], 125)
        self.assertTrue(np.isnan(features.ball_speed_mps[[0, 2, 3, 4]]).all())
        self.assertEqual(features.nearest_team[2], -1)

    def test_missing_runs_reset_and_include_trailing_gap(self):
        data = self.make_data(
            [[np.nan, np.nan]] * 6,
            [[[0, 0], [1, 1]]] * 6, [1, 1, 0, 2, 2, 2],
        )
        np.testing.assert_array_equal(
            extract_features(data).missing_run_frames, [1, 2, 0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
