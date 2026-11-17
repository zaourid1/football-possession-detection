"""State-machine behaviour on invented scenarios, without match labels."""

import unittest

import numpy as np

from possession_data import TrackingData
from possession_features import PossessionFeatures
from possession_model import ModelConfig, predict_possession, tag_evidence


class ModelTests(unittest.TestCase):
    def scenario(self, owners):
        n = len(owners)
        data = TrackingData(np.tile([50., 30.], (n, 1)),
                            np.zeros((n, 2, 2)), np.array([0, 1]), np.ones(n), 10)
        distances = np.full((n, 2), np.nan)
        for frame, owner in enumerate(owners):
            if owner >= 0:
                distances[frame] = [1, 5] if owner == 0 else [5, 1]
            else:
                data.ball_xy[frame] = np.nan
        features = PossessionFeatures(distances, np.asarray(owners),
                                      distances[:, 1] - distances[:, 0],
                                      np.where(np.asarray(owners) >= 0, 2., np.nan),
                                      np.zeros(n, dtype=int))
        return data, features

    def predict(self, data, features, events=(), config=None):
        return predict_possession(data, features, events, (105.3, 68), config)[0]

    def test_brief_opponent_touch_and_missing_gap_preserve_possession(self):
        data, features = self.scenario([0] * 5 + [1] + [-1] * 3 + [0] * 3)
        labels = self.predict(data, features)
        self.assertTrue((labels[3:] == "H").all())
        self.assertNotIn("D", labels)

    def test_sustained_opponent_control_changes_team(self):
        data, features = self.scenario([0] * 5 + [1] * 5)
        labels = self.predict(data, features)
        self.assertEqual(labels[5], "H")
        self.assertEqual(labels[-1], "A")

    def test_memory_expires_and_resets_between_halves(self):
        data, features = self.scenario([0] * 5 + [-1] * 6)
        labels = self.predict(data, features, config=ModelConfig(memory_s=0.3))
        self.assertEqual(labels[-1], "X")
        data.half[5:] = 2
        self.assertEqual(self.predict(data, features)[5], "X")

    def test_tag_interval_is_not_automatically_dead_ball(self):
        data, features = self.scenario([0] * 30)
        event = {"code": "SET PIECES", "start_frame": 0, "end_frame": 30,
                 "tags": {"Team": "Barcelona"}}
        self.assertNotIn("D", self.predict(data, features, [event]))
        features.ball_speed_mps[:25] = 0
        labels = self.predict(data, features, [event])
        self.assertEqual(labels[24], "D")
        self.assertEqual(labels[-1], "H")

    def test_defensive_tags_invert_and_conflicts_cancel(self):
        events = [
            {"code": "DEFENDING IN MIDDLE THIRD", "start_frame": 0, "end_frame": 4,
             "tags": {"Team": "Barcelona"}},
            {"code": "BUILD UP", "start_frame": 2, "end_frame": 4,
             "tags": {"Team": "Barcelona"}},
        ]
        teams, _ = tag_evidence(events, 4)
        np.testing.assert_array_equal(teams, [1, 1, -1, -1])


if __name__ == "__main__":
    unittest.main()
