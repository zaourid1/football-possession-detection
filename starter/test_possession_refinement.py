"""Invented scenarios establish behaviour, not real-match accuracy."""

import unittest

import numpy as np

import test_possession_model as fixtures
from possession_model import ModelConfig, predict_possession
from possession_refinement import bridge_missing_gaps, predict_refined


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ModelTests()

    def test_intermittent_control_can_accumulate(self):
        owners = [0, 0, -1] * 6
        data, features = self.fixture.scenario(owners)
        old, _ = predict_possession(data, features, [], (105.3, 68))
        new, _ = predict_refined(data, features, [], (105.3, 68))
        # Invented truth is home throughout, despite intermittent observations.
        self.assertGreater(np.count_nonzero(new == "H"), np.count_nonzero(old == "H"))

    def test_single_opponent_touch_does_not_switch(self):
        data, features = self.fixture.scenario([0] * 12 + [1] + [0] * 5)
        labels, _ = predict_refined(data, features, [], (105.3, 68))
        self.assertTrue((labels[10:] == "H").all())

    def test_turnover_eventually_switches_but_has_confirmation_latency(self):
        data, features = self.fixture.scenario([0] * 12 + [1] * 12)
        old, _ = predict_possession(data, features, [], (105.3, 68))
        new, _ = predict_refined(data, features, [], (105.3, 68))
        self.assertEqual(new[-1], "A")
        # Honest tradeoff: retained home votes delay a clean abrupt turnover.
        self.assertGreater(np.flatnonzero(new == "A")[0], np.flatnonzero(old == "A")[0])

    def test_phase_only_cannot_reach_control_threshold(self):
        data, features = self.fixture.scenario([-1] * 30)
        events = [{"code": "BUILD UP", "start_frame": 0, "end_frame": 30,
                   "tags": {"Team": "Barcelona"}}]
        labels, _ = predict_refined(data, features, events, (105.3, 68))
        self.assertTrue((labels == "X").all())

    def gap(self, right_owner=0, gap_length=10):
        data, features = self.fixture.scenario([0] + [-1] * gap_length + [right_owner])
        labels = np.array(["H"] + ["X"] * gap_length + ["H" if right_owner == 0 else "A"])
        reasons = np.array(["confirmed_control"] + ["insufficient_evidence"] * gap_length
                           + ["confirmed_control"], dtype="<U32")
        return data, features, labels, reasons

    def test_bridge_agreeing_control_anchors(self):
        data, features, labels, reasons = self.gap()
        result, why = bridge_missing_gaps(labels, reasons, data, features, [])
        self.assertTrue((result == "H").all())
        self.assertTrue((why[1:-1] == "bridged_same_team").all())
        self.assertTrue((labels[1:-1] == "X").all())  # input unchanged

    def test_bridge_rejects_turnover_long_gap_and_half_boundary(self):
        for scenario in (self.gap(right_owner=1), self.gap(gap_length=61)):
            data, features, labels, reasons = scenario
            result, _ = bridge_missing_gaps(labels, reasons, data, features, [])
            np.testing.assert_array_equal(result, labels)
        data, features, labels, reasons = self.gap()
        data.half[-1] = 2
        result, _ = bridge_missing_gaps(labels, reasons, data, features, [])
        np.testing.assert_array_equal(result, labels)

    def test_bridge_rejects_restart_conflicting_tag_and_dead_ball(self):
        for code, team in (("SET PIECES", "N/A"), ("BUILD UP", "Atletic Club")):
            data, features, labels, reasons = self.gap()
            events = [{"code": code, "start_frame": 2, "end_frame": 5,
                       "tags": {"Team": team}}]
            result, _ = bridge_missing_gaps(labels, reasons, data, features, events)
            np.testing.assert_array_equal(result, labels)
        data, features, labels, reasons = self.gap()
        labels[4] = "D"
        result, _ = bridge_missing_gaps(labels, reasons, data, features, [])
        np.testing.assert_array_equal(result, labels)


if __name__ == "__main__":
    unittest.main()
