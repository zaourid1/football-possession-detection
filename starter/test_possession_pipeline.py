"""Integrated control/pass/team tests use invented observations only."""

from types import SimpleNamespace
import unittest

import numpy as np

from possession_data import TrackingData
from possession_features import extract_features
from possession_pipeline import TeamConfig, decode_team_possession, predict_pipeline
from possession_refinement import bridge_missing_gaps
import test_possession_passes as pass_fixtures


class PipelineTests(unittest.TestCase):
    def scene(self, n=60, fps=10):
        ball = np.column_stack([30 + np.arange(n) * 0.05, np.full(n, 25.)])
        players = np.stack([ball + [1., 0.], ball + [8., 0.]], axis=1)
        return TrackingData(ball, players, np.array([0, 1]), np.ones(n, dtype=int), fps)

    def evidence(self, owners, chain=0):
        owners = np.asarray(owners)
        n = len(owners)
        contacts = SimpleNamespace(team=owners, observed=owners >= 0,
                                   confidence=np.where(owners >= 0, 1., 0.))
        passes = SimpleNamespace(events=[], team=np.full(n, 0 if chain else -1),
                                 chain_count=np.full(n, chain), confidence=np.full(n, 1. if chain else 0.))
        return contacts, passes

    def test_interception_and_carry_need_no_passes(self):
        data = self.scene(30)
        contacts, passes = self.evidence([0] * 15 + [1] * 15)
        labels, _ = decode_team_possession(data, contacts, passes, [], np.zeros(30, bool))
        self.assertEqual(labels[10], "H")
        self.assertEqual(labels[-1], "A")

    def test_phase_tags_cannot_create_ownership_or_force_turnover(self):
        data = self.scene(30)
        event = {"code": "BUILD UP", "start_frame": 0, "end_frame": 30,
                 "tags": {"Team": "Atletic Club"}}
        contacts, passes = self.evidence([-1] * 30)
        labels, _ = decode_team_possession(data, contacts, passes, [event], np.zeros(30, bool))
        self.assertTrue((labels == "X").all())
        contacts, passes = self.evidence([0] * 5 + [-1] * 25)
        labels, _ = decode_team_possession(data, contacts, passes, [event], np.zeros(30, bool))
        self.assertNotIn("A", labels)

    def test_team_memory_outlives_carrier_but_expires(self):
        data = self.scene(25)
        contacts, passes = self.evidence([0] * 5 + [-1] * 20)
        config = TeamConfig(memory_s=0.5, max_supported_memory_s=1.)
        labels, reasons = decode_team_possession(data, contacts, passes, [], np.zeros(25, bool), config)
        self.assertEqual(labels[7], "H")
        self.assertEqual(labels[-1], "X")
        self.assertIn("evidence_expired", reasons)

    def test_pass_chain_extends_but_does_not_remove_memory_limit(self):
        data = self.scene(25)
        contacts, plain = self.evidence([0] * 5 + [-1] * 20)
        _, chain = self.evidence([0] * 5 + [-1] * 20, chain=2)
        chain.events = [SimpleNamespace(completed=True, confirmation_frame=4,
                                        team=0, confidence=1.)]
        # Pass counting can reset on missing observations, while the separate
        # team-support snapshot still decays through the gap.
        chain.chain_count[5:] = 0
        chain.confidence[5:] = 0
        config = TeamConfig(memory_s=0.3, max_supported_memory_s=1.)
        old, _ = decode_team_possession(data, contacts, plain, [], np.zeros(25, bool), config)
        new, _ = decode_team_possession(data, contacts, chain, [], np.zeros(25, bool), config)
        self.assertEqual(old[11], "X")
        self.assertEqual(new[11], "H")
        self.assertEqual(new[-1], "X")

    def test_uncertain_pass_does_not_extend_team_memory(self):
        data = self.scene(25)
        contacts, passes = self.evidence([0] * 5 + [-1] * 20)
        passes.events = [SimpleNamespace(completed=False, confirmation_frame=4,
                                         team=0, confidence=0.3)]
        config = TeamConfig(memory_s=0.3, max_supported_memory_s=1.)
        labels, _ = decode_team_possession(data, contacts, passes, [], np.zeros(25, bool), config)
        self.assertEqual(labels[11], "X")

    def test_old_pass_support_cannot_return_after_opponent_control(self):
        data = self.scene(40)
        contacts, passes = self.evidence([0] * 5 + [1] * 10 + [0] * 5 + [-1] * 20)
        passes.chain_count[4] = 2
        passes.confidence[4] = 1.
        passes.events = [SimpleNamespace(completed=True, confirmation_frame=4,
                                         team=0, confidence=1.)]
        config = TeamConfig(memory_s=0.3, max_supported_memory_s=1.)
        labels, _ = decode_team_possession(data, contacts, passes, [], np.zeros(40, bool), config)
        self.assertEqual(labels[12], "A")
        self.assertEqual(labels[19], "H")
        self.assertEqual(labels[24], "X")

    def test_pass_flying_near_opponent_retains_team_ownership(self):
        data, _ = pass_fixtures.PassTests().scene()
        data.player_xy[:, 3] = [6., 0.]
        result = predict_pipeline(data, [], (105.3, 68))
        self.assertNotIn("A", result.labels)
        self.assertTrue((result.labels[5:14] == "H").all())
        self.assertTrue(any(event.completed for event in result.passes.events))

    def test_dead_ball_and_new_half_clear_team_evidence(self):
        data = self.scene(25)
        contacts, passes = self.evidence([0] * 8 + [-1] * 17)
        dead = np.zeros(25, bool)
        dead[8:10] = True
        labels, _ = decode_team_possession(data, contacts, passes, [], dead)
        self.assertTrue((labels[8:10] == "D").all())
        self.assertTrue((labels[10:] == "X").all())
        data.half[8:] = 2
        labels, _ = decode_team_possession(data, contacts, passes, [], np.zeros(25, bool))
        self.assertTrue((labels[8:] == "X").all())

    def test_uncontrolled_opponent_proximity_does_not_veto_bridge(self):
        data = self.scene(6)
        data.ball_xy[1:4] = np.nan
        features = extract_features(data)
        features.nearest_team[4] = 1
        labels = np.array(["H", "X", "X", "X", "H", "H"])
        reasons = np.full(6, "confirmed_control", dtype="<U32")
        old, _ = bridge_missing_gaps(labels, reasons, data, features, [])
        np.testing.assert_array_equal(old, labels)
        control = np.array([0, -1, -1, -1, -1, 0])
        new, _ = bridge_missing_gaps(labels, reasons, data, features, [], controlled_team=control)
        self.assertTrue((new == "H").all())
        control[4] = 1
        denied, _ = bridge_missing_gaps(labels, reasons, data, features, [], controlled_team=control)
        np.testing.assert_array_equal(denied, labels)

    def test_full_pipeline_is_deterministic_and_preserves_inputs(self):
        data = self.scene(80)
        original = data.ball_xy.copy()
        one = predict_pipeline(data, [], (105.3, 68))
        two = predict_pipeline(data, [], (105.3, 68))
        np.testing.assert_array_equal(one.labels, two.labels)
        np.testing.assert_array_equal(data.ball_xy, original)
        self.assertEqual(one.labels[-1], "H")
        self.assertFalse(one.passes.events)

    def test_full_pipeline_resets_contacts_and_passes_at_stoppage(self):
        data = self.scene(100)
        data.ball_xy[:] = [1., 20.]
        data.player_xy[:, 0] = [2., 20.]
        data.player_xy[:, 1] = [8., 20.]
        event = {"code": "SET PIECES", "start_frame": 0, "end_frame": 100,
                 "tags": {"Team": "N/A"}}
        result = predict_pipeline(data, [event], (105.3, 68))
        self.assertIn("D", result.labels)
        dead = result.labels == "D"
        self.assertTrue((result.contacts.team[dead] == -1).all())
        self.assertTrue((result.passes.chain_count[dead] == 0).all())


if __name__ == "__main__":
    unittest.main()
