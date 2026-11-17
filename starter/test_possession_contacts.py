"""Invented control observations; no restricted match assets or labels."""

import unittest

import numpy as np

from possession_contacts import (
    ContactConfig,
    _classify_touch_events,
    _group_touch_candidates,
    derive_interactions,
    detect_contacts,
)
from possession_data import TrackingData
from possession_features import extract_features


class ContactTests(unittest.TestCase):
    def scene(self, n=50, fps=25):
        ball = np.tile([20., 30.], (n, 1))
        players = np.tile([[21., 30.], [50., 30.]], (n, 1, 1))
        return TrackingData(ball, players, np.array([0, 1]), np.ones(n, dtype=int), fps)

    def test_stable_carry_and_looser_retention(self):
        data = self.scene()
        data.ball_xy[:, 0] += np.arange(50) * 0.04
        data.player_xy[:, 0, 0] += np.arange(50) * 0.04
        data.player_xy[20:, 0, 0] += np.minimum(np.arange(30) * 0.1, 1.7)
        evidence = detect_contacts(data, extract_features(data))
        self.assertTrue((evidence.team[10:] == 0).all())
        self.assertEqual(len(np.unique(evidence.episode_id[10:])), 1)
        self.assertTrue(evidence.observed[-1])

    def test_new_control_outside_entry_radius_is_not_established(self):
        data = self.scene()
        data.player_xy[:, 0, 0] += 1.7
        evidence = detect_contacts(data, extract_features(data))
        self.assertTrue((evidence.team == -1).all())

    def test_missing_carrier_is_held_briefly_then_expires(self):
        data = self.scene()
        data.ball_xy[20:] = np.nan
        evidence = detect_contacts(data, extract_features(data))
        self.assertEqual(evidence.team[20], 0)
        self.assertFalse(evidence.observed[20])
        self.assertGreater(evidence.confidence[20], evidence.confidence[24])
        self.assertEqual(evidence.team[25], -1)

    def test_half_and_stoppage_reset_episodes(self):
        data = self.scene()
        data.half[25:] = 2
        blocked = np.zeros(50, dtype=bool)
        blocked[15:20] = True
        evidence = detect_contacts(data, extract_features(data), blocked)
        self.assertTrue((evidence.team[15:20] == -1).all())
        self.assertEqual(evidence.team[25], -1)
        self.assertNotEqual(evidence.episode_id[10], evidence.episode_id[24])
        self.assertNotEqual(evidence.episode_id[24], evidence.episode_id[35])

    def test_slot_handoff_keeps_same_local_episode(self):
        data = self.scene()
        duplicate = data.player_xy[:, :1].copy()
        data.player_xy = np.concatenate([data.player_xy, duplicate], axis=1)
        data.track_team = np.array([0, 1, 0])
        data.player_xy[:25, 2] = np.nan
        data.player_xy[25:, 0] = np.nan
        evidence = detect_contacts(data, extract_features(data))
        self.assertEqual(evidence.slot[24], 0)
        self.assertEqual(evidence.slot[25], 2)
        self.assertEqual(evidence.episode_id[24], evidence.episode_id[25])

    def test_visible_fast_release_does_not_hold_carrier(self):
        data = self.scene()
        data.ball_xy[20:, 0] += np.arange(1, 31) * 0.8
        evidence = detect_contacts(data, extract_features(data))
        self.assertFalse(evidence.observed[23])
        self.assertEqual(evidence.team[23], -1)

    def test_velocity_windows_reject_gaps_and_jumps(self):
        data = self.scene()
        data.ball_xy[20] = np.nan
        data.ball_xy[35, 0] += 50
        features = derive_interactions(data, extract_features(data))
        self.assertTrue(np.isnan(features.ball_velocity[20:24]).all())
        self.assertTrue(np.isnan(features.ball_velocity[35:39]).all())
        self.assertTrue(features.ball_motion_valid[30])

    def test_ball_impulse_next_to_player_is_a_touch_candidate(self):
        data = self.scene(n=30, fps=10)
        data.ball_xy[:, 0] = np.arange(30, dtype=float)
        data.ball_xy[11:, 0] = 10 + np.arange(1, 20) * 0.2
        data.player_xy[:, 0] = [10.5, 30.0]
        data.player_xy[11:, 0, 0] = data.ball_xy[11:, 0] + 0.5
        interactions = derive_interactions(
            data, extract_features(data), ContactConfig(velocity_window_s=0.1))
        self.assertTrue(interactions.touch_candidate[10, 0])
        self.assertGreater(interactions.velocity_change[10], 3.0)

    def test_constant_ball_motion_is_not_a_touch_candidate(self):
        data = self.scene(n=30, fps=10)
        data.ball_xy[:, 0] = np.arange(30, dtype=float) * 0.2
        data.player_xy[:, 0, 0] = data.ball_xy[:, 0] + 0.5
        interactions = derive_interactions(
            data, extract_features(data), ContactConfig(velocity_window_s=0.1))
        self.assertFalse(interactions.touch_candidate.any())

    def test_touch_impulse_seeds_but_does_not_alone_confirm_control(self):
        data = self.scene(n=30, fps=10)
        data.ball_xy[:, 0] = np.arange(30, dtype=float)
        data.ball_xy[11:, 0] = 10 + np.arange(1, 20) * 0.2
        data.player_xy[:, 0] = [10.5, 30.0]
        data.player_xy[11:, 0, 0] = data.ball_xy[11:, 0] + 0.5
        config = ContactConfig(velocity_window_s=0.1, confirm_s=0.3)
        evidence = detect_contacts(data, extract_features(data), config=config)
        self.assertEqual(evidence.team[10], 0)
        self.assertEqual(evidence.episode_id[10], evidence.episode_id[12])
        self.assertIn("controlled", [touch.classification for touch in evidence.touches])

        # Remove the controlled follow-through. The single impulse remains a
        # possible touch, but it must not become a possession episode.
        data.player_xy[11:, 0] = np.nan
        evidence = detect_contacts(data, extract_features(data), config=config)
        self.assertEqual(evidence.team[10], -1)
        self.assertNotIn("controlled", [touch.classification for touch in evidence.touches])

    def test_repeated_impulses_without_followthrough_never_confirm_control(self):
        data = self.scene(n=30, fps=10)
        data.ball_xy[:, 0] = 20 + (np.arange(30) % 2)
        data.player_xy[:, 0] = [20.5, 30.0]
        config = ContactConfig(velocity_window_s=0.1, confirm_s=0.3)
        evidence = detect_contacts(data, extract_features(data), config=config)
        self.assertTrue((evidence.team == -1).all())
        self.assertNotIn("controlled", [touch.classification for touch in evidence.touches])

    def test_opponent_redirection_recovered_by_prior_team_is_deflection_like(self):
        data = self.scene(n=30, fps=10)
        data.ball_xy[:, 0] = np.arange(30, dtype=float)
        data.ball_xy[11:, 0] = 10 - np.arange(1, 20) * 0.8
        data.player_xy[:, 0] = [40.0, 30.0]
        data.player_xy[:, 1] = [10.5, 30.0]
        config = ContactConfig(velocity_window_s=0.1)
        interactions = derive_interactions(data, extract_features(data), config)
        blocked = np.zeros(30, dtype=bool)
        candidates = _group_touch_candidates(data, interactions, blocked, config)
        self.assertTrue(candidates)

        owner = np.full(30, -1, dtype=np.int8)
        observed = np.zeros(30, dtype=bool)
        owner[7], observed[7] = 0, True
        owner[14], observed[14] = 0, True
        touches = _classify_touch_events(
            data, interactions, candidates, owner, observed, blocked, config)
        away_touches = [touch for touch in touches if touch.team == 1]
        self.assertIn("deflection_like", [touch.classification for touch in away_touches])

        blocked[12] = True
        interrupted = _classify_touch_events(
            data, interactions, candidates, owner, observed, blocked, config)
        self.assertNotIn("deflection_like", [touch.classification for touch in interrupted])

        blocked[:] = False
        owner[7] = 1
        same_team = _classify_touch_events(
            data, interactions, candidates, owner, observed, blocked, config)
        self.assertNotIn("deflection_like", [touch.classification for touch in same_team])

    def test_touch_without_reliable_followthrough_stays_unresolved(self):
        data = self.scene(n=30, fps=10)
        data.ball_xy[:, 0] = np.arange(30, dtype=float)
        data.ball_xy[11, 0] = 10.2
        data.ball_xy[12:, 0] = np.nan
        data.player_xy[:, 0] = [10.5, 30.0]
        evidence = detect_contacts(
            data, extract_features(data), config=ContactConfig(velocity_window_s=0.1))
        self.assertTrue(evidence.touches)
        self.assertNotIn("controlled", [touch.classification for touch in evidence.touches])
        self.assertNotIn("deflection_like", [touch.classification for touch in evidence.touches])

    def test_invalid_hysteresis_config_is_rejected(self):
        with self.assertRaises(ValueError):
            ContactConfig(retain_radius_m=1)
        with self.assertRaises(ValueError):
            ContactConfig(confirm_s=0)
        with self.assertRaises(ValueError):
            ContactConfig(touch_radius_m=3)


if __name__ == "__main__":
    unittest.main()
