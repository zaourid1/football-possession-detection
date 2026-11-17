"""Pass-event tests on invented trajectories and contact episodes."""

from types import SimpleNamespace
import unittest

import numpy as np

from possession_data import TrackingData
from possession_features import extract_features
from possession_contacts import detect_contacts
from possession_passes import PassConfig, detect_passes


class PassTests(unittest.TestCase):
    def scene(self, two_passes=False):
        n = 35
        ball = np.zeros((n, 2), dtype=float)
        ball[4:15, 0] = np.arange(11)
        ball[15:, 0] = 10
        if two_passes:
            ball[18:29, 0] = 10 + np.arange(11)
            ball[29:, 0] = 20
        players = np.zeros((n, 4, 2), dtype=float)
        players[:, :, 0] = [0, 10, 20, 40]
        data = TrackingData(ball, players, np.array([0, 0, 0, 1]), np.ones(n, dtype=int), 10)
        teams = np.full(n, -1, dtype=int)
        slots = np.full(n, -1, dtype=int)
        episodes = np.full(n, -1, dtype=int)
        for episode, start, end, slot in ((0, 0, 5, 0), (1, 14, 19, 1), (2, 28, 35, 2)):
            if episode == 2 and not two_passes:
                continue
            teams[start:end] = 0
            slots[start:end] = slot
            episodes[start:end] = episode
        velocity = np.zeros((n, 2), dtype=float)
        velocity[1:] = np.diff(ball, axis=0) * 10
        contacts = SimpleNamespace(
            team=teams, slot=slots, episode_id=episodes, observed=teams >= 0,
            confidence=np.where(teams >= 0, 1., 0.),
            interactions=SimpleNamespace(
                ball_velocity=velocity, ball_motion_valid=np.ones(n, dtype=bool),
                carrier_velocity=np.zeros((n, 2, 2)),
            ),
        )
        return data, contacts

    def test_complete_pass_and_support_after_reception_only(self):
        data, contacts = self.scene()
        evidence = detect_passes(data, contacts)
        self.assertEqual(len(evidence.events), 1)
        event = evidence.events[0]
        self.assertTrue(event.completed)
        self.assertEqual((event.team, event.departure_frame, event.reception_frame), (0, 4, 14))
        self.assertEqual(evidence.chain_count[13], 0)
        self.assertEqual(evidence.chain_count[14], 1)
        self.assertEqual(evidence.team[14], 0)
        self.assertGreater(evidence.confidence[14], evidence.confidence[20])

    def test_two_pass_chain(self):
        data, contacts = self.scene(two_passes=True)
        evidence = detect_passes(data, contacts)
        self.assertEqual(len(evidence.events), 2)
        self.assertEqual(evidence.chain_count[28], 2)

    def test_actual_contact_detector_feeds_completed_pass(self):
        data, _ = self.scene()
        contacts = detect_contacts(data, extract_features(data))
        evidence = detect_passes(data, contacts)
        self.assertTrue(any(event.completed for event in evidence.events))

    def test_opponent_control_interrupts_a_pass_candidate(self):
        data, contacts = self.scene()
        contacts.team[9:11] = 1
        contacts.episode_id[9:11] = 99
        contacts.slot[9:11] = 3
        contacts.observed[9:11] = True
        contacts.confidence[9:11] = 1
        self.assertFalse(detect_passes(data, contacts).events)

    def test_missing_flight_is_only_possible_not_completed(self):
        data, contacts = self.scene()
        data.ball_xy[9] = np.nan
        contacts.interactions.ball_motion_valid[9:11] = False
        evidence = detect_passes(data, contacts)
        self.assertEqual(len(evidence.events), 1)
        self.assertFalse(evidence.events[0].completed)
        self.assertLess(evidence.events[0].confidence, .4)
        self.assertFalse(evidence.chain_count.any())

    def test_restart_or_new_half_blocks_pass(self):
        for mode in ("blocked", "half"):
            data, contacts = self.scene()
            blocked = np.zeros(len(data.half), dtype=bool)
            if mode == "blocked":
                blocked[9] = True
            else:
                data.half[9:] = 2
            self.assertFalse(detect_passes(data, contacts, blocked).events)

    def test_slot_switch_or_dribble_does_not_count(self):
        data, contacts = self.scene()
        # The original player's continuous plausible trajectory follows the ball.
        # Duplicating it in the receiver slot must not turn a dribble into a pass.
        data.player_xy[:, 0] = data.ball_xy
        data.player_xy[:, 1] = data.ball_xy
        self.assertFalse(detect_passes(data, contacts).events)

    def test_same_episode_never_counts_even_if_slot_changes(self):
        data, contacts = self.scene()
        contacts.episode_id[14:19] = 0
        self.assertFalse(detect_passes(data, contacts).events)

    def test_chain_resets_on_missing_frame_opponent_or_block(self):
        for mode in ("missing", "opponent", "blocked"):
            data, contacts = self.scene()
            blocked = np.zeros(len(data.half), dtype=bool)
            if mode == "missing":
                data.ball_xy[22] = np.nan
            elif mode == "opponent":
                contacts.team[22], contacts.observed[22] = 1, True
            else:
                blocked[22] = True
            evidence = detect_passes(data, contacts, blocked)
            self.assertEqual(evidence.chain_count[21], 1)
            self.assertEqual(evidence.chain_count[22], 0)

    def test_short_or_long_flight_rejected(self):
        data, contacts = self.scene()
        self.assertFalse(detect_passes(data, contacts, config=PassConfig(max_flight_s=.5)).events)
        self.assertFalse(detect_passes(data, contacts, config=PassConfig(min_travel_m=20)).events)

    def test_invalid_configuration_or_contact_confidence_rejected(self):
        with self.assertRaises(ValueError):
            PassConfig(min_visible_fraction=2)
        data, contacts = self.scene()
        contacts.confidence[0] = np.nan
        with self.assertRaises(ValueError):
            detect_passes(data, contacts)


if __name__ == "__main__":
    unittest.main()
