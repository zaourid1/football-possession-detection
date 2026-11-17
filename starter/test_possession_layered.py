"""Layering contracts on invented labels and physical evidence, no match stats."""

from types import SimpleNamespace
import unittest

import numpy as np

from possession_layered import combine_layers, predict_layered
import test_possession_pipeline as pipeline_fixtures


class LayeredTests(unittest.TestCase):
    def fixture(self, strict, observed=(), half=None):
        labels = np.array(list(strict))
        n = len(labels)
        measured = np.zeros(n, dtype=bool)
        measured[list(observed)] = True
        reasons = np.full(n, "team_memory", dtype="<U32")
        reasons[measured] = "confirmed_control"
        data = SimpleNamespace(half=np.ones(n, dtype=int) if half is None else np.array(half))
        physical = SimpleNamespace(
            labels=labels, reasons=reasons,
            contacts=SimpleNamespace(observed=measured,
                                     confidence=np.where(measured, 1., 0.),
                                     team=np.where(labels == "H", 0, np.where(labels == "A", 1, -1))),
            passes=SimpleNamespace(events=[]),
        )
        return data, physical

    def combine(self, reference, data, physical):
        reference = np.array(list(reference))
        return combine_layers(data, reference, np.full(len(reference), "memory"), physical)

    def test_strict_abstentions_never_erase_existing_answers(self):
        data, physical = self.fixture("XXXX")
        labels, _, _ = self.combine("HADX", data, physical)
        np.testing.assert_array_equal(labels, list("HADX"))

    def test_measured_override_persists_through_strict_memory_then_expires(self):
        data, physical = self.fixture("AAAAX", observed=(0,))
        labels, _, sources = self.combine("HHHHH", data, physical)
        np.testing.assert_array_equal(labels, list("AAAAH"))
        self.assertEqual(sources[1], "physical_memory")

    def test_late_anchor_cannot_rewrite_earlier_frames(self):
        data, physical = self.fixture("AAAAA", observed=(3,))
        labels, _, _ = self.combine("HHHHH", data, physical)
        np.testing.assert_array_equal(labels, list("HHHAA"))

    def test_held_carrier_or_uncertain_pass_cannot_start_correction(self):
        data, physical = self.fixture("AAAA")
        physical.contacts.confidence[:] = .5
        physical.passes.events = [SimpleNamespace(completed=False, confidence=.3,
                                                  confirmation_frame=0, team=1)]
        labels, _, _ = self.combine("HHHH", data, physical)
        np.testing.assert_array_equal(labels, list("HHHH"))

    def test_completed_pass_can_anchor_correction(self):
        data, physical = self.fixture("AAAA")
        physical.passes.events = [SimpleNamespace(completed=True, confidence=.8,
                                                  confirmation_frame=1, team=1)]
        labels, _, sources = self.combine("HHHH", data, physical)
        np.testing.assert_array_equal(labels, list("HAAA"))
        self.assertEqual(sources[1], "completed_pass")

    def test_reference_dead_ball_is_preserved_and_clears_override(self):
        data, physical = self.fixture("AAAAA", observed=(0,))
        labels, _, _ = self.combine("HHDHH", data, physical)
        np.testing.assert_array_equal(labels, list("AADHH"))

    def test_half_and_ungraded_boundaries_clear_override(self):
        data, physical = self.fixture("AAAAAA", observed=(0,), half=[1, 1, 0, 2, 2, 2])
        labels, _, _ = self.combine("HHHHHH", data, physical)
        np.testing.assert_array_equal(labels, list("AAXHHH"))

    def test_unexplained_dead_ball_cannot_override_but_physical_stop_can(self):
        data, physical = self.fixture("DDDD")
        physical.reasons[2:] = "stoppage_evidence"
        labels, _, _ = self.combine("HHHH", data, physical)
        np.testing.assert_array_equal(labels, list("HHDD"))

    def test_integrated_layering_is_deterministic_and_preserves_coverage(self):
        data = pipeline_fixtures.PipelineTests().scene(80)
        one = predict_layered(data, [], (105.3, 68))
        two = predict_layered(data, [], (105.3, 68))
        np.testing.assert_array_equal(one.labels, two.labels)
        self.assertTrue(np.all(one.labels[one.reference_labels != "X"] != "X"))
        self.assertEqual(one.labels[-1], "H")


if __name__ == "__main__":
    unittest.main()
