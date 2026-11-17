"""Integrated offline possession pipeline motivated by PAPER_REVIEW.md.

Existing predictors remain unchanged. No ground truth or human-labelled match
events are read. Defaults are interpretable heuristics, not fitted parameters.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from possession_data import TrackingData
from possession_features import extract_features
from possession_contacts import ContactConfig, ContactEvidence, detect_contacts
from possession_passes import PassConfig, PassEvidence, detect_passes
from possession_model import ModelConfig, tag_evidence
from possession_refinement import bridge_missing_gaps
from possession_stoppages import StillnessConfig, StoppageEvidence, detect_stillness


@dataclass(frozen=True)
class TeamConfig:
    """Carrier expiry is separate from these team-level time limits."""

    evidence_window_s: float = 0.8
    confirm_s: float = 0.24
    lead_s: float = 0.12
    memory_s: float = 4.0
    max_supported_memory_s: float = 8.0
    pass_memory_bonus_s: float = 1.0
    phase_weight: float = 0.15

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Team settings must be finite and nonnegative")
        if self.evidence_window_s <= 0 or self.confirm_s <= 0:
            raise ValueError("Evidence window and confirmation must be positive")
        if self.max_supported_memory_s < self.memory_s:
            raise ValueError("Supported memory cannot be shorter than base memory")


@dataclass(frozen=True)
class PipelineConfig:
    contact: ContactConfig = field(default_factory=ContactConfig)
    passes: PassConfig = field(default_factory=PassConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    stoppage: ModelConfig = field(default_factory=ModelConfig)
    stillness: StillnessConfig = field(default_factory=StillnessConfig)
    bridge: bool = True


@dataclass
class PipelineResult:
    labels: np.ndarray
    reasons: np.ndarray
    contacts: ContactEvidence
    passes: PassEvidence
    stoppages: StoppageEvidence


def _stoppage_mask(data, features, stillness, pitch, config, max_ball_speed):
    """Confirm stoppages from physical evidence; missing ball alone is neutral."""
    x, y = data.ball_xy.T
    margin = config.outside_margin_m
    outside = data.ball_available & (
        (x < -margin) | (x > pitch[0] + margin)
        | (y < -margin) | (y > pitch[1] + margin))
    evidence = outside | stillness.stationary_restart
    dead = np.zeros(len(data.half), dtype=bool)
    stopped = False
    stop_run = resume_run = 0
    previous_half = 0
    for frame, half in enumerate(data.half):
        if half != previous_half:
            stopped = False
            stop_run = resume_run = 0
        previous_half = half
        if half == 0:
            continue
        stop_run = stop_run + 1 if evidence[frame] else 0
        speed = features.ball_speed_mps[frame]
        moving = (np.isfinite(speed) and config.restart_speed_mps <= speed <= max_ball_speed
                  and not evidence[frame])
        resume_run = resume_run + 1 if moving else 0
        if stop_run >= max(1, round(config.stoppage_confirm_s * data.fps)):
            stopped = True
        elif resume_run >= max(1, round(config.restart_confirm_s * data.fps)):
            stopped = False
        dead[frame] = stopped
    return dead, evidence


def decode_team_possession(data, contacts, passes, events, dead, config=None):
    """Accumulate physical contacts; passes support ownership without defining it.

    Phase tags can reinforce an existing observation but cannot independently
    establish or switch ownership. Missing carriers do not immediately end team
    possession. Offline pass support is attached to observed receptions, never
    delayed until a fixed pass count is reached.
    """
    config = config or TeamConfig()
    n, fps = len(data.half), data.fps
    phase, _ = tag_evidence(events, n)
    labels = np.full(n, "X", dtype="<U1")
    reasons = np.full(n, "ungraded", dtype="<U32")
    votes = deque()
    totals = np.zeros(2)
    owner = -1
    last_physical = -1
    previous_half = 0
    chain_owner, chain_strength, last_pass = -1, 0.0, -1
    evidence_window = max(1, round(config.evidence_window_s * fps))
    # Pass detections may be confirmed after the physical reception. Never use
    # an uncertain flight as if it were a measured successful reception.
    pass_at = {}
    for event in passes.events:
        if event.completed:
            pass_at[event.confirmation_frame] = event

    for frame, half in enumerate(data.half):
        if half != previous_half or dead[frame]:
            owner, last_physical = -1, -1
            votes.clear()
            totals[:] = 0
            chain_owner, chain_strength, last_pass = -1, 0.0, -1
        previous_half = half
        if half == 0:
            continue
        if dead[frame]:
            labels[frame], reasons[frame] = "D", "stoppage_evidence"
            continue

        vote = np.zeros(2)
        observed_team = int(contacts.team[frame]) if contacts.observed[frame] else -1
        if observed_team >= 0 and observed_team != chain_owner:
            chain_owner, chain_strength, last_pass = -1, 0.0, -1
        if observed_team >= 0:
            vote[observed_team] = contacts.confidence[frame]
            if phase[frame] == observed_team:
                vote[observed_team] += config.phase_weight
        pass_event = pass_at.get(frame)
        if pass_event is not None:
            vote[pass_event.team] += pass_event.confidence
            chain_owner = pass_event.team
            chain_strength = float(passes.chain_count[frame]) * float(passes.confidence[frame])
            last_pass = frame
        votes.append(vote)
        totals += vote
        if len(votes) > evidence_window:
            totals -= votes.popleft()
        winner = int(np.argmax(totals))
        physical = observed_team == winner or (pass_event is not None and pass_event.team == winner)
        confirmed = (physical and totals[winner] >= max(1, config.confirm_s * fps)
                     and totals[winner] - totals[1 - winner] >= max(1, config.lead_s * fps))
        reason = "team_memory"
        if confirmed:
            owner = winner
            last_physical = frame
            reason = "confirmed_control" if observed_team == winner else "confirmed_pass"
        elif owner >= 0 and observed_team == owner:
            last_physical = frame

        if owner >= 0:
            # Chain support belongs to this offline observation sequence.
            # Phase agreement can extend a known owner, never revive X.
            # Preserve bounded, decaying TEAM confidence through missing-ball
            # frames even when pass counting resets because the flight is unseen.
            age_s = (frame - last_pass) / fps if last_pass >= 0 else np.inf
            strength = (chain_strength * np.exp(-age_s / max(config.max_supported_memory_s, 1e-6))
                        if chain_owner == owner else 0.)
            extra = config.pass_memory_bonus_s * strength
            if phase[frame] == owner:
                extra += config.pass_memory_bonus_s
            horizon = min(config.max_supported_memory_s, config.memory_s + extra)
            if frame - last_physical > horizon * fps:
                owner = -1
                reason = "evidence_expired"
        labels[frame] = "H" if owner == 0 else "A" if owner == 1 else "X"
        reasons[frame] = reason if owner >= 0 or reason == "evidence_expired" else "insufficient_evidence"
    return labels, reasons


def predict_pipeline(data: TrackingData, events, pitch, config=None, features=None):
    """Single explicit path: geometry -> stillness -> contacts -> passes -> teams -> bridge."""
    config = config or PipelineConfig()
    events = list(events)
    features = features if features is not None else extract_features(data)
    if len(pitch) != 2 or not np.all(np.isfinite(pitch)) or np.any(np.asarray(pitch) <= 0):
        raise ValueError("pitch must contain positive finite length and width in metres")
    stillness = detect_stillness(data, events, pitch, config.stillness)
    dead, stop_evidence = _stoppage_mask(
        data, features, stillness, pitch, config.stoppage, config.contact.max_ball_speed_mps)
    blocked = dead | stop_evidence | (data.half == 0)
    contacts = detect_contacts(data, features, blocked=blocked, config=config.contact)
    passes = detect_passes(data, contacts, blocked=blocked, config=config.passes)
    labels, reasons = decode_team_possession(data, contacts, passes, events, dead, config.team)
    if config.bridge:
        observed_team = np.where(contacts.observed, contacts.team, -1)
        labels, reasons = bridge_missing_gaps(
            labels, reasons, data, features, events, config.stoppage,
            controlled_team=observed_team)
    return PipelineResult(labels, reasons, contacts, passes, stillness)
