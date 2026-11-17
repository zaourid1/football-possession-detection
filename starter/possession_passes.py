"""Conservative pass evidence from controlled departure, flight, and reception.

This module does not equate a changed track slot with a pass, and does not
require passes before a team can gain possession. Inputs are the offline contact
episodes from possession_contacts; completed events require an observed flight.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class PassConfig:
    max_flight_s: float = 4.0
    min_flight_s: float = 0.12
    min_travel_m: float = 4.0
    min_peak_speed_mps: float = 4.0
    min_visible_fraction: float = 0.8
    min_receiver_separation_m: float = 3.0
    chain_memory_s: float = 8.0

    def __post_init__(self):
        values = (self.max_flight_s, self.min_flight_s, self.min_travel_m,
                  self.min_peak_speed_mps, self.min_receiver_separation_m,
                  self.chain_memory_s)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("Pass distances, speeds, and durations must be finite and positive")
        if self.min_flight_s > self.max_flight_s:
            raise ValueError("min_flight_s must not exceed max_flight_s")
        if not np.isfinite(self.min_visible_fraction) or not 0 < self.min_visible_fraction <= 1:
            raise ValueError("min_visible_fraction must be in (0, 1]")


@dataclass(frozen=True)
class PassEvent:
    team: int
    departure_frame: int
    reception_frame: int
    confirmation_frame: int
    confidence: float
    completed: bool
    reason: str


@dataclass
class PassEvidence:
    """Per-frame support from recent completed passes, not final ownership.

    team is -1 where there is no completed-pass support. chain_count is reset by
    missing observations, opposing control, stoppages, or a new half. Confidence
    decays over time and expires after chain_memory_s. Possible passes are kept
    in events for diagnostics but never increment the completed-pass chain.
    """

    events: List[PassEvent]
    chain_count: np.ndarray
    team: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class _Episode:
    id: int
    team: int
    start: int
    end: int
    confidence: float


def _episodes(contacts, half, blocked):
    """Group observed contacts, keeping held carrier frames out of anchors."""
    observed = (np.asarray(contacts.observed, dtype=bool)
                & (np.asarray(contacts.team) >= 0)
                & (np.asarray(contacts.episode_id) >= 0)
                & (half != 0) & ~blocked)
    episodes = []
    for frame in np.flatnonzero(observed):
        episode_id = int(contacts.episode_id[frame])
        team = int(contacts.team[frame])
        confidence = float(contacts.confidence[frame])
        if (episodes and episodes[-1].id == episode_id and episodes[-1].team == team
                and half[episodes[-1].end] == half[frame]
                and not np.any(blocked[episodes[-1].end:frame + 1])):
            previous = episodes[-1]
            episodes[-1] = _Episode(episode_id, team, previous.start, frame,
                                    min(previous.confidence, confidence))
        else:
            episodes.append(_Episode(episode_id, team, frame, frame, confidence))
    return episodes


def _same_continuous_carrier(data, contacts, departure, reception):
    """A continuously plausible slot at both contacts is evidence of a dribble.

    A slot jump invalidates this check; slot identity alone is never sufficient.
    The check only rejects a pass, it does not establish a player's identity.
    """
    slot = int(contacts.slot[departure])
    receiver_slot = int(contacts.slot[reception])
    if slot < 0 or receiver_slot < 0:
        return False
    path = data.player_xy[departure:reception + 1, slot]
    receiver = data.player_xy[reception, receiver_slot]
    if not np.isfinite(path).all() or not np.isfinite(receiver).all():
        return False
    # Broad plausibility bound; exact physical identity is unavailable.
    plausible = np.all(np.linalg.norm(np.diff(path, axis=0), axis=1) * data.fps <= 12.0)
    return plausible and np.linalg.norm(path[-1] - receiver) <= 1.0


def detect_passes(data, contacts, blocked=None, config=None):
    """Infer completed/possible passes and independent pass-chain support.

    Contact entry may be backdated by its confirmation interval, so this is an
    offline detector. confirmation_frame records the reception onset in that
    offline timeline, not a promise of real-time availability.
    """
    config = config or PassConfig()
    n_frames = len(data.half)
    if not np.isfinite(data.fps) or data.fps <= 0:
        raise ValueError("fps must be finite and positive")
    half = np.asarray(data.half)
    blocked = np.zeros(n_frames, dtype=bool) if blocked is None else np.asarray(blocked, dtype=bool)
    if blocked.shape != (n_frames,):
        raise ValueError("blocked must have one value per frame")
    for field in (contacts.team, contacts.slot, contacts.episode_id,
                  contacts.observed, contacts.confidence):
        if np.asarray(field).shape != (n_frames,):
            raise ValueError("Contact arrays must have one value per frame")
    if not np.isin(contacts.team, [-1, 0, 1]).all():
        raise ValueError("Contact teams must be -1, 0, or 1")
    if (not np.isfinite(contacts.confidence).all()
            or np.any(np.asarray(contacts.confidence) < 0)
            or np.any(np.asarray(contacts.confidence) > 1)):
        raise ValueError("Contact confidence must lie in [0, 1]")

    available = data.ball_available
    episodes = _episodes(contacts, half, blocked)
    events = []
    for source, receiver in zip(episodes, episodes[1:]):
        departure, reception = source.end, receiver.start
        if source.team != receiver.team or source.id == receiver.id:
            continue
        flight_s = (reception - departure) / data.fps
        if not config.min_flight_s <= flight_s <= config.max_flight_s:
            continue
        span = slice(departure, reception + 1)
        if half[departure] == 0 or not np.all(half[span] == half[departure]) or blocked[span].any():
            continue
        if not available[departure] or not available[reception]:
            continue
        coverage = float(np.mean(available[span]))
        if coverage < config.min_visible_fraction:
            continue
        travel = float(np.linalg.norm(data.ball_xy[reception] - data.ball_xy[departure]))
        if travel < config.min_travel_m:
            continue
        velocity = np.asarray(contacts.interactions.ball_velocity)[departure + 1:reception + 1]
        motion_valid = np.asarray(contacts.interactions.ball_motion_valid)[departure + 1:reception + 1]
        speeds = np.linalg.norm(velocity, axis=1)
        usable = motion_valid & np.isfinite(speeds)
        if not usable.any() or np.max(speeds[usable]) < config.min_peak_speed_mps:
            continue
        sender_slot, receiver_slot = int(contacts.slot[departure]), int(contacts.slot[reception])
        if sender_slot < 0 or receiver_slot < 0:
            continue
        sender_position = data.player_xy[departure, sender_slot]
        receiver_position = data.player_xy[reception, receiver_slot]
        sender_velocity = contacts.interactions.carrier_velocity[departure, source.team]
        if not np.isfinite(sender_position).all() or not np.isfinite(receiver_position).all():
            continue
        if _same_continuous_carrier(data, contacts, departure, reception):
            continue
        has_sender_velocity = np.isfinite(sender_velocity).all()
        projected_sender = sender_position + sender_velocity * flight_s if has_sender_velocity else sender_position
        separation = np.linalg.norm(receiver_position - projected_sender)
        if separation < config.min_receiver_separation_m:
            continue
        reliability = float(np.mean(usable))
        completed = bool(available[span].all() and reliability >= config.min_visible_fraction
                         and has_sender_velocity)
        confidence = float(min(source.confidence, receiver.confidence) * coverage * reliability)
        if not completed:
            confidence *= 0.4
        reason = "observed_departure_reception" if completed else "uncertain_flight_or_carrier"
        events.append(PassEvent(source.team, departure, reception, reception,
                                confidence, completed, reason))

    chain_count = np.zeros(n_frames, dtype=np.int32)
    support_team = np.full(n_frames, -1, dtype=np.int8)
    support_confidence = np.zeros(n_frames, dtype=np.float32)
    arrivals = {event.reception_frame: event for event in events if event.completed}
    owner, count, last_pass, confidence, previous_half = -1, 0, -1, 0.0, 0
    for frame in range(n_frames):
        if half[frame] != previous_half or half[frame] == 0 or blocked[frame]:
            owner, count, last_pass, confidence = -1, 0, -1, 0.0
        previous_half = half[frame]
        if half[frame] == 0 or blocked[frame]:
            continue
        if not available[frame]:
            count, last_pass, confidence = 0, -1, 0.0
        if contacts.observed[frame] and contacts.team[frame] >= 0:
            observed_team = int(contacts.team[frame])
            if observed_team != owner:
                owner, count, last_pass, confidence = observed_team, 0, -1, 0.0
        event = arrivals.get(frame)
        if event is not None:
            if event.team != owner:
                owner, count = event.team, 0
            count += 1
            last_pass, confidence = frame, event.confidence
        if count and frame - last_pass <= config.chain_memory_s * data.fps:
            chain_count[frame] = count
            support_team[frame] = owner
            age_s = (frame - last_pass) / data.fps
            support_confidence[frame] = confidence * np.exp(-age_s / config.chain_memory_s)
        elif count:
            count, last_pass, confidence = 0, -1, 0.0

    return PassEvidence(events, chain_count, support_team, support_confidence)
