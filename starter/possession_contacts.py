"""Local control episodes from proximity, hysteresis and relative movement.

Track slots are local hypotheses, never permanent player identities. These
heuristics are independent of team-possession memory and use no match labels.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from possession_data import TrackingData


@dataclass(frozen=True)
class ContactConfig:
    entry_radius_m: float = 2.0
    retain_radius_m: float = 3.0
    entry_advantage_m: float = 0.75
    retain_advantage_m: float = 0.0
    confirm_s: float = 0.16
    carrier_hold_s: float = 0.2
    velocity_window_s: float = 0.12
    max_ball_speed_mps: float = 40.0
    max_player_speed_mps: float = 12.0
    entry_relative_speed_mps: float = 6.0
    retain_relative_speed_mps: float = 8.0
    touch_radius_m: float = 1.5
    min_touch_velocity_change_mps: float = 3.0
    min_touch_speed_change_mps: float = 2.0
    min_touch_direction_change_deg: float = 25.0
    touch_group_s: float = 0.16
    touch_followthrough_s: float = 0.6
    deflection_context_s: float = 2.0
    deflection_relative_speed_mps: float = 6.0
    deflection_separation_gain_m: float = 0.5

    def __post_init__(self):
        values = tuple(self.__dict__.values())
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Contact settings must be finite")
        if (self.entry_radius_m <= 0 or self.retain_radius_m < self.entry_radius_m
                or self.entry_advantage_m < self.retain_advantage_m
                or self.retain_advantage_m < 0 or self.confirm_s <= 0
                or self.carrier_hold_s < 0 or self.velocity_window_s <= 0
                or self.max_ball_speed_mps <= 0 or self.max_player_speed_mps <= 0
                or self.entry_relative_speed_mps <= 0
                or self.retain_relative_speed_mps < self.entry_relative_speed_mps
                or self.touch_radius_m <= 0
                or self.touch_radius_m > self.entry_radius_m
                or self.min_touch_velocity_change_mps <= 0
                or self.min_touch_speed_change_mps < 0
                or not 0 <= self.min_touch_direction_change_deg <= 180
                or self.touch_group_s <= 0 or self.touch_followthrough_s <= 0
                or self.deflection_context_s <= 0
                or self.deflection_relative_speed_mps <= 0
                or self.deflection_separation_gain_m <= 0):
            raise ValueError("Invalid contact settings or hysteresis ordering")


@dataclass
class InteractionFeatures:
    nearest_slot: np.ndarray
    ball_velocity: np.ndarray
    carrier_velocity: np.ndarray
    relative_speed: np.ndarray
    ball_motion_valid: np.ndarray
    ball_velocity_after: np.ndarray
    velocity_change: np.ndarray
    direction_change_deg: np.ndarray
    touch_candidate: np.ndarray
    carrier_velocity_after: np.ndarray
    post_touch_relative_speed: np.ndarray
    separation_gain_m: np.ndarray


@dataclass(frozen=True)
class TouchEvent:
    """One grouped ball-velocity impulse near a local player.

    ``deflection_like`` is an operational possession label: it means an
    opponent redirected the ball without establishing control and the prior
    team recovered it. It does not claim to know the player's intent.
    """

    team: int
    slot: int
    start_frame: int
    end_frame: int
    frame: int
    classification: str
    confidence: float


@dataclass
class ContactEvidence:
    team: np.ndarray
    slot: np.ndarray
    episode_id: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    interactions: InteractionFeatures
    touches: List[TouchEvent]


@dataclass(frozen=True)
class _TouchCandidateEvent:
    team: int
    slot: int
    start_frame: int
    end_frame: int
    frame: int


def _velocities(xy, half, fps, lag, max_speed):
    """Backward displacement over an entirely observed, jump-free window."""
    n = len(xy)
    velocity = np.full_like(xy, np.nan, dtype=np.float32)
    if n <= lag:
        return velocity
    finite = np.isfinite(xy).all(axis=-1)
    steps = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) * fps
    same_half = (half[1:] == half[:-1]) & (half[1:] != 0)
    shape = (n - 1,) + (1,) * (finite.ndim - 1)
    step_valid = (finite[1:] & finite[:-1] & (steps <= max_speed)
                  & same_half.reshape(shape))
    invalid = np.concatenate((np.zeros_like(finite[:1], dtype=np.int32),
                              np.cumsum(~step_valid, axis=0)), axis=0)
    valid = invalid[lag:] == invalid[:-lag]
    candidate = (xy[lag:] - xy[:-lag]) * (fps / lag)
    velocity[lag:] = np.where(valid[..., None], candidate, np.nan)
    return velocity


def derive_interactions(data: TrackingData, features, config=None):
    """Add local carrier slots and reliable metre/second velocity evidence.

    Missing coordinates, implausible steps, and half boundaries invalidate the
    entire velocity window. No interpolation or speed-unit guessing is used.
    """
    config = config or ContactConfig()
    if not np.isfinite(data.fps) or data.fps <= 0:
        raise ValueError("Frame rate must be finite and positive")
    n = len(data.half)
    ball_available = data.ball_available
    lag = max(1, round(config.velocity_window_s * data.fps))
    ball_velocity = _velocities(data.ball_xy, data.half, data.fps, lag,
                                config.max_ball_speed_mps)
    # The backward velocity at t + lag describes the same observed interval as
    # the forward velocity at t. Reusing it keeps both sides of a possible
    # touch subject to exactly the same gap, jump, and half-boundary checks.
    ball_velocity_after = np.full_like(ball_velocity, np.nan)
    ball_velocity_after[:-lag] = ball_velocity[lag:]
    player_velocity = _velocities(data.player_xy, data.half, data.fps, lag,
                                  config.max_player_speed_mps)
    player_velocity_after = np.full_like(player_velocity, np.nan)
    player_velocity_after[:-lag] = player_velocity[lag:]
    nearest_slot = np.full((n, 2), -1, dtype=np.int32)
    carrier_velocity = np.full((n, 2, 2), np.nan, dtype=np.float32)
    carrier_velocity_after = np.full((n, 2, 2), np.nan, dtype=np.float32)
    separation_gain = np.full((n, 2), np.nan, dtype=np.float32)
    for team in (0, 1):
        slots = np.flatnonzero(data.track_team == team)
        if len(slots) == 0:
            continue
        players = data.player_xy[:, slots]
        valid = np.isfinite(players).all(axis=2) & ball_available[:, None]
        distance = np.linalg.norm(players - data.ball_xy[:, None], axis=2)
        distance = np.where(valid, distance, np.inf)
        selected = slots[np.argmin(distance, axis=1)]
        known = np.isfinite(np.min(distance, axis=1))
        nearest_slot[known, team] = selected[known]
        frames = np.flatnonzero(known)
        carrier_velocity[known, team] = player_velocity[frames, selected[known]]
        carrier_velocity_after[known, team] = player_velocity_after[frames, selected[known]]
        future = frames + lag
        has_future = future < n
        future_frames = future[has_future]
        future_slots = selected[known][has_future]
        future_player = data.player_xy[future_frames, future_slots]
        future_ball = data.ball_xy[future_frames]
        future_valid = (np.isfinite(future_player).all(axis=1)
                        & np.isfinite(future_ball).all(axis=1)
                        & (data.half[future_frames] == data.half[frames[has_future]])
                        & (data.half[future_frames] != 0))
        future_distance = np.linalg.norm(future_player - future_ball, axis=1)
        current_distance = features.team_distance_m[frames[has_future], team]
        gain = np.where(future_valid, future_distance - current_distance, np.nan)
        separation_gain[frames[has_future], team] = gain
    relative = np.linalg.norm(carrier_velocity - ball_velocity[:, None], axis=2)
    post_touch_relative = np.linalg.norm(
        carrier_velocity_after - ball_velocity_after[:, None], axis=2)
    before_speed = np.linalg.norm(ball_velocity, axis=1)
    after_speed = np.linalg.norm(ball_velocity_after, axis=1)
    velocity_change = np.linalg.norm(ball_velocity_after - ball_velocity, axis=1)
    speed_change = np.abs(after_speed - before_speed)
    direction_change = np.full(n, np.nan, dtype=np.float32)
    directional = (np.isfinite(before_speed) & np.isfinite(after_speed)
                   & (before_speed > 0.25) & (after_speed > 0.25))
    dot = np.sum(ball_velocity * ball_velocity_after, axis=1)
    cosine = np.divide(dot, before_speed * after_speed,
                       out=np.zeros(n, dtype=np.float32), where=directional)
    direction_change[directional] = np.degrees(
        np.arccos(np.clip(cosine[directional], -1.0, 1.0)))

    impulse = (
        np.isfinite(velocity_change)
        & (velocity_change >= config.min_touch_velocity_change_mps)
        & ((speed_change >= config.min_touch_speed_change_mps)
           | (direction_change >= config.min_touch_direction_change_deg))
    )
    touch_candidate = np.zeros((n, 2), dtype=bool)
    for team in (0, 1):
        margin = features.distance_margin_m * (1 if team == 0 else -1)
        touch_candidate[:, team] = (
            impulse
            & (nearest_slot[:, team] >= 0)
            & np.isfinite(margin)
            & (margin >= config.entry_advantage_m)
            & (features.team_distance_m[:, team] <= config.touch_radius_m)
        )
    return InteractionFeatures(nearest_slot, ball_velocity, carrier_velocity,
                               relative, np.isfinite(ball_velocity).all(axis=1),
                               ball_velocity_after, velocity_change,
                               direction_change, touch_candidate,
                               carrier_velocity_after, post_touch_relative,
                               separation_gain)


def _group_touch_candidates(data, interactions, blocked, config):
    """Collapse overlapping velocity windows into one local touch event."""
    group_gap = max(1, round(config.touch_group_s * data.fps))
    events = []
    for team in (0, 1):
        frames = np.flatnonzero(interactions.touch_candidate[:, team] & ~blocked)
        if not len(frames):
            continue
        group = [int(frames[0])]
        for frame_value in frames[1:]:
            frame = int(frame_value)
            previous = group[-1]
            same_slot = (interactions.nearest_slot[frame, team]
                         == interactions.nearest_slot[previous, team])
            same_half = data.half[frame] == data.half[previous] != 0
            uninterrupted = not np.any(blocked[previous:frame + 1])
            if (frame - previous <= group_gap and same_slot and same_half
                    and uninterrupted):
                group.append(frame)
            else:
                events.append(_touch_event_from_group(team, group, interactions))
                group = [frame]
        events.append(_touch_event_from_group(team, group, interactions))
    return sorted(events, key=lambda event: (event.frame, event.team))


def _touch_event_from_group(team, frames, interactions):
    """Represent a grouped impulse by its largest reliable velocity change."""
    changes = interactions.velocity_change[frames]
    representative = int(frames[int(np.nanargmax(changes))])
    return _TouchCandidateEvent(
        team=team,
        slot=int(interactions.nearest_slot[representative, team]),
        start_frame=int(frames[0]),
        end_frame=int(frames[-1]),
        frame=representative,
    )


def _classify_touch_events(data, interactions, candidates, owner, observed,
                           blocked, config):
    """Classify grouped impulses from confirmed post-touch behaviour.

    Control requires a real confirmed episode backdated to the impulse. A
    deflection-like event requires an earlier opposing owner, rapid separation
    from the touching player, and confirmed recovery by that earlier team.
    Everything ambiguous remains unresolved.
    """
    follow = max(1, round(config.touch_followthrough_s * data.fps))
    context = max(1, round(config.deflection_context_s * data.fps))
    touches = []
    for event in candidates:
        frame, team = event.frame, event.team
        classification, confidence = "unresolved", 0.0
        if observed[frame] and owner[frame] == team:
            classification, confidence = "controlled", 1.0
        else:
            left = max(0, event.start_frame - context)
            prior_team = -1
            for prior in range(event.start_frame - 1, left - 1, -1):
                if (blocked[prior]
                        or data.half[prior] != data.half[event.start_frame]):
                    break
                if observed[prior] and owner[prior] >= 0:
                    prior_team = int(owner[prior])
                    break
            right = min(len(data.half), event.end_frame + follow + 1)
            post = slice(event.end_frame + 1, right)
            valid_window = (
                right > event.end_frame + 1
                and data.half[frame] != 0
                and np.all(data.half[event.start_frame:right] == data.half[frame])
                and not np.any(blocked[event.start_frame:right])
            )
            touching_control = np.any(observed[post] & (owner[post] == team))
            prior_recovers = prior_team >= 0 and np.any(
                observed[post] & (owner[post] == prior_team))
            relative = interactions.post_touch_relative_speed[frame, team]
            separation = interactions.separation_gain_m[frame, team]
            clear_departure = (
                np.isfinite(relative)
                and relative >= config.deflection_relative_speed_mps
                and np.isfinite(separation)
                and separation >= config.deflection_separation_gain_m
            )
            if (valid_window and prior_team != team and prior_recovers
                    and not touching_control and clear_departure):
                classification, confidence = "deflection_like", 0.6
        touches.append(TouchEvent(
            team=team,
            slot=event.slot,
            start_frame=event.start_frame,
            end_frame=event.end_frame,
            frame=frame,
            classification=classification,
            confidence=confidence,
        ))
    return touches


def detect_contacts(data, features, blocked=None, config=None):
    """Create offline control episodes with strict entry and looser retention.

    Confirmed onset is backdated only over its observed confirmation run.
    `observed` distinguishes actual contact evidence from a brief held carrier.
    A visible fast release ends contact immediately. One grouped impulse may
    seed an onset, but only ordinary low-relative-speed follow-through can
    confirm it. Unconfirmed redirections are classified conservatively after
    the contact pass as deflection-like or unresolved.
    """
    config = config or ContactConfig()
    n = len(data.half)
    interactions = derive_interactions(data, features, config)
    ball_available = data.ball_available
    blocked = np.zeros(n, dtype=bool) if blocked is None else np.asarray(blocked, dtype=bool)
    if blocked.shape != (n,):
        raise ValueError("blocked must have one value per frame")
    touch_candidates = _group_touch_candidates(data, interactions, blocked, config)
    touch_seed_team = np.full(n, -1, dtype=np.int8)
    for event in touch_candidates:
        # A positive team-distance advantage makes cross-team collisions
        # unlikely. If malformed evidence still collides, neither team seeds.
        if touch_seed_team[event.frame] == -1:
            touch_seed_team[event.frame] = event.team
        elif touch_seed_team[event.frame] != event.team:
            touch_seed_team[event.frame] = -2
    owner = np.full(n, -1, dtype=np.int8)
    carrier = np.full(n, -1, dtype=np.int32)
    episodes = np.full(n, -1, dtype=np.int32)
    observed = np.zeros(n, dtype=bool)
    confidence = np.zeros(n, dtype=np.float32)
    confirm = max(1, round(config.confirm_s * data.fps))
    hold = max(0, round(config.carrier_hold_s * data.fps))
    active_team = active_slot = active_episode = -1
    next_episode = 0
    last_observed = -1
    last_position = np.full(2, np.nan)
    candidate = -1
    candidate_frames = []
    candidate_seeded = False
    candidate_control_frames = 0
    previous_half = 0

    def compatible(frame, team, entry):
        slot = interactions.nearest_slot[frame, team]
        margin = features.distance_margin_m[frame] * (1 if team == 0 else -1)
        radius = config.entry_radius_m if entry else config.retain_radius_m
        advantage = config.entry_advantage_m if entry else config.retain_advantage_m
        speed = config.entry_relative_speed_mps if entry else config.retain_relative_speed_mps
        relative = interactions.relative_speed[frame, team]
        if (not entry and not np.isfinite(relative) and active_team == team
                and slot >= 0 and slot != active_slot and last_observed >= 0
                and interactions.ball_motion_valid[frame]):
            elapsed = (frame - last_observed) / data.fps
            local_velocity = (data.player_xy[frame, slot] - last_position) / elapsed
            if np.linalg.norm(local_velocity) <= config.max_player_speed_mps:
                relative = np.linalg.norm(interactions.ball_velocity[frame] - local_velocity)
        movement_compatible = (np.isfinite(relative) and relative <= speed)
        return (slot >= 0 and np.isfinite(margin) and margin >= advantage
                and features.team_distance_m[frame, team] <= radius
                and movement_compatible)

    def same_carrier(frame, team):
        slot = interactions.nearest_slot[frame, team]
        if slot < 0 or last_observed < 0:
            return False
        # Slot identity can change; local spatial continuity is authoritative.
        elapsed = (frame - last_observed) / data.fps
        step = np.linalg.norm(data.player_xy[frame, slot] - last_position)
        return step <= config.max_player_speed_mps * elapsed + 0.75

    for frame, half in enumerate(data.half):
        if half != previous_half or blocked[frame] or half == 0:
            active_team = active_slot = active_episode = -1
            last_observed = -1
            candidate, candidate_frames = -1, []
            candidate_seeded = False
            candidate_control_frames = 0
        previous_half = half
        if half == 0 or blocked[frame]:
            continue

        retained = (active_team >= 0 and compatible(frame, active_team, False)
                    and same_carrier(frame, active_team))
        if retained:
            slot = interactions.nearest_slot[frame, active_team]
            owner[frame], carrier[frame], episodes[frame] = active_team, slot, active_episode
            observed[frame], confidence[frame] = True, 1.0
            active_slot = slot
            last_observed, last_position = frame, data.player_xy[frame, slot].copy()
            candidate, candidate_frames = -1, []
            candidate_seeded = False
            candidate_control_frames = 0
            continue

        entry_team = int(features.nearest_team[frame])
        ordinary_entry = entry_team >= 0 and compatible(frame, entry_team, True)
        touch_seed = entry_team >= 0 and touch_seed_team[frame] == entry_team
        entry = ordinary_entry or touch_seed
        if entry:
            slot = int(interactions.nearest_slot[frame, entry_team])
            # Candidate confirmation must itself be one plausible local carrier.
            continuity = True
            if candidate_frames and candidate == entry_team:
                prev = candidate_frames[-1]
                prev_slot = interactions.nearest_slot[prev, entry_team]
                step = np.linalg.norm(data.player_xy[frame, slot] - data.player_xy[prev, prev_slot])
                continuity = step <= config.max_player_speed_mps / data.fps + 0.75
            if candidate == entry_team and continuity and ordinary_entry:
                candidate_frames.append(frame)
                candidate_control_frames += 1
            elif touch_seed:
                candidate, candidate_frames = entry_team, [frame]
                candidate_seeded = True
                candidate_control_frames = 0
            elif ordinary_entry:
                candidate, candidate_frames = entry_team, [frame]
                candidate_seeded = False
                candidate_control_frames = 1
            else:
                candidate, candidate_frames = -1, []
                candidate_seeded = False
                candidate_control_frames = 0
            confirmed = (candidate_control_frames >= confirm if candidate_seeded
                         else len(candidate_frames) >= confirm)
            if confirmed:
                active_team, active_slot, active_episode = entry_team, slot, next_episode
                next_episode += 1
                for start in candidate_frames:
                    owner[start] = entry_team
                    carrier[start] = interactions.nearest_slot[start, entry_team]
                    episodes[start] = active_episode
                    observed[start], confidence[start] = True, 1.0
                last_observed, last_position = frame, data.player_xy[frame, slot].copy()
                candidate, candidate_frames = -1, []
                candidate_seeded = False
                candidate_control_frames = 0
                continue
        else:
            candidate, candidate_frames = -1, []
            candidate_seeded = False
            candidate_control_frames = 0

        # Hold only uncertain observations, not a measured departure or opponent
        # control candidate. Team ownership is handled separately by the caller.
        visible_motion = interactions.ball_motion_valid[frame]
        uncertain = not ball_available[frame] or not visible_motion
        if (active_team >= 0 and uncertain and not entry
                and 0 < frame - last_observed <= hold):
            owner[frame], carrier[frame], episodes[frame] = active_team, active_slot, active_episode
            confidence[frame] = max(0.0, 1 - (frame - last_observed) / (hold + 1))
        else:
            active_team = active_slot = active_episode = -1

    touches = _classify_touch_events(
        data, interactions, touch_candidates, owner, observed, blocked, config)
    return ContactEvidence(owner, carrier, episodes, observed, confidence,
                           interactions, touches)
