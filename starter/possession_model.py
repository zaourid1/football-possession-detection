"""Section 3: conservative possession state machine.

Heuristic defaults are engineering assumptions, not fitted to public labels.
Run locally: python3 starter/possession_model.py (aggregate diagnostics only).
"""

from dataclasses import dataclass
from collections import deque

import numpy as np

from possession_data import load_tracking
from possession_features import extract_features
from games import GAME, GAMES
from load_smart_tagging import load as load_tags


@dataclass(frozen=True)
class ModelConfig:
    control_radius_m: float = 2.5
    distance_advantage_m: float = 0.75
    max_control_speed_mps: float = 12.0
    control_confirm_s: float = 0.32
    tag_confirm_s: float = 1.5
    stationary_speed_mps: float = 0.35
    stoppage_confirm_s: float = 2.0
    restart_speed_mps: float = 1.5
    restart_confirm_s: float = 0.2
    outside_margin_m: float = 0.5
    memory_s: float = 4.0
    evidence_window_s: float = 0.8
    phase_weight: float = 0.15
    evidence_lead_s: float = 0.16
    bridge_max_s: float = 6.0
    bridge_anchor_s: float = 0.5


def tag_evidence(events, n_frames):
    """Return phase-team votes and restart context on the video timeline.

    Attacking phases vote for their named team; defending phases vote for the
    opponent. Set pieces supply context only, never a whole interval of D.
    Conflicting phase votes cancel; N/A and unknown codes supply no team vote.
    """
    votes = np.zeros((n_frames, 2), dtype=np.int16)
    restart = np.zeros(n_frames, dtype=bool)
    teams = {"barcelona": 0, "atletic club": 1, "athletic club": 1}
    attack = {"build up", "progression", "attacking transition", "creating chances",
              "ball in final third", "ball in the box", "long ball"}
    defense = {"defensive transition", "defending in defensive third",
               "defending in middle third", "defending in attacking third"}
    for event in events:
        start = max(0, min(n_frames, int(event["start_frame"])))
        end = max(0, min(n_frames, int(event["end_frame"])))
        code = (event["code"] or "").strip().lower()
        if code == "set pieces":
            restart[start:end] = True
            continue
        team = teams.get((event["tags"].get("Team") or "").strip().lower())
        if team is not None and code in attack | defense:
            owner = 1 - team if code in defense else team
            votes[start:end, owner] += 1
    phase_team = np.full(n_frames, -1, dtype=np.int8)
    phase_team[votes[:, 0] > votes[:, 1]] = 0
    phase_team[votes[:, 1] > votes[:, 0]] = 1
    return phase_team, restart


def predict_possession(data, features, events, pitch, config=None, *, accumulated=False,
                       stationary_evidence=None):
    """Return labels and per-frame reasons; retain uncertainty as X.

    Default evidence is causal. Supplying centred stationary_evidence makes
    this an offline variant. Confirmation delays are explicit; no retrospective
    boundary adjustment is performed. Memory resets between graded halves.
    """
    config = config or ModelConfig()
    fps = data.fps
    n_frames = len(data.half)
    if stationary_evidence is not None:
        stationary_evidence = np.asarray(stationary_evidence, dtype=bool)
        if stationary_evidence.shape != (n_frames,):
            raise ValueError("stationary_evidence must have one value per frame")
    phase_team, restart_context = tag_evidence(events, n_frames)
    labels = np.full(n_frames, "X", dtype="<U1")
    reasons = np.full(n_frames, "ungraded", dtype="<U32")
    available = data.ball_available
    state = "X"
    candidate = -1
    candidate_source = ""
    candidate_run = stop_run = resume_run = 0
    last_support = -1
    previous_half = 0
    evidence = deque()
    totals = np.zeros(2, dtype=float)

    for frame, half in enumerate(data.half):
        if half != previous_half:
            state, candidate, candidate_source = "X", -1, ""
            candidate_run = stop_run = resume_run = 0
            last_support = -1
            evidence.clear()
            totals[:] = 0
        previous_half = half
        if half == 0:
            continue

        speed = features.ball_speed_mps[frame]
        x, y = data.ball_xy[frame]
        outside = available[frame] and (
            x < -config.outside_margin_m or x > pitch[0] + config.outside_margin_m
            or y < -config.outside_margin_m or y > pitch[1] + config.outside_margin_m)
        stationary_restart = (restart_context[frame] and np.isfinite(speed)
                              and speed <= config.stationary_speed_mps)
        if stationary_evidence is not None:
            stationary_restart = bool(stationary_evidence[frame])
        stop_run = stop_run + 1 if outside or stationary_restart else 0

        team = int(features.nearest_team[frame])
        margin = features.distance_margin_m[frame]
        # Both teams must be observed: a missing opponent is not an advantage.
        control = (team >= 0 and np.isfinite(margin)
                   and abs(margin) >= config.distance_advantage_m
                   and features.team_distance_m[frame, team] <= config.control_radius_m
                   and np.isfinite(speed) and speed <= config.max_control_speed_mps
                   and not outside)
        supported = team if control else int(phase_team[frame])
        source = "control" if control else "phase"
        if supported >= 0 and supported == candidate and source == candidate_source:
            candidate_run += 1
        elif supported >= 0:
            candidate, candidate_source, candidate_run = supported, source, 1
        else:
            candidate, candidate_source, candidate_run = -1, "", 0

        if accumulated:
            vote = np.zeros(2, dtype=float)
            if control:
                vote[team] = 1.0
                if phase_team[frame] == team:
                    vote[team] += config.phase_weight
            elif phase_team[frame] >= 0:
                vote[phase_team[frame]] = config.phase_weight
            evidence.append(vote)
            totals += vote
            if len(evidence) > max(1, round(config.evidence_window_s * fps)):
                totals -= evidence.popleft()

        moving = np.isfinite(speed) and speed >= config.restart_speed_mps and not outside
        # High-frequency jitter at a robustly stationary ball is not a restart.
        if stationary_evidence is not None and stationary_restart:
            moving = False
        resume_run = resume_run + 1 if moving else 0
        reason = "memory"
        if stop_run >= max(1, round(config.stoppage_confirm_s * fps)):
            state, reason = "D", "stoppage_evidence"
        else:
            threshold = config.control_confirm_s if source == "control" else config.tag_confirm_s
            confirmed = supported >= 0 and candidate_run >= max(1, round(threshold * fps))
            if accumulated:
                winner = int(np.argmax(totals))
                confirmed = (
                    supported == winner
                    and totals[winner] >= max(1, config.control_confirm_s * fps)
                    and totals[winner] - totals[1 - winner]
                    >= max(1, config.evidence_lead_s * fps)
                )
            can_resume = state != "D" or resume_run >= max(1, round(config.restart_confirm_s * fps))
            if confirmed and can_resume:
                state = "H" if supported == 0 else "A"
                reason = "confirmed_" + source
                last_support = frame
            elif state in ("H", "A"):
                owner = 0 if state == "H" else 1
                if supported == owner:
                    last_support = frame
                if frame - last_support > config.memory_s * fps:
                    state, reason = "X", "evidence_expired"
            elif state == "D" and resume_run >= max(1, round(config.restart_confirm_s * fps)):
                state, reason = "X", "restart_owner_unknown"
            elif state == "X":
                reason = "insufficient_evidence"
        labels[frame], reasons[frame] = state, reason

    return labels, reasons


if __name__ == "__main__":
    data = load_tracking()
    features = extract_features(data)
    labels, reasons = predict_possession(data, features, load_tags(), GAMES[GAME]["pitch"])
    graded = data.half != 0
    print("Graded label counts:", dict(zip(*np.unique(labels[graded], return_counts=True))))
    print("Decision reasons:", dict(zip(*np.unique(reasons[graded], return_counts=True))))
