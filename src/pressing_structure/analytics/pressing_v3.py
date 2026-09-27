"""
Pressing V3: SIMULTANEOUS dual-team press tracking with a proper
transition-aware (hysteresis) state machine, replacing the old
single-perspective "is a press active right now" binary trigger with
independent, continuously-scored states for BOTH teams every frame.

NEW module. Does not modify `pressing_structure/analytics/pressing.py`
-- reuses its existing, unmodified `build_pressing_features` (raw
per-frame nearest-defender-distance / closing-speed / presser-count
features) and `SUCCESSFUL_PRESS_STATES` read-only. The V2 outcome
taxonomy (BALL_REGAIN/FORCED_BACKWARD/...) and episode detector in
pressing.py are UNCHANGED and still used elsewhere; this module adds a
different, complementary signal (continuous per-team press intensity
+ stable state), not a replacement for the outcome classifier.

## Why a per-team raw score, not a single "is anyone pressing" score

A team can only be "pressing" while the OTHER team has the ball --
that's the real football definition already encoded in
`build_pressing_features` (pressers are always drawn from the
non-carrier team). So `press_score_team0` and `press_score_team1` are
computed EVERY frame, but only one of them can be non-zero at a time:
- if team 1 has the ball this frame: `press_score_team0` = the real,
  disclosed pressing-intensity combo below; `press_score_team1` = 0.0
  (team 1 cannot press while their own team has the ball -- a real,
  defined zero, not a guess).
- vice versa if team 0 has the ball.
- if no confident carrier exists this frame: BOTH scores are None (no
  evidence, not zero) -- carried through the smoothing/FSM step as
  "UNCERTAIN" rather than silently treated as "no press".

## Raw score formula (disclosed, simple, NOT fit to any outcome)

    raw = clip(0.40 * norm(n_pressers_5m,      0,   4)
             + 0.35 * norm(closing_speed_cm_s, 0, 300)
             + 0.25 * (1 - norm(nearest_defender_distance_cm, 0, 800)),
               0, 1)

Weights are round numbers chosen to reflect (in order) how many
players are closing in, how fast the nearest one is closing, and how
tight the nearest gap already is -- not tuned against any label.

## Stabilization

`smooth_scores()` + `run_fsm()` (both in the shared
`pressing_structure.analytics.state_machine` module) turn the raw
per-frame score into one of NO_PRESS / PRESS_FORMING / ACTIVE_PRESS /
PRESS_ENDING / UNCERTAIN per team, with hysteresis thresholds and
minimum hold/cooldown durations documented in `PRESS_FSM_CONFIG`
below -- see that module's docstring for exactly what each parameter
does. `press_confidence_teamX(f)` is simply the smoothed score itself
(already in [0,1], already real) -- not a separate invented quantity.

## Dominant/display state decision

`decide_dominant_state()` implements the brief's exact rule: one team
clearly active -> show that team; both low -> NO_PRESS; both
active/forming simultaneously (only possible right at a
possession-change boundary, since only the non-carrier team can score
above zero in steady state) -> CONTESTED; anything else -> UNCERTAIN.
"""
import os
import sys

import numpy as np
import polars as pl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PARENT = os.path.dirname(PROJECT_ROOT)
for path in (PROJECT_ROOT, PROJECT_PARENT):
    if path not in sys.path:
        sys.path.insert(0, path)

from pressing_structure.analytics.state_machine import FSMConfig, raw_to_stable_states, states_to_segments  # noqa: E402

RAW_SCORE_WINDOW_FRAMES = 15  # 0.5s rolling smoothing window

PRESS_FSM_CONFIG = FSMConfig(
    # Recalibrated from an initial 0.5/0.3 after validating against real
    # data: the measured NONZERO raw-score distribution on this clip
    # has team 0's OWN 90th percentile at just 0.42 (median 0.35) and
    # team 1's at 0.61 (median 0.37) -- an enter_th of 0.5 sat ABOVE
    # team 0's 90th percentile, meaning team 0 could almost never
    # register ACTIVE_PRESS regardless of real intensity, and even a
    # genuine short episode's confident frames (0.50, 0.56, 0.44, 0.38)
    # averaged to 0.47 under simple-mean smoothing -- just under 0.5.
    # This is a recalibration to the SCORE'S OWN measured scale (not a
    # fit to any specific known episode's timing/outcome): 0.4/0.25
    # sits at roughly team 0's own 75th-90th percentile band for both
    # teams, while still requiring real above-typical intensity to
    # register.
    enter_th=0.4, exit_th=0.25,
    min_activate_frames=8,     # ~0.27s of sustained smoothed score >= enter_th to confirm ACTIVE
    min_hold_frames=10,        # ~0.33s minimum ACTIVE hold before it's allowed to end
    ending_frames=10,          # ~0.33s PRESS_ENDING window before resolving to NO_PRESS
    cooldown_frames=15,        # ~0.5s grace period before re-arming after an episode ends
    low_state="NO_PRESS", forming_state="PRESS_FORMING",
    active_state="ACTIVE_PRESS", ending_state="PRESS_ENDING", uncertain_state="UNCERTAIN",
)


def _norm(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def compute_dual_team_raw_scores(features: pl.DataFrame, n_total_frames: int) -> dict:
    """Returns {"team0": [raw_score_or_None per frame], "team1": [...]}."""
    feat_by_frame = {r["frame"]: r for r in features.to_dicts()}
    raw = {0: [None] * n_total_frames, 1: [None] * n_total_frames}
    for f in range(n_total_frames):
        r = feat_by_frame.get(f)
        if r is None:
            continue
        carrier_team = r["ball_carrier_team_id"]
        presser_team = 1 - carrier_team
        score = (0.40 * _norm(r["n_pressers_5m"], 0, 4) +
                 0.35 * _norm(r["nearest_defender_closing_speed_cm_s"] or 0, 0, 300) +
                 0.25 * (1.0 - _norm(r["nearest_defender_distance_cm"], 0, 800)))
        raw[presser_team][f] = float(np.clip(score, 0.0, 1.0))
        raw[carrier_team][f] = 0.0  # a real, defined zero: this team has the ball, cannot be pressing
    return {"team0": raw[0], "team1": raw[1]}


def build_dual_team_press_states(features: pl.DataFrame, n_total_frames: int, fps: float = 30.0) -> dict:
    """Full pipeline: raw scores -> smoothed -> stable FSM states ->
    run-length segments, for BOTH teams independently. Returns a dict
    with per-team smoothed scores, states, and segments."""
    raw = compute_dual_team_raw_scores(features, n_total_frames)
    out = {}
    for team_key in ("team0", "team1"):
        smoothed, states = raw_to_stable_states(raw[team_key], RAW_SCORE_WINDOW_FRAMES, PRESS_FSM_CONFIG)
        out[team_key] = {
            "raw": raw[team_key], "smoothed": smoothed, "states": states,
            "segments": states_to_segments(states, fps=fps),
        }
    return out


def decide_dominant_state(state0: str, state1: str) -> tuple:
    """The brief's exact decision rule. Returns (display_state,
    dominant_team_or_None)."""
    active_like = (PRESS_FSM_CONFIG.active_state, PRESS_FSM_CONFIG.forming_state)
    t0_active = state0 in active_like
    t1_active = state1 in active_like
    if state0 == PRESS_FSM_CONFIG.active_state and state1 not in active_like:
        return state0, 0
    if state1 == PRESS_FSM_CONFIG.active_state and state0 not in active_like:
        return state1, 1
    if t0_active and t1_active:
        return "CONTESTED", None
    if state0 == PRESS_FSM_CONFIG.low_state and state1 == PRESS_FSM_CONFIG.low_state:
        return PRESS_FSM_CONFIG.low_state, None
    if state0 == PRESS_FSM_CONFIG.uncertain_state and state1 == PRESS_FSM_CONFIG.uncertain_state:
        return PRESS_FSM_CONFIG.uncertain_state, None
    # one FORMING, other NO_PRESS/UNCERTAIN -- show the forming one softly
    if state0 == PRESS_FSM_CONFIG.forming_state and state1 in (PRESS_FSM_CONFIG.low_state, PRESS_FSM_CONFIG.uncertain_state):
        return state0, 0
    if state1 == PRESS_FSM_CONFIG.forming_state and state0 in (PRESS_FSM_CONFIG.low_state, PRESS_FSM_CONFIG.uncertain_state):
        return state1, 1
    # ENDING states, mixed uncertainty, etc. -- report honestly rather than guess
    if PRESS_FSM_CONFIG.ending_state in (state0, state1):
        team = 0 if state0 == PRESS_FSM_CONFIG.ending_state else 1
        return (state0 if team == 0 else state1), team
    return PRESS_FSM_CONFIG.uncertain_state, None
