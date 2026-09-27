"""
Pressing-structure analytics (Project 1) -- reusable, project-scoped to
`pressing_structure/` only. Builds on:
  - `pressing_structure.analytics.cleaned_tracking_view` (this project's
    own cleaned/segmented positions -- see that module's docstring)
  - `analytics.possession` (EXISTING PhD module, REUSED UNCHANGED --
    ball-carrier identity is never re-invented here; if the PhD
    possession model says UNCERTAIN/FREE_BALL/BALL_LOST, this module
    honors that and does not compute pressing features for that frame)
  - `analytics.pitch_control` (EXISTING PhD module, REUSED UNCHANGED --
    "pitch-control advantage around the ball" samples the SAME
    time-to-intercept surface already built and QA'd for the main
    dashboard, not a new model)
  - `outputs/analytics/testVideo1_120s_v3/{passes,turnovers}.parquet`
    (EXISTING, CANONICAL, DERIVED data -- read-only, used only to decide
    a press's OUTCOME; no new pass/turnover detection logic here)

See `pressing_structure/docs/METHODOLOGY.md` for exact definitions,
thresholds, and what is explicitly NOT implemented in this first pass.
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

from legacy_analytics.pitch_control import compute_pitch_control_grid  # noqa: E402
from legacy_analytics.possession import compute_possession, control_episodes  # noqa: E402

# --- thresholds (all explicit, all documented in METHODOLOGY.md) ---
PRESSER_RADII_CM = {"3m": 300.0, "5m": 500.0, "8m": 800.0}
LOCAL_BALANCE_RADIUS_CM = 1000.0          # 10m -- "local numerical balance"
SUPPORT_RADIUS_CM = 1500.0                # 15m -- attacking passing-option search radius
LANE_BLOCK_DISTANCE_CM = 200.0            # 2m -- a defender within this of the carrier-teammate
                                           # line segment is considered to block that lane
PRESS_TRIGGER_RADIUS_CM = PRESSER_RADII_CM["5m"]
PRESS_TRIGGER_CLOSING_SPEED_CM_S = 100.0  # defender must be closing at >=1 m/s to count as "pressing"
PRESS_MIN_PRESSERS = 1
PRESS_END_GRACE_FRAMES = 15               # 0.5s -- avoid flickering START/END on single-frame noise
OUTCOME_WINDOW_SEC = 3.0                  # how long after press end to look for a turnover/escape


def _possession_input_view(cleaned: pl.DataFrame) -> pl.DataFrame:
    """Renames cleaned_tracking_view's columns to match
    `analytics.possession.compute_possession`'s expected schema
    (display_x_pitch/display_y_pitch/display_team_id/
    display_object_type) -- a pure rename, no logic change, so the
    EXISTING possession module runs completely unmodified."""
    return cleaned.rename({
        "x_pitch": "display_x_pitch", "y_pitch": "display_y_pitch",
        "display_team_id": "display_team_id", "display_object_type": "display_object_type",
    })


def build_possession(cleaned: pl.DataFrame, ball: pl.DataFrame) -> pl.DataFrame:
    """Thin wrapper around the EXISTING `analytics.possession.
    compute_possession` -- reused, not reimplemented."""
    view = _possession_input_view(cleaned)
    return compute_possession(ball, view)


def _players_at_frame(cleaned: pl.DataFrame, frame: int) -> list[dict]:
    rows = cleaned.filter(
        (pl.col("frame") == frame) & pl.col("x_pitch").is_not_null() &
        pl.col("display_object_type").is_in(["player", "goalkeeper"])
    )
    return rows.to_dicts()


def _passing_lane_open(carrier_xy, teammate_xy, defenders: list[dict]) -> bool:
    """A lane is 'open' if no defender sits within LANE_BLOCK_DISTANCE_CM
    of the straight segment from carrier to teammate -- a simple,
    disclosed geometric proxy, NOT a modeled interception probability."""
    cx, cy = carrier_xy
    tx, ty = teammate_xy
    seg = np.array([tx - cx, ty - cy])
    seg_len2 = seg.dot(seg)
    if seg_len2 == 0:
        return True
    for d in defenders:
        p = np.array([d["x_pitch"] - cx, d["y_pitch"] - cy])
        t = np.clip(p.dot(seg) / seg_len2, 0.0, 1.0)
        closest = t * seg
        dist = np.hypot(p[0] - closest[0], p[1] - closest[1])
        if dist < LANE_BLOCK_DISTANCE_CM:
            return False
    return True


def compute_pressing_frame_features(cleaned: pl.DataFrame, poss_row: dict, frame: int) -> dict | None:
    """One row of pressing features for `frame`, or None if no
    confident ball carrier exists (possession_state != CONTROLLED) --
    never guessed."""
    if poss_row["possession_state"] != "CONTROLLED" or poss_row["possessing_track_id"] is None:
        return None
    carrier_id = poss_row["possessing_track_id"]
    carrier_team = poss_row["possessing_team_id"]
    if carrier_team is None:
        return None

    players = _players_at_frame(cleaned, frame)
    carrier_rows = [p for p in players if p["track_id"] == carrier_id]
    if not carrier_rows:
        return None
    carrier = carrier_rows[0]
    cx, cy = carrier["x_pitch"], carrier["y_pitch"]

    defenders = [p for p in players if p["display_team_id"] is not None and p["display_team_id"] != carrier_team]
    attackers = [p for p in players if p["display_team_id"] == carrier_team and p["track_id"] != carrier_id]
    if not defenders:
        return None  # can't assess pressing with no known opposing players this frame

    dists = []
    for d in defenders:
        dist = float(np.hypot(d["x_pitch"] - cx, d["y_pitch"] - cy))
        closing = None
        if d.get("vx_cm_s") is not None and d.get("vy_cm_s") is not None and dist > 0:
            # component of defender velocity directed toward the carrier
            dx, dy = (cx - d["x_pitch"]) / dist, (cy - d["y_pitch"]) / dist
            if carrier.get("vx_cm_s") is not None and carrier.get("vy_cm_s") is not None and d.get("motion_valid", True) and carrier.get("motion_valid", True):
                closing = (d["vx_cm_s"] - carrier["vx_cm_s"]) * dx + (d["vy_cm_s"] - carrier["vy_cm_s"]) * dy
        dists.append({"track_id": d["track_id"], "dist_cm": dist, "closing_speed_cm_s": closing})
    dists.sort(key=lambda r: r["dist_cm"])
    nearest = dists[0]

    n3 = sum(1 for r in dists if r["dist_cm"] <= PRESSER_RADII_CM["3m"])
    n5 = sum(1 for r in dists if r["dist_cm"] <= PRESSER_RADII_CM["5m"])
    n8 = sum(1 for r in dists if r["dist_cm"] <= PRESSER_RADII_CM["8m"])

    n_def_local = sum(1 for r in dists if r["dist_cm"] <= LOCAL_BALANCE_RADIUS_CM)
    n_att_local = sum(1 for p in attackers
                       if np.hypot(p["x_pitch"] - cx, p["y_pitch"] - cy) <= LOCAL_BALANCE_RADIUS_CM)
    numerical_balance = n_att_local - n_def_local  # positive = attacking side outnumbers locally

    support = [p for p in attackers if np.hypot(p["x_pitch"] - cx, p["y_pitch"] - cy) <= SUPPORT_RADIUS_CM]
    n_open_lanes = sum(1 for p in support if _passing_lane_open((cx, cy), (p["x_pitch"], p["y_pitch"]), defenders))
    n_blocked_lanes = len(support) - n_open_lanes

    # defending team width/depth (global, all currently-visible defending outfield players)
    def_xs = [d["x_pitch"] for d in defenders if d["display_object_type"] == "player"]
    def_ys = [d["y_pitch"] for d in defenders if d["display_object_type"] == "player"]
    def_width_cm = max(def_ys) - min(def_ys) if len(def_ys) >= 2 else None
    def_depth_cm = max(def_xs) - min(def_xs) if len(def_xs) >= 2 else None

    # local compactness: mean pairwise distance among defenders within 15m of the carrier
    local_defs = [d for d in defenders if np.hypot(d["x_pitch"] - cx, d["y_pitch"] - cy) <= SUPPORT_RADIUS_CM]
    compactness_cm = None
    if len(local_defs) >= 2:
        pd = [np.hypot(a["x_pitch"] - b["x_pitch"], a["y_pitch"] - b["y_pitch"])
              for i, a in enumerate(local_defs) for b in local_defs[i + 1:]]
        compactness_cm = float(np.mean(pd))

    # pitch control AT the carrier's own location -- reuses the EXISTING
    # PhD analytics.pitch_control model unmodified; sampled at the
    # nearest grid cell to the carrier's position.
    team_a = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0}
              for p in players if p["display_team_id"] == 0]
    team_b = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0}
              for p in players if p["display_team_id"] == 1]
    pc = compute_pitch_control_grid(team_a, team_b)
    pitch_control_at_carrier = None
    if pc["valid"]:
        col = int(np.clip(cx / 12000.0 * pc["grid"].shape[1], 0, pc["grid"].shape[1] - 1))
        row = int(np.clip(cy / 7000.0 * pc["grid"].shape[0], 0, pc["grid"].shape[0] - 1))
        control_a = float(pc["grid"][row, col])
        pitch_control_at_carrier = control_a if carrier_team == 0 else 1.0 - control_a

    return {
        "frame": frame,
        "ball_carrier_track_id": carrier_id, "ball_carrier_team_id": carrier_team,
        "ball_carrier_x": cx, "ball_carrier_y": cy,
        "nearest_defender_track_id": nearest["track_id"],
        "nearest_defender_distance_cm": nearest["dist_cm"],
        "nearest_defender_closing_speed_cm_s": nearest["closing_speed_cm_s"],
        "n_pressers_3m": n3, "n_pressers_5m": n5, "n_pressers_8m": n8,  # deprecated aliases
        "nearby_opponents_3m": n3, "nearby_opponents_5m": n5, "nearby_opponents_8m": n8,
        "n_attack_support_local": n_att_local, "n_defenders_local": n_def_local,
        "local_numerical_balance": numerical_balance,
        "n_passing_options": len(support), "n_open_lanes": n_open_lanes, "n_blocked_lanes": n_blocked_lanes,
        "defending_team_width_cm": def_width_cm, "defending_team_depth_cm": def_depth_cm,
        "local_compactness_cm": compactness_cm,
        "pitch_control_for_carrier_team": pitch_control_at_carrier,
        "nearest_opponent_distance_proxy_cm": nearest["dist_cm"],
        "escape_space_score_cm": nearest["dist_cm"],  # deprecated alias only  # explicit proxy, see METHODOLOGY.md
        "possession_confidence": poss_row["possession_confidence"],
    }


def build_pressing_features(cleaned: pl.DataFrame, ball: pl.DataFrame, poss: pl.DataFrame) -> pl.DataFrame:
    """Per-frame pressing feature table -- only for frames with a
    confident ball carrier AND at least one visible opposing player.
    Frames failing either condition are simply absent (never
    fabricated)."""
    poss_by_frame = {r["frame"]: r for r in poss.to_dicts()}
    rows = []
    for frame in sorted(poss_by_frame.keys()):
        feat = compute_pressing_frame_features(cleaned, poss_by_frame[frame], frame)
        if feat is not None:
            rows.append(feat)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def detect_pressing_events(features: pl.DataFrame, fps: float = 30.0,
                            trigger_radius_cm: float = PRESS_TRIGGER_RADIUS_CM,
                            trigger_closing_speed: float = PRESS_TRIGGER_CLOSING_SPEED_CM_S,
                            min_pressers: int = PRESS_MIN_PRESSERS,
                            grace_frames: int = PRESS_END_GRACE_FRAMES) -> pl.DataFrame:
    """Rule-based PRESS START -> ACTIVE -> END segmentation. A frame
    counts as 'under press' if it has >= min_pressers defenders within
    trigger_radius_cm of the carrier, AND the nearest one has a positive
    closing speed >= trigger_closing_speed_cm_s (moving TOWARD the
    carrier, not just standing nearby). `grace_frames` bridges brief
    single-frame dropouts so a real press isn't sliced into a dozen
    micro-episodes by measurement jitter -- this is NOT the same as
    fabricating data; the underlying features are already present in
    every bridged frame, only the episode boundary is smoothed."""
    if features.height == 0:
        return pl.DataFrame()
    rows = features.sort("frame").to_dicts()
    under_press = []
    for r in rows:
        is_pressed = (r["n_pressers_5m"] >= min_pressers and
                      (r["nearest_defender_closing_speed_cm_s"] or 0) >= trigger_closing_speed and
                      r["nearest_defender_distance_cm"] <= trigger_radius_cm)
        under_press.append(is_pressed)

    episodes = []
    i = 0
    n = len(rows)
    while i < n:
        if not under_press[i]:
            i += 1
            continue
        start = i
        j = i
        while j < n:
            if j > start and (rows[j]["frame"] - rows[j - 1]["frame"] > grace_frames or
                              rows[j]["ball_carrier_team_id"] != rows[start]["ball_carrier_team_id"]):
                break
            if under_press[j]:
                j += 1
                continue
            # look ahead within grace_frames for a resumption
            k = j
            gap_ok = False
            while k < n and rows[k]["frame"] - rows[j - 1]["frame"] <= grace_frames and rows[k]["ball_carrier_team_id"] == rows[start]["ball_carrier_team_id"]:
                if under_press[k]:
                    gap_ok = True
                    break
                k += 1
            if gap_ok:
                j = k + 1
                continue
            break
        end = j - 1
        episodes.append((start, end))
        i = j

    event_rows = []
    for eid, (s, e) in enumerate(episodes):
        seg = rows[s:e + 1]
        event_rows.append({
            "press_event_id": eid,
            "start_frame": seg[0]["frame"], "end_frame": seg[-1]["frame"],
            "start_time_sec": seg[0]["frame"] / fps, "end_time_sec": seg[-1]["frame"] / fps,
            "duration_sec": (seg[-1]["frame"] - seg[0]["frame"] + 1) / fps,
            "carrier_team_id": seg[0]["ball_carrier_team_id"],
            "n_frames_in_episode": len(seg),
            "mean_pressers_5m": float(np.mean([r["n_pressers_5m"] for r in seg])),
            "min_nearest_defender_distance_cm": float(min(r["nearest_defender_distance_cm"] for r in seg)),
            "mean_local_numerical_balance": float(np.mean([r["local_numerical_balance"] for r in seg])),
        })
    return pl.DataFrame(event_rows) if event_rows else pl.DataFrame()


def classify_press_outcomes(events: pl.DataFrame, passes: pl.DataFrame, turnovers: pl.DataFrame,
                             fps: float = 30.0, window_sec: float = OUTCOME_WINDOW_SEC) -> pl.DataFrame:
    """Outcome is decided ONLY from the EXISTING, already-validated
    passes/turnovers datasets -- never re-derived from raw positions.
    SUCCESS: a turnover involving the pressed team as source occurs
    within [end_frame, end_frame + window]. FAILED: the pressed team
    completes a pass (in passes.parquet) whose end_frame falls in that
    same window (they kept the ball -- escaped the press). Otherwise
    UNCERTAIN -- never forced."""
    if events.height == 0:
        return events
    out = []
    for e in events.to_dicts():
        window_end = e["end_frame"] + window_sec * fps
        team = e["carrier_team_id"]
        to = turnovers.filter(
            (pl.col("source_team_id") == team) &
            (pl.col("start_frame") >= e["start_frame"]) & (pl.col("start_frame") <= window_end)
        )
        esc = passes.filter(
            (pl.col("team_id") == team) &
            (pl.col("end_frame") >= e["end_frame"]) & (pl.col("end_frame") <= window_end)
        )
        if to.height > 0:
            outcome = "SUCCESS"
        elif esc.height > 0:
            outcome = "FAILED"
        else:
            outcome = "UNCERTAIN"
        out.append({**e, "outcome": outcome})
    return pl.DataFrame(out)


# ============================================================================
# V2 outcome taxonomy (pressing_structure V2). The old SUCCESS/FAILED/
# UNCERTAIN classifier above is UNCHANGED and still used wherever v1
# outputs are read -- this is a NEW function, not a rewrite, so v1's
# `pressing_dashboard.mp4` / `pressing_events.csv` remain reproducible
# exactly as before.
# ============================================================================
FORWARD_PROGRESS_THRESHOLD_CM = 300.0
# ^ 3m -- a round, disclosed number, not fit to the data. A completed
# pass whose forward-direction displacement (toward the OPPONENT's
# goal, i.e. away from the pressed team's own goal) exceeds this in
# either direction is called BACKWARD/FORWARD; between +-300cm is
# treated as genuinely lateral (limited progress either way).

PRESS_OUTCOME_V2_STATES = ("BALL_REGAIN", "FORCED_BACKWARD", "FORCED_LATERAL", "ESCAPED_PRESS", "UNCERTAIN")
SUCCESSFUL_PRESS_STATES = ("BALL_REGAIN", "FORCED_BACKWARD", "FORCED_LATERAL")
# ^ Documented tactical interpretation (not a silent redefinition of
# "success"): a press that either wins the ball back outright, or denies
# the pressed team clean forward progress (forces them backward or
# sideways), counts as a tactically successful pressing action even
# when possession itself isn't immediately regained. ESCAPED_PRESS is
# the only outcome NOT counted as successful, since the pressed team
# both kept the ball AND advanced through/past the pressing zone.


def classify_press_outcomes_v2(events: pl.DataFrame, passes: pl.DataFrame, turnovers: pl.DataFrame,
                                fps: float = 30.0, window_sec: float = OUTCOME_WINDOW_SEC,
                                forward_threshold_cm: float = FORWARD_PROGRESS_THRESHOLD_CM) -> pl.DataFrame:
    """Refined, football-aware outcome taxonomy (Part 1A of the V2
    brief). Still decided ONLY from the EXISTING passes/turnovers data
    -- no new ball/position re-derivation.

    - BALL_REGAIN: a turnover with the pressed team as source starts
      within [press end, press end + window_sec]. (Same evidence as v1's
      SUCCESS.)
    - FORCED_BACKWARD / FORCED_LATERAL / ESCAPED_PRESS: the pressed team
      completes a pass (source_team_id == pressed team) ending in that
      same window. The pass's forward-direction displacement
      (`end_x`-`start_x` reoriented so positive = toward the OPPONENT's
      goal) decides which: <= -forward_threshold_cm -> FORCED_BACKWARD;
      >= +forward_threshold_cm -> ESCAPED_PRESS; in between ->
      FORCED_LATERAL.
    - UNCERTAIN: neither a turnover nor a completed pass is found in the
      window -- e.g. the ball went out of play, the clip ended, or
      detection simply didn't pick anything up. Never forced.

    If BOTH a turnover and an escaping pass appear in the window (should
    be rare -- they're mutually exclusive events), the turnover is
    treated as authoritative (it means the ball changed teams, so any
    "pass" the old detector also found there is very likely a
    misattributed/overlapping candidate, not a real escape)."""
    if events.height == 0:
        return events
    out = []
    for e in events.to_dicts():
        window_end = e["end_frame"] + window_sec * fps
        team = e["carrier_team_id"]
        from tactical_shared.coordinates import DEFAULT_PITCH
        forward_sign = DEFAULT_PITCH.attacking_sign(team, e["end_frame"])

        to = turnovers.filter(
            (pl.col("source_team_id") == team) &
            (pl.col("start_frame") >= e["start_frame"]) & (pl.col("start_frame") <= window_end)
        )
        esc = passes.filter(
            (pl.col("team_id") == team) &
            (pl.col("end_frame") >= e["end_frame"]) & (pl.col("end_frame") <= window_end)
        ).sort("end_frame")

        forward_progress_cm = None
        if to.height > 0:
            outcome_v2 = "BALL_REGAIN"
        elif esc.height > 0:
            p = esc.to_dicts()[0]
            if None not in (p["start_x"], p["end_x"]):
                forward_progress_cm = forward_sign * (p["end_x"] - p["start_x"])
                if forward_progress_cm <= -forward_threshold_cm:
                    outcome_v2 = "FORCED_BACKWARD"
                elif forward_progress_cm >= forward_threshold_cm:
                    outcome_v2 = "ESCAPED_PRESS"
                else:
                    outcome_v2 = "FORCED_LATERAL"
            else:
                outcome_v2 = "UNCERTAIN"  # pass found but geometry incomplete -- don't guess
        else:
            outcome_v2 = "UNCERTAIN"

        out.append({**e, "outcome_v2": outcome_v2,
                    "forward_progress_cm": forward_progress_cm,
                    "is_successful_press": outcome_v2 in SUCCESSFUL_PRESS_STATES if outcome_v2 != "UNCERTAIN" else None})
    return pl.DataFrame(out)
