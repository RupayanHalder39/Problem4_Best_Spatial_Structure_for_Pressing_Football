"""
Research-phase pressing episode feature extraction (Part 3 of the
2026-09-08 research-phase kickoff). Pure feature aggregation only --
reuses the SAME already-existing, already-tested analytics functions
(`pressing_v4.frame_features`, `pressure_field.team_pressure_field`,
`escape_directions`, `passing_lane_metrics`, `pressure_gradient`,
`contour_coverage_fractions`) that the frozen dashboard itself uses.
No pressure-field formula or FSM threshold is touched or reimplemented
here.

One feature row per press EPISODE (`pressing_v4.build_pressing_v4`'s
own `episodes` list), anchored at the episode's `onset_frame` -- the
real, FSM-detected moment pressure first crosses the `enter` threshold,
not an arbitrarily chosen frame. Features are grouped exactly as
specified in `RESEARCH_DATASET_PLAN.md`: GEOMETRY, MOTION,
PASSING/ESCAPE, FIELD, CONTEXT.

Every feature here is computable from CURRENT tracking (positions,
velocities, motion_valid) -- nothing requiring future data, formation
metadata, or event annotations that don't already exist.
"""
import numpy as np

from pressing_structure.analytics.pressing_v4 import relative_closing, evidence_quality, frame_features
from pressing_structure.analytics.pressure_field import (
    team_pressure_field, escape_directions, passing_lane_metrics,
    pressure_gradient, sample_field, contour_coverage_fractions,
)
from tactical_shared.coordinates import DEFAULT_PITCH

try:
    from scipy.spatial import ConvexHull
    _HAVE_SCIPY_HULL = True
except ImportError:  # pragma: no cover -- scipy is a declared project dependency; documented fallback only
    _HAVE_SCIPY_HULL = False

RADIUS_BANDS_CM = (300.0, 500.0, 800.0)  # 3m/5m/8m -- reuses this project's own already-used bands (frame_features' 3m/5m, PROXIMITY_WEIGHT's own 800cm reference), never a new radius choice
ESCAPE_RADIUS_CM = 800.0
LOW_PRESSURE_THRESHOLD = 0.35  # reuses pressure_field.CONTOUR_LOW's own calibrated meaning (~1.5s single-defender reach) -- not a new number invented for this module
N_ESCAPE_CANDIDATES = 8


def _eligible(players_this_frame):
    return [p for p in players_this_frame if p.get("x_pitch") is not None and p["display_object_type"] in ("player", "goalkeeper")]


def _dist(a, b):
    return float(np.hypot(a["x_pitch"] - b["x_pitch"], a["y_pitch"] - b["y_pitch"]))


# ---------------------------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------------------------
def geometry_features(carrier, defenders):
    """`defenders`: the pressing team's eligible players this frame
    (never pre-filtered to only `participant_ids` -- distance bands are
    computed against every real defender, participant status is a
    SEPARATE, FSM-defined concept re-exposed as `n_participants` below).
    Returns None-safe (an empty/insufficient defender list yields all
    None, never a fabricated 0)."""
    if not defenders:
        return dict(nearest_presser_distance_cm=None, second_presser_distance_cm=None,
                     mean_presser_distance_cm=None, n_within_3m=None, n_within_5m=None, n_within_8m=None,
                     first_presser_angle_deg=None, second_presser_angle_deg=None,
                     local_numerical_superiority=None, compactness_cm=None, convex_hull_area_cm2=None)
    dists = sorted(((_dist(carrier, d), d) for d in defenders), key=lambda t: t[0])
    nearest_d, nearest = dists[0]
    second_d = dists[1][0] if len(dists) >= 2 else None

    def angle_deg(d):
        dx, dy = d["x_pitch"] - carrier["x_pitch"], d["y_pitch"] - carrier["y_pitch"]
        return float(np.degrees(np.arctan2(dy, dx)))

    hull_area = None
    if _HAVE_SCIPY_HULL and len(defenders) >= 3:
        pts = np.array([[d["x_pitch"], d["y_pitch"]] for d in defenders])
        try:
            hull_area = float(ConvexHull(pts).volume)  # 2D ConvexHull.volume IS the enclosed area (scipy convention)
        except Exception:
            hull_area = None  # degenerate (collinear) point set -- honest gap, never a fabricated area

    local = [d for dist, d in dists if dist <= RADIUS_BANDS_CM[2]]
    compactness = (float(np.mean([_dist(a, b) for i, a in enumerate(local) for b in local[i + 1:]]))
                   if len(local) >= 2 else None)

    return dict(
        nearest_presser_distance_cm=nearest_d,
        second_presser_distance_cm=second_d,
        mean_presser_distance_cm=float(np.mean([d for d, _ in dists])),
        n_within_3m=sum(d <= RADIUS_BANDS_CM[0] for d, _ in dists),
        n_within_5m=sum(d <= RADIUS_BANDS_CM[1] for d, _ in dists),
        n_within_8m=sum(d <= RADIUS_BANDS_CM[2] for d, _ in dists),
        first_presser_angle_deg=angle_deg(nearest),
        second_presser_angle_deg=angle_deg(dists[1][1]) if len(dists) >= 2 else None,
        local_numerical_superiority=None,  # filled by caller from frame_features' support_minus_opponents (identical convention, never recomputed differently)
        compactness_cm=compactness,
        convex_hull_area_cm2=hull_area,
    )


# ---------------------------------------------------------------------------
# MOTION
# ---------------------------------------------------------------------------
def motion_features(carrier, defenders, participant_ids):
    """`participant_ids`: the FSM's own real `participant_ids` (opponents
    within 5m closing at >=100cm/s, from `frame_features` -- reused,
    never recomputed with a different rule)."""
    nearest = min(defenders, key=lambda d: _dist(carrier, d)) if defenders else None
    nearest_closing = relative_closing(nearest, carrier) if nearest is not None else None
    presser_vel_toward_carrier = None
    if nearest is not None and nearest.get("motion_valid") and _dist(carrier, nearest) > 0:
        dx, dy = carrier["x_pitch"] - nearest["x_pitch"], carrier["y_pitch"] - nearest["y_pitch"]
        d = _dist(carrier, nearest)
        presser_vel_toward_carrier = float((nearest.get("vx_cm_s", 0.0) * dx + nearest.get("vy_cm_s", 0.0) * dy) / d)

    participants = [d for d in defenders if d["track_id"] in (participant_ids or [])]
    convergence_vals = [c for c in (relative_closing(p, carrier) for p in participants) if c is not None]
    support_convergence = float(np.mean(convergence_vals)) if convergence_vals else None

    carrier_speed = None
    if carrier.get("motion_valid") and carrier.get("vx_cm_s") is not None and carrier.get("vy_cm_s") is not None:
        carrier_speed = float(np.hypot(carrier["vx_cm_s"], carrier["vy_cm_s"]))

    return dict(
        nearest_presser_closing_cm_s=nearest_closing,
        presser_velocity_toward_carrier_cm_s=presser_vel_toward_carrier,
        n_participants=len(participants),
        supporting_player_convergence_cm_s=support_convergence,
        carrier_speed_cm_s=carrier_speed,
    )


# ---------------------------------------------------------------------------
# PASSING / ESCAPE
# ---------------------------------------------------------------------------
def passing_escape_features(field, gx, gy, carrier, teammates, opponents, pitch=DEFAULT_PITCH):
    """`field`: the DEFENDING (opponent-to-carrier) team's
    `team_pressure_field` output -- SAME array the dashboard washes,
    never recomputed differently."""
    lanes = [passing_lane_metrics(field, gx, gy, carrier, tm, opponents) for tm in teammates]
    escape_pts = escape_directions(field, gx, gy, (carrier["x_pitch"], carrier["y_pitch"]), pitch,
                                    radius_cm=ESCAPE_RADIUS_CM, n_candidates=N_ESCAPE_CANDIDATES, n_select=1)
    best_escape_pressure = escape_pts[0][2] if escape_pts else None
    lane_pressures = [ln["mean_pressure"] for ln in lanes]
    min_defender_dists = [ln["min_defender_distance_cm"] for ln in lanes if ln["min_defender_distance_cm"] is not None]

    return dict(
        n_teammates_considered=len(teammates),
        mean_lane_pressure=float(np.mean(lane_pressures)) if lane_pressures else None,
        max_lane_pressure=float(np.max(lane_pressures)) if lane_pressures else None,
        min_defender_to_lane_distance_cm=min(min_defender_dists) if min_defender_dists else None,
        n_low_pressure_lanes=sum(p < LOW_PRESSURE_THRESHOLD for p in lane_pressures),
        best_escape_direction_pressure=best_escape_pressure,
    )


# ---------------------------------------------------------------------------
# FIELD
# ---------------------------------------------------------------------------
def field_features(field, gx, gy, carrier):
    p_carrier = sample_field(field, gx, gy, carrier["x_pitch"], carrier["y_pitch"])
    coverage = contour_coverage_fractions(field)
    gx_grad, gy_grad = pressure_gradient(field, gx, gy, carrier["x_pitch"], carrier["y_pitch"])
    grad_mag = float(np.hypot(gx_grad, gy_grad))
    # asymmetry: pressure sampled a fixed 200cm (one grid step) either
    # side of the carrier along the pitch-length axis -- a simple,
    # disclosed directional-imbalance proxy, not a claim of a "correct"
    # asymmetry model.
    p_ahead = sample_field(field, gx, gy, carrier["x_pitch"] + 200.0, carrier["y_pitch"])
    p_behind = sample_field(field, gx, gy, carrier["x_pitch"] - 200.0, carrier["y_pitch"])
    return dict(
        pressure_at_carrier=p_carrier,
        local_mean_pressure=float(field.mean()),
        local_max_pressure=float(field.max()),
        pressure_gradient_magnitude=grad_mag,
        pressure_asymmetry=float(p_ahead - p_behind),
        contour_coverage_low=coverage[sorted(coverage)[0]],
        contour_coverage_high=coverage[sorted(coverage)[-1]],
    )


# ---------------------------------------------------------------------------
# CONTEXT
# ---------------------------------------------------------------------------
def context_features(carrier, players_this_frame, carrier_team, pitch=DEFAULT_PITCH, confidence=None, coverage=None):
    depth_cm = pitch.depth(carrier_team, carrier["x_pitch"])
    touchline_dist_cm = min(carrier["y_pitch"], pitch.width_cm - carrier["y_pitch"])
    third = "defensive" if depth_cm < pitch.length_cm / 3 else "middle" if depth_cm < 2 * pitch.length_cm / 3 else "attacking"
    return dict(
        pitch_zone_third=third,
        pitch_zone_channel=("left" if carrier["y_pitch"] < pitch.width_cm / 3 else
                             "central" if carrier["y_pitch"] < 2 * pitch.width_cm / 3 else "right"),
        attacking_sign=pitch.attacking_sign(carrier_team),
        distance_to_goal_cm=float(pitch.length_cm - depth_cm),
        touchline_proximity_cm=float(touchline_dist_cm),
        players_visible_this_frame=len(_eligible(players_this_frame)),
        data_confidence=confidence,
        data_coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Episode-level assembly
# ---------------------------------------------------------------------------
def extract_episode_features(episode, players_by_frame, roles, gx, gy, pitch=DEFAULT_PITCH):
    """One feature row for one `pressing_v4.build_pressing_v4` episode,
    anchored at `episode['onset_frame']` (the FSM's own real detected
    onset, never a hand-picked frame). Returns None if the anchor frame
    lacks the positional evidence needed (honest gap, never fabricated)."""
    f = episode["onset_frame"]
    role = roles[f]
    presser_team = episode["team"]
    players = players_by_frame.get(f, [])
    eligible = _eligible(players)
    carrier = next((p for p in eligible if p["track_id"] == role.get("carrier_track")), None)
    if carrier is None or role.get("carrier_team") is None:
        return None
    carrier_team = role["carrier_team"]
    defenders = [p for p in eligible if p["display_team_id"] == presser_team]
    teammates = [p for p in eligible if p["display_team_id"] == carrier_team and p["track_id"] != carrier["track_id"]]

    field = team_pressure_field(defenders, gx, gy)
    # Reuse the SAME `frame_features` the dashboard/FSM itself already
    # computes for (presser_team) at this frame -- never a second,
    # differently-derived numerical-superiority/coverage/confidence.
    feat = frame_features(players, role, presser_team)

    geom = geometry_features(carrier, defenders)
    geom["local_numerical_superiority"] = feat.get("support_minus_opponents")

    row = dict(episode_id=episode["episode_id"], frame=f, time_sec=f / 30.0,
               pressing_team=presser_team, ball_carrier_team=carrier_team, carrier_track_id=carrier["track_id"])
    row.update({f"geom_{k}": v for k, v in geom.items()})
    row.update({f"motion_{k}": v for k, v in motion_features(carrier, defenders, feat.get("participant_ids") or []).items()})
    row.update({f"escape_{k}": v for k, v in passing_escape_features(field, gx, gy, carrier, teammates, defenders, pitch).items()})
    row.update({f"field_{k}": v for k, v in field_features(field, gx, gy, carrier).items()})
    row.update({f"ctx_{k}": v for k, v in context_features(carrier, players, carrier_team, pitch, confidence=feat.get("confidence"), coverage=feat.get("coverage")).items()})
    return row


def attribute_outcomes_all_episodes(episodes, passes, turnovers, pitch=DEFAULT_PITCH, fps=30.0, players_by_frame=None):
    """Generalization of `pressing_v4.attribute_outcomes`: the SAME
    matching logic and the SAME `evidence_quality` gate, reused
    unchanged -- the only difference is candidacy is evaluated against
    EVERY episode via `end_frame`/`termination_time`, not gated on
    `active_start`. `pressing_v4.attribute_outcomes` itself is NOT
    modified (the frozen dashboard/V4 pipeline's own `events` list is
    untouched) -- this is an additive, research-only wrapper. See
    `RESEARCH_DATASET_PLAN.md` section 9 for why this generalization
    exists (0/19 real episodes on this clip ever reach `active_start`,
    so the un-generalized function yields zero outcomes here)."""
    out = [{**e, "heuristic_outcome": "UNCERTAIN", "outcome_reason": "NO_MATCHED_EVIDENCE", "outcome_evidence_id": None,
            "outcome_evidence_time": None, "outcome_confirmation_time": None, "forward_progress_cm": None} for e in episodes]
    evidence = [("pass", p) for p in passes] + [("turnover", p) for p in turnovers]
    for kind, p in sorted(evidence, key=lambda kp: (kp[1]["end_frame"], kp[1]["start_frame"], kp[0], kp[1].get("pass_id", 0))):
        source = p.get("source_team_id") if kind == "turnover" else p.get("team_id")
        cands = [e for e in out if e["outcome_evidence_id"] is None and 1 - e["team"] == source
                 and e["end_frame"] <= p["start_frame"] and p["end_frame"] <= e["end_frame"] + 3 * fps
                 and pitch.period(e["end_frame"]) == pitch.period(p["end_frame"])]
        if not cands:
            continue
        e = max(cands, key=lambda e: e["end_frame"])
        valid, reason = evidence_quality(p, kind, players_by_frame)
        label, progress = "UNCERTAIN", None
        if valid:
            if kind == "turnover":
                label = "BALL_REGAIN" if p.get("receiver_team_id") == e["team"] else "UNCERTAIN"
            else:
                progress = pitch.progress(source, p["start_x"], p["end_x"], p["start_frame"])
                label = "BACKWARD_ACTION" if progress <= -300 else "FORWARD_PROGRESSION" if progress >= 300 else "LATERAL_ACTION"
        e.update(heuristic_outcome=label, outcome_reason=reason or "OBSERVED_POST_PRESS_ACTION_NOT_CAUSAL_SUCCESS",
                  outcome_evidence_id=f"{kind}:{p.get('scene_id', 0)}:{p['pass_id']}",
                  outcome_evidence_time=p["start_frame"] / fps, outcome_confirmation_time=p["end_frame"] / fps,
                  forward_progress_cm=progress)
    return out


# Mapping from the pipeline's coarse heuristic labels to the finer
# human-labeling taxonomy (PRESS_LABELING_PROTOCOL.md section 5) --
# documented here (not silently duplicated) since both files must stay
# in sync; imported by the export script and by evaluate_outcomes.py.
HEURISTIC_TO_TAXONOMY = {
    "BALL_REGAIN": "BALL_REGAIN",
    "BACKWARD_ACTION": "FORCED_BACKWARD",
    "LATERAL_ACTION": "FORCED_LATERAL",
    "FORWARD_PROGRESSION": "PRESS_ESCAPED",  # documented limitation: cannot distinguish escape from adequate-forward-pass without a human judgment
    "UNCERTAIN": "UNCERTAIN",
}
