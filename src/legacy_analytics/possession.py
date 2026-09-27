"""
Reusable, project-level BALL-POSSESSION analytics (Phase 11 of the pass-
analytics diagnosis, see PLAN.md). Determines, per frame, which player (if
any) plausibly controls the ball -- pure geometry over already-computed
tracking + ball-trajectory data, no model inference here.

This module exists because the OLD pass detector (`analytics/
pass_detection.py`'s original trajectory-segment approach) never modeled
possession at all -- it looked for a geometrically plausible ball path and
guessed a nearby player only after the fact. A manual visual audit of the
full testVideo1_120s clip (`outputs/analytics/testVideo1_120s/
pass_manual_audit.csv` + `pass_failure_analysis.csv`) found this was the
SINGLE LARGEST failure category (6 of 15 manually-confirmed real passes,
40%): the old detector simply never tried to connect ball motion to a
source/receiver player identity. `analytics/pass_detection.py`'s new
possession-transition mode (Phase 14) consumes this module's output.

## Control-radius calibration (Phase 12)

Tested against the 15 manually-confirmed real passes: measured the
nearest-player-to-ball pitch distance at each pass's source (release) and
receiver (reception) moment -- 30 endpoint measurements total, range
35-284cm (`outputs/analytics/testVideo1_120s/pass_ball_evidence_audit.csv`
and this diagnosis's own scratch analysis). 100cm and 150cm were BOTH too
strict: e.g. one receiver-control moment measured 284cm, one source-
control moment measured 265cm -- real, visually-confirmed control
moments, not noise. This is consistent with `analytics/ball_tracker.py`'s
own documented finding that the ball's 2D pitch-mapping carries real
perspective/height-off-ground error (a 2D homography assumes a ground-
plane object; a moving/rolling ball is not always exactly on the ground,
and pitch-mapping jitter compounds this). **Chosen**: `CONTROL_RADIUS_CM
= 200` (confident control) with an `EXTENDED_RADIUS_CM = 400` lower-
confidence "plausible control / reception in progress" band that tapers
to zero, beyond which the ball is `FREE_BALL`. This is fit to THIS clip's
measured camera/detection/pitch-mapping precision, not a universal
football constant -- re-calibrate against a fresh manual audit before
reusing on materially different footage.
"""
import numpy as np
import polars as pl

CONTROL_RADIUS_CM = 200.0
EXTENDED_RADIUS_CM = 400.0
DEFAULT_PERSISTENCE_FRAMES = 5
# ^ Phase 13: a single UNCERTAIN/BALL_LOST frame sandwiched between two
# CONTROLLED frames of the SAME track_id is very likely coordinate jitter,
# not a real release-and-reacquire -- bridge gaps up to this length. This
# is NOT the same mechanism as the ball tracker's own short-gap fill
# (`analytics/temporal_utils.py`); it operates on the possession LABEL,
# never fabricates a ball position.

POSSESSION_CONTROLLED = "CONTROLLED"
POSSESSION_FREE_BALL = "FREE_BALL"
POSSESSION_UNCERTAIN = "UNCERTAIN"
POSSESSION_BALL_LOST = "BALL_LOST"


def compute_possession(ball_traj: pl.DataFrame, display_df: pl.DataFrame,
                        control_radius_cm: float = CONTROL_RADIUS_CM,
                        extended_radius_cm: float = EXTENDED_RADIUS_CM,
                        persistence_frames: int = DEFAULT_PERSISTENCE_FRAMES,
                        frame_min: int = None, frame_max: int = None,
                        include_goalkeeper: bool = True) -> pl.DataFrame:
    """`ball_traj`: `analytics/ball_motion.py`'s trajectory schema (needs
    `frame`, `x_pitch`, `y_pitch`, `is_observed`). `display_df`:
    `analytics/position_fill.py::build_full_display_frame()`'s stabilized
    per-frame player positions (needs `frame`, `track_id`,
    `display_object_type`, `display_team_id`, `display_x_pitch`,
    `display_y_pitch`). Possession considers players AND goalkeepers by
    default (unlike the heatmap module, which excludes the goalkeeper for
    an unrelated spatial-occupancy reason -- a goalkeeper can obviously be
    in possession of the ball).

    Returns one row per frame in [frame_min, frame_max] (defaults to the
    full range spanned by `display_df`): frame, timestamp_sec, ball_state,
    possessing_track_id, possessing_team_id, distance_to_ball,
    possession_confidence, possession_state, is_bridged (True only for
    frames filled by the Phase 13 short-gap persistence pass -- never
    hidden, same provenance-transparency convention as the rest of this
    project)."""
    roles = ["player"] + (["goalkeeper"] if include_goalkeeper else [])
    players = display_df.filter(
        pl.col("display_object_type").is_in(roles) & pl.col("display_x_pitch").is_not_null()
    ).select(["frame", "track_id", "display_team_id", "display_x_pitch", "display_y_pitch"])

    players_by_frame: dict[int, list[dict]] = {}
    for r in players.iter_rows(named=True):
        players_by_frame.setdefault(r["frame"], []).append(r)

    ball_by_frame = {r["frame"]: r for r in ball_traj.to_dicts()}

    if frame_min is None:
        frame_min = int(display_df["frame"].min())
    if frame_max is None:
        frame_max = int(display_df["frame"].max())

    fps = 30.0
    rows = []
    for f in range(frame_min, frame_max + 1):
        b = ball_by_frame.get(f)
        if b is None or not b.get("is_observed"):
            rows.append(_row(f, fps, POSSESSION_BALL_LOST, "BALL_LOST", None, None, None, 0.0))
            continue
        bx, by = b.get("x_pitch"), b.get("y_pitch")
        if bx is None or by is None:
            # observed in image space but no per-frame homography this
            # frame -- we genuinely cannot relate it to player positions
            rows.append(_row(f, fps, "OBSERVED_NO_PITCH", POSSESSION_UNCERTAIN, None, None, None, 0.1))
            continue

        cands = players_by_frame.get(f, [])
        if not cands:
            rows.append(_row(f, fps, "OBSERVED", POSSESSION_UNCERTAIN, None, None, None, 0.1))
            continue

        best = min(cands, key=lambda p: (p["display_x_pitch"] - bx) ** 2 + (p["display_y_pitch"] - by) ** 2)
        dist = float(np.hypot(best["display_x_pitch"] - bx, best["display_y_pitch"] - by))

        if dist <= control_radius_cm:
            conf = 1.0 - 0.3 * (dist / control_radius_cm)  # 1.0 -> 0.7
            rows.append(_row(f, fps, "OBSERVED", POSSESSION_CONTROLLED, best["track_id"],
                              best["display_team_id"], dist, conf,
                              best["display_x_pitch"], best["display_y_pitch"]))
        elif dist <= extended_radius_cm:
            frac = (dist - control_radius_cm) / (extended_radius_cm - control_radius_cm)
            conf = 0.7 - 0.5 * frac  # 0.7 -> 0.2
            rows.append(_row(f, fps, "OBSERVED", POSSESSION_UNCERTAIN, best["track_id"],
                              best["display_team_id"], dist, conf,
                              best["display_x_pitch"], best["display_y_pitch"]))
        else:
            rows.append(_row(f, fps, "OBSERVED", POSSESSION_FREE_BALL, None, None, dist, 0.0, None, None))

    row_schema = {"frame": pl.Int64, "timestamp_sec": pl.Float64, "ball_state": pl.Utf8,
                  "possessing_track_id": pl.Int64, "possessing_team_id": pl.Int64,
                  "distance_to_ball": pl.Float64, "possession_confidence": pl.Float64,
                  "possession_state": pl.Utf8, "is_bridged": pl.Boolean,
                  "player_x_pitch": pl.Float64, "player_y_pitch": pl.Float64}
    df = pl.DataFrame(rows, schema=row_schema)
    return _bridge_short_gaps(df, persistence_frames)


def _row(frame, fps, ball_state, possession_state, track_id, team_id, dist, conf, px=None, py=None):
    return {"frame": frame, "timestamp_sec": frame / fps, "ball_state": ball_state,
            "possessing_track_id": track_id, "possessing_team_id": team_id,
            "distance_to_ball": dist, "possession_confidence": conf,
            "possession_state": possession_state, "is_bridged": False,
            "player_x_pitch": px, "player_y_pitch": py}


def _bridge_short_gaps(df: pl.DataFrame, persistence_frames: int) -> pl.DataFrame:
    """Phase 13: a run of length <= persistence_frames where
    possession_state is NOT CONTROLLED, sandwiched between two CONTROLLED
    frames of the SAME track_id, is relabeled CONTROLLED for that track
    (confidence discounted 20% since it's bridged, not directly
    measured) -- coordinate jitter should not flicker possession. A gap
    bounded by DIFFERENT track_ids, or exceeding persistence_frames, is
    left alone: a genuine departure must transition promptly, never
    artificially extended."""
    rows = df.sort("frame").to_dicts()
    n = len(rows)
    i = 0
    while i < n:
        if rows[i]["possession_state"] == POSSESSION_CONTROLLED:
            j = i + 1
            gap_start = j
            while j < n and rows[j]["possession_state"] != POSSESSION_CONTROLLED and (j - gap_start) < persistence_frames:
                j += 1
            if (j < n and rows[j]["possession_state"] == POSSESSION_CONTROLLED
                    and rows[j]["possessing_track_id"] == rows[i]["possessing_track_id"]
                    and j > gap_start):
                track = rows[i]["possessing_track_id"]
                team = rows[i]["possessing_team_id"]
                lo_conf = min(rows[i]["possession_confidence"], rows[j]["possession_confidence"])
                for k in range(gap_start, j):
                    rows[k]["possession_state"] = POSSESSION_CONTROLLED
                    rows[k]["possessing_track_id"] = track
                    rows[k]["possessing_team_id"] = team
                    rows[k]["possession_confidence"] = round(lo_conf * 0.8, 3)
                    rows[k]["is_bridged"] = True
            i = j
        else:
            i += 1
    row_schema = {"frame": pl.Int64, "timestamp_sec": pl.Float64, "ball_state": pl.Utf8,
                  "possessing_track_id": pl.Int64, "possessing_team_id": pl.Int64,
                  "distance_to_ball": pl.Float64, "possession_confidence": pl.Float64,
                  "possession_state": pl.Utf8, "is_bridged": pl.Boolean,
                  "player_x_pitch": pl.Float64, "player_y_pitch": pl.Float64}
    return pl.DataFrame(rows, schema=row_schema)


def control_episodes(possession_df: pl.DataFrame, min_confidence: float = 0.0) -> list[dict]:
    """Collapses CONTROLLED (incl. bridged) runs of the same track_id into
    episodes: [{track_id, team_id, start_frame, end_frame, mean_confidence}].
    This is the primary input to the new possession-transition pass
    detector (`analytics/pass_detection.py`)."""
    rows = possession_df.filter(
        (pl.col("possession_state") == POSSESSION_CONTROLLED) &
        (pl.col("possession_confidence") >= min_confidence)
    ).sort("frame").to_dicts()

    episodes = []
    cur = None
    for r in rows:
        if cur is not None and r["possessing_track_id"] == cur["track_id"] and r["frame"] - cur["end_frame"] <= 1:
            cur["end_frame"] = r["frame"]
            cur["confidences"].append(r["possession_confidence"])
            if r["player_x_pitch"] is not None:
                cur["end_x"], cur["end_y"] = r["player_x_pitch"], r["player_y_pitch"]
        else:
            if cur is not None:
                episodes.append(cur)
            cur = {"track_id": r["possessing_track_id"], "team_id": r["possessing_team_id"],
                   "start_frame": r["frame"], "end_frame": r["frame"],
                   "start_x": r["player_x_pitch"], "start_y": r["player_y_pitch"],
                   "end_x": r["player_x_pitch"], "end_y": r["player_y_pitch"],
                   "confidences": [r["possession_confidence"]]}
    if cur is not None:
        episodes.append(cur)
    for e in episodes:
        e["mean_confidence"] = round(sum(e["confidences"]) / len(e["confidences"]), 3)
        del e["confidences"]
    return episodes
