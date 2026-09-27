"""
Reusable pass-detection analytics.

Two detectors live in this module now (Phase 14 of the pass-analytics
diagnosis, PLAN.md):

1. `detect_passes()` -- the ORIGINAL trajectory-segment-only detector: a
   "pass candidate" is a contiguous ball detection segment whose net
   pitch-space displacement clears a minimum distance. Deliberately does
   NOT try to classify every ball touch; short wandering segments
   (dribbles, control touches) are excluded by design. Kept, unmodified,
   as the "old" arm of the Phase 20 old-vs-new comparison
   (`outputs/analytics/testVideo1_120s/pass_detector_comparison.md`).
   **Measured limitation** (manual audit, `pass_failure_analysis.csv`):
   this detector NEVER models possession/player identity at all, and
   REQUIRES the whole event inside one continuous ball-tracking segment
   -- together these two properties accounted for 11 of 15 manually-
   confirmed real passes being missed entirely on testVideo1_120s.

2. `detect_passes_possession()` -- the NEW possession-transition
   detector, built on `analytics/possession.py`. Primary definition:
   SOURCE PLAYER CONTROLS BALL -> BALL LEAVES SOURCE -> BALL TRAVELS ->
   RECEIVER CONTROLS BALL -> SOURCE TEAM == RECEIVER TEAM -> COMPLETED
   PASS. Ball trajectory geometry becomes SUPPORTING evidence
   (`trajectory_support`), not the primary requirement -- a pass is not
   rejected just because the ball itself went briefly unobserved between
   two clear possession episodes.

`detect_passes_hybrid()` combines both: every possession-evidenced pass,
plus any old-detector pass that has no overlapping possession-based
match (tagged `evidence_type="TRAJECTORY_ONLY"`), so a genuine pass the
possession radius happened to miss (e.g. no player ever came inside
`EXTENDED_RADIUS_CM`) is not silently dropped either.

HONESTY CONSTRAINT (ground vs. aerial): the export has no true ball
height / z-coordinate -- `x_pitch`/`y_pitch` come from a GROUND-PLANE
homography, so an airborne ball's image position, when run through that
same homography, does not correspond to any real ground point. The
`classify_pass_type()` heuristic below exploits exactly that: an
airborne ball tends to produce apparent-speed spikes and lower
straightness in its (incorrectly ground-projected) trajectory, since the
homography is mapping a moving-through-3D-space point onto a fixed 2D
plane. This is a WEAK, INDIRECT, UNVALIDATED-UNTIL-CHECKED signal, not a
height measurement. `pass_type` is always one of
{"ground_like", "aerial_like", "unknown"} and `method` always says
"2D heuristic (apparent-speed + straightness proxy) -- not a true
height measurement" so no downstream consumer can mistake it for 3D
ball tracking.
"""
import polars as pl

from legacy_analytics.ball_motion import build_ball_trajectory, summarize_ball_segments

DEFAULT_MIN_DISTANCE_CM = 500.0     # ~5m -- below this, treat as a touch/dribble, not a pass
DEFAULT_SOURCE_TARGET_RADIUS_CM = 300.0  # 3m -- how close a player must be to credit them
AERIAL_SPEED_SPIKE_CM_S = 1500.0    # ~54 km/h apparent speed -- see module docstring
DEFAULT_MAX_DURATION_SEC = 2.5      # a genuine single pass/strike -- not a multi-touch passage
DEFAULT_MIN_STRAIGHTNESS = 0.75     # a pass is a fairly direct transfer, not a wandering carry
# ^ Evidence for these two: after hardening analytics/ball_tracker.py's
# reacquisition logic (2-consecutive-consistency confirmation, pitch-
# space physical speed gating), ball segments on testVideo1_20s became
# naturally much shorter/more atomic -- the previously-flagged 4.6s
# "multi-touch passage" segment is now split into 5 separate segments by
# the tracker itself (each a genuine continuous-tracking run). The
# resulting real candidate segments all had duration <=1.17s and
# straightness >=0.83 (see ball_tracking_summary.md Part 7); these
# thresholds keep generous headroom above that while still rejecting a
# segment that, on different footage, might still span an extended
# multi-touch passage the tracker didn't naturally break up.
GROUND_STRAIGHTNESS_MIN = 0.85      # smooth+straight -> more consistent with a grounded pass


def _nearest_player(tracking_df: pl.DataFrame, frame: int, x: float, y: float,
                     radius_cm: float) -> dict | None:
    """Nearest player/goalkeeper (BOTTOM_CENTER pitch position, i.e. the
    already-exported x_pitch/y_pitch) to (x, y) at `frame`, within
    radius_cm. Returns None if nobody qualifies -- callers must accept
    that source/target attribution can be unknown rather than guessed."""
    cand = tracking_df.filter(
        (pl.col("frame") == frame) &
        (pl.col("object_type").is_in(["player", "goalkeeper"])) &
        pl.col("x_pitch").is_not_null()
    )
    if cand.height == 0:
        return None
    best = None
    best_d = radius_cm
    for r in cand.to_dicts():
        d = ((r["x_pitch"] - x) ** 2 + (r["y_pitch"] - y) ** 2) ** 0.5
        if d <= best_d:
            best_d = d
            best = r
    if best is None:
        return None
    return {"track_id": best["track_id"], "team_id": best["team_id"], "distance_cm": best_d}


def classify_pass_type(mean_speed_cm_s: float | None, max_speed_cm_s: float | None,
                        straightness: float | None, n_detections: int) -> tuple:
    """Returns (pass_type, confidence, method). See module docstring for
    the honesty constraint -- this NEVER claims true 3D height."""
    method = "2D heuristic (apparent-speed + straightness proxy) -- not a true height measurement"
    if n_detections < 3 or straightness is None or mean_speed_cm_s is None:
        return "unknown", 0.0, method
    spike = (max_speed_cm_s or 0) >= AERIAL_SPEED_SPIKE_CM_S
    smooth_and_straight = straightness >= GROUND_STRAIGHTNESS_MIN and not spike
    if spike:
        conf = min(1.0, (max_speed_cm_s - AERIAL_SPEED_SPIKE_CM_S) / AERIAL_SPEED_SPIKE_CM_S + 0.3)
        return "aerial_like", round(min(conf, 0.6), 2), method  # capped -- weak signal
    if smooth_and_straight:
        return "ground_like", round(min(0.3 + 0.4 * straightness, 0.6), 2), method  # capped -- weak signal
    return "unknown", 0.2, method


def detect_passes(tracking_df: pl.DataFrame, traj_df: pl.DataFrame | None = None,
                   min_distance_cm: float = DEFAULT_MIN_DISTANCE_CM,
                   source_target_radius_cm: float = DEFAULT_SOURCE_TARGET_RADIUS_CM,
                   max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
                   min_straightness: float = DEFAULT_MIN_STRAIGHTNESS) -> pl.DataFrame:
    """Returns one row per candidate pass: a segment representing a
    DISCRETE ball transfer, not just "moved far during one continuous
    segment" -- net_distance_cm >= min_distance_cm (clear onset+movement),
    >=2 detections, duration_sec <= max_duration_sec (a single pass, not
    an extended multi-touch passage), AND straightness >= min_straightness
    (a fairly direct transfer, not a wandering carry/dribble that
    happened to net a large displacement). See module-level constants for
    the evidence behind these thresholds. Columns: pass_id, start/end
    frame+time, start/end xy, distance_cm, duration_sec, estimated_speed,
    team_id, source_track_id, target_track_id, pass_type,
    pass_type_confidence, method. Team/source/target are left null when
    no player is close enough to credit -- never guessed."""
    if traj_df is None:
        traj_df = build_ball_trajectory(tracking_df)
    segs = summarize_ball_segments(traj_df)

    schema = {
        "pass_id": pl.Int64, "start_frame": pl.Int64, "end_frame": pl.Int64,
        "start_time": pl.Float64, "end_time": pl.Float64,
        "start_x": pl.Float64, "start_y": pl.Float64, "end_x": pl.Float64, "end_y": pl.Float64,
        "distance_cm": pl.Float64, "duration_sec": pl.Float64, "estimated_speed_cm_s": pl.Float64,
        "team_id": pl.Int64, "source_track_id": pl.Int64, "target_track_id": pl.Int64,
        "pass_type": pl.Utf8, "pass_type_confidence": pl.Float64, "method": pl.Utf8,
    }
    if segs.height == 0:
        return pl.DataFrame(schema=schema)

    candidates = segs.filter(
        (pl.col("net_distance_cm") >= min_distance_cm) & (pl.col("n_detections") >= 2) &
        (pl.col("duration_sec") <= max_duration_sec) &
        (pl.col("straightness").is_not_null()) & (pl.col("straightness") >= min_straightness)
    )
    if candidates.height == 0:
        return pl.DataFrame(schema=schema)

    rows = []
    for pid, seg in enumerate(candidates.to_dicts()):
        pass_type, conf, method = classify_pass_type(
            seg["mean_speed_cm_s"], seg["max_speed_cm_s"], seg["straightness"], seg["n_detections"])

        source = _nearest_player(tracking_df, seg["start_frame"], seg["start_x"], seg["start_y"],
                                  source_target_radius_cm)
        target = _nearest_player(tracking_df, seg["end_frame"], seg["end_x"], seg["end_y"],
                                  source_target_radius_cm)
        team_id = None
        if source is not None:
            team_id = source["team_id"]
        elif target is not None:
            team_id = target["team_id"]

        rows.append({
            "pass_id": pid,
            "start_frame": seg["start_frame"], "end_frame": seg["end_frame"],
            "start_time": seg["start_time"], "end_time": seg["end_time"],
            "start_x": seg["start_x"], "start_y": seg["start_y"],
            "end_x": seg["end_x"], "end_y": seg["end_y"],
            "distance_cm": seg["net_distance_cm"], "duration_sec": seg["duration_sec"],
            "estimated_speed_cm_s": seg["mean_speed_cm_s"],
            "team_id": team_id,
            "source_track_id": source["track_id"] if source else None,
            "target_track_id": target["track_id"] if target else None,
            "pass_type": pass_type, "pass_type_confidence": conf, "method": method,
        })
    return pl.DataFrame(rows, schema=schema)


def active_pass_at_frame(passes_df: pl.DataFrame, frame: int,
                          max_display_frames: int = 60, hold_frames: int = 15) -> dict | None:
    """Returns the pass dict active/visible at `frame`, or None.

    Display window: the LAST `max_display_frames` of [start_frame,
    end_frame], extended `hold_frames` past the end so the arrow is
    visible long enough to read. Capped rather than showing the arrow
    for the segment's full raw duration -- some detected segments span
    several seconds (manual inspection found one covering 4.6s / 140
    frames, more consistent with an extended passage of play containing
    several touches than a single struck pass -- see
    ball_tracking_summary.md). Drawing a straight start->end arrow for
    that whole span would visually imply one continuous flight that
    didn't happen; capping the display window to the end of the segment
    keeps the arrow honest about what it's showing."""
    if passes_df.height == 0:
        return None
    cand = passes_df.filter(
        (pl.col("end_frame") - max_display_frames <= frame) & (pl.col("end_frame") + hold_frames >= frame) &
        (pl.col("start_frame") <= frame)
    )
    if cand.height == 0:
        return None
    return cand.to_dicts()[0]


DEFAULT_MAX_MISSING_BALL_FRACTION = 0.85
# ^ Re-tuned during the Voronoi/pass-quality reopening (user: "still not
# satisfactory... accurate results"). Measured on testVideo1_120s_v2's
# 14 possession-evidenced passes: 12 had missing_ball_fraction <=0.79,
# one outlier (pass_id=13, the old numbering) measured 0.912 -- i.e. the
# ball was observed in only ~9% of that transfer's frames. A same-team
# possession-episode transition with THAT little trajectory confirmation
# is much weaker evidence of one real pass than of "the ball went
# somewhere during a longer passage of unobserved play" -- excluded by
# default rather than drawn as an equally-confident arrow.
DEFAULT_MAX_TRANSFER_GAP_SEC = 5.0
# ^ Phase 17 evidence: the manual audit's longest SAME-track-to-SAME-team
# possession-episode gap that visually corresponded to a plausible single
# passage was well under this; an 11.1s gap observed between two same-
# team control episodes (35.6s-46.7s) visually turned out to be a dense,
# multi-touch attacking sequence (`pass_manual_audit.csv` row M11/M32/
# M33), not one pass -- collapsing an 11s gap into a single arrow would
# misrepresent it. Capping at 5s rejects that kind of multi-touch chain
# as "too long to be one transfer" while still covering every genuine
# single-pass gap actually measured in the audit (all under 2s).
DEFAULT_MIN_PASS_CONFIDENCE = 0.35
# ^ Raised from 0.15 during the Voronoi/pass-quality reopening. At 0.15
# every possession-evidenced candidate on testVideo1_120s_v2 passed
# regardless of quality; the two weakest measured confidences (0.377,
# 0.474) both corresponded to the two longest/most ball-sparse
# transfers (missing_ball_fraction 0.79 and 0.91 respectively) --
# exactly the "long ball segment stretched into a guessed pass" pattern
# the user's brief explicitly warns against. 0.35 sits just below the
# next-weakest genuinely-observed pass (0.377 still clears the OLD 0.15
# but is now a deliberate near-miss worth re-examining, not silently
# admitted) while cutting the least-evidenced candidates.


def _ball_endpoint(traj_df: pl.DataFrame | None, frame: int, window: int, prefer_after: bool):
    """Nearest OBSERVED ball pitch position to `frame`, searched outward
    up to `window` frames in the preferred direction first. Returns
    (x, y, frame_found) or (None, None, None)."""
    if traj_df is None:
        return None, None, None
    for d in range(0, window + 1):
        for f in ([frame + d, frame - d] if prefer_after else [frame - d, frame + d]):
            row = traj_df.filter((pl.col("frame") == f) & pl.col("is_observed"))
            if row.height > 0:
                r = row.to_dicts()[0]
                if r.get("x_pitch") is not None:
                    return r["x_pitch"], r["y_pitch"], f
    return None, None, None


def detect_passes_possession(episodes: list[dict], traj_df: pl.DataFrame | None = None,
                              fps: float = 30.0,
                              max_transfer_gap_sec: float = DEFAULT_MAX_TRANSFER_GAP_SEC,
                              min_confidence: float = DEFAULT_MIN_PASS_CONFIDENCE,
                              max_missing_ball_fraction: float = DEFAULT_MAX_MISSING_BALL_FRACTION,
                              min_distance_cm: float = DEFAULT_MIN_DISTANCE_CM,
                              scene_id: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Phase 14/15/18/19. `episodes`: `analytics/possession.py::
    control_episodes()` output, already sorted or not (sorted here).
    Returns (passes_df, turnovers_df) -- turnovers (different-team
    possession transitions) are computed for completeness/QA but are
    NEVER included in `passes_df` (Phase 19: a turnover must not corrupt
    a team's completed-pass map).

    Team attribution (Phase 18): uses each episode's own stabilized
    `team_id` directly -- never guessed from direction/geometry. A
    transfer where either endpoint's team is unknown is recorded with
    `team_id=None` (excluded from both pass maps downstream, same rule
    `analytics/pass_map.py` already enforces)."""
    eps = sorted(episodes, key=lambda e: e["start_frame"])
    schema = {
        "pass_id": pl.Int64, "scene_id": pl.Int64, "start_frame": pl.Int64, "end_frame": pl.Int64,
        "start_time": pl.Float64, "end_time": pl.Float64,
        "source_track_id": pl.Int64, "target_track_id": pl.Int64, "team_id": pl.Int64,
        "start_x": pl.Float64, "start_y": pl.Float64, "end_x": pl.Float64, "end_y": pl.Float64,
        "distance_cm": pl.Float64, "duration_sec": pl.Float64,
        "trajectory_support": pl.Utf8,
        "source_possession_confidence": pl.Float64, "receiver_possession_confidence": pl.Float64,
        "missing_ball_fraction": pl.Float64, "confidence": pl.Float64, "evidence_type": pl.Utf8,
        "start_is_player_proxy": pl.Boolean, "end_is_player_proxy": pl.Boolean,
    }
    pass_rows, turnover_rows = [], []
    pid, tid = 0, 0

    for i in range(len(eps) - 1):
        a, b = eps[i], eps[i + 1]
        if a["track_id"] == b["track_id"]:
            continue  # same player re-registers control after a jitter gap -- not a transfer
        gap_frames = b["start_frame"] - a["end_frame"]
        if gap_frames <= 0 or gap_frames / fps > max_transfer_gap_sec:
            continue  # overlapping/malformed, or too long to be one plausible transfer (see const doc)

        duration_sec = gap_frames / fps
        # trajectory support: how much of the transfer window has an
        # actual observed ball position, and where its endpoints are
        window_obs = 0
        if traj_df is not None:
            window_obs = traj_df.filter(
                (pl.col("frame") >= a["end_frame"]) & (pl.col("frame") <= b["start_frame"]) & pl.col("is_observed")
            ).height
        missing_frac = 1.0 - min(1.0, window_obs / max(1, gap_frames))
        support = "full" if missing_frac <= 0.3 else ("partial" if missing_frac < 1.0 else "none")

        sx, sy, _ = _ball_endpoint(traj_df, a["end_frame"], 10, prefer_after=False)
        ex, ey, _ = _ball_endpoint(traj_df, b["start_frame"], 10, prefer_after=True)
        start_proxy = sx is None
        end_proxy = ex is None
        # Fall back to the source/receiver PLAYER's own pitch position
        # (captured by analytics/possession.py::control_episodes() at
        # the episode's edge) as a documented geometric proxy when the
        # ball itself has no observation right at the endpoint -- never
        # a fabricated curved path, just the two endpoint anchors (Phase
        # 17's explicit allowance: "start->end arrow can still represent
        # the inferred transfer IF confidence is high enough").
        if start_proxy:
            sx, sy = a.get("end_x"), a.get("end_y")
        if end_proxy:
            ex, ey = b.get("start_x"), b.get("start_y")

        duration_mult = 1.0 if duration_sec <= 1.5 else max(0.4, 1.0 - 0.5 * (duration_sec - 1.5) / (max_transfer_gap_sec - 1.5))
        support_mult = {"full": 1.0, "partial": 0.75, "none": 0.5}[support]
        base_conf = (a["mean_confidence"] + b["mean_confidence"]) / 2.0
        confidence = round(base_conf * duration_mult * support_mult, 3)

        same_team = a["team_id"] is not None and b["team_id"] is not None and a["team_id"] == b["team_id"]
        team_known = a["team_id"] is not None and b["team_id"] is not None

        pass_distance_cm = (float(((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5)
                             if None not in (sx, sy, ex, ey) else None)

        if same_team:
            if confidence < min_confidence or missing_frac > max_missing_ball_fraction:
                continue
            if pass_distance_cm is not None and pass_distance_cm < min_distance_cm:
                continue  # too short to be a deliberate transfer -- see DEFAULT_MIN_DISTANCE_CM's doc
            evidence_type = "TRAJECTORY_PLUS_POSSESSION" if support in ("full", "partial") else "POSSESSION_TRANSITION"
            pass_rows.append({
                "pass_id": pid, "scene_id": scene_id,
                "start_frame": a["end_frame"], "end_frame": b["start_frame"],
                "start_time": round(a["end_frame"] / fps, 3), "end_time": round(b["start_frame"] / fps, 3),
                "source_track_id": a["track_id"], "target_track_id": b["track_id"], "team_id": a["team_id"],
                "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
                "distance_cm": pass_distance_cm,
                "duration_sec": round(duration_sec, 3), "trajectory_support": support,
                "source_possession_confidence": a["mean_confidence"], "receiver_possession_confidence": b["mean_confidence"],
                "missing_ball_fraction": round(missing_frac, 3), "confidence": confidence,
                "evidence_type": evidence_type,
                "start_is_player_proxy": start_proxy, "end_is_player_proxy": end_proxy,
            })
            pid += 1
        elif team_known:
            # different-team transition -- a turnover/interception, NOT a pass (Phase 19)
            turnover_rows.append({
                "pass_id": tid, "scene_id": scene_id,
                "start_frame": a["end_frame"], "end_frame": b["start_frame"],
                "start_time": round(a["end_frame"] / fps, 3), "end_time": round(b["start_frame"] / fps, 3),
                "source_track_id": a["track_id"], "target_track_id": b["track_id"],
                "team_id": None, "source_team_id": a["team_id"], "receiver_team_id": b["team_id"],
                "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
                "distance_cm": pass_distance_cm,
                "duration_sec": round(duration_sec, 3), "trajectory_support": support,
                "source_possession_confidence": a["mean_confidence"], "receiver_possession_confidence": b["mean_confidence"],
                "missing_ball_fraction": round(missing_frac, 3), "confidence": confidence,
                "evidence_type": "TURNOVER_CANDIDATE",
                "start_is_player_proxy": start_proxy, "end_is_player_proxy": end_proxy,
            })
            tid += 1
        # else: one or both teams unknown -- Phase 18's "if source/receiver
        # unavailable -> unknown": not recorded as either a pass or a
        # turnover, since we cannot even tell which category it is.

    passes_df = pl.DataFrame(pass_rows, schema=schema) if pass_rows else pl.DataFrame(schema=schema)
    turnover_schema = dict(schema)
    turnover_schema["source_team_id"] = pl.Int64
    turnover_schema["receiver_team_id"] = pl.Int64
    turnovers_df = pl.DataFrame(turnover_rows, schema=turnover_schema) if turnover_rows else pl.DataFrame(schema=turnover_schema)
    return passes_df, turnovers_df


def detect_passes_hybrid(old_passes_df: pl.DataFrame, new_passes_df: pl.DataFrame,
                          overlap_tolerance_frames: int = 15) -> pl.DataFrame:
    """Phase 20's "C. hybrid detector": every possession-evidenced pass
    from `new_passes_df`, PLUS any `old_passes_df` (trajectory-segment)
    pass whose [start_frame, end_frame] does not overlap (within
    `overlap_tolerance_frames`) any new-detector pass -- tagged
    `evidence_type="TRAJECTORY_ONLY"`. This way a real pass the
    possession radius happened to miss (no player ever came within
    `EXTENDED_RADIUS_CM`) is not silently dropped just because the newer
    detector is the preferred one.

    **Re-tuned (pass-quality reopening)**: an old-detector addition is
    now only kept if it has BOTH `source_track_id` AND `target_track_id`
    -- a "pass" must have a known passer AND receiver to be a football
    event at all; a geometric ball-trajectory segment with no player
    attribution on either end is evidence of BALL MOVEMENT, not evidence
    of a PASS (the user's own words: "not just a long ball segment").
    Measured on testVideo1_120s_v2: ALL 6 of the old detector's non-
    overlapping additions had a null source or target (2 had BOTH null)
    -- none would have survived this bar, which is itself informative:
    the old detector's genuinely well-attributed finds already overlap a
    new-detector pass (redundant, correctly not double-counted); what's
    LEFT unique to the old detector is disproportionately its weakest
    evidence. Confidence for a surviving old-detector addition is no
    longer a flat placeholder -- computed from the segment's own
    straightness/duration (tighter, more direct segments score higher),
    capped below what a possession-confirmed pass can reach, since it
    still lacks independent possession confirmation."""
    if new_passes_df.height == 0 and old_passes_df.height == 0:
        return new_passes_df

    def overlaps(o, new_rows):
        for n in new_rows:
            if not (o["end_frame"] < n["start_frame"] - overlap_tolerance_frames or
                    o["start_frame"] > n["end_frame"] + overlap_tolerance_frames):
                return True
        return False

    new_rows = new_passes_df.to_dicts()
    extra = []
    next_pid = (max((r["pass_id"] for r in new_rows), default=-1)) + 1
    for o in old_passes_df.to_dicts():
        if o.get("source_track_id") is None or o.get("target_track_id") is None:
            continue  # no known passer+receiver -- not football-meaningful, see docstring
        if overlaps(o, new_rows):
            continue
        straightness_conf = 0.3 + 0.3 * max(0.0, 1.0 - (o["duration_sec"] or 2.5) / 2.5)
        extra.append({
            "pass_id": next_pid, "scene_id": 0,
            "start_frame": o["start_frame"], "end_frame": o["end_frame"],
            "start_time": o["start_time"], "end_time": o["end_time"],
            "source_track_id": o["source_track_id"], "target_track_id": o["target_track_id"],
            "team_id": o["team_id"],
            "start_x": o["start_x"], "start_y": o["start_y"], "end_x": o["end_x"], "end_y": o["end_y"],
            "distance_cm": o["distance_cm"], "duration_sec": o["duration_sec"],
            "trajectory_support": "full",
            "source_possession_confidence": None, "receiver_possession_confidence": None,
            "missing_ball_fraction": 0.0, "confidence": round(min(0.6, straightness_conf), 3),
            "evidence_type": "TRAJECTORY_ONLY",
            "start_is_player_proxy": False, "end_is_player_proxy": False,
        })
        next_pid += 1

    if not extra:
        return new_passes_df
    extra_df = pl.DataFrame(extra, schema=new_passes_df.schema)
    return pl.concat([new_passes_df, extra_df]).sort("start_frame")
