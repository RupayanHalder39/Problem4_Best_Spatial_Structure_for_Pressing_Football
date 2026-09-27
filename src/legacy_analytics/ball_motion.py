"""
Reusable ball-motion analytics, built ONLY from already-exported
`tracking.parquet` ball rows (`object_type == "ball"`). No model
inference happens here -- this is pure post-hoc analysis of saved
detections, meant to be consumed by `analytics/pass_detection.py`, the
dashboard renderer, and (eventually) Journal work.

RAW vs DERIVED: every function here returns a NEW DataFrame; the input
`tracking.parquet` is read-only and never modified. `is_raw_detection`
marks which rows came directly from the export vs. which were filled in
by the short-gap interpolation described below -- callers that care about
data provenance (e.g. Journal 1 feature extraction) should always check
this flag rather than assuming every row is a real detection.

Two DIFFERENT kinds of "gap" exist in the ball data, and this module
treats them very differently (same distinction established for players in
Phase 7, `outputs/demo/testVideo1_20s/demo_validation_summary.md` Part A):

1. A frame WHERE A BALL ROW EXISTS but `x_pitch`/`y_pitch` is null (the
   frame's homography was rejected by the Phase 6C temporal-consistency
   gate -- this affects ALL objects in that frame, not just the ball).
   These gaps are short (measured on testVideo1_20s: within the one dense
   ball-detection run, 364/365/370/372/373/375/376 are single-or-double-
   frame null-pitch gaps between raw detections one frame apart) and are
   bridged with short-gap linear interpolation, `max_gap=3`, matching the
   already-validated player policy.

2. A frame with NO ball row at all (the ball simply wasn't detected --
   very common; ball coverage on testVideo1_20s is only 6.4% of frames,
   6/8 of the gaps between detection clusters are 19-309 frames long).
   These are NEVER filled or interpolated across -- a missing row is not
   evidence the ball existed at some in-between position, so nothing is
   fabricated for it. `split_into_runs()` uses these boundaries to define
   `segment_id`, so a "movement segment" never silently bridges a real
   absence.
"""
import numpy as np
import polars as pl

from legacy_analytics.homography_replay import apply_transformers_to_points
from legacy_analytics.temporal_utils import (causal_ema, causal_rolling_majority,  # noqa: F401
                                       displacement_cm, fill_short_gaps,
                                       split_into_runs)

DEFAULT_MAX_GAP = 3          # frames; matches the validated player policy
DEFAULT_TRAIL_FRAMES = 24    # ~0.8s at 30fps -- Part 7's "last 0.5-1.0s of trusted motion"


def load_ball_series(tracking_df: pl.DataFrame) -> pl.DataFrame:
    """Raw ball rows only, deduplicated to at most one row per frame
    (the general multi-class detector occasionally yields two ball-like
    candidates in one frame -- e.g. testVideo1_20s frame 382 -- so we
    keep the higher-confidence one and drop the other; this is the only
    row-selection step, nothing about position is altered)."""
    ball = tracking_df.filter(pl.col("object_type") == "ball")
    ball = ball.sort(["frame", "confidence"], descending=[False, True])
    ball = ball.unique(subset=["frame"], keep="first").sort("frame")
    return ball.select(["frame", "timestamp_sec", "x_image", "y_image",
                         "x_pitch", "y_pitch", "confidence"])


def build_ball_trajectory(tracking_df: pl.DataFrame, max_gap: int = DEFAULT_MAX_GAP,
                           gap_method: str = "interp") -> pl.DataFrame:
    """One row per frame that has a raw ball detection (never fabricates
    rows for frames with no detection at all). Adds:
      display_x_pitch/display_y_pitch : short-gap-filled pitch position
      is_raw_pitch                    : True if x_pitch was non-null in
                                         the export (not filled)
      segment_id                      : increments each time the gap to
                                         the previous raw detection frame
                                         exceeds max_gap -- a segment is a
                                         contiguous run of genuine ball
                                         detections
      dx_cm/dy_cm/step_distance_cm/dt_sec/speed_cm_s/direction_deg :
                                         causal frame-to-frame motion,
                                         computed only within a segment
                                         (never across a segment break)
    """
    ball = load_ball_series(tracking_df)
    if ball.height == 0:
        return ball.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("display_x_pitch"),
            pl.lit(None, dtype=pl.Float64).alias("display_y_pitch"),
            pl.lit(False).alias("is_raw_pitch"),
            pl.lit(None, dtype=pl.Int64).alias("segment_id"),
        ])

    frames = ball["frame"].to_list()
    xs = ball["x_pitch"].to_list()
    ys = ball["y_pitch"].to_list()
    is_raw_pitch = [x is not None for x in xs]

    out_x, out_y, filled = fill_short_gaps(frames, xs, ys, method=gap_method, max_gap=max_gap)

    runs = split_into_runs(frames, max_gap=max_gap)
    segment_id = [0] * len(frames)
    for seg_idx, (s, e) in enumerate(runs):
        for i in range(s, e + 1):
            segment_id[i] = seg_idx

    dx_cm = [None] * len(frames)
    dy_cm = [None] * len(frames)
    step_dist = [None] * len(frames)
    dt_sec = [None] * len(frames)
    speed = [None] * len(frames)
    direction_deg = [None] * len(frames)
    ts = ball["timestamp_sec"].to_list()
    for i in range(1, len(frames)):
        if segment_id[i] != segment_id[i - 1]:
            continue
        if out_x[i] is None or out_x[i - 1] is None:
            continue
        dx = out_x[i] - out_x[i - 1]
        dy = out_y[i] - out_y[i - 1]
        dx_cm[i] = dx
        dy_cm[i] = dy
        d = displacement_cm(out_x[i - 1], out_y[i - 1], out_x[i], out_y[i])
        step_dist[i] = d
        dt = ts[i] - ts[i - 1]
        dt_sec[i] = dt
        speed[i] = d / dt if dt > 0 else None
        direction_deg[i] = float((180.0 / 3.141592653589793) * __import__("math").atan2(dy, dx))

    return ball.with_columns([
        pl.Series("display_x_pitch", out_x),
        pl.Series("display_y_pitch", out_y),
        pl.Series("is_raw_pitch", is_raw_pitch),
        pl.Series("segment_id", segment_id),
        pl.Series("dx_cm", dx_cm),
        pl.Series("dy_cm", dy_cm),
        pl.Series("step_distance_cm", step_dist),
        pl.Series("dt_sec", dt_sec),
        pl.Series("speed_cm_s", speed),
        pl.Series("direction_deg", direction_deg),
    ])


def summarize_ball_segments(traj_df: pl.DataFrame) -> pl.DataFrame:
    """One row per contiguous ball-detection segment: start/end
    frame+time, net straight-line distance, total path length walked
    (sum of per-step distances), a 0-1 "straightness" ratio
    (net_distance/path_length -- 1.0 = a single straight line, low values
    = the ball wandered/was dribbled), and mean/max speed. Used by
    `pass_detection.py` to decide which segments are pass-like; exposed
    here too since it's independently useful (e.g. for later Journal
    work on ball-carry vs. pass patterns)."""
    if traj_df.height == 0 or "segment_id" not in traj_df.columns:
        return pl.DataFrame(schema={
            "segment_id": pl.Int64, "start_frame": pl.Int64, "end_frame": pl.Int64,
            "start_time": pl.Float64, "end_time": pl.Float64, "n_detections": pl.Int64,
            "start_x": pl.Float64, "start_y": pl.Float64, "end_x": pl.Float64, "end_y": pl.Float64,
            "net_distance_cm": pl.Float64, "path_length_cm": pl.Float64,
            "straightness": pl.Float64, "duration_sec": pl.Float64,
            "mean_speed_cm_s": pl.Float64, "max_speed_cm_s": pl.Float64,
        })

    rows = []
    for seg_id in sorted(traj_df["segment_id"].unique().to_list()):
        seg = traj_df.filter(pl.col("segment_id") == seg_id).sort("frame")
        valid = seg.filter(pl.col("display_x_pitch").is_not_null())
        if valid.height == 0:
            continue
        xs = valid["display_x_pitch"].to_list()
        ys = valid["display_y_pitch"].to_list()
        speeds = [s for s in seg["speed_cm_s"].to_list() if s is not None]
        # path_length: consecutive-hop distance across ALL valid pitch
        # points in chronological order (not the possibly-gapped
        # step_distance_cm column, which only covers ADJACENT rows in
        # the source table and silently skips a hop whenever an
        # intermediate frame's homography was unavailable -- that
        # under-counts the path and can make straightness exceed 1,
        # which is mathematically impossible for a true net/path ratio).
        path_length = sum(displacement_cm(xs[i], ys[i], xs[i + 1], ys[i + 1]) for i in range(len(xs) - 1))
        net = displacement_cm(xs[0], ys[0], xs[-1], ys[-1])
        rows.append({
            "segment_id": seg_id,
            "start_frame": int(seg["frame"].min()), "end_frame": int(seg["frame"].max()),
            "start_time": float(seg["timestamp_sec"].min()), "end_time": float(seg["timestamp_sec"].max()),
            "n_detections": seg.height,
            "start_x": xs[0], "start_y": ys[0], "end_x": xs[-1], "end_y": ys[-1],
            "net_distance_cm": net, "path_length_cm": path_length,
            "straightness": (net / path_length) if path_length > 0 else None,
            "duration_sec": float(seg["timestamp_sec"].max() - seg["timestamp_sec"].min()),
            "mean_speed_cm_s": (sum(speeds) / len(speeds)) if speeds else None,
            "max_speed_cm_s": max(speeds) if speeds else None,
        })
    return pl.DataFrame(rows)


def get_trail(traj_df: pl.DataFrame, current_frame: int,
              max_trail_frames: int = DEFAULT_TRAIL_FRAMES) -> list[dict]:
    """Returns recent ball positions for rendering a fading trail +
    direction arrow, restricted to the SAME segment as current_frame (so
    a trail never draws a line across a real detection gap). Each item:
    {frame, x, y, fade} with fade in (0, 1], 1.0 = current_frame. Returns
    [] if current_frame has no ball position."""
    if traj_df.height == 0:
        return []
    row = traj_df.filter(pl.col("frame") == current_frame)
    if row.height == 0 or row["display_x_pitch"][0] is None:
        return []
    seg_id = row["segment_id"][0]
    seg = traj_df.filter((pl.col("segment_id") == seg_id) &
                          (pl.col("frame") <= current_frame) &
                          (pl.col("display_x_pitch").is_not_null())).sort("frame")
    seg = seg.tail(max_trail_frames)
    frames = seg["frame"].to_list()
    xs = seg["display_x_pitch"].to_list()
    ys = seg["display_y_pitch"].to_list()
    obs = seg["is_observed"].to_list() if "is_observed" in seg.columns else [True] * len(frames)
    n = len(frames)
    out = []
    for i, (f, x, y, o) in enumerate(zip(frames, xs, ys, obs)):
        fade = 0.15 + 0.85 * ((i + 1) / n) if n > 1 else 1.0
        out.append({"frame": f, "x": x, "y": y, "fade": fade, "is_observed": bool(o)})
    return out


def get_image_trail(traj_df: pl.DataFrame, current_frame: int,
                     max_trail_frames: int = DEFAULT_TRAIL_FRAMES) -> list[dict]:
    """Same idea as get_trail() but in IMAGE pixel space (x_image/
    y_image) for drawing on the match-video panel. For the original
    (Phase 8) raw-detector trajectory every row is a genuine raw
    detection; for the tracker-based trajectory (Phase 9,
    `build_ball_trajectory_from_tracker`) some rows may be short-gap
    interpolated -- `is_observed` (default True if the column is absent,
    for backward compatibility) tells the caller which is which so it can
    render them differently rather than hiding the distinction."""
    if traj_df.height == 0:
        return []
    row = traj_df.filter(pl.col("frame") == current_frame)
    if row.height == 0:
        return []
    seg_id = row["segment_id"][0]
    seg = traj_df.filter((pl.col("segment_id") == seg_id) & (pl.col("frame") <= current_frame)).sort("frame")
    seg = seg.tail(max_trail_frames)
    frames = seg["frame"].to_list()
    xs = seg["x_image"].to_list()
    ys = seg["y_image"].to_list()
    obs = seg["is_observed"].to_list() if "is_observed" in seg.columns else [True] * len(frames)
    n = len(frames)
    out = []
    for i, (f, x, y, o) in enumerate(zip(frames, xs, ys, obs)):
        fade = 0.15 + 0.85 * ((i + 1) / n) if n > 1 else 1.0
        out.append({"frame": f, "x": x, "y": y, "fade": fade, "is_observed": bool(o)})
    return out


MIN_DIRECTION_DISPLACEMENT_CM = 60.0  # ~0.6m -- below this, treat as jitter, not real motion


def current_direction_vector(traj_df: pl.DataFrame, current_frame: int,
                              lookback_frames: int = 6, min_points: int = 3,
                              min_observed_fraction: float = 0.5) -> tuple:
    """Returns a normalized (ux, uy) direction vector for the arrow, or
    (None, None) if there isn't enough EVIDENCE to show one -- Part 8's
    explicit rule: "wrong arrow is worse than no arrow." Requires ALL of:
      - at least `min_points` same-segment trail points (a 2-point
        direction is too easily jitter-driven);
      - the displacement across the lookback window clears
        `MIN_DIRECTION_DISPLACEMENT_CM` (nontrivial velocity -- a
        near-stationary ball has no meaningful "direction");
      - at least `min_observed_fraction` of the lookback points are real
        OBSERVATIONS, not short-term display predictions -- an arrow
        should reflect what was actually seen, not mostly extrapolation.
    """
    trail = get_trail(traj_df, current_frame, max_trail_frames=lookback_frames)
    if len(trail) < min_points:
        return (None, None)
    observed_frac = sum(1 for p in trail if p.get("is_observed", True)) / len(trail)
    if observed_frac < min_observed_fraction:
        return (None, None)
    dx = trail[-1]["x"] - trail[0]["x"]
    dy = trail[-1]["y"] - trail[0]["y"]
    mag = (dx ** 2 + dy ** 2) ** 0.5
    if mag < MIN_DIRECTION_DISPLACEMENT_CM:
        return (None, None)
    return (dx / mag, dy / mag)


def build_ball_trajectory_from_tracker(tracked_image_space_df: pl.DataFrame,
                                        transformers: dict, max_gap: int = DEFAULT_MAX_GAP) -> pl.DataFrame:
    """Ball-perception Stage 7/10: turns the causal single-object
    tracker's image-space output (`analytics/ball_tracker.py::
    track_ball_image_space`, already short-gap-filled in image space by
    `scripts/track_ball_image_space.py` via `temporal_utils.fill_short_
    gaps`) into the SAME trajectory schema `summarize_ball_segments`/
    `get_trail`/`get_image_trail`/`current_direction_vector` above
    already consume -- so none of those functions needed to change.

    `transformers`: {frame: ViewTransformer or None}, from
    `analytics/homography_replay.py::replay_transformers` -- the EXACT
    same per-frame accepted homography used for players, never a second
    independently-fit one. Ball anchor is CENTER (unchanged policy);
    `tracked_image_space_df`'s x_image/y_image are already CENTER-ish
    detection-box centers from the specialized detector.

    Only rows where the tracker has SOME position (observed or
    interpolated) are pitch-mapped; frames still missing after gap-fill
    stay null in x_pitch/y_pitch/display_x_pitch/display_y_pitch -- never
    fabricated. `is_observed`/`is_interpolated`/`track_quality` are
    carried through so no consumer can mistake a recovered point for a
    real detection."""
    df = tracked_image_space_df.sort("frame")
    frames = df["frame"].to_list()
    xs_obs = df["x_image"].to_list()          # observed-only (None if not observed this frame)
    ys_obs = df["y_image"].to_list()
    xs_disp = df["x_image_filled"].to_list()  # observed + short-gap-filled
    ys_disp = df["y_image_filled"].to_list()
    is_observed = df["is_observed"].to_list()
    is_interp = df["is_interpolated"].to_list()
    conf = df["confidence"].to_list()
    track_quality = df["track_quality"].to_list()
    is_restart = df["is_track_restart"].to_list() if "is_track_restart" in df.columns else [False] * len(frames)
    restart_frames = {f for f, r in zip(frames, is_restart) if r}

    pitch_obs = apply_transformers_to_points(transformers, list(zip(frames, xs_obs, ys_obs)))
    pitch_disp = apply_transformers_to_points(transformers, list(zip(frames, xs_disp, ys_disp)))

    # Segment boundaries come from TWO independent signals, either one
    # forces a break: (1) the usual frame-gap rule (a real absence longer
    # than max_gap), and (2) the tracker's own `is_track_restart` flag --
    # a frame where the tracker re-anchored after being lost, even if the
    # raw frame gap to it happens to be short. (2) is what stops a
    # reacquisition from visually connecting to the stale pre-loss
    # position (Part 6's explicit requirement) -- frame-gap alone cannot
    # tell "smooth continuation" apart from "fresh anchor that happens to
    # follow shortly after."
    valid_frames = [f for f, x in zip(frames, xs_disp) if x is not None]
    runs = split_into_runs(valid_frames, max_gap=max_gap)
    frame_to_seg = {}
    seg_idx = -1
    for s, e in runs:
        for i in range(s, e + 1):
            f = valid_frames[i]
            if i == s or f in restart_frames:
                seg_idx += 1
            frame_to_seg[f] = seg_idx
    segment_id = [frame_to_seg.get(f, -1) for f in frames]

    n = len(frames)
    dx_cm = [None] * n
    dy_cm = [None] * n
    step_dist = [None] * n
    dt_sec = [None] * n
    speed = [None] * n
    direction_deg = [None] * n
    fps_guess = 1.0 / (df["timestamp_sec"][1] - df["timestamp_sec"][0]) if "timestamp_sec" in df.columns and n > 1 else 30.0
    ts = df["timestamp_sec"].to_list() if "timestamp_sec" in df.columns else [f / fps_guess for f in frames]

    import math
    for i in range(1, n):
        if segment_id[i] != segment_id[i - 1] or segment_id[i] == -1:
            continue
        x0, y0 = pitch_disp[i - 1]
        x1, y1 = pitch_disp[i]
        if x0 is None or x1 is None:
            continue
        dx, dy = x1 - x0, y1 - y0
        dx_cm[i], dy_cm[i] = dx, dy
        d = displacement_cm(x0, y0, x1, y1)
        step_dist[i] = d
        dt = ts[i] - ts[i - 1]
        dt_sec[i] = dt
        speed[i] = d / dt if dt > 0 else None
        direction_deg[i] = float((180.0 / math.pi) * math.atan2(dy, dx))

    out = pl.DataFrame({
        "frame": frames, "timestamp_sec": ts,
        "x_image": xs_disp, "y_image": ys_disp,
        "x_pitch": [p[0] for p in pitch_obs], "y_pitch": [p[1] for p in pitch_obs],
        "display_x_pitch": [p[0] for p in pitch_disp], "display_y_pitch": [p[1] for p in pitch_disp],
        "confidence": conf, "is_raw_pitch": is_observed,
        "is_observed": is_observed, "is_interpolated": is_interp, "track_quality": track_quality,
        "segment_id": segment_id, "dx_cm": dx_cm, "dy_cm": dy_cm, "step_distance_cm": step_dist,
        "dt_sec": dt_sec, "speed_cm_s": speed, "direction_deg": direction_deg,
    })
    # Drop frames where the tracker has NO position at all (not observed,
    # not short-gap-filled) -- matches the sparse-row convention every
    # other function in this module already assumes (a row's mere
    # existence means "the ball was here", per Phase 8's load_ball_series
    # docstring). This does not lose information: the full per-frame
    # observed/interpolated/missing counts are reported separately by
    # scripts/track_ball_image_space.py before this filtering happens.
    return out.filter(pl.col("x_image").is_not_null())


DEFAULT_MAX_PREDICT_FRAMES = 10   # ~0.33s -- see rationale below
DISPLAY_STATE_OBSERVED = "OBSERVED"
DISPLAY_STATE_PREDICTED = "PREDICTED"


def add_short_term_display_prediction(traj_df: pl.DataFrame, transformers: dict,
                                       max_predict_frames: int = DEFAULT_MAX_PREDICT_FRAMES,
                                       max_speed_cm_s: float = 3500.0) -> pl.DataFrame:
    """DISPLAY-ONLY continuity aid (Part 5/6 of the ball-continuity
    reopening -- `ball_continuity_summary.md`). Extends the trail a few
    frames past the END of a segment using constant-velocity
    extrapolation, so a viewer sees a brief, honestly-labeled "PREDICTED"
    tail instead of the ball vanishing mid-motion. Never modifies an
    existing row; every new row is tagged `display_state="PREDICTED"`,
    `is_display_prediction=True` -- it is never mistaken for a real
    detection downstream (`is_observed`/`is_interpolated` stay False on
    these rows) and NEVER gets written into any "raw"/"scientific"
    artifact -- this function's OUTPUT is display data, consumed only by
    the renderer.

    Measured evidence for the window (`ball_continuity_summary.md`, Part
    4): on testVideo1_60s, ZERO of the 37 real missing-ball gaps are <=5
    frames (the minimum observed gap is 7 frames) -- a literal 2/3/5-
    frame window would never fire. Extending to `max_predict_frames=10`
    (0.33s) still only reaches the shortest ~8% of real gaps. This is
    reported honestly, not hidden: short-term prediction measurably
    helps only a minority of gaps here; it is not a fix for the
    underlying detection-sparsity problem (out of scope for a display
    layer -- see Part 27's explicit "do not loosen the tracker just to
    raise coverage").

    Qualifying conditions before predicting past a segment's last frame
    (ALL required):
      - the segment's last row is a real OBSERVATION (`is_observed`),
        not itself an interpolated/predicted point -- never chain
        predictions off a prediction;
      - at least 2 recent observed points exist in the segment to
        estimate a velocity (a single point has no direction);
      - the extrapolated pitch-space speed stays under `max_speed_cm_s`
        (the same physical bound the tracker itself uses);
      - the predicted point stays within the pitch bounds (0-12000,
        0-7000cm) -- a prediction that leaves the pitch is not plausible;
      - stops immediately if the NEXT real segment already starts within
        the window (never overwrite/exceed into real data)."""
    df = traj_df.sort("frame")
    all_rows = df.to_dicts()
    by_segment: dict[int, list[dict]] = {}
    for r in all_rows:
        by_segment.setdefault(r["segment_id"], []).append(r)

    next_start_after = {}
    starts_sorted = sorted(min(r["frame"] for r in seg) for seg in by_segment.values())
    for seg in by_segment.values():
        seg_start = min(r["frame"] for r in seg)
        later = [s for s in starts_sorted if s > seg_start]
        next_start_after[seg_start] = min(later) if later else None

    fps = 30.0
    if len(all_rows) > 1:
        ts_sorted = sorted(r["timestamp_sec"] for r in all_rows)
        deltas = [b - a for a, b in zip(ts_sorted, ts_sorted[1:]) if b > a]
        if deltas:
            fps = 1.0 / min(deltas)

    new_rows = []
    for seg_id, seg in by_segment.items():
        seg = sorted(seg, key=lambda r: r["frame"])
        last = seg[-1]
        if not last["is_observed"]:
            continue
        observed_in_seg = [r for r in seg if r["is_observed"]]
        if len(observed_in_seg) < 2:
            continue
        prev = observed_in_seg[-2]
        dt = last["timestamp_sec"] - prev["timestamp_sec"]
        if dt <= 0 or last["display_x_pitch"] is None or prev["display_x_pitch"] is None:
            continue
        vx = (last["display_x_pitch"] - prev["display_x_pitch"]) / dt
        vy = (last["display_y_pitch"] - prev["display_y_pitch"]) / dt
        speed = float(np.hypot(vx, vy))
        if speed > max_speed_cm_s:
            continue  # the observed motion itself was already implausibly fast -- don't extend it

        vx_img = vy_img = None
        if last["x_image"] is not None and prev["x_image"] is not None:
            vx_img = (last["x_image"] - prev["x_image"]) / dt
            vy_img = (last["y_image"] - prev["y_image"]) / dt

        seg_start = min(r["frame"] for r in seg)
        ceiling_frame = next_start_after.get(seg_start)
        for k in range(1, max_predict_frames + 1):
            f = last["frame"] + k
            if ceiling_frame is not None and f >= ceiling_frame:
                break
            t = last["timestamp_sec"] + k / fps
            px = last["display_x_pitch"] + vx * (k / fps)
            py = last["display_y_pitch"] + vy * (k / fps)
            if not (0 <= px <= 12000 and 0 <= py <= 7000):
                break
            img_x = last["x_image"] + vx_img * (k / fps) if vx_img is not None else None
            img_y = last["y_image"] + vy_img * (k / fps) if vy_img is not None else None
            new_rows.append({
                "frame": f, "timestamp_sec": t,
                "x_image": img_x, "y_image": img_y,
                "x_pitch": None, "y_pitch": None,
                "display_x_pitch": px, "display_y_pitch": py,
                "confidence": None, "is_raw_pitch": False,
                "is_observed": False, "is_interpolated": False, "track_quality": last.get("track_quality"),
                "segment_id": seg_id, "dx_cm": vx * (1 / fps), "dy_cm": vy * (1 / fps),
                "step_distance_cm": speed / fps, "dt_sec": 1 / fps, "speed_cm_s": speed,
                "direction_deg": last.get("direction_deg"),
                "display_state": DISPLAY_STATE_PREDICTED, "is_display_prediction": True,
            })

    # OBSERVED covers real detections; interpolated (short-gap, bounded
    # on both sides by real data) is still not a raw observation --
    # label it PREDICTED too so the 3-state taxonomy (Part 5) stays
    # exactly 3 states, while `is_interpolated` remains available for
    # anyone who needs to tell interpolation and extrapolation apart.
    base = df.with_columns([
        pl.when(pl.col("is_observed")).then(pl.lit(DISPLAY_STATE_OBSERVED))
        .otherwise(pl.lit(DISPLAY_STATE_PREDICTED)).alias("display_state"),
        pl.lit(False).alias("is_display_prediction"),
    ])

    if not new_rows:
        return base
    pred_df = pl.DataFrame(new_rows, schema=base.schema)
    return pl.concat([base, pred_df], how="vertical_relaxed").sort("frame")
