"""
Reusable CLEANED tracking view for the two new research prototypes
(pressing_structure, offside_break) ONLY -- not part of, and does not
modify, the existing PhD pipeline's `analytics/` package.

Builds entirely on top of the EXISTING, UNCHANGED PhD stabilization
utilities:
  - `analytics.role_stability.stabilize_roles_and_team` (via
    `analytics.position_fill.build_full_display_frame`)
  - `analytics.position_fill.fill_player_position_gaps` (short-gap fill,
    max_gap=3 frames, never a long gap)
  - `analytics.temporal_utils.causal_ema` (smoothing)

Those utilities fix role/team flicker and fill short position NULLs.
They do NOT address a separate, measured problem: `outputs/tracking/
testVideo1_120s/quality_summary.md` found 50% of raw tracks contain at
least one frame-to-frame pitch speed >20 m/s (2000 cm/s) -- a candidate
ID-swap/teleport artifact, not real player motion. Smoothing straight
through such a jump would blend two different physical identities into
one fake continuous trajectory. This module adds exactly one new thing:
**explicit rejection, never silent interpolation, of that jump.**

## Method

1. `build_full_display_frame(tracking_df, apply_smoothing=False)` --
   reused as-is, gives role/team-stabilized, short-gap-filled (but not
   yet smoothed) positions.
2. Per track_id, frame-to-frame pitch speed is computed on those
   filled-but-unsmoothed positions. Any transition exceeding
   `IMPOSSIBLE_SPEED_CM_S` (2000 cm/s -- the SAME threshold already
   used by `scripts/quality_report.py`, not a new invented number)
   starts a new `cleaned_segment_id` for that track, exactly like the
   ball tracker's `is_track_restart` concept elsewhere in this project.
   The flagged transition is NEVER bridged, interpolated across, or
   smoothed across.
3. EMA smoothing (`causal_ema`, reused) is applied PER SEGMENT (never
   across a rejected transition).
4. Per-segment causal velocity (vx_cm_s, vy_cm_s, speed_cm_s) is
   computed by finite difference on the final smoothed positions --
   needed by both downstream projects (closing speed, run velocity).

Output columns (one row per original tracking.parquet row that has a
track_id; ball rows are handled separately, see `load_ball_view`):
frame, timestamp_sec, track_id, cleaned_segment_id, is_track_restart,
display_object_type, display_team_id, x_pitch, y_pitch,
is_position_filled, vx_cm_s, vy_cm_s, speed_cm_s.

Never fabricates a position for a frame the object wasn't detected in,
never bridges a long gap, never hides a rejected jump -- `is_track_restart`
is always present and inspectable.
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

from legacy_analytics.position_fill import build_full_display_frame  # noqa: E402
from legacy_analytics.temporal_utils import causal_ema  # noqa: E402

IMPOSSIBLE_SPEED_CM_S = 2000.0
# ^ Reused verbatim from scripts/quality_report.py's own flagging
# threshold (~20 m/s) -- not re-derived, so a "rejected here" jump is
# the same jump the quality report already told us about, for the same
# reason (generous headroom above genuine elite sprint, ~8.24 m/s
# measured 95th percentile -- see outputs/analytics/testVideo1_120s_v3/
# pitch_control_calibration.md).
DEFAULT_EMA_ALPHA = 0.35  # matches analytics/position_fill.py's own default


def _segment_track(sub: pl.DataFrame, fps: float) -> pl.DataFrame:
    """One track_id's rows, sorted by frame. Returns the same rows with
    cleaned_segment_id / is_track_restart / smoothed x_pitch,y_pitch /
    vx_cm_s,vy_cm_s,speed_cm_s added."""
    frames = sub["frame"].to_list()
    xs = sub["display_x_pitch"].to_list()
    ys = sub["display_y_pitch"].to_list()

    seg_ids = [0] * len(frames)
    restarts = [False] * len(frames)
    prev_valid = None  # (frame, x, y)
    seg = 0
    for i, (f, x, y) in enumerate(zip(frames, xs, ys)):
        if x is None or y is None:
            seg_ids[i] = seg
            continue
        if prev_valid is not None:
            pf, px, py = prev_valid
            dt = (f - pf) / fps
            if dt > 0:
                speed = float(np.hypot(x - px, y - py)) / dt
                if speed > IMPOSSIBLE_SPEED_CM_S:
                    seg += 1
                    restarts[i] = True
        seg_ids[i] = seg
        prev_valid = (f, x, y)

    out = sub.with_columns([
        pl.Series("cleaned_segment_id", seg_ids),
        pl.Series("is_track_restart", restarts),
    ])

    # smooth + velocity PER SEGMENT -- never bridges a rejected jump
    parts = []
    for (sid,), seg_df in out.group_by(["cleaned_segment_id"], maintain_order=True):
        seg_df = seg_df.sort("frame")
        sm_x, sm_y = causal_ema(seg_df["display_x_pitch"].to_list(),
                                 seg_df["display_y_pitch"].to_list(), alpha=DEFAULT_EMA_ALPHA)
        fr = seg_df["frame"].to_numpy()
        sx = np.array([v if v is not None else np.nan for v in sm_x])
        sy = np.array([v if v is not None else np.nan for v in sm_y])
        vx = np.full(len(fr), np.nan)
        vy = np.full(len(fr), np.nan)
        for i in range(1, len(fr)):
            dt = (fr[i] - fr[i - 1]) / fps
            if dt > 0 and not np.isnan(sx[i]) and not np.isnan(sx[i - 1]):
                vx[i] = (sx[i] - sx[i - 1]) / dt
                vy[i] = (sy[i] - sy[i - 1]) / dt
        speed = np.hypot(vx, vy)
        parts.append(seg_df.with_columns([
            pl.Series("x_pitch", [None if np.isnan(v) else float(v) for v in sx], dtype=pl.Float64),
            pl.Series("y_pitch", [None if np.isnan(v) else float(v) for v in sy], dtype=pl.Float64),
            pl.Series("vx_cm_s", [None if np.isnan(v) else float(v) for v in vx], dtype=pl.Float64),
            pl.Series("vy_cm_s", [None if np.isnan(v) else float(v) for v in vy], dtype=pl.Float64),
            pl.Series("speed_cm_s", [None if np.isnan(v) else float(v) for v in speed], dtype=pl.Float64),
        ]))
    return pl.concat(parts, how="diagonal_relaxed").sort("frame")


def build_cleaned_view(tracking_df: pl.DataFrame, fps: float = 30.0) -> pl.DataFrame:
    """Main entry point. `tracking_df`: the RAW, UNCHANGED
    `tracking.parquet` DataFrame (never mutated -- a fresh DataFrame is
    returned)."""
    display_df = build_full_display_frame(tracking_df, apply_smoothing=False)
    tracked = display_df.filter(pl.col("track_id").is_not_null())

    parts = []
    for (tid,), sub in tracked.group_by(["track_id"], maintain_order=True):
        sub = sub.sort("frame")
        parts.append(_segment_track(sub, fps))
    cleaned = pl.concat(parts, how="diagonal_relaxed").sort(["frame", "track_id"])

    cleaned = cleaned.with_columns((pl.col("frame") / fps).alias("timestamp_sec"))
    return cleaned.select([
        "frame", "timestamp_sec", "track_id", "cleaned_segment_id", "is_track_restart",
        "display_object_type", "display_team_id", "x_pitch", "y_pitch",
        "display_is_filled", "vx_cm_s", "vy_cm_s", "speed_cm_s",
    ]).rename({"display_is_filled": "is_position_filled"})


def load_ball_view(analytics_dir: str) -> pl.DataFrame:
    """Loads the EXISTING, CANONICAL, already-hardened ball trajectory
    (`outputs/analytics/testVideo1_120s_v3/ball_trajectory.parquet`) --
    read-only, never recomputed here. Preserves its OBSERVED/PREDICTED/
    LOST distinction (`is_observed` column) exactly as the PhD pipeline
    already validated it; this module adds no new ball logic."""
    path = os.path.join(analytics_dir, "ball_trajectory.parquet")
    return pl.read_parquet(path)


def cleaned_view_summary(cleaned: pl.DataFrame) -> dict:
    """Small, honest summary of what cleaning actually did -- for the
    methodology doc, not for hiding anything."""
    n_tracks = cleaned["track_id"].n_unique()
    n_segments = cleaned.select(["track_id", "cleaned_segment_id"]).unique().height
    n_restarts = int(cleaned["is_track_restart"].sum())
    return {
        "n_raw_tracks": n_tracks,
        "n_cleaned_segments": n_segments,
        "n_rejected_transitions": n_restarts,
        "mean_segments_per_track": round(n_segments / n_tracks, 2) if n_tracks else None,
    }
