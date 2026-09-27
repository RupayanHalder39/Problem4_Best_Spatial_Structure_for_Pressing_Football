"""
Reusable short-gap pitch-position filling for player/goalkeeper/referee
tracks, extracted from the Phase 7 professor-demo work (`demo_display_
layer.fill_position_gaps`) so every consumer (the old single-panel demo,
the new analytics pipeline -- `formation.py` and `team_shape.py` need
filled positions to get usable per-frame sample coverage, since ~29% of
frames on testVideo1_20s have every object's x_pitch/y_pitch null from a
single rejected homography) shares one implementation.

Fills ONLY rows that already exist in the export (the object WAS
detected) but have null x_pitch/y_pitch (the frame's homography was
rejected) -- never synthesizes a row for a frame where the object wasn't
even detected. See analytics/ball_motion.py's docstring for the identical
distinction applied to the ball.
"""
import polars as pl

from legacy_analytics.temporal_utils import causal_ema, fill_short_gaps

DEFAULT_MAX_INTERP_GAP = 3
DEFAULT_EMA_ALPHA = 0.35


def fill_player_position_gaps(df: pl.DataFrame, method: str = "interp",
                               max_gap: int = DEFAULT_MAX_INTERP_GAP) -> pl.DataFrame:
    """Adds `display_x_pitch`/`display_y_pitch`/`display_is_filled` to a
    copy of `df`, per track_id. Does not modify `x_pitch`/`y_pitch`."""
    assert method in ("hold", "interp")
    tracked = df.filter(pl.col("track_id").is_not_null()).sort(["track_id", "frame"])
    untracked = df.filter(pl.col("track_id").is_null()).with_columns([
        pl.col("x_pitch").alias("display_x_pitch"),
        pl.col("y_pitch").alias("display_y_pitch"),
        pl.lit(False).alias("display_is_filled"),
    ])

    parts = []
    for tid in tracked["track_id"].unique(maintain_order=True).to_list():
        sub = tracked.filter(pl.col("track_id") == tid)
        frames = sub["frame"].to_list()
        xs = sub["x_pitch"].to_list()
        ys = sub["y_pitch"].to_list()
        out_x, out_y, filled = fill_short_gaps(frames, xs, ys, method=method, max_gap=max_gap)
        parts.append(sub.with_columns([
            pl.Series("display_x_pitch", out_x),
            pl.Series("display_y_pitch", out_y),
            pl.Series("display_is_filled", filled),
        ]))

    tracked_out = pl.concat(parts) if parts else tracked.with_columns([
        pl.lit(None, dtype=pl.Float64).alias("display_x_pitch"),
        pl.lit(None, dtype=pl.Float64).alias("display_y_pitch"),
        pl.lit(False).alias("display_is_filled"),
    ])
    return pl.concat([tracked_out, untracked], how="diagonal_relaxed").sort(["frame", "track_id"])


def smooth_player_positions(df: pl.DataFrame, alpha: float = DEFAULT_EMA_ALPHA) -> pl.DataFrame:
    """Causal EMA smoothing of display_x_pitch/display_y_pitch per
    track_id. Requires fill_player_position_gaps() to have run first."""
    tracked = df.filter(pl.col("track_id").is_not_null()).sort(["track_id", "frame"])
    untracked = df.filter(pl.col("track_id").is_null())

    parts = []
    for tid in tracked["track_id"].unique(maintain_order=True).to_list():
        sub = tracked.filter(pl.col("track_id") == tid)
        sm_x, sm_y = causal_ema(sub["display_x_pitch"].to_list(), sub["display_y_pitch"].to_list(), alpha=alpha)
        parts.append(sub.with_columns([
            pl.Series("display_x_pitch", sm_x),
            pl.Series("display_y_pitch", sm_y),
        ]))

    tracked_out = pl.concat(parts) if parts else tracked
    return pl.concat([tracked_out, untracked], how="diagonal_relaxed").sort(["frame", "track_id"])


def build_full_display_frame(tracking_df: pl.DataFrame, role_window: int = 9, team_window: int = 9,
                              gap_method: str = "interp", max_gap: int = DEFAULT_MAX_INTERP_GAP,
                              apply_smoothing: bool = True, ema_alpha: float = DEFAULT_EMA_ALPHA) -> pl.DataFrame:
    """Full DEMO/ANALYTICS DISPLAY DATA pipeline in one call: role
    stabilization -> team stabilization -> gap fill -> optional mild EMA
    smoothing. Returns a new DataFrame; `tracking_df` is never mutated."""
    from legacy_analytics.role_stability import stabilize_roles_and_team
    out = stabilize_roles_and_team(tracking_df, role_window=role_window, team_window=team_window)
    out = fill_player_position_gaps(out, method=gap_method, max_gap=max_gap)
    if apply_smoothing:
        out = smooth_player_positions(out, alpha=ema_alpha)
    return out
