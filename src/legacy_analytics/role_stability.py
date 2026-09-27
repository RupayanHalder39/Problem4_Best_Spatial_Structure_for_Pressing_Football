"""
Reusable player role/team-id temporal stabilization, extracted from the
Phase 7 professor-demo QA work (`outputs/demo/testVideo1_20s/
demo_validation_summary.md` Parts C/D) so it has exactly one
implementation shared by every consumer -- the old single-panel demo
(`scripts/demo_display_layer.py`), the new dashboard analytics pipeline,
and anything else built on top of tracking.parquet later.

CRITICAL: operates on a DERIVED DataFrame, never mutates the raw
`object_type`/`team_id` columns. Adds `display_object_type`/
`display_team_id` columns only.

Evidence for the window default (window=9, causal, no lookahead): see
demo_validation_summary.md Part C -- on testVideo1_20s this eliminated
100% of the goalkeeper track's role/team flicker (12->0, 16->0 flips)
without ever being able to flip a genuinely-stable track, since a
majority vote over 9 observations cannot be swayed by 1-2 noisy frames.
"""
import polars as pl

from legacy_analytics.temporal_utils import causal_rolling_majority

DEFAULT_ROLE_WINDOW = 9
DEFAULT_TEAM_WINDOW = 9


def stabilize_roles(df: pl.DataFrame, window: int = DEFAULT_ROLE_WINDOW) -> pl.DataFrame:
    """Adds `display_object_type` = causal rolling-majority vote of
    `object_type` per track_id. Ball rows (track_id is null) pass
    through unchanged. Does not touch `object_type`."""
    tracked = df.filter(pl.col("track_id").is_not_null()).sort(["track_id", "frame"])
    untracked = df.filter(pl.col("track_id").is_null()).with_columns(
        pl.col("object_type").alias("display_object_type")
    )

    parts = []
    for tid in tracked["track_id"].unique(maintain_order=True).to_list():
        sub = tracked.filter(pl.col("track_id") == tid)
        stabilized = causal_rolling_majority(sub["object_type"].to_list(), window)
        parts.append(sub.with_columns(pl.Series("display_object_type", stabilized)))
    tracked_out = pl.concat(parts) if parts else tracked.with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("display_object_type"))

    return pl.concat([tracked_out, untracked], how="diagonal_relaxed").sort(["frame", "track_id"])


def stabilize_team(df: pl.DataFrame, window: int = DEFAULT_TEAM_WINDOW) -> pl.DataFrame:
    """Adds `display_team_id` = causal rolling-majority vote of `team_id`
    per track_id. Does not touch `team_id`. Expects `df` to already carry
    whatever other columns the caller wants preserved (e.g. after
    stabilize_roles())."""
    tracked = df.filter(pl.col("track_id").is_not_null()).sort(["track_id", "frame"])
    untracked = df.filter(pl.col("track_id").is_null()).with_columns(
        pl.col("team_id").alias("display_team_id")
    )

    parts = []
    for tid in tracked["track_id"].unique(maintain_order=True).to_list():
        sub = tracked.filter(pl.col("track_id") == tid)
        stabilized = causal_rolling_majority(sub["team_id"].to_list(), window)
        parts.append(sub.with_columns(pl.Series("display_team_id", stabilized)))
    tracked_out = pl.concat(parts) if parts else tracked.with_columns(
        pl.lit(None, dtype=pl.Int64).alias("display_team_id"))

    return pl.concat([tracked_out, untracked], how="diagonal_relaxed").sort(["frame", "track_id"])


def stabilize_roles_and_team(df: pl.DataFrame, role_window: int = DEFAULT_ROLE_WINDOW,
                              team_window: int = DEFAULT_TEAM_WINDOW) -> pl.DataFrame:
    """Convenience wrapper: stabilize_roles() then stabilize_team()."""
    return stabilize_team(stabilize_roles(df, window=role_window), window=team_window)
