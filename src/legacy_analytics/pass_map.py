"""
Reusable, project-level pass-map analytics. Consumes `passes.parquet`
(`analytics/pass_detection.py`'s output) -- does NOT re-derive passes
itself, and the renderer must not derive pass arrows independently
either (Part 9/10's explicit rule: one underlying `pass_id` drives the
match overlay, the full radar, AND the pass map, never three separate
computations).

Pass maps are HISTORY/EVENT maps (Part 11) -- "which passes have
happened up to now," not the current game state (that's the full radar's
job, unchanged).

Cumulative vs. rolling window: re-compared on testVideo1_120s (v2 pass-
map fix), which has 3 total accepted passes (2 with a known team, 1
with `team_id=null`) -- i.e. AT MOST 1 pass per team's map in this whole
120s window. A rolling-60s window would actively make this worse: team
0's only pass ends at t=8.1s, so a rolling-60s map would show it only
until t=68.1s and then sit empty for the rest of the clip, even though
that pass genuinely happened and is the only tactical history this team
has. **Chosen (confirmed, not just re-asserted): cumulative-with-
opacity-fade**, `window_sec=None` by default -- a pass, once accepted,
is part of this team's history for the rest of the 120s. No clutter
risk exists at this pass count; re-run this comparison on footage with
many more passes/team before assuming cumulative still reads cleanly.

**Persistence fix (v2, then re-tuned)**: the original `MIN_OPACITY=0.15`
fade floor made an old pass arrow nearly invisible against the green
pitch after `fade_sec` (confirmed by direct pixel inspection of the v1
render at t=118s). Raised to 0.45 in v2. Phase 24 of the pass-pipeline
diagnosis asked for a further comparison against 0.55/0.65/0.75 once the
new possession-based detector produced a realistic multi-pass dataset
(rendered all three side-by-side on `outputs/analytics/
testVideo1_120s_v2` data, `visualization/tactical_maps.py::
draw_pass_map`'s dark-outline style). 0.55 was already clearly legible;
0.75 made the oldest and newest passes nearly indistinguishable in
brightness, undercutting Phase 24's other requirement ("recent passes
should be more prominent"). **Chosen: `MIN_OPACITY = 0.65`** -- every
arrow obvious at a glance, while a visible recency gradient remains.
`MAX_OPACITY` stays 1.0 (newest pass fully opaque).
"""
import polars as pl

DEFAULT_FADE_SEC = 30.0   # a pass older than this fades to the minimum opacity
MIN_OPACITY = 0.65
MAX_OPACITY = 1.0


def passes_visible_at(passes_df: pl.DataFrame, current_frame: int, team_id: int | None = None,
                       window_sec: float | None = None, fps: float = 30.0) -> pl.DataFrame:
    """Passes that have already ENDED by `current_frame` (a pass isn't
    "history" until it's finished) -- optionally filtered to one team
    (never guesses a team for a pass whose `team_id` is null; such passes
    are simply excluded from both team maps, per Part 34's "do not force
    it into Team 1/2 pass map" rule) and optionally restricted to a
    rolling window of the last `window_sec` seconds (None = cumulative)."""
    df = passes_df.filter(pl.col("end_frame") <= current_frame)
    if team_id is not None:
        df = df.filter(pl.col("team_id") == team_id)
    if window_sec is not None:
        cutoff_frame = current_frame - window_sec * fps
        df = df.filter(pl.col("end_frame") >= cutoff_frame)
    return df.sort("end_frame")


def pass_opacity(passes_df_visible: pl.DataFrame, current_frame: int, fps: float = 30.0,
                  fade_sec: float = DEFAULT_FADE_SEC) -> list[float]:
    """One opacity value per row (same order as the input), newest pass
    near MAX_OPACITY, fading linearly to MIN_OPACITY by `fade_sec`
    seconds after it ended -- never below MIN_OPACITY (an old pass stays
    faintly visible rather than vanishing, since this is cumulative
    history, not a rolling window with a hard cutoff)."""
    out = []
    for r in passes_df_visible.to_dicts():
        age_sec = (current_frame - r["end_frame"]) / fps
        frac = max(0.0, min(1.0, age_sec / fade_sec))
        out.append(MAX_OPACITY - frac * (MAX_OPACITY - MIN_OPACITY))
    return out


def build_pass_map(passes_df: pl.DataFrame, scene_id: int = 0) -> pl.DataFrame:
    """Thin, documented pass-through: ensures a `scene_id` column exists
    (defaults to 0 for a single-scene clip; a multi-scene full-video run
    should pass each pass's actual scene_id so a pass is never treated as
    visible/history across a scene cut -- see Part 22/23). Does not
    invent or drop any column `passes.parquet` already provides."""
    if passes_df.height == 0:
        return passes_df.with_columns(pl.lit(scene_id).alias("scene_id"))
    if "scene_id" not in passes_df.columns:
        return passes_df.with_columns(pl.lit(scene_id).alias("scene_id"))
    return passes_df
