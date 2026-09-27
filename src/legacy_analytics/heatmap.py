"""
Reusable, project-level spatial-occupancy analytics. NOT possession, NOT
a trained model -- a simple grid-binned histogram of where a team's
outfield players have stood, optionally Gaussian-smoothed for readability.
Correct terminology used everywhere this is surfaced: **TEAM SPATIAL
OCCUPANCY HEATMAP** ("player occupancy" as the dashboard subtitle) --
never "possession," which this data cannot support (it has no ball-
control information at all).

No future leakage (Part 16): `compute_occupancy_grid(..., frame_end=t)`
only ever aggregates frames `<= t` -- called once per displayed frame
during rendering, or the caller must re-call it per frame rather than
computing one grid from the whole clip and reusing it everywhere.
"""
import numpy as np
import polars as pl

from legacy_analytics.position_fill import build_full_display_frame

PITCH_LENGTH_CM = 12000.0
PITCH_WIDTH_CM = 7000.0
DEFAULT_GRID_COLS = 48
DEFAULT_GRID_ROWS = 28   # 250cm x 250cm cells at the defaults -- square cells
DEFAULT_WINDOW_SEC = 60.0
# ^ Grid resolution (v2 refinement): compared 24x14/sigma=1.0 (original),
# 48x28/sigma=0.6, 48x28/sigma=0.9, and 60x35/sigma=0.5 side-by-side, both
# at native render resolution and downscaled to the actual 640x394 bottom-
# panel size the dashboard uses (a fine grid that only looks good before
# the panel-resize downscale isn't actually useful). 24x14/sigma=1.0
# produced one large, low-detail blob (the reported complaint). 60x35/
# sigma=0.5 was noticeably speckled/noisy even after downscale. 48x28/
# sigma=0.6 struck the best balance: individual occupancy streaks are
# visible (not one blob), still reads as a smooth map rather than raw
# pixel noise, and holds up after the panel downscale. Chosen as the new
# default; the halved cell size (250cm vs 500cm) is the main driver, the
# reduced smoothing (0.6 vs 1.0 sigma) is secondary.
# ^ Part 16 asked for a visual comparison of cumulative vs. rolling-60s.
# Chosen: ROLLING 60s by default. Rationale (not just habit): a
# cumulative occupancy grid over a multi-minute video converges toward
# "roughly the team's average shape," which is real and used in
# broadcast analysis, but is less useful for a LIVE dashboard tile that
# should reflect the CURRENT phase of play (a team pinned in its own
# half for the last minute should show that, not be diluted by data from
# 4 minutes ago). Exposed as a parameter, not hardcoded -- pass
# `window_sec=None` for cumulative if a specific use case wants it, and
# re-validate visually before trusting either choice on a very different
# video (this was only tested on testVideo1's ~60-90s scale).


def compute_occupancy_grid(tracking_df: pl.DataFrame, team_id: int, frame_end: int,
                            window_sec: float | None = DEFAULT_WINDOW_SEC, fps: float = 30.0,
                            grid_cols: int = DEFAULT_GRID_COLS, grid_rows: int = DEFAULT_GRID_ROWS,
                            include_goalkeeper: bool = False, gaussian_sigma_cells: float = 0.6,
                            display_df: pl.DataFrame | None = None) -> np.ndarray:
    """Returns a (grid_rows, grid_cols) float array, normalized to [0, 1]
    (max cell = 1.0). Uses STABILIZED role/team + short-gap-filled
    positions (`analytics/position_fill.py`), same as every other
    analytics module -- never the raw, flicker-prone columns.

    Excludes: referees, ball, the opposing team, and (by default,
    `include_goalkeeper=False`) the goalkeeper -- Part 15's explicit
    concern: a goalkeeper standing near one small area for the whole
    clip could dominate a cell and wash out the outfield pattern that's
    actually informative. Measured both ways on testVideo1_60s (not just
    assumed): on THIS sample the goalkeeper's own peak cell count (46)
    was far below the outfield peak (508) and didn't change the overall
    max at all -- because this team's goalkeeper was only tracked for
    230 of the clip's 1800 frames (sparse ball-side-of-play detection in
    this excerpt), so inclusion happened not to distort this particular
    60s window. Excluding by default anyway, on the general reasoning
    that a goalkeeper's positional variance is much lower than an
    outfield player's over LONGER footage (a full match would track them
    far more continuously near their own goal) -- re-measure on a longer
    clip before assuming this is a non-issue there too."""
    if display_df is None:
        display_df = build_full_display_frame(tracking_df)

    roles = ["player"] + (["goalkeeper"] if include_goalkeeper else [])
    frame_start = 0 if window_sec is None else max(0, frame_end - int(window_sec * fps))
    rows = display_df.filter(
        (pl.col("frame") >= frame_start) & (pl.col("frame") <= frame_end) &
        (pl.col("display_team_id") == team_id) &
        (pl.col("display_object_type").is_in(roles)) &
        pl.col("display_x_pitch").is_not_null()
    )

    grid = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    if rows.height == 0:
        return grid

    xs = rows["display_x_pitch"].to_numpy()
    ys = rows["display_y_pitch"].to_numpy()
    col = np.clip((xs / PITCH_LENGTH_CM * grid_cols).astype(int), 0, grid_cols - 1)
    row = np.clip((ys / PITCH_WIDTH_CM * grid_rows).astype(int), 0, grid_rows - 1)
    np.add.at(grid, (row, col), 1.0)

    if gaussian_sigma_cells > 0:
        grid = _gaussian_blur(grid, gaussian_sigma_cells)

    peak = grid.max()
    if peak > 0:
        grid = grid / peak
    return grid


def _gaussian_blur(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Simple separable Gaussian smoothing -- no external dependency
    beyond numpy, no training, just a standard readability pass so the
    grid doesn't look like a raw blocky histogram."""
    radius = max(1, int(round(sigma * 3)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=grid)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=out)
    return out


def grid_to_pitch_cells(grid: np.ndarray) -> list[dict]:
    """Flattens a grid into a list of {x0,y0,x1,y1,value} cells in pitch
    cm coordinates, for the renderer to draw without needing to know the
    grid's shape/scale itself."""
    rows, cols = grid.shape
    cell_w = PITCH_LENGTH_CM / cols
    cell_h = PITCH_WIDTH_CM / rows
    out = []
    for r in range(rows):
        for c in range(cols):
            v = float(grid[r, c])
            if v <= 0:
                continue
            out.append({"x0": c * cell_w, "y0": r * cell_h, "x1": (c + 1) * cell_w,
                        "y1": (r + 1) * cell_h, "value": v})
    return out
