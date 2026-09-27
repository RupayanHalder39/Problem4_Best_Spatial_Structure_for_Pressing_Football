"""
Reusable, simple team-shape metrics (centroid, width, depth, compactness,
line positions) -- foundation analytics only, deliberately not a Journal-
scale tactical model. Uses the same STABILIZED role/team + filled
positions as `analytics/formation.py` so the two stay consistent.
"""
import polars as pl

from legacy_analytics.formation import estimate_attack_direction
from legacy_analytics.position_fill import build_full_display_frame


def compute_team_shape_for_window(display_df: pl.DataFrame, team_id: int,
                                   frame_start: int, frame_end: int,
                                   attack_direction: int | None = None) -> dict:
    """Simple, interpretable per-team shape metrics averaged over
    [frame_start, frame_end], EXCLUDING the goalkeeper (outfield shape
    only, per spec) and computed on the already team/role-stabilized,
    short-gap-filled `display_x_pitch`/`display_y_pitch` columns."""
    if attack_direction is None:
        attack_direction = estimate_attack_direction(display_df, team_id)

    window = display_df.filter(
        (pl.col("frame") >= frame_start) & (pl.col("frame") <= frame_end) &
        (pl.col("display_team_id") == team_id) &
        (pl.col("display_object_type") == "player") &
        pl.col("display_x_pitch").is_not_null()
    )
    if window.height == 0:
        return {"team_id": team_id, "window_start_frame": frame_start, "window_end_frame": frame_end,
                "visible_player_frames": 0, "centroid_x": None, "centroid_y": None,
                "width_cm": None, "depth_cm": None, "defensive_line_x": None,
                "midfield_line_x": None, "attacking_line_x": None, "compactness_cm": None,
                "method": "insufficient data"}

    xs = window["display_x_pitch"].to_list()
    ys = window["display_y_pitch"].to_list()
    line_x = [x if attack_direction == 1 else (12000.0 - x) for x in xs]

    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)
    width_cm = max(ys) - min(ys)          # spread across the pitch WIDTH (touchline to touchline)
    depth_cm = max(line_x) - min(line_x)  # spread along the attack axis (own goal to opponent goal)

    sorted_line_x = sorted(line_x)
    n = len(sorted_line_x)
    defensive_line_x = sum(sorted_line_x[:max(1, n // 3)]) / max(1, n // 3)
    attacking_line_x = sum(sorted_line_x[-max(1, n // 3):]) / max(1, n // 3)
    mid_slice = sorted_line_x[n // 3: n - n // 3] or sorted_line_x
    midfield_line_x = sum(mid_slice) / len(mid_slice)

    mean_x = sum(line_x) / n
    mean_y = centroid_y
    compactness_cm = (sum(((lx - mean_x) ** 2 + (y - mean_y) ** 2) ** 0.5
                           for lx, y in zip(line_x, ys)) / n)

    return {"team_id": team_id, "window_start_frame": frame_start, "window_end_frame": frame_end,
            "visible_player_frames": window.height, "centroid_x": round(centroid_x, 1),
            "centroid_y": round(centroid_y, 1), "width_cm": round(width_cm, 1),
            "depth_cm": round(depth_cm, 1), "defensive_line_x": round(defensive_line_x, 1),
            "midfield_line_x": round(midfield_line_x, 1), "attacking_line_x": round(attacking_line_x, 1),
            "compactness_cm": round(compactness_cm, 1),
            "method": "mean centroid / min-max spread / tercile line positions over the window, "
                      "outfield players only (goalkeeper excluded), attack-axis-oriented"}


def compute_team_shape_series(tracking_df: pl.DataFrame, team_id: int,
                               window_frames: int = 90, step_frames: int | None = None,
                               display_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Same windowing convention as formation.estimate_formation_series,
    so the two can be shown/joined side by side."""
    step_frames = step_frames or window_frames
    if display_df is None:
        display_df = build_full_display_frame(tracking_df)
    attack_direction = estimate_attack_direction(display_df, team_id)

    fmin, fmax = int(tracking_df["frame"].min()), int(tracking_df["frame"].max())
    rows = []
    f = fmin
    while f <= fmax:
        f_end = min(f + window_frames - 1, fmax)
        rows.append(compute_team_shape_for_window(display_df, team_id, f, f_end,
                                                    attack_direction=attack_direction))
        f += step_frames
    return pl.DataFrame(rows)


def team_shape_at_frame(shape_series: pl.DataFrame, frame: int) -> dict:
    if shape_series.height == 0:
        return {}
    row = shape_series.filter((pl.col("window_start_frame") <= frame) & (pl.col("window_end_frame") >= frame))
    if row.height == 0:
        row = shape_series.sort((pl.col("window_start_frame") - frame).abs()).head(1)
    return row.to_dicts()[0]
