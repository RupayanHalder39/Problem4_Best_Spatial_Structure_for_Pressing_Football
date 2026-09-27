"""
Reusable, project-level Voronoi / pitch-control analytics. Pure
geometry over already-computed stabilized player pitch positions
(`analytics/position_fill.py::build_full_display_frame()`'s
`display_x_pitch`/`display_y_pitch`) -- no model inference, no training,
same RAW/DERIVED separation as every other module here.

## What this approximates

An ALL-PLAYER Voronoi diagram (Phase 6's "A"): every point on the pitch
is assigned to whichever player is geographically nearest, and each
resulting cell is colored by that player's team. This is a standard,
well-understood proxy for **"nearest-player territory"** -- it is NOT a
physically-modeled pitch-control field (a real pitch-control model would
weight by player speed/acceleration/reaction time, e.g. Spearman 2018);
this module does not claim that. Cell area is reported as **"nearest-
player territory"**, and the derived team split as **pitch-control
approximation by proximity**, never bare "possession" or "control" --
same honesty convention as `analytics/heatmap.py`'s occupancy map.

## Team-only Voronoi (Phase 6's "B")

`compute_voronoi()` accepts ANY player list -- pass only one team's
players for the secondary "internal spacing" view. The dashboard uses
the ALL-PLAYER form (`voronoi_all_players()`) as the primary/default
representation per the brief ("likely best default").

## Degenerate inputs

Fewer than 4 distinct player positions cannot form a bounded Voronoi
diagram reliably (and two coincident positions make `scipy.spatial.
Voronoi` degenerate). Both cases return `{"cells": [], "valid": False,
"reason": ...}` rather than fabricating a plausible-looking wrong
result -- callers must check `valid` before using team-area figures.
"""
import numpy as np
import polars as pl
from scipy.spatial import Voronoi

PITCH_LENGTH_CM = 12000.0
PITCH_WIDTH_CM = 7000.0
_FAR = 200000.0  # mirror/padding points, far outside the pitch, to bound every real cell
_PAD_POINTS = [(-_FAR, -_FAR), (-_FAR, _FAR), (PITCH_LENGTH_CM + _FAR, -_FAR), (PITCH_LENGTH_CM + _FAR, _FAR)]


def _clip_polygon(poly: list[tuple], x0: float, y0: float, x1: float, y1: float) -> list[tuple]:
    """Sutherland-Hodgman clip of a convex polygon to the axis-aligned
    rectangle [x0,x1]x[y0,y1]. No external dependency (shapely is not
    installed in this project's venv) -- a standard, exact algorithm for
    the convex-vs-convex case, which a Voronoi cell always is."""
    def clip_edge(points, inside_fn, intersect_fn):
        out = []
        n = len(points)
        for i in range(n):
            cur, prev = points[i], points[i - 1]
            cur_in, prev_in = inside_fn(cur), inside_fn(prev)
            if cur_in:
                if not prev_in:
                    out.append(intersect_fn(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect_fn(prev, cur))
        return out

    edges = [
        (lambda p: p[0] >= x0, lambda a, b: (x0, a[1] + (b[1] - a[1]) * (x0 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= x1, lambda a, b: (x1, a[1] + (b[1] - a[1]) * (x1 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= y0, lambda a, b: (a[0] + (b[0] - a[0]) * (y0 - a[1]) / (b[1] - a[1]), y0)),
        (lambda p: p[1] <= y1, lambda a, b: (a[0] + (b[0] - a[0]) * (y1 - a[1]) / (b[1] - a[1]), y1)),
    ]
    poly = list(poly)
    for inside_fn, intersect_fn in edges:
        if not poly:
            return []
        poly = clip_edge(poly, inside_fn, intersect_fn)
    return poly


def _polygon_area(poly: list[tuple]) -> float:
    if len(poly) < 3:
        return 0.0
    x = np.array([p[0] for p in poly])
    y = np.array([p[1] for p in poly])
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def compute_voronoi(players: list[dict], pitch_length: float = PITCH_LENGTH_CM,
                     pitch_width: float = PITCH_WIDTH_CM) -> dict:
    """`players`: [{track_id, team_id, x_pitch, y_pitch}, ...] (any
    subset -- both teams for the all-player view, one team for the
    team-only view). Returns {"valid": bool, "reason": str|None,
    "cells": [{track_id, team_id, polygon: [(x,y),...], area_cm2}],
    "team_area_cm2": {team_id: float}, "team_area_fraction": {team_id:
    float}, "total_area_cm2": float}. `polygon` is already clipped to
    the pitch rectangle -- callers never need to clip again."""
    pts = [(p["x_pitch"], p["y_pitch"]) for p in players]
    if len(pts) < 4:
        return {"valid": False, "reason": f"only {len(pts)} players (need >=4)", "cells": [],
                "team_area_cm2": {}, "team_area_fraction": {}, "total_area_cm2": 0.0}
    unique_pts = {(round(x, 1), round(y, 1)) for x, y in pts}
    if len(unique_pts) < len(pts):
        return {"valid": False, "reason": "coincident player positions (degenerate)", "cells": [],
                "team_area_cm2": {}, "team_area_fraction": {}, "total_area_cm2": 0.0}

    all_pts = np.array(pts + _PAD_POINTS)
    try:
        vor = Voronoi(all_pts)
    except Exception as e:  # pragma: no cover -- qhull failures on pathological input
        return {"valid": False, "reason": f"Voronoi construction failed: {e}", "cells": [],
                "team_area_cm2": {}, "team_area_fraction": {}, "total_area_cm2": 0.0}

    cells = []
    team_area = {}
    for i, p in enumerate(players):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue  # still unbounded even with padding -- skip rather than guess
        raw_poly = [(vor.vertices[v][0], vor.vertices[v][1]) for v in region]
        clipped = _clip_polygon(raw_poly, 0.0, 0.0, pitch_length, pitch_width)
        area = _polygon_area(clipped)
        if area <= 0:
            continue
        cells.append({"track_id": p["track_id"], "team_id": p.get("team_id"),
                       "polygon": clipped, "area_cm2": area})
        tid = p.get("team_id")
        team_area[tid] = team_area.get(tid, 0.0) + area

    total_area = sum(team_area.values())
    team_fraction = {k: (v / total_area if total_area > 0 else 0.0) for k, v in team_area.items()}
    return {"valid": True, "reason": None, "cells": cells, "team_area_cm2": team_area,
            "team_area_fraction": team_fraction, "total_area_cm2": total_area}


def voronoi_all_players(display_df: pl.DataFrame, frame: int,
                         include_goalkeeper: bool = True) -> dict:
    """Convenience wrapper (Phase 6's default "ALL-PLAYER VORONOI,
    colored by team"): pulls both teams' outfield players + goalkeepers
    (referees and the ball excluded -- neither controls pitch space in
    this sense) from `display_df` at `frame` and calls `compute_voronoi`.
    Goalkeepers ARE included by default here (unlike the heatmap module)
    -- a keeper is a real spatial-control agent on the pitch; excluding
    them would leave a false "gap" of unclaimed territory in the box."""
    roles = ["player"] + (["goalkeeper"] if include_goalkeeper else [])
    rows = display_df.filter(
        (pl.col("frame") == frame) &
        pl.col("display_object_type").is_in(roles) &
        pl.col("display_x_pitch").is_not_null()
    ).select(["track_id", "display_team_id", "display_x_pitch", "display_y_pitch"]).rename(
        {"display_team_id": "team_id", "display_x_pitch": "x_pitch", "display_y_pitch": "y_pitch"}
    )
    return compute_voronoi(rows.to_dicts())


def voronoi_team_only(display_df: pl.DataFrame, frame: int, team_id: int,
                       include_goalkeeper: bool = True) -> dict:
    """Phase 6's secondary "B" internal-spacing view: Voronoi over just
    one team's own players (partitions the WHOLE pitch by nearest
    same-team player, not the true contested space -- useful for judging
    a team's own coverage/spacing, not for a Team A vs Team B split)."""
    roles = ["player"] + (["goalkeeper"] if include_goalkeeper else [])
    rows = display_df.filter(
        (pl.col("frame") == frame) & (pl.col("display_team_id") == team_id) &
        pl.col("display_object_type").is_in(roles) &
        pl.col("display_x_pitch").is_not_null()
    ).select(["track_id", "display_team_id", "display_x_pitch", "display_y_pitch"]).rename(
        {"display_team_id": "team_id", "display_x_pitch": "x_pitch", "display_y_pitch": "y_pitch"}
    )
    return compute_voronoi(rows.to_dicts())


def pitch_control_summary_series(tracking_df: pl.DataFrame, display_df: pl.DataFrame,
                                  stride: int = 5) -> pl.DataFrame:
    """Reusable DERIVED summary (Phase 7): team pitch-control area/
    fraction sampled every `stride` frames across the whole clip (full
    per-frame polygons are NOT persisted -- they're cheap to recompute
    live, same precedent as `analytics/team_graph.py`'s per-frame graph;
    persisting hundreds of polygon vertices per frame for every frame
    would bloat storage for no reuse benefit). Columns: frame,
    timestamp_sec, valid, team_0_area_cm2, team_1_area_cm2,
    team_0_fraction, team_1_fraction, dominant_team_id."""
    frames = sorted(tracking_df["frame"].unique().to_list())[::stride]
    rows = []
    for f in frames:
        result = voronoi_all_players(display_df, f)
        row = {"frame": f, "timestamp_sec": round(f / 30.0, 3), "valid": result["valid"]}
        if result["valid"]:
            a0 = result["team_area_cm2"].get(0, 0.0)
            a1 = result["team_area_cm2"].get(1, 0.0)
            f0 = result["team_area_fraction"].get(0, 0.0)
            f1 = result["team_area_fraction"].get(1, 0.0)
            row.update({"team_0_area_cm2": a0, "team_1_area_cm2": a1,
                        "team_0_fraction": f0, "team_1_fraction": f1,
                        "dominant_team_id": (0 if a0 >= a1 else 1) if (a0 or a1) else None})
        else:
            row.update({"team_0_area_cm2": None, "team_1_area_cm2": None,
                        "team_0_fraction": None, "team_1_fraction": None, "dominant_team_id": None})
        rows.append(row)
    return pl.DataFrame(rows)
