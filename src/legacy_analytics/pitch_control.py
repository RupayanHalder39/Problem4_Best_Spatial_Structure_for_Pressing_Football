"""
Reusable, project-level CONTINUOUS pitch-control analytics -- distinct
from `analytics/voronoi.py`, not a re-labeling of it.

## Voronoi vs. pitch control (the distinction that matters here)

`analytics/voronoi.py` answers "whose is this point, right now, based
on who is CLOSEST" -- a discrete polygon ownership map with hard
boundaries (nearest-player territory).

THIS module answers a different, genuinely harder question: "which
TEAM could get a player to this point FASTEST" -- a continuous
probability field with smooth gradients, no hard edges. Two players
equidistant-but-facing-opposite-directions have different real control
of a point; Voronoi cannot express that (it only knows position), this
module can (it uses velocity too). The dashboard shows BOTH, correctly
distinguished: Voronoi on the right-side match video (Phase 6 of the
Voronoi work), this continuous field on the "PITCH CONTROL" radar panel
(Phase 13 of that same work explicitly asked for them to not be
duplicates of each other -- this module is what makes that true).

## Method (simplified time-to-intercept + logistic blend)

For each grid point p and each player i:
  effective_position_i = position_i + velocity_i * REACTION_TIME_SEC
  (a short constant-velocity extrapolation -- "where they'll actually
  start moving toward p from," not simply where they stand right now)
  time_to_reach_i(p) = |p - effective_position_i| / PLAYER_MAX_SPEED_CM_S

Per team, take the FASTEST responder: T_team(p) = min_i time_to_reach_i(p).
Team A's control probability at p is a logistic function of the time
ADVANTAGE: control_A(p) = sigmoid((T_B(p) - T_A(p)) / CONTROL_SIGMA_SEC)
-- smoothly 0.5 when both teams could arrive equally fast, smoothly
approaching 1 (or 0) as one team's advantage grows, continuous
everywhere (no polygon edges).

**Honesty constraint**: this is a SIMPLIFIED, single-fastest-responder
model, not the full multi-player influence integral of Spearman (2018)
or Fernandez & Bornn (2018) -- it does not account for every player's
partial influence, ball position, or a real reaction-time distribution.
It is a standard, well-understood simplification (the same "who gets
there first" core idea), reported as exactly that -- never as a
validated professional pitch-control model.

## Speed calibration (measured, not textbook)

`PLAYER_MAX_SPEED_CM_S = 800.0` -- the 95th percentile of this specific
clip's own frame-to-frame player displacement speed (8.24 m/s measured
directly from `tracking.parquet`, `outputs/analytics/testVideo1_120s_v3/
pitch_control_calibration.md`), not an assumed textbook constant. The
99th+ percentiles (12.9-53.5 m/s) are almost certainly tracking
jitter/mis-association, not real human speed, and were excluded from
the calibration the same way the ball tracker's own speed gate treats
implausible spikes as noise rather than evidence.
"""
import numpy as np
import polars as pl

PITCH_LENGTH_CM = 12000.0
PITCH_WIDTH_CM = 7000.0
PLAYER_MAX_SPEED_CM_S = 800.0
REACTION_TIME_SEC = 0.3
CONTROL_SIGMA_SEC = 0.5
DEFAULT_GRID_COLS = 48
DEFAULT_GRID_ROWS = 28


def estimate_player_velocities(display_df: pl.DataFrame, frame: int, fps: float = 30.0,
                                lookback_frames: int = 5) -> dict:
    """Finite-difference velocity estimate per track_id: compares
    `frame` to the nearest available earlier frame within
    `lookback_frames` (never looks forward -- causal, like every other
    per-frame estimator in this project). Returns {track_id: (vx, vy)}
    in cm/s; a track with no earlier frame available is simply absent
    (callers treat missing velocity as (0, 0) -- a stationary
    assumption, not a guessed direction)."""
    cur = display_df.filter(
        (pl.col("frame") == frame) & pl.col("display_x_pitch").is_not_null()
    ).select(["track_id", "display_x_pitch", "display_y_pitch"])
    velocities = {}
    if cur.height == 0:
        return velocities
    for lb in range(1, lookback_frames + 1):
        prev = display_df.filter(
            (pl.col("frame") == frame - lb) & pl.col("display_x_pitch").is_not_null()
        ).select(["track_id", "display_x_pitch", "display_y_pitch"])
        if prev.height == 0:
            continue
        joined = cur.join(prev, on="track_id", suffix="_prev")
        dt = lb / fps
        for r in joined.iter_rows(named=True):
            if r["track_id"] in velocities:
                continue
            vx = (r["display_x_pitch"] - r["display_x_pitch_prev"]) / dt
            vy = (r["display_y_pitch"] - r["display_y_pitch_prev"]) / dt
            velocities[r["track_id"]] = (vx, vy)
        if len(velocities) == cur.height:
            break
    return velocities


def compute_pitch_control_grid(team_a_players: list[dict], team_b_players: list[dict],
                                grid_cols: int = DEFAULT_GRID_COLS, grid_rows: int = DEFAULT_GRID_ROWS,
                                max_speed_cm_s: float = PLAYER_MAX_SPEED_CM_S,
                                reaction_time_sec: float = REACTION_TIME_SEC,
                                sigma_sec: float = CONTROL_SIGMA_SEC,
                                pitch_length: float = PITCH_LENGTH_CM,
                                pitch_width: float = PITCH_WIDTH_CM) -> dict:
    """`team_a_players`/`team_b_players`: [{track_id, x_pitch, y_pitch,
    vx, vy}, ...] (vx/vy in cm/s, default 0 if unknown). Returns
    {"valid": bool, "reason": str|None, "grid": (rows,cols) ndarray of
    Team A's control probability in [0,1] (Team B = 1-grid),
    "cell_centers_x"/"cell_centers_y": 1D ndarrays, "team_a_mean_control":
    float, "team_b_mean_control": float, "team_a_majority_fraction":
    float (share of cells where Team A's probability > 0.5)}."""
    if not team_a_players or not team_b_players:
        return {"valid": False, "reason": "one team has zero visible players", "grid": None,
                "cell_centers_x": None, "cell_centers_y": None,
                "team_a_mean_control": None, "team_b_mean_control": None, "team_a_majority_fraction": None}

    cell_w = pitch_length / grid_cols
    cell_h = pitch_width / grid_rows
    xs = (np.arange(grid_cols) + 0.5) * cell_w
    ys = (np.arange(grid_rows) + 0.5) * cell_h
    grid_x, grid_y = np.meshgrid(xs, ys)  # (rows, cols)

    def team_min_time(players):
        min_t = np.full(grid_x.shape, np.inf)
        for p in players:
            eff_x = p["x_pitch"] + p.get("vx", 0.0) * reaction_time_sec
            eff_y = p["y_pitch"] + p.get("vy", 0.0) * reaction_time_sec
            dist = np.hypot(grid_x - eff_x, grid_y - eff_y)
            t = dist / max_speed_cm_s
            min_t = np.minimum(min_t, t)
        return min_t

    t_a = team_min_time(team_a_players)
    t_b = team_min_time(team_b_players)
    control_a = 1.0 / (1.0 + np.exp(-(t_b - t_a) / sigma_sec))

    return {
        "valid": True, "reason": None, "grid": control_a,
        "cell_centers_x": xs, "cell_centers_y": ys,
        "team_a_mean_control": float(control_a.mean()),
        "team_b_mean_control": float(1.0 - control_a.mean()),
        "team_a_majority_fraction": float((control_a > 0.5).mean()),
    }


def pitch_control_from_display_df(display_df: pl.DataFrame, frame: int, fps: float = 30.0,
                                   include_goalkeeper: bool = True, **kwargs) -> dict:
    """Convenience wrapper: pulls both teams' current positions +
    estimated velocities from `display_df` at `frame` and calls
    `compute_pitch_control_grid`."""
    roles = ["player"] + (["goalkeeper"] if include_goalkeeper else [])
    rows = display_df.filter(
        (pl.col("frame") == frame) & pl.col("display_object_type").is_in(roles) &
        pl.col("display_x_pitch").is_not_null()
    ).select(["track_id", "display_team_id", "display_x_pitch", "display_y_pitch"])
    velocities = estimate_player_velocities(display_df, frame, fps=fps)

    team_a, team_b = [], []
    for r in rows.iter_rows(named=True):
        vx, vy = velocities.get(r["track_id"], (0.0, 0.0))
        entry = {"track_id": r["track_id"], "x_pitch": r["display_x_pitch"],
                 "y_pitch": r["display_y_pitch"], "vx": vx, "vy": vy}
        if r["display_team_id"] == 0:
            team_a.append(entry)
        elif r["display_team_id"] == 1:
            team_b.append(entry)
    return compute_pitch_control_grid(team_a, team_b, **kwargs)
