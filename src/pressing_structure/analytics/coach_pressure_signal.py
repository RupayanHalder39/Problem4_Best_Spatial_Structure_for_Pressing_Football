"""
Coach-mode continuous team-pressure signal (`opp_ballw`), pressure
dominance, and the outcome-bar dynamic-axis helper.

Everything in this module is NEW and ADDITIVE: it is a presentation/
derived-signal layer built on top of the UNCHANGED, existing
`team_pressure_field`/`sample_field` (pressure_field.py). No existing
pressing analytics formula, threshold, or FSM logic (pressing_v4.py)
is imported, duplicated, retuned, or modified here.

This implements exactly the definition validated by the two completed
feasibility audits:
  audit_reports/2026-09-09_coach_graph_feasibility/
    COACH_GRAPH_FEASIBILITY_AUDIT.md
  audit_reports/2026-09-09_continuous_pressure_signal_audit/
    CONTINUOUS_PRESSURE_SIGNAL_AUDIT.md

opp_ballw(team T, frame f):
    1. team_field = team_pressure_field(team T's own tracked players, gx, gy)
       (existing, unchanged function -- no carrier/eligibility gating).
    2. Sample team_field AT each tracked OPPONENT (team 1-T) player's
       own pitch position (existing `sample_field`, unchanged).
    3. If the ball's pitch position is observed this frame: weight
       each opponent by a fixed, disclosed Gaussian on distance to the
       ball (sigma=BALL_PROXIMITY_SIGMA_CM=1000cm=10m) and take the
       weighted mean.
    4. If the ball is NOT observed this frame: fall back explicitly to
       the UNWEIGHTED mean over the same opponent-sampled values (this
       fallback is part of the audited recommendation, not an ad hoc
       addition) -- the caller can always tell which path was taken
       via the returned `mode`.
    5. If team T has no tracked players, or there are no tracked
       opponents to sample at, the value is MISSING (`None`) -- never
       fabricated, never defaulted to 0.
"""
import math

from pressing_structure.analytics.pressure_field import sample_field, team_pressure_field

BALL_PROXIMITY_SIGMA_CM = 1000.0  # 10m -- audited, disclosed, fixed; never tuned per clip

MODE_BALL_WEIGHTED = "BALL_WEIGHTED"
MODE_UNWEIGHTED_FALLBACK = "UNWEIGHTED_FALLBACK"
MODE_MISSING = "MISSING"


def opp_ballw_value(team_field, opponents, ball_xy, gx, gy, sigma_cm=BALL_PROXIMITY_SIGMA_CM):
    """opp_ballw for ONE team, ONE frame.

    `team_field`: that team's own `team_pressure_field(...)` output (an
    array in [0,1)), or None if that team has no tracked geometry this
    frame. `opponents`: the OTHER team's tracked player rows (each a
    dict with 'x_pitch'/'y_pitch') -- must be the opposing team, never
    this team's own players. `ball_xy`: (x_cm, y_cm) pitch coordinate,
    or None if the ball is not observed this frame.

    Returns `(value_or_None, mode)`.
    """
    if team_field is None or not opponents:
        return None, MODE_MISSING
    vals = [sample_field(team_field, gx, gy, p["x_pitch"], p["y_pitch"]) for p in opponents]
    if ball_xy is None:
        return sum(vals) / len(vals), MODE_UNWEIGHTED_FALLBACK
    bx, by = ball_xy
    weights = [
        math.exp(-((p["x_pitch"] - bx) ** 2 + (p["y_pitch"] - by) ** 2) / (2 * sigma_cm ** 2))
        for p in opponents
    ]
    total_w = sum(weights)
    if total_w <= 1e-9:
        # Numerically-zero weights (all opponents effectively infinitely far
        # from the ball on this grid) -- fall back rather than divide by ~0.
        return sum(vals) / len(vals), MODE_UNWEIGHTED_FALLBACK
    weighted = sum(w * v for w, v in zip(weights, vals)) / total_w
    return weighted, MODE_BALL_WEIGHTED


def compute_opp_ballw_series(players_by_frame, ball_by_frame, frame_times, gx, gy, sigma_cm=BALL_PROXIMITY_SIGMA_CM):
    """Builds the full-clip opp_ballw series for BOTH teams in one pass.

    `players_by_frame`: {frame_idx: [player_row, ...]}, each row a dict
    with 'x_pitch'/'y_pitch'/'display_team_id'/'display_object_type'
    (the same shape `pressing_v4.build_pressing_v4` itself consumes).
    `ball_by_frame`: {frame_idx: ball_row_or_missing}, each row (when
    present) a dict with 'is_observed'/'x_pitch'/'y_pitch'.
    `frame_times`: ordered `[(frame_idx, time_sec), ...]`.

    Returns `{0: {"values": [(t, v_or_None), ...], "modes": [mode, ...]},
              1: {...}}`. Team 0's value samples TEAM 0's OWN field at
    TEAM 1's player positions, and vice versa -- never mixed.
    """
    out = {0: {"values": [], "modes": []}, 1: {"values": [], "modes": []}}
    for f, t in frame_times:
        players = players_by_frame.get(f, [])
        ball = ball_by_frame.get(f)
        ball_xy = (
            (ball["x_pitch"], ball["y_pitch"])
            if ball and ball.get("is_observed") and ball.get("x_pitch") is not None
            else None
        )
        team_players = {
            side: [
                p for p in players
                if p.get("display_team_id") == side and p.get("display_object_type") in ("player", "goalkeeper")
                and p.get("x_pitch") is not None
            ]
            for side in (0, 1)
        }
        fields = {
            side: (team_pressure_field(team_players[side], gx, gy) if team_players[side] else None)
            for side in (0, 1)
        }
        for side in (0, 1):
            opponents = team_players[1 - side]
            v, mode = opp_ballw_value(fields[side], opponents, ball_xy, gx, gy, sigma_cm=sigma_cm)
            out[side]["values"].append((t, v))
            out[side]["modes"].append(mode)
    return out


def compute_dominance(full_a, full_b):
    """D_raw(t) = I_A(t) - I_B(t), naturally in [-1, 1] when I_A/I_B are
    each in [0, 1]. `full_a`/`full_b`: `[(t, v_or_None), ...]` on the
    SAME time grid. Missing either side -> missing dominance (never
    fabricated, never assumed to be 0 or the other side's value)."""
    out = []
    for (ta, va), (tb, vb) in zip(full_a, full_b):
        assert abs(ta - tb) < 1e-6, "dominance requires both series on the same time grid"
        out.append((ta, None if (va is None or vb is None) else va - vb))
    return out


def compute_max_intensity(full_a, full_b):
    """max(I_A, I_B) per frame -- used only for low-pressure gating.
    Missing either side -> missing (the gate must never assume a
    missing side was "low")."""
    out = []
    for (ta, va), (tb, vb) in zip(full_a, full_b):
        assert abs(ta - tb) < 1e-6, "max-intensity requires both series on the same time grid"
        out.append((ta, None if (va is None or vb is None) else max(va, vb)))
    return out


LOW_PRESSURE = "LOW_PRESSURE"  # sentinel: "known to be low/neutral pressure", distinct from a real gap (None)


def gate_dominance(full_d, full_max_intensity, threshold):
    """Returns a NEW series (never mutates `full_d`) where any point
    with `max_intensity < threshold` is replaced with the sentinel
    `LOW_PRESSURE` -- it never fabricates a value of 0.0 (a real,
    meaningful dominance reading) and never silently drops the point.
    A real gap (`None`) always stays `None`, gated or not.
    `threshold=None` -> returns `full_d` unchanged (ungated passthrough).
    This is an EXPERIMENTAL, disclosed, non-universal gate -- callers
    must pass a threshold explicitly; nothing here treats any value as
    a validated universal constant."""
    if threshold is None:
        return list(full_d)
    out = []
    for (td, vd), (tm, vm) in zip(full_d, full_max_intensity):
        assert abs(td - tm) < 1e-6, "gating requires dominance and max-intensity on the same time grid"
        if vd is None:
            out.append((td, None))
        elif vm is not None and vm < threshold:
            out.append((td, LOW_PRESSURE))
        else:
            out.append((td, vd))
    return out


# ---------------------------------------------------------------------------
# Outcome-bar dynamic Y-axis: audited nice-ceiling table + monotonic
# (never-shrinks) expansion. Same design validated in the first
# feasibility audit (COACH_GRAPH_FEASIBILITY_AUDIT.md, sections D.2/D.3).
NICE_AXIS_SCALE = (1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000)


def nice_bar_axis(max_count):
    """Smallest table ceiling STRICTLY greater than `max_count` (falls
    back to the next power-of-ten beyond the table for very large
    counts -- not expected on this project's real per-match data, but
    handled rather than raising)."""
    if max_count <= 0:
        return 1
    for s in NICE_AXIS_SCALE:
        if s > max_count:
            return s
    return int(10 ** math.ceil(math.log10(max_count)))


def monotonic_nice_axis(max_count_so_far, current_ymax):
    """Never shrinks: only grows `current_ymax` when the real count
    exceeds it, and even then only to the next nice ceiling. Kept as
    the explicit, defensive design for the (currently hypothetical, 0
    occurrences in real data today) case where a later
    human/adjudicated relabel changes an outcome's category after it
    was already counted -- see the first audit's D.3 finding: on a
    strictly non-decreasing count stream this is mathematically
    identical to calling `nice_bar_axis(max_count_so_far)` directly,
    but this explicit form is the one that keeps holding once counts
    are no longer guaranteed non-decreasing."""
    if max_count_so_far >= current_ymax:
        return nice_bar_axis(max_count_so_far)
    return current_ymax


# ---------------------------------------------------------------------------
# 2026-09-09 -- COACH-MODE Graph 1/2 DISPLAY-DENSITY decimation. This is a
# presentation-only downsampling stage applied to the FINAL (already
# causally-EMA'd) series, immediately before drawing -- it never touches
# the underlying opp_ballw computation, the EMA smoothing itself, event
# detection, outcome timing, or the pressure field. Order is always:
#   full-resolution real series -> causal EMA -> display decimation -> draw
# never the reverse (decimating BEFORE EMA would change what the EMA
# itself smooths over, silently altering the analytical result).
DEFAULT_DISPLAY_HZ = 10.0


def decimate_for_display(hist, display_hz=DEFAULT_DISPLAY_HZ, max_gap_sec=0.5):
    """Downsamples `hist` (`[(t, v_or_None_or_sentinel), ...]`, real
    timestamps, already EMA'd if desired) to at most one plotted point
    per `1/display_hz`-second time bucket (`bucket = floor(t /
    (1/display_hz))` -- a deterministic, TIMESTAMP-based rule, correct
    regardless of the source frame rate, never an every-Nth-array-index
    rule). Within each bucket, keeps the LAST real sample (the most
    recent causal value at that display instant) -- never an average,
    never a future-centered choice, never a fabricated in-between value.

    `display_hz >= 30` (or `None`) is a full-density passthrough
    (returns `hist` unchanged) -- used for `--coach-graph-display-hz 30`
    (debug/full-density mode).

    Missing-segment preservation (critical, tested explicitly): a real
    gap in the input -- a `None`-valued sample, OR a timestamp jump
    exceeding `max_gap_sec` between two valid samples -- decimation
    NEVER bridges it. A `None` sample's own timestamp is re-emitted
    verbatim in the output (so the exact count of genuine missing
    segments is unchanged by decimation); a pure timestamp-jump gap
    (no `None` sample recorded, just two valid runs far apart in real
    time) survives automatically, because decimation only THINS a
    run's own internal samples -- it never moves, merges, or removes a
    run's first/last real timestamp, so two runs that were more than
    `max_gap_sec` apart before decimation are still that far apart
    (or farther, never closer) after it."""
    if display_hz is None or display_hz >= 30:
        return list(hist)
    bucket_sec = 1.0 / display_hz

    # Split into ("run", [(t,v),...]) / ("gap", t_or_None) segments,
    # exactly mirroring the gap-detection rule the drawing helpers
    # themselves already use (break on a None sample OR a time jump).
    segments = []
    run, last_t = [], None
    for t, v in hist:
        gap = last_t is not None and (t - last_t > max_gap_sec)
        if v is None or gap:
            if run:
                segments.append(("run", run))
            run = []
            segments.append(("gap", t if v is None else None))
        if v is not None:
            run.append((t, v))
        last_t = t
    if run:
        segments.append(("run", run))

    out = []
    for kind, payload in segments:
        if kind == "gap":
            if payload is not None:
                out.append((payload, None))
            continue
        bucket_last = {}
        for t, v in payload:
            b = math.floor(t / bucket_sec)
            bucket_last[b] = (t, v)  # later same-bucket sample overwrites -> keeps the LAST one
        for b in sorted(bucket_last):
            out.append(bucket_last[b])
    out.sort(key=lambda p: p[0])
    return out


def count_missing_segments(hist, max_gap_sec=0.5):
    """Counts genuine missing segments in `hist` -- a maximal run of
    consecutive `None`/absent coverage, delimited either by an explicit
    `None` sample or by a real timestamp jump exceeding `max_gap_sec`
    between two valid samples. Used only for numerical validation (this
    round's own "missing segment count must remain identical before vs
    after decimation" requirement), never by the drawing code itself."""
    count = 0
    in_gap = False
    last_t = None
    for t, v in hist:
        jump = last_t is not None and (t - last_t > max_gap_sec)
        if v is None or jump:
            if not in_gap:
                count += 1
                in_gap = True
        else:
            in_gap = False
        last_t = t
    return count
