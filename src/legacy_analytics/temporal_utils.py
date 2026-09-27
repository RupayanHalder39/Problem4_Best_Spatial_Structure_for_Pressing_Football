"""
Generic, dependency-free temporal helpers shared by every analytics
module that needs to stabilize a noisy per-frame label or bridge a short
detection gap. Nothing here is football-specific -- `analytics/
role_stability.py` (player role/team) and `analytics/ball_motion.py`
(ball position) both build on these instead of re-implementing the same
rolling-window logic, per the project rule against duplicating temporal
stabilization code across scripts.

These are pure functions over plain Python lists so they can be reused
identically whether the caller is grouping by track_id (players) or has
no grouping key at all (the single ball series).
"""
from collections import deque

import numpy as np


def causal_rolling_majority(values: list, window: int) -> list:
    """values: list of raw labels (any hashable, None allowed and passed
    through unchanged). Returns a new list, same length, where each
    position is the majority vote of the last `window` non-None
    observations up to and including this one. Ties keep the previous
    stabilized value (persistence), or the raw value if there is no
    previous stabilized value yet. Strictly causal: never looks at
    future observations."""
    out = []
    buf = deque(maxlen=window)
    prev_stable = None
    for v in values:
        if v is None:
            out.append(None)
            continue
        buf.append(v)
        counts = {}
        for x in buf:
            counts[x] = counts.get(x, 0) + 1
        best = max(counts.values())
        winners = [k for k, c in counts.items() if c == best]
        if len(winners) == 1:
            stable = winners[0]
        else:
            stable = prev_stable if prev_stable in winners else v
        prev_stable = stable
        out.append(stable)
    return out


def fill_short_gaps(frames: list, xs: list, ys: list, method: str = "interp",
                     max_gap: int = 3) -> tuple:
    """frames/xs/ys are parallel lists for ONE entity's timeline (already
    sorted by frame). xs[i]/ys[i] may be None where a row exists but the
    position is unknown (e.g. homography rejected that frame, or -- for
    the ball -- simply not the row we're filling). Frames that don't
    appear in `frames` at all (no row, i.e. no detection) are a different
    phenomenon and are handled by the caller -- this function only fills
    None-valued positions between two known-valid neighbors that are
    <=max_gap FRAME NUMBERS apart, never fabricating anything past a
    longer gap. Returns (out_x, out_y, filled_flags)."""
    assert method in ("hold", "interp")
    n = len(xs)
    out_x, out_y, filled = list(xs), list(ys), [False] * n
    valid_idx = [i for i in range(n) if xs[i] is not None]
    for a, b in zip(valid_idx, valid_idx[1:]):
        frame_gap = frames[b] - frames[a]
        n_missing_between = (b - a) - 1
        if n_missing_between == 0:
            continue
        if frame_gap > max_gap + 1:
            continue  # long gap -- leave null, never fabricate
        for i in range(a + 1, b):
            if method == "hold":
                out_x[i], out_y[i] = xs[a], ys[a]
            else:
                t = (frames[i] - frames[a]) / (frames[b] - frames[a])
                out_x[i] = xs[a] + t * (xs[b] - xs[a])
                out_y[i] = ys[a] + t * (ys[b] - ys[a])
            filled[i] = True
    return out_x, out_y, filled


def fill_short_gaps_respecting_restarts(frames: list, xs: list, ys: list, is_restart: list,
                                         method: str = "interp", max_gap: int = 3) -> tuple:
    """Same as fill_short_gaps(), but never fills a gap that would bridge
    INTO a frame flagged `is_restart[i]=True` (a genuine tracking
    discontinuity, e.g. analytics/ball_tracker.py's track-reacquisition
    events) -- a restart means the underlying track re-anchored, so
    interpolating a point between the old position and it would fabricate
    a smooth transition that never happened. `is_restart`: list of bool,
    same length as frames/xs/ys."""
    n = len(frames)
    out_x, out_y, filled = list(xs), list(ys), [False] * n
    valid_idx = [i for i in range(n) if xs[i] is not None]
    for a, b in zip(valid_idx, valid_idx[1:]):
        if is_restart[b]:
            continue
        frame_gap = frames[b] - frames[a]
        n_missing_between = (b - a) - 1
        if n_missing_between == 0 or frame_gap > max_gap + 1:
            continue
        for i in range(a + 1, b):
            if method == "hold":
                out_x[i], out_y[i] = xs[a], ys[a]
            else:
                t = (frames[i] - frames[a]) / (frames[b] - frames[a])
                out_x[i] = xs[a] + t * (xs[b] - xs[a])
                out_y[i] = ys[a] + t * (ys[b] - ys[a])
            filled[i] = True
    return out_x, out_y, filled


def causal_ema(xs: list, ys: list, alpha: float = 0.35) -> tuple:
    """Causal EMA smoothing over a position series. Resets (starts fresh)
    after any None -- never bridges a still-null gap. Returns (sm_x, sm_y)."""
    sm_x, sm_y = [], []
    prev = None
    for x, y in zip(xs, ys):
        if x is None:
            sm_x.append(None)
            sm_y.append(None)
            prev = None
            continue
        if prev is None:
            sm_x.append(x)
            sm_y.append(y)
        else:
            sm_x.append(alpha * x + (1 - alpha) * prev[0])
            sm_y.append(alpha * y + (1 - alpha) * prev[1])
        prev = (sm_x[-1], sm_y[-1])
    return sm_x, sm_y


def split_into_runs(frames: list, max_gap: int) -> list:
    """Splits a sorted list of frame numbers into runs where consecutive
    members are <=max_gap frames apart. Returns a list of (start_idx,
    end_idx) index pairs into `frames` (inclusive). Used to find
    contiguous "movement segments" without bridging long absences."""
    if not frames:
        return []
    runs = []
    start = 0
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > max_gap:
            runs.append((start, i - 1))
            start = i
    runs.append((start, len(frames) - 1))
    return runs


def displacement_cm(x1, y1, x2, y2) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))
