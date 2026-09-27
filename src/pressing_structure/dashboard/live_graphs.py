"""
Shared, reusable scrolling-line-graph renderer for the V2 dashboards
(pressing_structure AND offside_break -- imported by both, defined
once here to avoid duplicating the same cv2 plotting code twice).

Pure rendering: takes a precomputed (time, value) history and draws a
trailing scrolling window ending at "now", with a fixed (not
per-frame-rescaled) y-axis for visual stability, a current-time marker
at the right edge, and gaps (None values, or a time gap exceeding
`max_gap_sec`) drawn as broken line segments -- never interpolated
across a real data gap.
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
GRID_COLOR = (220, 220, 220)
AXIS_COLOR = (120, 120, 120)
MARKER_COLOR = (0, 0, 0)
BG_COLOR = (250, 250, 250)


def compute_stable_y_range(values: list, pad_frac: float = 0.1) -> tuple:
    """Fixed y-axis range for the WHOLE clip, from the 1st-99th
    percentile of all non-null values (robust to rare outlier spikes)
    plus padding -- computed ONCE, reused every frame, so the axis
    never jitters/rescales as the video plays."""
    vals = [v for v in values if v is not None]
    if not vals:
        return (0.0, 1.0)
    lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * pad_frac
    return (max(0.0, lo - pad), hi + pad)


def draw_scrolling_graph(history: list, cur_time: float, window_sec: float,
                          width: int, height: int, title: str, y_label: str,
                          y_range: tuple, unit: str = "", line_color: tuple = (0, 120, 255),
                          max_gap_sec: float = 0.5) -> np.ndarray:
    """`history`: sorted list of (time_sec, value_or_None) covering AT
    LEAST [cur_time - window_sec, cur_time]. Draws only that trailing
    window. `y_range`: FIXED (t_lo, t_hi) from `compute_stable_y_range`
    -- never recomputed per frame here, so the axis is stable across
    the whole render."""
    img = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    pad_l, pad_r, pad_t, pad_b = 60, 14, 30, 26
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    t_lo, t_hi = cur_time - window_sec, cur_time
    y_lo, y_hi = y_range

    def to_px(t, v):
        x = pad_l + int((t - t_lo) / max(1e-6, (t_hi - t_lo)) * plot_w)
        y = pad_t + plot_h - int((v - y_lo) / max(1e-6, (y_hi - y_lo)) * plot_h)
        return x, y

    # axes + gridlines (3 horizontal guides)
    cv2.rectangle(img, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), AXIS_COLOR, 1)
    for frac in (0.25, 0.5, 0.75):
        gy = pad_t + int(plot_h * (1 - frac))
        cv2.line(img, (pad_l, gy), (pad_l + plot_w, gy), GRID_COLOR, 1, cv2.LINE_AA)
        val = y_lo + frac * (y_hi - y_lo)
        cv2.putText(img, f"{val:.0f}", (4, gy + 4), FONT, 0.35, AXIS_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, f"{y_hi:.0f}", (4, pad_t + 8), FONT, 0.35, AXIS_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, f"{y_lo:.0f}", (4, pad_t + plot_h + 4), FONT, 0.35, AXIS_COLOR, 1, cv2.LINE_AA)

    # plot the trailing window, breaking the line at gaps/Nones
    windowed = [(t, v) for t, v in history if t_lo - max_gap_sec <= t <= t_hi]
    segment = []
    last_t = None
    for t, v in windowed:
        gap = (last_t is not None) and (t - last_t > max_gap_sec)
        if v is None or gap:
            if len(segment) >= 2:
                pts = np.array(segment, dtype=np.int32)
                cv2.polylines(img, [pts], False, line_color, 2, cv2.LINE_AA)
            segment = []
        if v is not None:
            segment.append(to_px(t, np.clip(v, y_lo, y_hi)))
        last_t = t
    if len(segment) >= 2:
        pts = np.array(segment, dtype=np.int32)
        cv2.polylines(img, [pts], False, line_color, 2, cv2.LINE_AA)
    elif len(segment) == 1:
        cv2.circle(img, segment[0], 3, line_color, -1, cv2.LINE_AA)

    # current-time marker at the right edge ("now")
    mx, _ = to_px(cur_time, y_lo)
    cv2.line(img, (mx, pad_t), (mx, pad_t + plot_h), MARKER_COLOR, 1, cv2.LINE_AA)

    cv2.putText(img, f"{title} ({unit})" if unit else title, (pad_l, 18), FONT, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(img, y_label, (pad_l, height - 6), FONT, 0.32, AXIS_COLOR, 1, cv2.LINE_AA)
    return img


def _plot_series(img, history, cur_time, t_lo, t_hi, y_lo, y_hi, to_px, color, max_gap_sec):
    windowed = [(t, v) for t, v in history if t_lo - max_gap_sec <= t <= t_hi]
    segment = []
    last_t = None
    for t, v in windowed:
        gap = (last_t is not None) and (t - last_t > max_gap_sec)
        if v is None or gap:
            if len(segment) >= 2:
                cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            segment = []
        if v is not None:
            segment.append(to_px(t, np.clip(v, y_lo, y_hi)))
        last_t = t
    if len(segment) >= 2:
        cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    elif len(segment) == 1:
        cv2.circle(img, segment[0], 3, color, -1, cv2.LINE_AA)


def draw_dual_scrolling_graph(history_a: list, history_b: list, cur_time: float, window_sec: float,
                               width: int, height: int, title: str, y_label: str, y_range: tuple,
                               label_a: str, label_b: str, unit: str = "",
                               color_a: tuple = (0, 120, 255), color_b: tuple = (0, 150, 0),
                               max_gap_sec: float = 0.5) -> np.ndarray:
    """Same scrolling-window convention as `draw_scrolling_graph`, but
    overlays TWO series (e.g. attacker depth vs. defensive-line depth)
    on one fixed y-axis so a crossing between them is directly visible,
    with a small colored-swatch legend instead of overlapping text."""
    img = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    pad_l, pad_r, pad_t, pad_b = 60, 14, 30, 26
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    t_lo, t_hi = cur_time - window_sec, cur_time
    y_lo, y_hi = y_range

    def to_px(t, v):
        x = pad_l + int((t - t_lo) / max(1e-6, (t_hi - t_lo)) * plot_w)
        y = pad_t + plot_h - int((v - y_lo) / max(1e-6, (y_hi - y_lo)) * plot_h)
        return x, y

    cv2.rectangle(img, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), AXIS_COLOR, 1)
    for frac in (0.25, 0.5, 0.75):
        gy = pad_t + int(plot_h * (1 - frac))
        cv2.line(img, (pad_l, gy), (pad_l + plot_w, gy), GRID_COLOR, 1, cv2.LINE_AA)
        val = y_lo + frac * (y_hi - y_lo)
        cv2.putText(img, f"{val:.0f}", (4, gy + 4), FONT, 0.35, AXIS_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, f"{y_hi:.0f}", (4, pad_t + 8), FONT, 0.35, AXIS_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, f"{y_lo:.0f}", (4, pad_t + plot_h + 4), FONT, 0.35, AXIS_COLOR, 1, cv2.LINE_AA)

    _plot_series(img, history_a, cur_time, t_lo, t_hi, y_lo, y_hi, to_px, color_a, max_gap_sec)
    _plot_series(img, history_b, cur_time, t_lo, t_hi, y_lo, y_hi, to_px, color_b, max_gap_sec)

    mx, _ = to_px(cur_time, y_lo)
    cv2.line(img, (mx, pad_t), (mx, pad_t + plot_h), MARKER_COLOR, 1, cv2.LINE_AA)

    cv2.putText(img, f"{title} ({unit})" if unit else title, (pad_l, 18), FONT, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    # legend: colored swatches, never overlapping text, placed top-right of the plot area
    lx = pad_l + plot_w - 4
    for lbl, color in ((label_b, color_b), (label_a, color_a)):
        tw = cv2.getTextSize(lbl, FONT, 0.34, 1)[0][0]
        lx -= tw
        cv2.putText(img, lbl, (lx, pad_t - 6), FONT, 0.34, (20, 20, 20), 1, cv2.LINE_AA)
        lx -= 16
        cv2.rectangle(img, (lx, pad_t - 14), (lx + 10, pad_t - 6), color, -1)
        lx -= 10
    cv2.putText(img, y_label, (pad_l, height - 6), FONT, 0.32, AXIS_COLOR, 1, cv2.LINE_AA)
    return img
