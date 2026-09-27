"""
Shared dark-theme rendering primitives for the "final" dashboard style
(pressing_structure AND offside_break -- defined once here, imported by
both, per this project's established cross-project reuse pattern).

Pure rendering helpers only -- no analytics logic lives here. Visual
language only (rounded cards, gradient header strip, metric tiles,
dark-panel graph chrome) modeled after two locked reference mockups the
user supplied; no data, labels, thresholds, or event categories are
implied or hardcoded here beyond generic chrome.
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG_DARK = (36, 24, 18)          # BGR, near-black navy
PANEL_DARK = (46, 32, 24)
CARD_DARK = (54, 38, 28)
HEADER_DARK = (30, 18, 12)
BORDER_COLOR = (90, 70, 55)
TEXT_WHITE = (240, 240, 240)
TEXT_DIM = (170, 160, 150)
ACCENT_CYAN = (255, 200, 60)     # BGR -- cyan-ish accent (bright in B channel high, matches reference cyan on screen)
ACCENT_MAGENTA = (200, 60, 220)  # BGR magenta/pink accent
GRID_DARK = (70, 55, 45)


def rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1, alpha=1.0):
    """Draw a filled or outlined rounded rectangle, optionally alpha-blended."""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img if alpha >= 1.0 else img.copy()
    if thickness < 0:
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, -1)
    else:
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    return img


def draw_card(w, h, bg=CARD_DARK, border=BORDER_COLOR, radius=12):
    """A standalone rounded card canvas (for compositing into a larger frame)."""
    img = np.full((h, w, 3), BG_DARK, dtype=np.uint8)
    rounded_rect(img, (2, 2), (w - 3, h - 3), bg, radius=radius)
    rounded_rect(img, (2, 2), (w - 3, h - 3), border, radius=radius, thickness=2)
    return img


def draw_header_strip(width, height, title, subtitle, right_text_lines, accent=ACCENT_CYAN):
    """Dark gradient-ish header bar with a bold title (left) and small
    right-aligned status lines (e.g. score, clock)."""
    img = np.full((height, width, 3), HEADER_DARK, dtype=np.uint8)
    for x in range(width):
        shade = int(HEADER_DARK[0] * (1.0 - 0.15 * x / width))
        img[:, x] = (shade, HEADER_DARK[1], HEADER_DARK[2])
    cv2.putText(img, title, (24, int(height * 0.62)), FONT, height / 46.0, TEXT_WHITE, 2, cv2.LINE_AA)
    if subtitle:
        tw = cv2.getTextSize(title, FONT, height / 46.0, 2)[0][0]
        cv2.putText(img, subtitle, (24 + tw + 18, int(height * 0.6)), FONT, height / 90.0, TEXT_DIM, 1, cv2.LINE_AA)
    x = width - 24
    for line, color in reversed(right_text_lines):
        tw = cv2.getTextSize(line, FONT, height / 80.0, 1)[0][0]
        x -= tw
        cv2.putText(img, line, (x, int(height * 0.62)), FONT, height / 80.0, color, 1, cv2.LINE_AA)
        x -= 24
    cv2.line(img, (0, height - 2), (width, height - 2), accent, 2)
    return img


def draw_metric_tile(w, h, label, value, sub=None, accent=ACCENT_CYAN):
    """A compact card: small caption label, big value, optional sub-line."""
    img = draw_card(w, h)
    cv2.putText(img, label.upper(), (14, 22), FONT, 0.36, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, str(value), (14, int(h * 0.62)), FONT, h / 60.0, TEXT_WHITE, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (14, h - 12), FONT, 0.34, accent, 1, cv2.LINE_AA)
    return img


def draw_dark_scrolling_graph(history, cur_time, window_sec, width, height, title, y_label,
                                y_range, unit="", line_color=ACCENT_CYAN, max_gap_sec=0.5):
    """Same scrolling-window semantics as `live_graphs.draw_scrolling_graph`
    (stable fixed y-axis, gaps broken not interpolated, current-time
    marker), re-themed dark for the final dashboard style."""
    img = draw_card(width, height, bg=PANEL_DARK)
    pad_l, pad_r, pad_t, pad_b = 54, 14, 28, 22
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    t_lo, t_hi = cur_time - window_sec, cur_time
    y_lo, y_hi = y_range

    def to_px(t, v):
        x = pad_l + int((t - t_lo) / max(1e-6, (t_hi - t_lo)) * plot_w)
        y = pad_t + plot_h - int((v - y_lo) / max(1e-6, (y_hi - y_lo)) * plot_h)
        return x, y

    cv2.rectangle(img, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), BORDER_COLOR, 1)
    for frac in (0.25, 0.5, 0.75):
        gy = pad_t + int(plot_h * (1 - frac))
        cv2.line(img, (pad_l, gy), (pad_l + plot_w, gy), GRID_DARK, 1, cv2.LINE_AA)
        val = y_lo + frac * (y_hi - y_lo)
        cv2.putText(img, f"{val:.0f}", (4, gy + 4), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, f"{y_hi:.0f}", (4, pad_t + 8), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, f"{y_lo:.0f}", (4, pad_t + plot_h + 4), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

    windowed = [(t, v) for t, v in history if t_lo - max_gap_sec <= t <= t_hi]
    segment, last_t = [], None
    for t, v in windowed:
        gap = (last_t is not None) and (t - last_t > max_gap_sec)
        if v is None or gap:
            if len(segment) >= 2:
                cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, line_color, 2, cv2.LINE_AA)
            segment = []
        if v is not None:
            segment.append(to_px(t, np.clip(v, y_lo, y_hi)))
        last_t = t
    if len(segment) >= 2:
        cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, line_color, 2, cv2.LINE_AA)
    elif len(segment) == 1:
        cv2.circle(img, segment[0], 3, line_color, -1, cv2.LINE_AA)

    mx, _ = to_px(cur_time, y_lo)
    cv2.line(img, (mx, pad_t), (mx, pad_t + plot_h), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(img, (mx, pad_t), (mx, pad_t + plot_h), TEXT_WHITE, 1, cv2.LINE_AA)

    cv2.putText(img, f"{title} ({unit})" if unit else title, (pad_l, 18), FONT, 0.38, TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, y_label, (pad_l, height - 4), FONT, 0.3, TEXT_DIM, 1, cv2.LINE_AA)
    return img


def draw_dark_dual_scrolling_graph(history_a, history_b, cur_time, window_sec, width, height,
                                     title, y_label, y_range, label_a, label_b, unit="",
                                     color_a=ACCENT_CYAN, color_b=ACCENT_MAGENTA, max_gap_sec=0.5):
    img = draw_card(width, height, bg=PANEL_DARK)
    pad_l, pad_r, pad_t, pad_b = 54, 14, 28, 22
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    t_lo, t_hi = cur_time - window_sec, cur_time
    y_lo, y_hi = y_range

    def to_px(t, v):
        x = pad_l + int((t - t_lo) / max(1e-6, (t_hi - t_lo)) * plot_w)
        y = pad_t + plot_h - int((v - y_lo) / max(1e-6, (y_hi - y_lo)) * plot_h)
        return x, y

    cv2.rectangle(img, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), BORDER_COLOR, 1)
    for frac in (0.25, 0.5, 0.75):
        gy = pad_t + int(plot_h * (1 - frac))
        cv2.line(img, (pad_l, gy), (pad_l + plot_w, gy), GRID_DARK, 1, cv2.LINE_AA)
        val = y_lo + frac * (y_hi - y_lo)
        cv2.putText(img, f"{val:.0f}", (4, gy + 4), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, f"{y_hi:.0f}", (4, pad_t + 8), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, f"{y_lo:.0f}", (4, pad_t + plot_h + 4), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

    def plot_series(hist, color):
        windowed = [(t, v) for t, v in hist if t_lo - max_gap_sec <= t <= t_hi]
        segment, last_t = [], None
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

    plot_series(history_a, color_a)
    plot_series(history_b, color_b)

    mx, _ = to_px(cur_time, y_lo)
    cv2.line(img, (mx, pad_t), (mx, pad_t + plot_h), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(img, (mx, pad_t), (mx, pad_t + plot_h), TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"{title} ({unit})" if unit else title, (pad_l, 18), FONT, 0.38, TEXT_WHITE, 1, cv2.LINE_AA)

    lx = pad_l + plot_w - 4
    for lbl, color in ((label_b, color_b), (label_a, color_a)):
        tw = cv2.getTextSize(lbl, FONT, 0.32, 1)[0][0]
        lx -= tw
        cv2.putText(img, lbl, (lx, pad_t - 6), FONT, 0.32, TEXT_WHITE, 1, cv2.LINE_AA)
        lx -= 16
        cv2.rectangle(img, (lx, pad_t - 14), (lx + 10, pad_t - 6), color, -1)
        lx -= 10
    cv2.putText(img, y_label, (pad_l, height - 4), FONT, 0.3, TEXT_DIM, 1, cv2.LINE_AA)
    return img


def draw_event_timeline_dark(width, height, span_sec, cur_time, segments, markers, legend, title="TIMELINE"):
    """`segments`: list of (start_sec, end_sec, color) filled bars.
    `markers`: list of (time_sec, color) vertical ticks.
    `legend`: list of (label, color) swatches drawn bottom-left."""
    img = np.full((height, width, 3), HEADER_DARK, dtype=np.uint8)
    rounded_rect(img, (2, 2), (width - 3, height - 3), PANEL_DARK, radius=8)
    bar_top, bar_bot = 8, height - 20
    for t0, t1, color in segments:
        x0 = int(t0 / span_sec * width)
        x1 = max(int(t1 / span_sec * width), x0 + 2)
        cv2.rectangle(img, (x0, bar_top), (x1, bar_bot), color, -1)
    for t, color in markers:
        x = int(t / span_sec * width)
        cv2.line(img, (x, bar_top), (x, bar_bot), color, 2)
    cx = int(np.clip(cur_time / span_sec * width, -width, 2 * width))
    cv2.line(img, (cx, 0), (cx, height), TEXT_WHITE, 2)
    cv2.putText(img, title, (8, height - 5), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    lx = 90
    for label, color in legend:
        cv2.rectangle(img, (lx, height - 15), (lx + 10, height - 6), color, -1)
        cv2.putText(img, label, (lx + 14, height - 6), FONT, 0.3, TEXT_DIM, 1, cv2.LINE_AA)
        lx += 20 + 7 * len(label)
    return img
