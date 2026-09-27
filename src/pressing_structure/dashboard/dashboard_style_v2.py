"""
Second-generation dark dashboard chrome (pressing_structure AND
offside_break -- shared, imported by both `render_*_dashboard_final_v2.py`
scripts). This is a STRUCTURAL rebuild of `dashboard_style.py`'s visual
language after user feedback that the first "final" attempt still read
as a debug/QA screen -- kept as a SEPARATE module so the first
`final_dashboard/` output and its render scripts are untouched.

Pure rendering helpers only -- no analytics logic, no data, no fake
values. Every primitive here takes real numbers/labels from the
caller.
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
# Deep navy palette (closer to the locked broadcast-style references
# than the earlier warm dark-gray attempt).
BG = (46, 26, 14)             # BGR -- RGB(14,26,46), near-black navy
PANEL_BG = (60, 38, 22)
CARD_BG = (70, 46, 28)
HEADER_BG = (32, 18, 9)
BORDER = (110, 80, 48)
BORDER_SOFT = (78, 56, 34)
TEXT_WHITE = (248, 248, 248)
TEXT_DIM = (165, 155, 140)
TEXT_MUTED = (120, 108, 96)
ACCENT_CYAN = (255, 225, 25)      # BGR -- bright, fully-saturated cyan on screen
ACCENT_MAGENTA = (230, 25, 255)   # bumped saturation after feedback that radar contrast was too soft
ACCENT_ORANGE = (0, 155, 255)
ACCENT_GREEN = (105, 225, 95)
ACCENT_RED = (60, 60, 235)
GRID = (75, 55, 38)


def rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        # A prior version drew two overlapping full-length rectangle
        # OUTLINES here, which is not a rounded-rect outline at all (no
        # corner arcs, doubled straight edges) -- visibly broken at
        # thickness>=2 (found on the compact state badge). Proper
        # outline: 4 straight edges between the flat sides + 4 corner
        # arcs.
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
    return img


def panel_frame(w, h, bg=CARD_BG, border=BORDER, radius=10):
    """A card canvas with a thin border and internal padding baked in
    (callers draw content starting a few px inside)."""
    img = np.full((h, w, 3), BG, dtype=np.uint8)
    rounded_rect(img, (1, 1), (w - 2, h - 2), bg, radius=radius)
    rounded_rect(img, (1, 1), (w - 2, h - 2), border, radius=radius, thickness=1)
    return img


def panel_title(img, text, sub=None, pad=12, accent=ACCENT_CYAN):
    """Panel title with a short colored accent bar before the text --
    matches the locked reference's '| TITLE' header treatment on every
    card/panel."""
    cv2.rectangle(img, (pad, 8), (pad + 3, 20), accent, -1)
    cv2.putText(img, text, (pad + 10, 20), FONT, 0.5, TEXT_WHITE, 1, cv2.LINE_AA)
    if sub:
        tw = cv2.getTextSize(text, FONT, 0.5, 1)[0][0]
        cv2.putText(img, sub, (pad + 20 + tw, 19), FONT, 0.36, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.line(img, (pad, 30), (img.shape[1] - pad, 30), BORDER_SOFT, 1)
    return img


def draw_header(width, height, title, right_text, accent=ACCENT_CYAN):
    """Clean header: bold title left, one human-facing status string
    right (e.g. elapsed time) -- no debug/version subtitle, no file
    paths, no internal taxonomy names."""
    img = np.full((height, width, 3), HEADER_BG, dtype=np.uint8)
    cv2.putText(img, title, (26, int(height * 0.66)), FONT, height / 42.0, TEXT_WHITE, 2, cv2.LINE_AA)
    if right_text:
        text, color = right_text
        tw = cv2.getTextSize(text, FONT, height / 62.0, 1)[0][0]
        cv2.putText(img, text, (width - 26 - tw, int(height * 0.66)), FONT, height / 62.0, color, 1, cv2.LINE_AA)
    cv2.line(img, (0, height - 2), (width, height - 2), accent, 2)
    return img


def apply_color_wash_preserve_lines(pitch_img, color_field, alpha):
    """Blend `color_field` (same shape as pitch_img) into pitch_img at
    `alpha`, then restore the pitch's own white line pixels at full
    brightness so the markings stay crisp regardless of wash opacity
    (`draw_pitch` renders lines as pure white on solid green -- a
    simple brightness threshold isolates them losslessly)."""
    line_mask = (pitch_img >= 200).all(axis=2)
    blended = cv2.addWeighted(color_field, alpha, pitch_img, 1 - alpha, 0)
    blended[line_mask] = pitch_img[line_mask]
    return blended


# Team-specific pressure-glow color stops (outer, mid, core), BGR --
# team 0 stays the original warm red->orange->yellow; team 1 gets a
# clearly distinct cool green family, per the explicit "Team 0 pressing
# = amber/orange, Team 1 pressing = lime/green" requirement. Team 1's
# stops pin the R channel low throughout (mirroring team 0's B-pinned-
# low pattern) so the family reads as a clean green rather than drifting
# toward yellow -- see `add_team_pressure_glow` below for why this
# alone wasn't enough and what else changed.
TEAM_PRESS_GLOW_STOPS = {
    0: ((0, 40, 230), (0, 130, 255), (0, 220, 255)),  # red -> orange -> yellow (amber family, B pinned ~0)
    1: ((30, 90, 0), (30, 180, 0), (60, 255, 60)),    # dark green -> green -> bright green (R pinned ~0)
}


def add_pressure_glow(pitch_img, center_px, intensity, max_radius_px=140, stops=None):
    """Radial pressure hotspot centered on a real pitch position (e.g.
    the ball carrier under press), sized/brightened by a real,
    disclosed intensity in [0,1] (e.g. presser count normalized) --
    NOT a fabricated value, just a visualization of an existing scalar.

    Deliberately a DIFFERENT visual mechanism from the team-control
    wash (`apply_color_wash_preserve_lines`): a three-stop radial
    gradient (hot core, cooler edge), additively screened on top rather
    than alpha-blended, so it always reads as a distinct hotspot
    layered over the control field. `stops` = (outer, mid, core) BGR
    tuples; defaults to the original red->orange->yellow if omitted --
    pass `TEAM_PRESS_GLOW_STOPS[team]` for the team-specific palette."""
    if center_px is None or intensity <= 0:
        return pitch_img
    outer, mid, core_color = stops or TEAM_PRESS_GLOW_STOPS[0]
    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = pitch_img.shape[:2]
    r = max_radius_px * (0.45 + 0.55 * intensity)
    glow = np.zeros((h, w, 3), dtype=np.float32)
    # outer ring, then mid, then hot core -- drawn widest/coolest first
    # so the hotter, smaller circles layer on top
    cv2.circle(glow, center_px, int(r), outer, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=r * 0.4)
    core = np.zeros((h, w, 3), dtype=np.float32)
    cv2.circle(core, center_px, int(r * 0.6), mid, -1, cv2.LINE_AA)
    core = cv2.GaussianBlur(core, (0, 0), sigmaX=r * 0.28)
    glow += core
    hot = np.zeros((h, w, 3), dtype=np.float32)
    cv2.circle(hot, center_px, int(r * 0.28), core_color, -1, cv2.LINE_AA)
    hot = cv2.GaussianBlur(hot, (0, 0), sigmaX=r * 0.18)
    glow += hot
    out = pitch_img.astype(np.float32) + glow * (0.85 * intensity)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_team_pressure_glow(pitch_img, center_px, intensity, max_radius_px=140, stops=None):
    """NEW (v3, pressing dual-team upgrade): a team-colored pressure
    hotspot for radar overlays that carry a real background color wash
    underneath (the v3 dashboards' pitch-control wash), where the
    original `add_pressure_glow`'s pure ADDITIVE compositing breaks
    down.

    Root cause found during Goal 5 QA: `add_pressure_glow` ADDS its
    three rings' BGR colors directly onto the existing pixel. When the
    underlying wash pixel is already bright in two of three channels
    (which the magenta/cyan pitch-control wash often is), the only
    channel with headroom left gets pushed up by the addition and the
    result clips toward WHITE regardless of which team's hue was added
    -- confirmed by a direct pixel-crop comparison showing team 0's
    (amber) and team 1's (green) hotspots both rendering as a near-
    identical whitish/yellowish blob at real, moderate intensities
    (~0.38-0.47), even after re-tuning team 1's stop colors to pin the
    same channel low that team 0 does. Retuning the stops alone could
    not fix it because the failure mode is in the BLEND, not the hue.

    Fix: alpha-composite (normal "over" blend) instead of add. Each
    ring contributes a weight (its blurred circular mask) and a color;
    at each pixel the three rings' colors are combined as a
    weight-normalized AVERAGE (so the displayed hue always stays
    within the team's own stop family, never drifting toward
    whatever's left over from the background), and that averaged color
    REPLACES the background proportionally to total local glow weight
    and `intensity` -- so a real team-colored tint is guaranteed to be
    visible at the hotspot regardless of how bright/saturated the
    pixel underneath already was.

    Kept as a SEPARATE function rather than changing
    `add_pressure_glow` in place: that function is still used as-is
    elsewhere (single-team, unwashed hotspots) and this project's
    safety constraints call for not changing established, working
    core-file behavior when a new, narrower function will do."""
    if center_px is None or intensity <= 0:
        return pitch_img
    outer, mid, core_color = stops or TEAM_PRESS_GLOW_STOPS[0]
    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = pitch_img.shape[:2]
    r = max_radius_px * (0.45 + 0.55 * intensity)

    def _layer(radius_frac, sigma_frac):
        m = np.zeros((h, w), dtype=np.float32)
        cv2.circle(m, center_px, max(1, int(r * radius_frac)), 1.0, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(1.0, r * sigma_frac))
        return m[..., None]

    m_outer = _layer(1.0, 0.4)
    m_mid = _layer(0.6, 0.28)
    m_core = _layer(0.28, 0.18)

    weight = m_outer + m_mid + m_core
    color_accum = (m_outer * np.array(outer, dtype=np.float32) +
                   m_mid * np.array(mid, dtype=np.float32) +
                   m_core * np.array(core_color, dtype=np.float32))
    tint = color_accum / np.clip(weight, 1e-6, None)  # weight-normalized hue -- stays in-family

    alpha = np.clip(weight, 0.0, 1.0) * (0.92 * intensity)  # how strongly the tint replaces the background
    out = pitch_img.astype(np.float32) * (1.0 - alpha) + tint * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_radar_legend_box(canvas, x, y, w, h, rows, gradient=None, gradient_labels=None):
    """Small bordered legend box overlaid on a radar/pitch panel --
    `rows`: list of (label, color) swatch rows. `gradient`: optional
    (color_hi, color_lo) pair rendered as a vertical gradient bar next
    to `gradient_labels` (top, mid, bottom text) describing the SCALE
    itself (e.g. 'Team A Control / High Pressure / Team B Control'),
    not any specific measured value."""
    rounded_rect(canvas, (x, y), (x + w, y + h), (18, 12, 8), radius=6)
    rounded_rect(canvas, (x, y), (x + w, y + h), BORDER, radius=6, thickness=1)
    ty = y + 16
    for label, color in rows:
        cv2.circle(canvas, (x + 12, ty - 3), 5, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 22, ty), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
        ty += 18
    if gradient is not None:
        gx0, gy0 = x + 10, ty + 4
        gw, gh = 14, h - (gy0 - y) - 10
        color_hi, color_lo = gradient
        for i in range(gh):
            frac = i / max(1, gh - 1)
            c = tuple(int(color_hi[k] * (1 - frac) + color_lo[k] * frac) for k in range(3))
            cv2.line(canvas, (gx0, gy0 + i), (gx0 + gw, gy0 + i), c, 1)
        cv2.rectangle(canvas, (gx0, gy0), (gx0 + gw, gy0 + gh), BORDER_SOFT, 1)
        if gradient_labels:
            top, mid, bot = gradient_labels
            cv2.putText(canvas, top, (gx0 + gw + 8, gy0 + 8), FONT, 0.28, TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(canvas, mid, (gx0 + gw + 8, gy0 + gh // 2 + 4), FONT, 0.28, TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(canvas, bot, (gx0 + gw + 8, gy0 + gh - 2), FONT, 0.28, TEXT_DIM, 1, cv2.LINE_AA)
    return canvas


def draw_hatched_zone(pitch_img, pt0, pt1, color, alpha=0.16, spacing=26):
    """Diagonally-hatched shaded rectangle (e.g. 'dangerous space
    behind the line') -- a light, sparse texture that reads as a
    distinct hazard zone without overpowering the players/pitch-control
    wash/defensive line drawn under and over it. Tuned lighter/sparser
    than the first pass after feedback that the hatching was too heavy;
    the zone's own color is kept vivid (caller's choice) so the region
    still reads clearly despite the low alpha."""
    x0, y0 = pt0
    x1, y1 = pt1
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    overlay = pitch_img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    for x in range(x0 - (y1 - y0), x1, spacing):
        cv2.line(overlay, (x, y0), (x + (y1 - y0), y1), tuple(int(c * 0.7) for c in color), 1, cv2.LINE_AA)
    mask = np.zeros(pitch_img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    blended = cv2.addWeighted(overlay, alpha, pitch_img, 1 - alpha, 0)
    out = pitch_img.copy()
    out[mask > 0] = blended[mask > 0]
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
    return out


def draw_metric_card(w, h, label, value, sub=None, accent=ACCENT_CYAN, value_available=True):
    """Compact metric card: uppercase dim label, big value (or a small
    muted 'N/A' that does NOT dominate the card when unavailable), a
    thin accent bar on the left edge, optional sub-line."""
    img = panel_frame(w, h)
    cv2.rectangle(img, (0, 6), (4, h - 6), accent, -1)
    cv2.putText(img, label.upper(), (16, 20), FONT, 0.33, TEXT_DIM, 1, cv2.LINE_AA)
    if value_available:
        cv2.putText(img, str(value), (16, int(h * 0.66)), FONT, h / 62.0, TEXT_WHITE, 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "N/A", (16, int(h * 0.6)), FONT, 0.5, TEXT_MUTED, 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (16, h - 10), FONT, 0.32, accent, 1, cv2.LINE_AA)
    return img


def _to_px(t, v, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph):
    x = x0 + int((t - t_lo) / max(1e-6, (t_hi - t_lo)) * pw)
    y = y0 + ph - int((v - y_lo) / max(1e-6, (y_hi - y_lo)) * ph)
    return x, y


def draw_big_graph(w, h, history, cur_time, window_sec, title, y_range, unit="",
                    line_color=ACCENT_CYAN, max_gap_sec=0.5, y_label="", tick_fmt=None):
    """Larger, cleaner scrolling-window graph card: proper margins,
    thicker line, current-time marker with outline for contrast on any
    background, title inside the panel, axis labels that don't clip.

    `tick_fmt`: format spec for the y-axis tick labels (e.g. "{:.2f}").
    Defaults to None, which auto-picks "{:.0f}" for ranges of several
    units or more (unchanged behavior for every existing caller: m/s,
    cm, degrees, etc.) but "{:.2f}" when the whole range spans <=1.5 --
    a 0-1 SCORE axis previously used the same ":.0f" as every other
    metric, which silently rounds 0.71->1 and 0.29->0 (audit C5) --
    fixed generically here rather than per call site so no other
    renderer needs to change to benefit."""
    img = panel_frame(w, h)
    panel_title(img, title, sub=(f"last {window_sec:.0f}s  •  {unit}" if unit else f"last {window_sec:.0f}s"))
    pad_l, pad_r, pad_t, pad_b = 46, 16, 42, 26
    x0, y0 = pad_l, pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    t_lo, t_hi = cur_time - window_sec, cur_time
    y_lo, y_hi = y_range
    fmt = tick_fmt or ("{:.2f}" if (y_hi - y_lo) <= 1.5 else "{:.0f}")

    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), BORDER_SOFT, 1)
    for frac in (0.25, 0.5, 0.75):
        gy = y0 + int(ph * (1 - frac))
        cv2.line(img, (x0, gy), (x0 + pw, gy), GRID, 1, cv2.LINE_AA)
        cv2.putText(img, fmt.format(y_lo + frac * (y_hi - y_lo)), (6, gy + 4), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, fmt.format(y_hi), (6, y0 + 8), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, fmt.format(y_lo), (6, y0 + ph + 4), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

    windowed = [(t, v) for t, v in history if t_lo - max_gap_sec <= t <= t_hi]
    segment, last_t = [], None
    for t, v in windowed:
        gap = (last_t is not None) and (t - last_t > max_gap_sec)
        if v is None or gap:
            if len(segment) >= 2:
                cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, line_color, 3, cv2.LINE_AA)
            segment = []
        if v is not None:
            segment.append(_to_px(t, np.clip(v, y_lo, y_hi), t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph))
        last_t = t
    if len(segment) >= 2:
        cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, line_color, 3, cv2.LINE_AA)
    elif len(segment) == 1:
        cv2.circle(img, segment[0], 4, line_color, -1, cv2.LINE_AA)

    mx, _ = _to_px(cur_time, y_lo, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
    cv2.line(img, (mx, y0), (mx, y0 + ph), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (mx, y0), (mx, y0 + ph), TEXT_WHITE, 2, cv2.LINE_AA)
    if y_label:
        cv2.putText(img, y_label, (x0, h - 8), FONT, 0.3, TEXT_DIM, 1, cv2.LINE_AA)
    return img


def draw_big_dual_graph(w, h, history_a, history_b, cur_time, window_sec, title, y_range,
                          label_a, label_b, unit="", color_a=ACCENT_CYAN, color_b=ACCENT_MAGENTA,
                          max_gap_sec=0.5, tick_fmt=None):
    """`tick_fmt`: see `draw_big_graph` -- same auto-pick ("{:.2f}" for a
    <=1.5-wide range, else "{:.0f}") so a 0-1 score axis shows fractional
    ticks instead of rounding to a repeated 0/1 (audit C5)."""
    img = panel_frame(w, h)
    panel_title(img, title, sub=f"last {window_sec:.0f}s")
    pad_l, pad_r, pad_t, pad_b = 46, 16, 42, 26
    x0, y0 = pad_l, pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    t_lo, t_hi = cur_time - window_sec, cur_time
    y_lo, y_hi = y_range
    fmt = tick_fmt or ("{:.2f}" if (y_hi - y_lo) <= 1.5 else "{:.0f}")

    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), BORDER_SOFT, 1)
    for frac in (0.25, 0.5, 0.75):
        gy = y0 + int(ph * (1 - frac))
        cv2.line(img, (x0, gy), (x0 + pw, gy), GRID, 1, cv2.LINE_AA)
        cv2.putText(img, fmt.format(y_lo + frac * (y_hi - y_lo)), (6, gy + 4), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, fmt.format(y_hi), (6, y0 + 8), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, fmt.format(y_lo), (6, y0 + ph + 4), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

    def plot(hist, color):
        windowed = [(t, v) for t, v in hist if t_lo - max_gap_sec <= t <= t_hi]
        segment, last_t = [], None
        for t, v in windowed:
            gap = (last_t is not None) and (t - last_t > max_gap_sec)
            if v is None or gap:
                if len(segment) >= 2:
                    cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, color, 3, cv2.LINE_AA)
                segment = []
            if v is not None:
                segment.append(_to_px(t, np.clip(v, y_lo, y_hi), t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph))
            last_t = t
        if len(segment) >= 2:
            cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, color, 3, cv2.LINE_AA)
        elif len(segment) == 1:
            cv2.circle(img, segment[0], 4, color, -1, cv2.LINE_AA)

    plot(history_a, color_a)
    plot(history_b, color_b)
    mx, _ = _to_px(cur_time, y_lo, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
    cv2.line(img, (mx, y0), (mx, y0 + ph), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (mx, y0), (mx, y0 + ph), TEXT_WHITE, 2, cv2.LINE_AA)

    lx = x0 + pw - 6
    for lbl, color in ((label_b, color_b), (label_a, color_a)):
        tw = cv2.getTextSize(lbl, FONT, 0.32, 1)[0][0]
        lx -= tw
        cv2.putText(img, lbl, (lx, 20), FONT, 0.32, TEXT_WHITE, 1, cv2.LINE_AA)
        lx -= 16
        cv2.rectangle(img, (lx, 12), (lx + 10, 20), color, -1)
        lx -= 10
    return img


def draw_support_panel(w, h, title, lines, accent=ACCENT_CYAN):
    """A supporting text panel (Row C): title + a short list of
    human-facing lines. `lines`: list of (text, color). Line spacing
    is computed from the panel's own height so up to ~3 lines fit
    without clipping the last one against the bottom border."""
    img = panel_frame(w, h)
    panel_title(img, title)
    content_h = h - 42
    step = max(18, min(24, content_h // max(1, len(lines))))
    y = 42 + int(step * 0.7)
    for text, color in lines:
        cv2.putText(img, text, (16, y), FONT, 0.4, color, 1, cv2.LINE_AA)
        y += step
    return img


def draw_timeline_v2(width, height, span_sec, cur_time, dots, legend, title, tick_every_sec=15.0):
    """Clean, sparse timeline: colored DOTS at event times on a single
    axis line (not dense per-frame bars), minute-mark ticks, a legend
    row, and a current-time marker. `dots`: list of (time_sec, color,
    radius)."""
    img = panel_frame(width, height, bg=PANEL_BG)
    pad_l, pad_r = 20, 20
    axis_y = int(height * 0.42)
    x0, x1 = pad_l, width - pad_r
    cv2.line(img, (x0, axis_y), (x1, axis_y), BORDER, 2, cv2.LINE_AA)
    t = 0.0
    while t <= span_sec + 1e-6:
        x = int(x0 + t / span_sec * (x1 - x0))
        cv2.line(img, (x, axis_y - 4), (x, axis_y + 4), BORDER_SOFT, 1)
        mm = int(t // 60)
        ss = int(t % 60)
        cv2.putText(img, f"{mm}:{ss:02d}", (x - 12, axis_y + 18), FONT, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)
        t += tick_every_sec
    for tsec, color, r in dots:
        x = int(x0 + tsec / span_sec * (x1 - x0))
        cv2.circle(img, (x, axis_y), r, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (x, axis_y), r - 1, color, -1, cv2.LINE_AA)
    cx = int(x0 + np.clip(cur_time / span_sec, 0, 1) * (x1 - x0))
    cv2.line(img, (cx, 4), (cx, height - 22), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (cx, 4), (cx, height - 22), TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, title, (pad_l, height - 6), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    title_w = cv2.getTextSize(title, FONT, 0.32, 1)[0][0]
    lx = pad_l + title_w + 28
    for label, color in legend:
        cv2.circle(img, (lx, height - 9), 5, color, -1, cv2.LINE_AA)
        (lw, _), _ = cv2.getTextSize(label, FONT, 0.3, 1)
        cv2.putText(img, label, (lx + 10, height - 5), FONT, 0.3, TEXT_DIM, 1, cv2.LINE_AA)
        lx += 14 + lw + 18
    return img


def draw_timeline_bucketed(width, height, span_sec, cur_time, bucket_colors, bucket_sec,
                            extra_dots, legend, title):
    """For high-frequency per-frame state (offside trap state): render
    ONE bar per `bucket_sec` window colored by the MAJORITY state in
    that window (a disclosed downsampling for legibility, not a change
    to the underlying per-frame data -- see offside_v2_summary.md for
    the real per-frame series). `bucket_colors`: list of colors, one
    per bucket, already computed by the caller. `extra_dots`: sparse
    (time_sec, color, radius) markers (e.g. run starts) drawn above
    the bucket strip."""
    img = panel_frame(width, height, bg=PANEL_BG)
    pad_l, pad_r = 20, 20
    strip_top, strip_bot = 30, height - 26
    x0, x1 = pad_l, width - pad_r
    n = len(bucket_colors)
    bw = (x1 - x0) / max(n, 1)
    for i, color in enumerate(bucket_colors):
        bx0 = int(x0 + i * bw)
        bx1 = int(x0 + (i + 1) * bw) + 1
        cv2.rectangle(img, (bx0, strip_top), (bx1, strip_bot), color, -1)
    for tsec, color, r in extra_dots:
        x = int(x0 + tsec / span_sec * (x1 - x0))
        cv2.circle(img, (x, strip_top - 6), r, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (x, strip_top - 6), r - 1, color, -1, cv2.LINE_AA)
    t = 0.0
    while t <= span_sec + 1e-6:
        x = int(x0 + t / span_sec * (x1 - x0))
        mm, ss = int(t // 60), int(t % 60)
        cv2.putText(img, f"{mm}:{ss:02d}", (x - 12, height - 6), FONT, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)
        t += 15.0
    cx = int(x0 + np.clip(cur_time / span_sec, 0, 1) * (x1 - x0))
    cv2.line(img, (cx, 2), (cx, strip_bot), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (cx, 2), (cx, strip_bot), TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, title, (pad_l, 16), FONT, 0.3, TEXT_DIM, 1, cv2.LINE_AA)
    title_w = cv2.getTextSize(title, FONT, 0.3, 1)[0][0]
    lx = pad_l + title_w + 24
    for label, color in legend:
        cv2.rectangle(img, (lx, 6), (lx + 10, 14), color, -1)
        (lw, _), _ = cv2.getTextSize(label, FONT, 0.28, 1)
        cv2.putText(img, label, (lx + 14, 15), FONT, 0.28, TEXT_DIM, 1, cv2.LINE_AA)
        lx += 14 + lw + 16
    return img


# ============================================================================
# Broadcast-polish primitives (icon KPI cards, glow banner, team-comparison
# table/bars, data-derived "insights" bullets) -- added after feedback that
# the rebuild still read as a debug screen, not a finished analytics
# product. Every value drawn by these is passed in by the caller from real
# analytics output; nothing here invents a number.
# ============================================================================

def _draw_glyph(img, kind, cx, cy, r, color):
    """Simple decorative glyphs (drawn from primitives, no image assets)
    -- purely visual variety between KPI cards, never claiming to
    depict a specific measured quantity beyond the card's own label."""
    if kind == "target":
        cv2.circle(img, (cx, cy), r, color, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), max(2, r // 2), color, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 1, color, -1, cv2.LINE_AA)
    elif kind == "check":
        pts = np.array([(cx - r // 2, cy), (cx - r // 6, cy + r // 2), (cx + r // 2, cy - r // 2)], dtype=np.int32)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
    elif kind == "ruler":
        cv2.arrowedLine(img, (cx - r, cy), (cx + r, cy), color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(img, (cx + r, cy), (cx - r, cy), color, 2, cv2.LINE_AA, tipLength=0.25)
    elif kind == "cluster":
        for dx, dy in [(-r // 2, -r // 3), (r // 2, -r // 3), (0, r // 2)]:
            cv2.circle(img, (cx + dx, cy + dy), max(2, r // 4), color, -1, cv2.LINE_AA)
    elif kind == "bolt":
        pts = np.array([(cx + r // 4, cy - r), (cx - r // 2, cy + r // 6), (cx, cy + r // 6),
                         (cx - r // 4, cy + r), (cx + r // 2, cy - r // 6), (cx, cy - r // 6)], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)
    elif kind == "shield":
        pts = np.array([(cx, cy - r), (cx + r, cy - r // 2), (cx + r, cy + r // 3),
                         (cx, cy + r), (cx - r, cy + r // 3), (cx - r, cy - r // 2)], dtype=np.int32)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    else:
        cv2.circle(img, (cx, cy), r, color, 2, cv2.LINE_AA)


def draw_icon_kpi_card(w, h, label, value, sub, accent=ACCENT_CYAN, glyph="target"):
    """Reference-style KPI card: a soft colored circular badge with a
    simple decorative glyph (not claiming to depict anything beyond the
    card's own real label/value), big bold value, small caption above,
    small sub-line below (real context, e.g. sample size) -- no fake
    vs-previous-match delta since only one clip exists here."""
    img = panel_frame(w, h, bg=CARD_BG)
    cx, cy, r = 28, h // 2, 17
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), r + 4, accent, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, img, 0.78, 0, dst=img)
    cv2.circle(img, (cx, cy), r + 4, accent, 2, cv2.LINE_AA)
    _draw_glyph(img, glyph, cx, cy, r // 2 + 2, accent)
    tx = cx + r + 20
    cv2.putText(img, label.upper(), (tx, 20), FONT, 0.31, TEXT_DIM, 1, cv2.LINE_AA)
    value = str(value)
    avail_w = max(10, w - tx - 12)
    scale = h / 56.0
    while scale > 0.3 and cv2.getTextSize(value, FONT, scale, 2)[0][0] > avail_w:
        scale -= 0.05
    cv2.putText(img, value, (tx, int(h * 0.74)), FONT, scale, TEXT_WHITE, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (tx, h - 8), FONT, 0.29, accent, 1, cv2.LINE_AA)
    return img


def composite_state_badge(frame, x, y, state_text, accent, role_text=None, w=216, h=58):
    """Compact, rounded, translucent state badge composited directly
    onto a video frame -- replaces an earlier oversized flat rectangle
    after feedback ('no oversized flat rectangles', 'smaller and more
    polished'). A colored left accent bar; prominent state text on one
    line, a smaller team-role line beneath it, BOTH inside the same
    translucent rounded card so contrast against the raw, unpredictable
    video background underneath is never an issue."""
    roi = frame[y:y + h, x:x + w]
    overlay = roi.copy()
    rounded_rect(overlay, (0, 0), (w - 1, h - 1), (18, 14, 10), radius=10)
    blended = cv2.addWeighted(overlay, 0.68, roi, 0.32, 0)
    rounded_rect(blended, (0, 0), (w - 1, h - 1), accent, radius=10, thickness=2)
    cv2.rectangle(blended, (0, 10), (4, h - 10), accent, -1)
    frame[y:y + h, x:x + w] = blended
    scale = min(0.5, 0.1 * w / max(1, len(state_text)))
    cv2.putText(frame, state_text, (x + 14, y + int(h * 0.44)), FONT, scale, TEXT_WHITE, 2, cv2.LINE_AA)
    if role_text:
        # Plain single-stroke text -- a thick outline behind small text
        # SMEARS at this font scale (adjacent glyphs' strokes bleed
        # together), found and fixed earlier. Light-on-dark: the badge's
        # own fill is a dark near-black tint (not a pale accent glow
        # like the first version), so this needs a LIGHT color, not the
        # dark one used previously -- dark-on-dark was invisible, found
        # while reviewing this exact preview frame.
        cv2.putText(frame, role_text, (x + 14, y + h - 12), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    return frame


def draw_glow_banner(w, h, text, accent, subtext=None):
    """Big glowing status banner (e.g. 'TRAP ACTIVE') -- a soft colored
    backdrop glow behind crisp bold text, matching the reference's
    punchy alert-style state banners."""
    img = np.full((h, w, 3), BG, dtype=np.uint8)
    glow = np.zeros((h, w, 3), dtype=np.float32)
    cv2.rectangle(glow, (0, 0), (w, h), accent, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=h * 0.6)
    img = np.clip(img.astype(np.float32) + glow * 0.35, 0, 255).astype(np.uint8)
    rounded_rect(img, (1, 1), (w - 2, h - 2), (0, 0, 0), radius=6, thickness=2)
    scale = min(h / 46.0, 0.062 * w / max(1, len(text)))
    cv2.putText(img, text, (14, int(h * 0.62)), FONT, scale, TEXT_WHITE, 2, cv2.LINE_AA)
    if subtext:
        cv2.putText(img, subtext, (14, h - 6), FONT, 0.3, accent, 1, cv2.LINE_AA)
    return img


def draw_team_table(w, h, title, columns, rows, team_colors):
    """Small data table: `columns` = list of header strings; `rows` =
    list of (team_label, [cell strings...]); `team_colors` = list of
    colors, one per row, for a leading color swatch. The team-label
    column gets its own fixed margin so the N data columns (headers
    AND cells, at identical x positions) fit fully inside the panel
    without clipping against the right border."""
    img = panel_frame(w, h, bg=CARD_BG)
    panel_title(img, title)
    n_cols = len(columns)
    label_col_w = 110
    data_w = w - label_col_w - 16
    col_w = data_w // n_cols
    col_x = [label_col_w + ci * col_w for ci in range(n_cols)]
    y0 = 42
    for i, col in enumerate(columns):
        cv2.putText(img, col, (col_x[i], y0), FONT, 0.28, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.line(img, (10, y0 + 6), (w - 10, y0 + 6), BORDER_SOFT, 1)
    row_h = max(18, (h - y0 - 12) // max(1, len(rows)))
    for ri, (team_label, cells) in enumerate(rows):
        ry = y0 + 6 + (ri + 1) * row_h - 6
        color = team_colors[ri] if ri < len(team_colors) else TEXT_WHITE
        cv2.circle(img, (16, ry - 4), 4, color, -1, cv2.LINE_AA)
        cv2.putText(img, team_label, (26, ry), FONT, 0.32, TEXT_WHITE, 1, cv2.LINE_AA)
        for ci, cell in enumerate(cells):
            cv2.putText(img, str(cell), (col_x[ci], ry), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    return img


def draw_bar_comparison(w, h, title, metrics, color_a, color_b, label_a="Team 0", label_b="Team 1"):
    """`metrics`: list of (metric_label, value_a, value_b, unit, max_value).
    Two horizontal bars per metric, one per team, real values only --
    value None renders as a muted 'n/a' instead of a zero-length bar
    that would misleadingly read as zero."""
    img = panel_frame(w, h, bg=CARD_BG)
    panel_title(img, title)
    n = max(1, len(metrics))
    row_h = (h - 42) // n
    bar_x0, bar_max_w = 130, w - 200
    for i, (label, va, vb, unit, vmax) in enumerate(metrics):
        y = 42 + i * row_h
        cv2.putText(img, label, (10, y + 14), FONT, 0.3, TEXT_DIM, 1, cv2.LINE_AA)
        for j, (val, color) in enumerate([(va, color_a), (vb, color_b)]):
            by = y + 4 + j * 12
            if val is None:
                cv2.putText(img, "n/a", (bar_x0, by + 9), FONT, 0.28, TEXT_MUTED, 1, cv2.LINE_AA)
                continue
            bw = int(np.clip(val / max(1e-6, vmax), 0, 1) * bar_max_w)
            cv2.rectangle(img, (bar_x0, by), (bar_x0 + max(bw, 2), by + 9), color, -1)
            cv2.putText(img, f"{val:.1f}{unit}", (bar_x0 + bw + 6, by + 9), FONT, 0.28, TEXT_WHITE, 1, cv2.LINE_AA)
    return img


def draw_insights_panel(w, h, title, bullets, accent=ACCENT_ORANGE):
    """Title + bulleted list of short, data-derived factual lines
    (counts, rates, sample sizes actually computed by the caller --
    never a subjective/fabricated tactical narrative)."""
    img = panel_frame(w, h, bg=CARD_BG)
    panel_title(img, title)
    content_h = h - 42
    step = max(16, min(22, content_h // max(1, len(bullets))))
    y = 42 + int(step * 0.65)
    for text in bullets:
        cv2.circle(img, (14, y - 4), 2, accent, -1, cv2.LINE_AA)
        cv2.putText(img, text, (22, y), FONT, 0.31, TEXT_DIM, 1, cv2.LINE_AA)
        y += step
    return img
