"""
Phase 7 / C6 fix (2026-09-07 continuation): ONE shared pitch<->pixel
transform for every radar layer (player/ball markers, defensive lines,
pitch-control wash, pressing hotspot, dangerous-space hatch, run arrows),
so a color sampled under a player marker is guaranteed to be the same
grid cell used to compute that player's own pitch-control value.

Root cause this replaces: `draw_pitch(padding=50, scale=0.1)` produces a
(length*scale + 2*padding) x (width*scale + 2*padding) canvas (1300x800
for a 12000x7000cm pitch) with the PLAYABLE rectangle occupying pixels
[padding, padding+length*scale) x [padding, padding+width*scale) --
exactly what `pitch_xy_to_px` already targets for markers. The existing
renderers' pitch-control wash, however, resized its grid to cover the
FULL padded canvas (`pc["grid"]` -> `cv2.resize(..., pitch.shape[:2])`),
i.e. pitch x=0 landed at pixel 0 instead of pixel `padding` -- a
`padding` pixel (5m at this scale) misregistration versus every marker.
"""
import cv2
import numpy as np


def pitch_xy_to_px(x, y, scale=0.1, padding=50):
    """Canonical pitch-cm -> pixel transform. Matches the pre-existing
    `_pitch_xy_to_px` used by every V3 renderer for player/ball/line
    markers exactly (same formula, same defaults) -- this module does
    not change where markers are drawn, only makes the wash agree
    with them."""
    return int(x * scale) + padding, int(y * scale) + padding


def playing_rect_px(pitch_length_cm, pitch_width_cm, scale=0.1, padding=50):
    """(x0, y0, x1, y1) pixel bounds of the PLAYABLE rectangle only
    (excludes the padding border) within a `draw_pitch` canvas."""
    x0, y0 = pitch_xy_to_px(0, 0, scale, padding)
    x1, y1 = pitch_xy_to_px(pitch_length_cm, pitch_width_cm, scale, padding)
    return x0, y0, x1, y1


def apply_registered_control_wash(pitch_img, grid, team_a_color, team_b_color,
                                   pitch_length_cm, pitch_width_cm, scale=0.1, padding=50, alpha=0.88):
    """Resamples a pitch-control `grid` (values in [0,1], team-A
    fraction) into ONLY the playable rectangle -- at the exact same
    pixel bounds `pitch_xy_to_px` uses for markers -- then alpha-blends
    it, leaving the padding border untouched. Returns a new image (does
    not mutate `pitch_img`)."""
    x0, y0, x1, y1 = playing_rect_px(pitch_length_cm, pitch_width_cm, scale, padding)
    pw, ph = x1 - x0, y1 - y0
    up = cv2.resize(grid.astype(np.float32), (pw, ph), interpolation=cv2.INTER_CUBIC)
    up = np.clip(up, 0, 1)[..., None]
    color_field = (up * np.array(team_a_color) + (1 - up) * np.array(team_b_color)).astype(np.uint8)
    out = pitch_img.copy()
    region = out[y0:y1, x0:x1]
    # Same white-line-preserving blend the V3 renderers already use
    # (dashboard_style_v2.apply_color_wash_preserve_lines) -- reproduced
    # here rather than imported, to keep this module dependency-free of
    # the dashboard package for anything that isn't a full V3 renderer.
    line_mask = (region >= 200).all(axis=2)
    blended = cv2.addWeighted(color_field, alpha, region, 1 - alpha, 0)
    blended[line_mask] = region[line_mask]
    out[y0:y1, x0:x1] = blended
    return out
