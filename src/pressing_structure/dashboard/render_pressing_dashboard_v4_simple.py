"""
Pressing dashboard V4 -- FINAL simplified broadcast layout (2026-09-07
implementation pass). Same V4 data source as `render_pressing_dashboard_v4.py`
(`tactical_shared.snapshot.build_snapshot` / `pressing_v4.build_pressing_v4`,
evidence-aware FSM, both teams computed independently every frame) and the
same 2304x1204 canvas / video-output compatibility. This is the VISUAL
VALIDATION layer for Paper 1 (Pressing Structure Value & Optimization) --
it answers "why does the system think pressure is increasing?", not the
paper's scientific contribution itself.

LAYOUT: exactly two rows.
  ROW 1: tactical video feed (left) + pressing radar (right), both teams
         shown simultaneously, one compact state badge on the video only.
  ROW 2: exactly four graphs -- Press Intensity, Nearest Opponent
         Distance to Ball Carrier, Relative Closing Speed (signed, zero
         reference line), Numerical Support Around Ball Carrier (signed,
         step-style, zero reference line). Every graph carries a large
         title, an explicit X-axis label ("Match Time (seconds)"), an
         explicit Y-axis label with units, a one-line interpretation
         subtitle, and readable absolute-time tick labels -- a person who
         knows football but not this code should be able to read each
         graph unaided.

TWO TIME MODES:
  - "rolling" (default; what the final --out_path video uses): a
    trailing 20s window, tick labels at ABSOLUTE match-second values
    (e.g. at t=48s: 28s,32s,36s,40s,44s,48s), current-time cursor at the
    right edge.
  - "paper": the full 0-<duration>s clip, ticks every 30s, current-time
    cursor at its real position -- for static paper-figure/report use.
  Both modes only ever plot data up to the current frame (causal, no
  future-derived spoilers) and preserve real gaps for missing evidence
  -- no interpolation, no forward-fill, no event stitching, ever.

DATA LIMITATION (disclosed, not hidden): ball-carrier possession is only
resolved (`possession_state == 'CONTROLLED'`) in 144/3600 frames (4.0%)
of this clip -- every pressing graph is honestly this sparse because the
UPSTREAM possession-resolution feed is this sparse, not because of
anything in this renderer. Thresholds are not touched to make graphs
fuller.

Kept as a SEPARATE file from `render_pressing_dashboard_v4.py` (not an
in-place edit): that script produced the already-approved canonical
120s render and must stay reproducible exactly as before. No analytics
logic changes -- every value here is read as-is from the frozen V4
snapshot. "Team 0"/"Team 1" internal indices are UNCHANGED -- only the
user-facing TEXT/COLOR is Team A (green) / Team B (gold-yellow).
"""
import argparse
import math
import os
import pickle
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PARENT = os.path.dirname(PROJECT_ROOT)
sys.path[:0] = [PROJECT_ROOT, PROJECT_PARENT]

import cv2
import numpy as np
import polars as pl

from legacy_analytics.possession import CONTROL_RADIUS_CM
from pressing_structure.analytics.cleaned_tracking_view import build_cleaned_view, load_ball_view
from pressing_structure.analytics.pressure_field import (
    CONTOUR_LEVELS, escape_directions, escape_value_field, grid_to_pitch_matrix, passing_lane_metrics,
    pitch_grid, sample_field, team_pressure_field,
)
from pressing_structure.dashboard.dashboard_style_v2 import (
    ACCENT_CYAN, BG, BORDER_SOFT, FONT, GRID, TEXT_DIM, TEXT_WHITE,
    composite_state_badge, draw_header, panel_frame,
)
from pressing_structure.dashboard.render_pressing_dashboard import _nearest_transformer
# 2026-09-09 -- COACH-MODE OUTCOME BARS (Graph 3 only): reuses the
# research-phase's existing, already-tested episode/outcome pipeline
# UNCHANGED (never reimplemented here) purely to precompute the
# cumulative-outcome series once per video build. No analytics
# formula/threshold/taxonomy is touched.
from pressing_structure.analytics.pressing_v4 import build_pressing_v4
from pressing_structure.analytics.pressing_features import attribute_outcomes_all_episodes, HEURISTIC_TO_TAXONOMY
# 2026-09-10 -- COACH-MODE Graph 1 (Pressure Dominance) + Graph 2
# (Pressure Intensity Trend), per the two completed feasibility audits
# (audit_reports/2026-09-09_coach_graph_feasibility/,
# audit_reports/2026-09-09_continuous_pressure_signal_audit/). All NEW,
# additive, read-only-derived from the UNCHANGED team_pressure_field/
# sample_field -- no pressing analytics formula/threshold is touched.
from pressing_structure.analytics.coach_pressure_signal import (
    BALL_PROXIMITY_SIGMA_CM, LOW_PRESSURE, MODE_BALL_WEIGHTED, MODE_MISSING, MODE_UNWEIGHTED_FALLBACK,
    compute_dominance, compute_max_intensity, compute_opp_ballw_series, count_missing_segments,
    decimate_for_display, gate_dominance, nice_bar_axis,
)
from tactical_shared.tracking import build_quality_view, current_roles
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from tactical_shared.coordinates import DEFAULT_PITCH
from tactical_shared.radar_geometry import pitch_xy_to_px, playing_rect_px
# 2026-09-08 -- FINAL VISUALIZATION UPGRADE: shared perspective-radar
# camera/projection helper (rendering only -- no analytics/thresholds
# imported from here; PRESSING analytics themselves remain unchanged
# this round, only how the radar is DRAWN).
from tactical_shared.perspective_radar import (
    RadarCameraConfig, DEFAULT_RADAR_CAMERA, compute_radar_homography, project_point_int,
    draw_pitch_markings_perspective, marker_scale_factor, warp_raster_perspective, warp_scalar_field_perspective,
)

CONFIG = SoccerPitchConfiguration()
SUPPORT_RADIUS_CM = 500.0
SUPPORT_RADIUS_PX = int(round(SUPPORT_RADIUS_CM * 0.1))  # matches pitch_xy_to_px's fixed scale=0.1
HEATMAP_MAX_ALPHA = 0.55    # (2G) low = nearly transparent, high capped well short of opaque
HEATMAP_ALPHA_GAIN = 1.4

# ---------------------------------------------------------------------------
# 2026-09-08 -- PRESSURE-FIELD VISUAL-SEMANTIC REFINEMENT. The field is
# SPATIAL DEFENSIVE INFLUENCE / TIME-TO-REACH PRESSURE -- a continuous
# function of current positions/velocities, genuinely different from the
# PRESS EVENT FSM state (which additionally requires SUSTAINED evidence
# over time; see the field-computation comment below). It was ALREADY
# shown unconditionally whenever a carrier is known (never gated on
# PRESS_FORMING/ACTIVE_PRESS) -- this round only changes how PROMINENT
# that always-on wash/contours look, driven by the PRESSER team's own
# real FSM state, via alpha (opacity) and contour-line thickness alone.
# The underlying field VALUES and colors-per-value are IDENTICAL in
# every state -- only visibility/emphasis changes, never the numbers.
#
# Chosen scale (within the suggested 0.35-0.45 / 0.70-0.85 / 1.0 ranges,
# tuned once for on-pitch readability -- see the round's own preview QA):
#   NO_PRESS     -> 0.40x alpha, thin (1px) contours -- "this is the
#                   current spatial pressure landscape", never implies
#                   a confirmed press.
#   PRESS_FORMING-> 0.78x alpha, medium (2px) contours.
#   ACTIVE_PRESS -> 1.00x alpha, thick (3px) contours -- the strongest
#                   emphasis, reserved for real, FSM-confirmed ACTIVE_PRESS
#                   (never manufactured here; if this clip's own analytics
#                   never reach ACTIVE_PRESS, this tier simply never fires).
#   PRESS_ENDING -> 0.65x alpha, medium (2px) -- winding down, still
#                   real evidence, between FORMING and NO_PRESS.
#   UNCERTAIN/CONTESTED/anything else -> same as NO_PRESS: low, neutral,
#                   never implies a confirmed press.
PRESSURE_STATE_ALPHA_SCALE = {
    # 2026-09-08 (final presentation freeze): raised from 0.40x to
    # 0.45x after a real A/B test on representative NO_PRESS frames
    # (full-res and 640px share-size) found 0.40 already clearly
    # visible with negligible headroom (max pixel diff to 0.50 was
    # 13/255) -- 0.45 was chosen as the explicit final approved value
    # for the shareable video, still well short of PRESS_FORMING's
    # 0.78x, so NO_PRESS/UNCERTAIN/CONTESTED remain visually distinct
    # from an active press. Palette (yellow/amber/light-orange for this
    # tier) and every other state's value are unchanged.
    "NO_PRESS": 0.45, "UNCERTAIN": 0.45, "CONTESTED": 0.45,
    "PRESS_ENDING": 0.65, "PRESS_FORMING": 0.78, "ACTIVE_PRESS": 1.00,
}
PRESSURE_STATE_CONTOUR_THICKNESS = {
    "NO_PRESS": 1, "UNCERTAIN": 1, "CONTESTED": 1,
    "PRESS_ENDING": 2, "PRESS_FORMING": 2, "ACTIVE_PRESS": 3,
}
PRESSURE_STATE_DEFAULT = "UNCERTAIN"  # safe fallback if an unexpected state string ever appears

# ---------------------------------------------------------------------------
# Simple football palette -- Team A = green, Team B = gold/yellow, ball =
# white, IDENTICAL across video, radar and graphs. Local pressure intensity
# uses its own separate, team-agnostic yellow->orange->red scale.
# ---------------------------------------------------------------------------
TEAM_A_COLOR = (50, 180, 50)     # BGR -- green
TEAM_B_COLOR = (10, 200, 235)    # BGR -- gold / yellow
BALL_COLOR = (255, 255, 255)
TEAM_COLOR = {0: TEAM_A_COLOR, 1: TEAM_B_COLOR}
TEAM_LETTER = {0: "A", 1: "B"}   # UI-label-only remap (0/1 stay the internal keys)

PRESS_LOW = (80, 230, 255)       # pale yellow  -- low local pressure
PRESS_MED = (0, 130, 255)        # orange       -- medium
PRESS_HIGH = (20, 20, 220)       # red          -- strongest
PRESSURE_STOPS = (PRESS_LOW, PRESS_MED, PRESS_HIGH)
LINE_COLOR = (0, 140, 255)       # orange -- single-line (team-agnostic) graphs P2/P3

STATE_COLOR = {"NO_PRESS": (110, 110, 110), "PRESS_FORMING": (0, 140, 255), "ACTIVE_PRESS": (30, 30, 220),
               "PRESS_ENDING": (60, 200, 220), "UNCERTAIN": (90, 90, 90), "CONTESTED": (30, 100, 180)}
CARRIER_RING_COLOR = (255, 255, 255)   # white -- carrier highlight, never a team color (avoids implying "team = carrier")
SUPPORT_RADIUS_COLOR = (200, 200, 200)  # faint gray -- the 5m local-support radius, purely spatial reference
PARTICIPANT_RING_COLOR = (255, 255, 255)  # white stronger outline for confirmed pressing participants
PARTICIPANT_LINK_COLOR = (210, 210, 210)  # subtle light gray -- pressing-shape connecting lines
NEAREST_PRESSER_RING_COLOR = (0, 140, 255)   # orange -- the one real nearest_opponent_track_id
CLOSING_COLOR = (60, 220, 90)      # green -- relative_closing_cm_s > threshold (defender closing)
ESCAPING_COLOR = (30, 30, 220)     # red   -- relative_closing_cm_s < -threshold (gap opening)
NEUTRAL_CLOSING_COLOR = (200, 200, 200)  # gray -- roughly stable
CLOSING_NEUTRAL_THRESHOLD_CM_S = 20.0

# (B1) contour line colors, one per CONTOUR_LEVELS entry (low/med/high).
CONTOUR_COLORS = (PRESS_LOW, PRESS_MED, PRESS_HIGH)

# ---------------------------------------------------------------------------
# 2026-09-08 -- STATE-AWARE PALETTE EMPHASIS (round F, building on round
# E's alpha/thickness-only emphasis above). The field VALUES fed into this
# LUT are never touched -- `v` in `_colorize_field` below is the exact same
# clipped raw field array as before this round. Only WHICH 3 BGR stops
# that value is interpolated between changes per state, through the same
# single 2-segment low->mid->high formula every mode reuses (never a
# second/duplicated color code path) -- i.e. `pressure_value + press_state
# -> display RGBA`, never `pressure_value * state_factor -> fake value`.
#
#   "subtle"  (NO_PRESS/UNCERTAIN/CONTESTED) -- low=soft yellow,
#             mid=amber, high=deepened amber/light orange. Never reaches
#             PRESS_HIGH (red) at ANY field value -- even a maxed-out
#             spatial field still reads as "warm, not alarming" while no
#             sustained press evidence exists (see p4 preview below).
#   "ending"  (PRESS_ENDING) -- low=yellow, mid=orange (same mid stop as
#             the full palette), high=a restrained orange-red blend
#             (55% PRESS_HIGH / 45% PRESS_MED) -- winding down, between
#             subtle and the full palette.
#   "forming" (PRESS_FORMING) -- the full existing low->orange->red LUT,
#             UNCHANGED from every state before this round -- a genuine
#             localized red core once evidence has actually accumulated.
#   "active"  (ACTIVE_PRESS) -- identical color stops to "forming" (the
#             literal "full existing yellow->orange->red palette" the
#             spec asks for); ACTIVE_PRESS is distinguished from
#             PRESS_FORMING purely via the EXISTING alpha/thickness
#             emphasis above, not a second color ramp -- the spec
#             describes both with the same "yellow->orange->red" wording,
#             so inventing a 4th color tier here would be undocumented,
#             unrequested behavior.
# Both new intermediate colors are themselves plain blends of the
# module's existing PRESS_LOW/MED/HIGH stops (never invented from
# nowhere), keeping every tier visually related to the others.
PRESS_AMBER = tuple(int(round(a * 0.55 + b * 0.45)) for a, b in zip(PRESS_LOW, PRESS_MED))         # "subtle" mid stop
PRESS_LIGHT_ORANGE = tuple(int(round(a * 0.35 + b * 0.65)) for a, b in zip(PRESS_LOW, PRESS_MED))  # "subtle" high-cap stop (never reaches red)
PRESS_ORANGE_RED = tuple(int(round(a * 0.45 + b * 0.55)) for a, b in zip(PRESS_MED, PRESS_HIGH))   # "ending" high-cap stop (restrained red)

PRESSURE_PALETTE_STOPS = {
    "subtle": (PRESS_LOW, PRESS_AMBER, PRESS_LIGHT_ORANGE),
    "ending": (PRESS_LOW, PRESS_MED, PRESS_ORANGE_RED),
    "forming": (PRESS_LOW, PRESS_MED, PRESS_HIGH),
    "active": (PRESS_LOW, PRESS_MED, PRESS_HIGH),
}
PRESSURE_STATE_PALETTE_MODE = {
    "NO_PRESS": "subtle", "UNCERTAIN": "subtle", "CONTESTED": "subtle",
    "PRESS_ENDING": "ending", "PRESS_FORMING": "forming", "ACTIVE_PRESS": "active",
}
PALETTE_MODE_DEFAULT = "active"  # backward-compatible: identical to the pre-round-F full LUT

# ---------------------------------------------------------------------------
# 2026-09-08 -- CONTINUOUS FIELD, TEAM-CONSISTENT DUAL TINT (round G). Used
# ONLY for `_select_pressure_field_team()`'s CASE 3 (neither the instant
# carrier nor the existing trailing-4s `historical_defender` hold gives a
# defending-team attribution) -- see that function's own docstring for the
# full evidence hierarchy. Reuses the SAME `_colorize_field` 2-segment LUT
# and the SAME `PRESSURE_PALETTE_STOPS` lookup as every other palette mode
# (never a second/divergent color code path) -- just two more entries,
# keyed by team id, that ramp from a shared neutral-dark low stop up to
# THAT team's own real identity color (`TEAM_A_COLOR`/`TEAM_B_COLOR`,
# exactly what dots/boxes for that team already use elsewhere) rather than
# the press-intensity yellow/orange/red ramp -- so a viewer immediately
# reads "this is Team A's/B's zone of reach", never "this team is
# pressing" (which the SEPARATE event badge, unchanged, would still say
# "Uncertain" for anyway, since case 3 by definition has no team-attributed
# FSM state to report).
_TINT_NEUTRAL = (35, 35, 35)  # near-black -- shared low stop for BOTH tints, so v=0 never reads as any particular hue
def _tint_stop(team_color, t):
    return tuple(int(round(a * (1 - t) + b * t)) for a, b in zip(_TINT_NEUTRAL, team_color))
TEAM_TINT_PALETTE_MODE = {0: "team0_tint", 1: "team1_tint"}
PRESSURE_PALETTE_STOPS["team0_tint"] = (_TINT_NEUTRAL, _tint_stop(TEAM_A_COLOR, 0.6), TEAM_A_COLOR)
PRESSURE_PALETTE_STOPS["team1_tint"] = (_TINT_NEUTRAL, _tint_stop(TEAM_B_COLOR, 0.6), TEAM_B_COLOR)
# Deliberately MORE subdued than even the single-team UNCERTAIN/NO_PRESS
# tier (0.40x) -- case 3 lacks not just event evidence but TEAM
# ATTRIBUTION itself, so it earns the lowest-confidence visual treatment
# of any rendered field. Contours are deliberately NOT drawn in dual mode
# (see `draw_left_feed`/`draw_radar`) -- two overlapping unlabeled contour
# sets would read as precise boundaries this mode explicitly disclaims.
DUAL_UNCERTAIN_ALPHA_SCALE = 0.30


def _select_pressure_field_team(snap_f):
    """Evidence hierarchy for WHICH team's base spatial-influence field to
    attribute as "the defending team" field, independent of any
    carrier-SPECIFIC overlay (escape arrows, presser ring, lanes, etc,
    which remain separately gated on a resolved carrier). Reuses ONLY
    already-existing, already-documented signals from `tactical_shared.
    tracking.current_roles()` -- never a new persistence/forward-fill
    mechanism introduced here:

      CASE 1 -- `current_carrier_team` resolved THIS instant (real ball-
                possession evidence, ~4.0% of frames in this clip) -> the
                team NOT carrying is the defending team.
      CASE 2 -- carrier not resolved this instant, but `historical_defender`
                (`current_roles()`'s own `historical_context_role` -- an
                EXISTING, already-computed, already-DISCLOSED trailing-4s
                MAJORITY vote of the non-carrying team, cleared at period
                boundaries and expiring after 4s of no evidence -- see
                `tactical_shared/tracking.py`'s own docstring) gives a
                clear majority -> that team is used (~42.6% of frames).
      CASE 3 -- neither available (~53.4% of frames) -> returns None.
                The caller falls back to the dual-team, no-single-team-
                attribution-implied rendering mode -- NEVER an arbitrary
                Team A/B guess, and never a hold beyond what
                `historical_context_role` itself already bounds to 4s."""
    ct = snap_f.get("current_carrier_team")
    if ct is not None:
        return 1 - ct
    hd = snap_f.get("historical_defender")
    if hd is not None:
        return hd
    return None

# 2026-09-08 V8 -- P1/P2 simplification: exactly the TOP 2 low-pressure
# escape directions (best=thicker, second=thinner -- never 3+, never
# called a "recommendation"), and at most 2 lowest-pressure passing
# lanes (best=brighter, second=dimmer) + 1 highly-constrained lane for
# contrast (dashed, so it never reads as "the same kind of thing" as
# the open lanes).
ESCAPE_ARROW_COLOR = (235, 235, 235)     # near-white -- "LOW-PRESSURE ESCAPE DIRECTION", never a team color
LANE_OPEN_COLOR = (235, 245, 250)        # bright cream/white -- best (lowest-pressure) lane
LANE_OPEN_SECOND_COLOR = (170, 180, 185)  # dimmer cream -- second-best lane
LANE_CONSTRAINED_COLOR = (30, 30, 220)   # red, DASHED -- highly constrained lane (contrast only)
N_ESCAPE_ARROWS = 2
N_OPEN_LANES = 2
N_CONSTRAINED_LANES = 1

# ROW1 (video+radar) kept at the exact same height as the approved V4
# correctness-preview dashboard; Row 2 is one taller graph strip (no
# KPI/table/notes/timeline rows).
HEADER_H, TOP_ROW_H, GRAPH_ROW_H, GAP = 60, 602, 536, 6
CANVAS_W = 2304
LEFT_W = round(CANVAS_W * 0.52)
RIGHT_W = CANVAS_W - LEFT_W
assert HEADER_H + TOP_ROW_H + GRAPH_ROW_H + GAP == 1204


def _colorize_field(field, max_alpha=HEATMAP_MAX_ALPHA, alpha_gain=HEATMAP_ALPHA_GAIN, alpha_scale=1.0,
                     palette_mode=PALETTE_MODE_DEFAULT):
    """(2C/2G) field values in [0,1) -> a per-pixel BGR color (2-segment
    low->mid->high LUT) + a per-pixel ALPHA that is near-zero at low
    pressure (dark pitch/video stays visible) and capped at `max_alpha`
    at high pressure (players/ball never fully occluded) -- never a flat
    single-color wash. The RAW field value `v` is never altered by
    either parameter below -- both are display-only.

    `alpha_scale` (round E): a state-driven OPACITY multiplier only.

    `palette_mode` (round F, new): selects WHICH 3 BGR stops (low, mid,
    high) `v` is interpolated between -- see `PRESSURE_PALETTE_STOPS`'s
    own comment for the per-mode rationale. Defaults to `"active"`,
    the ORIGINAL PRESS_LOW/MED/HIGH stops -- so any caller that does not
    pass `palette_mode` renders byte-identical to before this round."""
    v = np.clip(field, 0.0, 1.0)
    low_c, mid_c, high_c = PRESSURE_PALETTE_STOPS.get(palette_mode, PRESSURE_PALETTE_STOPS[PALETTE_MODE_DEFAULT])
    low = np.array(low_c, dtype=np.float32)
    med = np.array(mid_c, dtype=np.float32)
    high = np.array(high_c, dtype=np.float32)
    t1 = np.clip(v * 2.0, 0, 1)[..., None]
    color = low * (1 - t1) + med * t1
    t2 = np.clip(v * 2.0 - 1.0, 0, 1)[..., None]
    color = color * (1 - t2) + high * t2
    alpha = np.clip(v * alpha_gain, 0, 1) * max_alpha * alpha_scale
    return color.astype(np.float32), alpha.astype(np.float32)


def apply_pressure_wash_radar(pitch_img, field, pitch=DEFAULT_PITCH, alpha_scale=1.0, palette_mode=PALETTE_MODE_DEFAULT):
    """(2C) full-pitch wash on the radar's playable rectangle -- SAME
    registration technique already established for the pitch-control
    wash (`tactical_shared.radar_geometry.playing_rect_px`, the C6 fix:
    resample into the exact playable-rectangle pixel bounds
    `pitch_xy_to_px` uses for every marker, not the whole padded canvas)
    -- so a color sampled under a player is the same field cell used to
    compute that player's own influence. White pitch-line pixels are
    restored after blending so markings stay crisp."""
    x0, y0, x1, y1 = playing_rect_px(pitch.length_cm, pitch.width_cm)
    pw, ph = x1 - x0, y1 - y0
    color, alpha = _colorize_field(field, alpha_scale=alpha_scale, palette_mode=palette_mode)
    color_up = cv2.resize(color, (pw, ph), interpolation=cv2.INTER_CUBIC)
    alpha_up = cv2.resize(alpha, (pw, ph), interpolation=cv2.INTER_CUBIC)[..., None]
    out = pitch_img.copy()
    region = out[y0:y1, x0:x1].astype(np.float32)
    line_mask = (region >= 200).all(axis=2)
    blended = region * (1 - alpha_up) + color_up * alpha_up
    blended[line_mask] = region[line_mask]
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def apply_pressure_wash_video(frame, field, transformer, alpha_scale=1.0, palette_mode=PALETTE_MODE_DEFAULT):
    """(2F) the SAME pitch-space raster, perspective-warped directly
    into broadcast image space -- pitch-space pressure raster -> single
    composed homography (grid-array px -> pitch cm -> image px) ->
    warpPerspective -> alpha blend. A genuinely trapezoidal/perspective-
    correct shape, never a circle/ellipse approximation. Returns the
    frame UNCHANGED if the homography is unavailable/singular this
    frame (2F/A4: never fabricate a broadcast overlay)."""
    if transformer is None:
        return frame
    try:
        inv_m = np.linalg.inv(transformer.m)
    except np.linalg.LinAlgError:
        return frame
    h, w = frame.shape[:2]
    combined = (inv_m.astype(np.float64) @ grid_to_pitch_matrix()).astype(np.float32)
    color, alpha = _colorize_field(field, alpha_scale=alpha_scale, palette_mode=palette_mode)
    warped_color = cv2.warpPerspective(color, combined, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    warped_alpha = cv2.warpPerspective(alpha, combined, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)[..., None]
    out = frame.astype(np.float32) * (1 - warped_alpha) + warped_color * warped_alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_contours_radar(pitch_img, field, pitch=DEFAULT_PITCH, levels=CONTOUR_LEVELS, colors=None, thickness=1,
                         palette_mode=PALETTE_MODE_DEFAULT):
    """(B1) tactical contour lines at CALIBRATED levels (see
    `pressure_field.CONTOUR_LEVELS` -- corresponding to ~1.5s/0.75s/0.3s
    single-defender time-to-reach, not a silent/arbitrary choice), drawn
    over the SAME registration-correct playable rectangle the fill wash
    uses. The filled heatmap stays subtle; contours make the boundary
    itself easy to read. LEVELS themselves (what counts as low/med/high)
    never change. `thickness` (round E): state-driven emphasis only (see
    `PRESSURE_STATE_CONTOUR_THICKNESS`). `colors`/`palette_mode` (round
    F): if `colors` is not given explicitly, contour line colors are
    drawn from the SAME `PRESSURE_PALETTE_STOPS[palette_mode]` triple the
    fill wash uses for this state -- one shared color table, never a
    second divergent one."""
    if colors is None:
        colors = PRESSURE_PALETTE_STOPS.get(palette_mode, PRESSURE_PALETTE_STOPS[PALETTE_MODE_DEFAULT])
    x0, y0, x1, y1 = playing_rect_px(pitch.length_cm, pitch.width_cm)
    pw, ph = x1 - x0, y1 - y0
    field_up = cv2.resize(field.astype(np.float32), (pw, ph), interpolation=cv2.INTER_CUBIC)
    out = pitch_img.copy()
    for level, color in zip(levels, colors):
        mask = (field_up >= level).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shifted = [c + np.array([x0, y0]) for c in contours]
        cv2.drawContours(out, shifted, -1, color, thickness, cv2.LINE_AA)
    return out


def apply_pressure_wash_radar_perspective(pitch_img, field, H, pitch=DEFAULT_PITCH, alpha_scale=1.0,
                                           palette_mode=PALETTE_MODE_DEFAULT):
    """(Perspective radar) the SAME `_colorize_field` color/alpha this
    project already computes for the flat radar and the broadcast video,
    warped through the SAME shared `compute_radar_homography()` matrix
    every other perspective-radar object uses -- never a second,
    divergent color/alpha model."""
    from pressing_structure.analytics.pressure_field import GRID_STEP_CM
    h, w = pitch_img.shape[:2]
    color, alpha = _colorize_field(field, alpha_scale=alpha_scale, palette_mode=palette_mode)
    warped_color, warped_alpha = warp_raster_perspective(color, alpha, H, GRID_STEP_CM, w, h)
    out = pitch_img.astype(np.float32) * (1 - warped_alpha[..., None]) + warped_color * warped_alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_contours_radar_perspective(pitch_img, field, H, levels=CONTOUR_LEVELS, colors=None, thickness=1,
                                     palette_mode=PALETTE_MODE_DEFAULT):
    """(Perspective radar) SAME contour technique as `draw_contours_video`
    -- warp the RAW field through the shared homography, then threshold/
    findContours directly in canvas-pixel space (never a polyline
    reprojection of grid-space contours). `thickness`: state-driven
    emphasis only, same convention as `draw_contours_radar`. `colors`/
    `palette_mode`: same shared-palette convention as `draw_contours_radar`."""
    if colors is None:
        colors = PRESSURE_PALETTE_STOPS.get(palette_mode, PRESSURE_PALETTE_STOPS[PALETTE_MODE_DEFAULT])
    from pressing_structure.analytics.pressure_field import GRID_STEP_CM
    h, w = pitch_img.shape[:2]
    warped_field = warp_scalar_field_perspective(field, H, GRID_STEP_CM, w, h)
    out = pitch_img.copy()
    for level, color in zip(levels, colors):
        mask = (warped_field >= level).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, thickness, cv2.LINE_AA)
    return out


def draw_contours_video(frame, field, transformer, levels=CONTOUR_LEVELS, colors=None, thickness=1,
                         palette_mode=PALETTE_MODE_DEFAULT):
    """(B1/B7) SAME contour technique, warped into broadcast space from
    the identical raster the video wash uses -- never a separate
    image-space estimate. Frame unchanged if the homography is invalid.
    `thickness`: state-driven emphasis only, same convention as
    `draw_contours_radar`. `colors`/`palette_mode`: same shared-palette
    convention as `draw_contours_radar`."""
    if colors is None:
        colors = PRESSURE_PALETTE_STOPS.get(palette_mode, PRESSURE_PALETTE_STOPS[PALETTE_MODE_DEFAULT])
    if transformer is None:
        return frame
    try:
        inv_m = np.linalg.inv(transformer.m)
    except np.linalg.LinAlgError:
        return frame
    h, w = frame.shape[:2]
    combined = (inv_m.astype(np.float64) @ grid_to_pitch_matrix()).astype(np.float32)
    warped_field = cv2.warpPerspective(field.astype(np.float32), combined, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
    out = frame.copy()
    for level, color in zip(levels, colors):
        mask = (warped_field >= level).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, thickness, cv2.LINE_AA)
    return out


def _bbox_ground_point(tracking_rows, track_id):
    row = next((r for r in tracking_rows if r["track_id"] == track_id), None)
    if row is None:
        return None
    return (int((row["bbox_x1"] + row["bbox_x2"]) / 2), int(row["bbox_y2"])), row


def draw_left_feed(frame, tracking_rows, snap_f, transformer, carrier_track_id, presser_track_id, closing_color,
                    field, carrier_xy, escape_points, lanes, pressure_alpha_scale=1.0, pressure_contour_thickness=1,
                    pressure_palette_mode=PALETTE_MODE_DEFAULT, field_b=None, dual_field_mode=False):
    vis = frame.copy()
    for r in tracking_rows:
        if r["object_type"] not in ("player", "goalkeeper"):
            continue
        color = TEAM_COLOR.get(r["team_id"], (180, 180, 180))
        cv2.rectangle(vis, (int(r["bbox_x1"]), int(r["bbox_y1"])), (int(r["bbox_x2"]), int(r["bbox_y2"])), color, 2)

    # (2F/B7) full-pitch pressure field + (B1) contours, perspective-
    # warped into the broadcast video from the SAME pitch-space raster
    # the radar washes -- never a circular approximation. Skipped
    # (frame unchanged) whenever the homography is unavailable. ALWAYS
    # drawn whenever `field is not None` -- CONTINUOUS (round G): no
    # longer requires a resolved carrier, only sufficient team player
    # geometry (see `_select_pressure_field_team`); never gated on
    # PRESS_FORMING/ACTIVE_PRESS -- only `pressure_alpha_scale`/
    # `pressure_contour_thickness`/`pressure_palette_mode` (the field
    # team's own real FSM state) change how PROMINENT it looks.
    #
    # `dual_field_mode` (round G, CASE 3 of `_select_pressure_field_team`):
    # team attribution itself is unknown -- `field`/`field_b` are Team
    # A/B's OWN independently-computed fields, each washed with THAT
    # team's own tint color at a fixed, extra-subdued alpha, and with NO
    # contours (two overlapping unlabeled contour sets would read as
    # precise boundaries this mode explicitly disclaims). Either field
    # may be None on its own (that team simply has no valid geometry
    # this frame) without affecting the other.
    if dual_field_mode:
        if field is not None:
            vis = apply_pressure_wash_video(vis, field, transformer, alpha_scale=DUAL_UNCERTAIN_ALPHA_SCALE, palette_mode=TEAM_TINT_PALETTE_MODE[0])
        if field_b is not None:
            vis = apply_pressure_wash_video(vis, field_b, transformer, alpha_scale=DUAL_UNCERTAIN_ALPHA_SCALE, palette_mode=TEAM_TINT_PALETTE_MODE[1])
    elif field is not None:
        vis = apply_pressure_wash_video(vis, field, transformer, alpha_scale=pressure_alpha_scale, palette_mode=pressure_palette_mode)
        vis = draw_contours_video(vis, field, transformer, thickness=pressure_contour_thickness, palette_mode=pressure_palette_mode)

    # (B7/P2) passing lanes projected the same pitch-space-point way,
    # clipped to frame -- each entry carries its own (color, thickness, dashed).
    for (x0, y0), (x1, y1), color, thickness, dashed in (lanes or []):
        pts = _project_pitch_points(transformer, [(x0, y0), (x1, y1)])
        if pts is not None:
            _clipped_line(vis, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[1]).astype(int)), color, thickness, dashed)

    # (B3/B7/P1) exactly the TOP N_ESCAPE_ARROWS low-pressure escape
    # directions -- best (index 0) thicker, second thinner.
    if carrier_xy is not None and escape_points:
        c_pts = _project_pitch_points(transformer, [carrier_xy])
        if c_pts is not None:
            c_img = tuple(np.round(c_pts[0]).astype(int))
            for i, (ex, ey, _) in enumerate(escape_points):
                e_pts = _project_pitch_points(transformer, [(ex, ey)])
                if e_pts is not None:
                    e_img = tuple(np.round(e_pts[0]).astype(int))
                    thickness = 3 if i == 0 else 1
                    cv2.arrowedLine(vis, c_img, e_img, ESCAPE_ARROW_COLOR, thickness, cv2.LINE_AA, tipLength=0.25)

    carrier_pt = _bbox_ground_point(tracking_rows, carrier_track_id)
    if carrier_pt is not None:
        (cx, cy), row = carrier_pt
        rw = max(24, int((row["bbox_x2"] - row["bbox_x1"]) * 0.9))
        cv2.ellipse(vis, (cx, cy), (rw, max(8, rw // 3)), 0, 0, 360, CARRIER_RING_COLOR, 2, cv2.LINE_AA)

    presser_pt = _bbox_ground_point(tracking_rows, presser_track_id)
    if presser_pt is not None:
        (px, py), row = presser_pt
        rw = max(24, int((row["bbox_x2"] - row["bbox_x1"]) * 0.9))
        cv2.ellipse(vis, (px, py), (rw, max(8, rw // 3)), 0, 0, 360, NEAREST_PRESSER_RING_COLOR, 2, cv2.LINE_AA)
        if carrier_pt is not None:
            cv2.line(vis, (px, py), carrier_pt[0], closing_color or NEUTRAL_CLOSING_COLOR, 2, cv2.LINE_AA)

    dom = snap_f["pressing"]["dominant_state"]
    team = snap_f["pressing"]["dominant_team"]
    banner = dom.replace("_", " ").title()
    color = STATE_COLOR.get(dom, (100, 100, 100))
    role = (f"Team A: {snap_f['pressing'][0]['state'].replace('_',' ').title()}   "
            f"Team B: {snap_f['pressing'][1]['state'].replace('_',' ').title()}")
    if team is not None:
        role += f"   (dominant: Team {TEAM_LETTER[team]})"
    vis = composite_state_badge(vis, 10, 10, banner, color, role, w=460)
    out = np.full((vis.shape[0] + 32, vis.shape[1], 3), BG, dtype=np.uint8)
    out[32:, :] = vis
    cv2.rectangle(out, (12, 8), (15, 24), ACCENT_CYAN, -1)
    cv2.putText(out, "TACTICAL MATCH FEED", (22, 22), FONT, 0.46, TEXT_WHITE, 2, cv2.LINE_AA)
    return out


def draw_radar(players_this_frame, ball_xy, snap_f, w, h, carrier, presser, presser_team, participant_ids,
                balance_text, closing_color, field, escape_points, lanes, debug=False, carrier_field_pressure=None,
                radar_view="flat", H=None, pressure_alpha_scale=1.0, pressure_contour_thickness=1,
                pressure_palette_mode=PALETTE_MODE_DEFAULT, field_b=None, dual_field_mode=False):
    """Radar kept deliberately minimal: Team A/B players, ball, a local
    pressure glow, and a small set of REAL, disclosed spatial cues
    around the carrier -- no pitch-control RGB wash, no corner text
    blocks, no legend box.

    `carrier`: the player dict looked up by `snap_f["current_carrier_track"]`
    (or None -- an honest gap on the ~96% of frames with no confirmed
    possession). `presser`: the player dict looked up by that presser
    team's `features.nearest_opponent_track_id` -- the EXACT player the
    already-plotted `nearest_opponent_distance_cm`/Graph 2 describes
    (2026-09-07 V5 pass: previously blocked pending this exact ID being
    exposed; now a real, disclosed identity, not an approximation).
    `presser_team`: whichever team's `features.participant_ids` is real
    this frame. `participant_ids`: that team's REAL `participant_ids`.
    `balance_text`: e.g. "A +2" / "B +1" / "Even", from the exact same
    `support_minus_opponents` field Graph 4 plots, or None. `closing_color`:
    CLOSING/ESCAPING/NEUTRAL color derived from the exact same
    `relative_closing_cm_s` value Graph 3 plots."""
    # 2026-09-08 FINAL VISUALIZATION UPGRADE -- ONE shared projection for
    # every radar object this call draws (pitch markings, players, ball,
    # pressure wash/contours, lanes, escape arrows, presser/carrier links).
    # `project(x,y)` is the ONLY thing that changes between flat and
    # perspective mode; every drawing call below is otherwise identical
    # to before this round.
    perspective = radar_view == "perspective" and H is not None
    if perspective:
        w0, h0 = 1300, 800  # SAME intrinsic working canvas size draw_pitch() itself produces
        base = draw_pitch_markings_perspective(H, CONFIG, w0, h0)
        project = lambda x, y: project_point_int(H, x, y)
        scale_at = lambda x, y: marker_scale_factor(H, x, y, DEFAULT_PITCH)
    else:
        base = draw_pitch(config=CONFIG)
        w0, h0 = base.shape[1], base.shape[0]
        project = pitch_xy_to_px
        scale_at = lambda x, y: 1.0
    pitch_dark = cv2.addWeighted(base, 0.7, np.zeros_like(base), 0.3, 0)

    carrier_px = project(carrier["x_pitch"], carrier["y_pitch"]) if carrier is not None else None
    presser_px = project(presser["x_pitch"], presser["y_pitch"]) if presser is not None else None

    # (2A-2D) FULL-PITCH pressure field(s), computed once in render_frame
    # and shared (identical array(s)) with the video wash for consistency
    # -- SAME array either way, only the raster's target projection
    # differs. CONTINUOUS (round G): no longer requires a resolved
    # carrier -- see `_select_pressure_field_team`. `dual_field_mode`
    # (CASE 3 -- team attribution itself unknown): wash Team A's/B's OWN
    # fields separately with THAT team's own tint color, fixed extra-
    # subdued alpha, and NO contours -- same rationale as `draw_left_feed`.
    if dual_field_mode:
        if field is not None:
            if perspective:
                pitch_dark = apply_pressure_wash_radar_perspective(pitch_dark, field, H, alpha_scale=DUAL_UNCERTAIN_ALPHA_SCALE, palette_mode=TEAM_TINT_PALETTE_MODE[0])
            else:
                pitch_dark = apply_pressure_wash_radar(pitch_dark, field, alpha_scale=DUAL_UNCERTAIN_ALPHA_SCALE, palette_mode=TEAM_TINT_PALETTE_MODE[0])
        if field_b is not None:
            if perspective:
                pitch_dark = apply_pressure_wash_radar_perspective(pitch_dark, field_b, H, alpha_scale=DUAL_UNCERTAIN_ALPHA_SCALE, palette_mode=TEAM_TINT_PALETTE_MODE[1])
            else:
                pitch_dark = apply_pressure_wash_radar(pitch_dark, field_b, alpha_scale=DUAL_UNCERTAIN_ALPHA_SCALE, palette_mode=TEAM_TINT_PALETTE_MODE[1])
    elif field is not None:
        if perspective:
            pitch_dark = apply_pressure_wash_radar_perspective(pitch_dark, field, H, alpha_scale=pressure_alpha_scale, palette_mode=pressure_palette_mode)
            pitch_dark = draw_contours_radar_perspective(pitch_dark, field, H, thickness=pressure_contour_thickness, palette_mode=pressure_palette_mode)  # (B1)
        else:
            pitch_dark = apply_pressure_wash_radar(pitch_dark, field, alpha_scale=pressure_alpha_scale, palette_mode=pressure_palette_mode)
            pitch_dark = draw_contours_radar(pitch_dark, field, thickness=pressure_contour_thickness, palette_mode=pressure_palette_mode)  # (B1)

    # (B6/P2) at most 2 lowest-pressure lanes (best brighter/thicker,
    # second dimmer/thinner) + at most 1 highly-constrained lane
    # (dashed, for contrast) -- drawn before the escape arrows/carrier
    # ring so those stay visually on top.
    for (x0, y0), (x1, y1), color, thickness, dashed in (lanes or []):
        p0, p1 = project(x0, y0), project(x1, y1)
        if dashed:
            _dashed_line_simple(pitch_dark, p0, p1, color, thickness)
        else:
            cv2.line(pitch_dark, p0, p1, color, thickness, cv2.LINE_AA)

    # (B3/B4/P1) exactly the TOP N_ESCAPE_ARROWS LOW-PRESSURE ESCAPE
    # DIRECTIONS -- real sampled points, never labeled a recommendation;
    # best (index 0) thicker, second thinner, never 3+.
    if carrier_px is not None:
        for i, (ex, ey, _) in enumerate(escape_points or []):
            thickness = 3 if i == 0 else 1
            cv2.arrowedLine(pitch_dark, carrier_px, project(ex, ey), ESCAPE_ARROW_COLOR, thickness, cv2.LINE_AA, tipLength=0.25)

    if carrier_px is not None:
        # (4) local 5m support radius -- a fixed, purely geometric
        # reference circle, not a claim about who is inside it.
        radius_px = int(round(SUPPORT_RADIUS_PX * scale_at(carrier["x_pitch"], carrier["y_pitch"])))
        cv2.circle(pitch_dark, carrier_px, radius_px, SUPPORT_RADIUS_COLOR, 1, cv2.LINE_AA)

    if presser_px is not None and carrier_px is not None:
        # (2) nearest-presser connector + (3) closing-direction cue --
        # the EXACT opponent `nearest_opponent_track_id` names, real
        # relative_closing_cm_s sign, no exaggerated length.
        cv2.line(pitch_dark, presser_px, carrier_px, closing_color or NEUTRAL_CLOSING_COLOR, 2, cv2.LINE_AA)
    if presser_px is not None:
        cv2.circle(pitch_dark, presser_px, 10, NEAREST_PRESSER_RING_COLOR, 2, cv2.LINE_AA)
        if presser.get("vx_cm_s") is not None and presser.get("vy_cm_s") is not None:
            ex = int(presser_px[0] + presser["vx_cm_s"] * 0.03)
            ey = int(presser_px[1] + presser["vy_cm_s"] * 0.03)
            cv2.arrowedLine(pitch_dark, presser_px, (ex, ey), closing_color or NEUTRAL_CLOSING_COLOR, 2, cv2.LINE_AA, tipLength=0.35)

    # (7) pressing shape -- connect the REAL confirmed participants
    # (features.participant_ids) to each other, only when >=2 exist.
    if participant_ids and len(participant_ids) >= 2:
        pts = [project(p["x_pitch"], p["y_pitch"]) for p in players_this_frame if p["track_id"] in participant_ids]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                cv2.line(pitch_dark, pts[i], pts[j], PARTICIPANT_LINK_COLOR, 1, cv2.LINE_AA)

    for p in players_this_frame:
        px, py = project(p["x_pitch"], p["y_pitch"])
        s = scale_at(p["x_pitch"], p["y_pitch"])
        color = TEAM_COLOR.get(p["display_team_id"], (150, 150, 150))
        cv2.circle(pitch_dark, (px, py), int(round(8 * s)), color, -1, cv2.LINE_AA)
        # (5) nearby pressing support -- a stronger outline for REAL
        # confirmed participants only (never inferred).
        is_participant = bool(participant_ids and p["track_id"] in participant_ids)
        ring_r, ring_t = (int(round(11 * s)), 2) if is_participant else (int(round(8 * s)), 2)
        ring_color = PARTICIPANT_RING_COLOR if is_participant else (10, 10, 10)
        cv2.circle(pitch_dark, (px, py), ring_r, ring_color, ring_t, cv2.LINE_AA)
    if ball_xy is not None:
        bx, by = project(*ball_xy)
        br = int(round(6 * scale_at(*ball_xy)))
        cv2.circle(pitch_dark, (bx, by), br, BALL_COLOR, -1, cv2.LINE_AA)
        cv2.circle(pitch_dark, (bx, by), br, (15, 15, 15), 1, cv2.LINE_AA)

    if carrier_px is not None:
        # (1) ball carrier highlight -- same identity/position as the
        # video panel's ring (both come from the same `_find_carrier` call).
        carrier_r = int(round(13 * scale_at(carrier["x_pitch"], carrier["y_pitch"])))
        cv2.circle(pitch_dark, carrier_px, carrier_r, CARRIER_RING_COLOR, 2, cv2.LINE_AA)

    out = np.full((h, w, 3), BG, dtype=np.uint8)
    resized = cv2.resize(pitch_dark, (w, h - 34))
    out[34:h, 0:w] = resized
    cv2.rectangle(out, (12, 8), (15, 24), ACCENT_CYAN, -1)
    # 2026-09-08 -- renamed from "PRESSING RADAR": the panel shows the
    # continuous SPATIAL pressure field (always on whenever computable),
    # not the PRESS EVENT state -- the two are genuinely different
    # questions (see PRESSURE_STATE_ALPHA_SCALE's own comment above).
    # The event badge (NO PRESS/PRESS FORMING/ACTIVE PRESS/UNCERTAIN)
    # stays a separate, unchanged element on the video panel.
    cv2.putText(out, "PRESSURE / SPATIAL INFLUENCE RADAR", (22, 22), FONT, 0.40, TEXT_WHITE, 2, cv2.LINE_AA)

    if carrier_px is not None and balance_text:
        # (6) local numerical balance -- tiny concise label, exact same
        # support_minus_opponents field as Graph 4, anchored near the
        # carrier (offset in the resized/padded radar frame).
        lx = int(carrier_px[0] * (w / pitch_dark.shape[1])) + 14
        ly = int(carrier_px[1] * ((h - 34) / pitch_dark.shape[0])) + 34 - 14
        (tw_, th_), _ = cv2.getTextSize(balance_text, FONT, 0.34, 1)
        cv2.rectangle(out, (lx - 3, ly - th_ - 3), (lx + tw_ + 3, ly + 3), (20, 14, 10), -1)
        cv2.putText(out, balance_text, (lx, ly), FONT, 0.34, TEXT_WHITE, 1, cv2.LINE_AA)

    if debug and carrier_px is not None and carrier_field_pressure is not None:
        # (P4) RESEARCH/DEBUG MODE ONLY -- off by default, not part of
        # the clean final video. Deliberately NOT merged with the
        # existing Press Intensity metric -- shown side by side for comparison.
        text = f"Pressure at carrier: {carrier_field_pressure:.2f}"
        cv2.putText(out, text, (14, h - 12), FONT, 0.36, (0, 220, 255), 1, cv2.LINE_AA)
    return out


def _lookup_by_track(players_this_frame, track_id):
    """Exact identity lookup by track_id. Returns None if the id is
    None or not visible this frame -- an honest gap, never a fallback
    guess."""
    if track_id is None:
        return None
    return next((p for p in players_this_frame if p["track_id"] == track_id), None)


def _project_pitch_points(transformer, pts_cm):
    """Pitch (cm) -> broadcast image (px) projection via the SAME
    inverse-homography technique already used by
    `render_pressing_dashboard._draw_pitch_circle_on_image` (not
    reinvented). Returns None if the transformer is missing/singular --
    callers must skip drawing rather than fabricate a broadcast overlay."""
    if transformer is None:
        return None
    try:
        inv_m = np.linalg.inv(transformer.m)
    except np.linalg.LinAlgError:
        return None
    pts = np.array(pts_cm, dtype=np.float32).reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(pts, inv_m).reshape(-1, 2)
    return img_pts


def _dashed_line_simple(img, p0, p1, color, thickness=2, dash_len=12, gap_len=9):
    p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float)
    total = float(np.linalg.norm(p1 - p0))
    if total < 1e-6:
        return
    direction = (p1 - p0) / total
    d = 0.0
    while d < total:
        a = tuple(np.round(p0 + direction * d).astype(int))
        b = tuple(np.round(p0 + direction * min(d + dash_len, total)).astype(int))
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
        d += dash_len + gap_len


def _clipped_line(img, p0, p1, color, thickness, dashed=False):
    """cv2.clipLine restricts a projected pitch-space segment to the
    VISIBLE FRAME bounds (never draws outside the visible field)."""
    h, w = img.shape[:2]
    ok, c0, c1 = cv2.clipLine((0, 0, w, h), p0, p1)
    if not ok:
        return
    if dashed:
        _dashed_line_simple(img, c0, c1, color, thickness)
    else:
        cv2.line(img, c0, c1, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 2026-09-10 (revision) -- COACH-MODE BOTTOM ANALYTICS (Graph 1:
# Pressing Team Timeline). SUPERSEDES the 2026-09-09 per-frame
# `dominant_state`/`dominant_team` sampling approach (audit finding:
# `dominant()` requires the pair's SINGLE highest-rank state to
# literally equal PRESS_FORMING/ACTIVE_PRESS at that EXACT sampled
# frame, which only overlaps a small fraction of a real episode's own
# [onset_time, end_time] window -- producing thin per-frame spikes
# rather than the real detected episode's actual duration). The
# correct source, per this round's own audit, is `pressing_v4.
# build_pressing_v4`'s own `episodes` list (`onset_time`/`end_time`,
# already real, already detected, UNCHANGED FSM/threshold logic) --
# used DIRECTLY here, not re-derived from per-frame state samples.
#
#   +1 = Team A Pressing  <- t falls within a REAL Team A episode's own [onset_time, end_time]
#   -1 = Team B Pressing  <- t falls within a REAL Team B episode's own [onset_time, end_time]
#    0 = No Clear Press   <- t falls in NEITHER team's episode window, OR
#         (genuinely important) falls within BOTH simultaneously -- a
#         real overlap between a Team A and a Team B episode is treated
#         as CONTESTED/unresolved and mapped to neutral, never guessed
#         toward either side.
#
# Each episode contributes ONLY its own real [onset_time, end_time]
# span -- separate episodes are NEVER stitched together, gaps between
# episodes are NEVER filled, and no episode is ever extended past its
# own real `end_time` (the FSM's own `last_observed_frame`).
#
# AUDIT FINDING (disclosed, not hidden): on this real 120s clip, EVERY
# one of the 19 real episodes has a genuinely short duration -- from 0ms
# (a single-frame onset+immediate termination) up to a maximum of 200ms
# (6 frames, episode `T0-957`). Switching from per-frame sampling to
# real episode intervals is still the semantically CORRECT fix (it now
# honors the FSM's own detected boundaries exactly, rather than an
# approximation reconstructed from `dominant()`'s frame-level picks),
# but it does NOT make this clip's timeline look dramatically less
# sparse -- the underlying episodes really are this short. Manufacturing
# wider blocks would misrepresent the real data and is explicitly
# forbidden by this round's own instructions ("do not extend episodes
# beyond their real detected boundaries").
PRESSING_STATE_LEVELS = ("Team A Pressing", "No Clear Press", "Team B Pressing")


def episode_intervals_by_team(episodes):
    """Split `pressing_v4.build_pressing_v4`'s real `episodes` list into
    two lists of `(onset_time, end_time)` tuples, one per team -- the
    exact, unmodified boundaries the FSM detected. Never merged,
    stitched, or extended."""
    intervals_a = [(e["onset_time"], e["end_time"]) for e in episodes if e["team"] == 0]
    intervals_b = [(e["onset_time"], e["end_time"]) for e in episodes if e["team"] == 1]
    return intervals_a, intervals_b


def pressing_signal_at_time(t, intervals_a, intervals_b, eps=1e-9):
    """+1/-1/0 per the module comment above. `t` within BOTH a Team A
    and a Team B interval simultaneously (a real, genuine overlap) maps
    to 0 (neutral) -- checked explicitly, never resolved by picking
    whichever comes first."""
    in_a = any(onset - eps <= t <= end + eps for onset, end in intervals_a)
    in_b = any(onset - eps <= t <= end + eps for onset, end in intervals_b)
    if in_a and in_b:
        return 0
    if in_a:
        return 1
    if in_b:
        return -1
    return 0


# ---------------------------------------------------------------------------
# 2026-09-09 -- COACH-MODE BOTTOM ANALYTICS (Graph 3: Press Outcome
# Summary). The heuristic outcome-attribution pipeline
# (`pressing_features.attribute_outcomes_all_episodes`) can currently
# only ever produce one of exactly these 5 raw labels (verified against
# its own source: `BALL_REGAIN`/`BACKWARD_ACTION`/`LATERAL_ACTION`/
# `FORWARD_PROGRESSION`/`UNCERTAIN`), mapped via the SAME
# `HEURISTIC_TO_TAXONOMY` table the research phase already defined
# (`pressing_features.py`, `PRESS_LABELING_PROTOCOL.md` section 5) into
# exactly `{BALL_REGAIN, FORCED_BACKWARD, FORCED_LATERAL, PRESS_ESCAPED,
# UNCERTAIN}` -- i.e. the heuristic pipeline, as it exists today,
# CANNOT produce `FORCED_LONG_CLEARANCE`/`REDUCED_FORWARD_PROGRESSION`/
# `FORCED_LOW_VALUE_PASS`/`DANGEROUS_FORWARD_PROGRESSION` at all (those
# require a human judgment call per the labeling protocol) -- so the
# "silently merging other categories" concern does not arise for
# heuristic-sourced bars specifically; `UNCERTAIN` is the only category
# excluded from the 4 bars below, shown instead in a footer count.
OUTCOME_CATEGORIES = ("Regain", "Forced Backward", "Forced Lateral", "Press Escaped")
OUTCOME_TAXONOMY_TO_CATEGORY = {
    "BALL_REGAIN": "Regain",
    "FORCED_BACKWARD": "Forced Backward",
    "FORCED_LATERAL": "Forced Lateral",
    "PRESS_ESCAPED": "Press Escaped",
}


def outcome_known_time(episode_end_time, outcome_confirmation_time, taxonomy_label, evidence_window_sec=3.0):
    """The real-time moment this episode's outcome becomes knowable --
    used to build a CAUSAL cumulative count (never counts an outcome
    before its own deciding evidence has actually occurred). A matched,
    classified outcome (any of the 4 real categories) is known at its
    real `outcome_confirmation_time` (the matched pass/turnover's own
    `end_frame`/fps -- when the deciding action actually finished). An
    `UNCERTAIN` episode (no evidence matched within the window) is only
    CERTAIN to be `UNCERTAIN` once the full evidence-matching window
    (`episode_end_time + evidence_window_sec`, the SAME 3s window
    `attribute_outcomes_all_episodes` itself searches) has elapsed with
    nothing found -- using the episode's own end time instead would
    claim the outcome was known before it actually could be."""
    if taxonomy_label in OUTCOME_TAXONOMY_TO_CATEGORY and outcome_confirmation_time is not None:
        return outcome_confirmation_time
    return episode_end_time + evidence_window_sec


def resolve_outcome_label(heuristic_taxonomy_label, human_outcome=None, final_adjudicated_outcome=None):
    """Evidence priority, exactly as specified: (1) `final_adjudicated_outcome`,
    (2) `human_outcome`, (3) the heuristic pipeline's own taxonomy-mapped
    label. Returns `(taxonomy_label, tier)` -- `tier` lets the caller
    track whether any DISPLAYED outcome actually came from the
    unvalidated heuristic tier (vs. real human/adjudicated evidence),
    so the "heuristic outcomes -- unvalidated" note is only ever shown
    when it's actually true of what's on screen."""
    if final_adjudicated_outcome is not None:
        return final_adjudicated_outcome, "final_adjudicated"
    if human_outcome is not None:
        return human_outcome, "human"
    return heuristic_taxonomy_label, "heuristic"


def build_outcome_events(episodes_with_outcome, label_overrides=None):
    """`episodes_with_outcome`: `attribute_outcomes_all_episodes`'s own
    output, unchanged. `label_overrides`: optional `{episode_id: {
    "human_outcome": taxonomy_label_or_None, "final_adjudicated_outcome":
    taxonomy_label_or_None}}`, sourced from the real
    `pressing_episode_features.parquet` export when present (both
    columns are `None` for every episode on today's real dataset -- 0
    human labels exist -- so `resolve_outcome_label` currently always
    falls through to the heuristic tier for real data; this is the
    correct, forward-looking wiring for once labels DO exist, not a
    behavior change today).

    Two mapping steps, kept explicit and separate (never collapsed into
    one lookup table): (1) the pipeline's raw label -> the 9-label
    PRESS_LABELING_PROTOCOL.md taxonomy, via `pressing_features.
    HEURISTIC_TO_TAXONOMY` (unchanged, imported) -- used only as the
    heuristic-tier fallback; (2) the RESOLVED taxonomy label -> one of
    this dashboard's 4 DISPLAY categories, via
    `OUTCOME_TAXONOMY_TO_CATEGORY` (this module's own, for the bar
    chart only) -- `None` for `UNCERTAIN` or any taxonomy label the
    heuristic pipeline cannot currently produce anyway
    (`FORCED_LONG_CLEARANCE`/`REDUCED_FORWARD_PROGRESSION`/
    `FORCED_LOW_VALUE_PASS`/`DANGEROUS_FORWARD_PROGRESSION`), which the
    excluded-count footer covers instead of a 5th+ bar category.
    Returns `(events, any_heuristic_used)` -- `events` is a time-sorted
    list of `(known_time, team, category_or_None)`."""
    label_overrides = label_overrides or {}
    events = []
    any_heuristic_used = False
    for e in episodes_with_outcome:
        heuristic_taxonomy = HEURISTIC_TO_TAXONOMY.get(e["heuristic_outcome"], "UNCERTAIN")
        override = label_overrides.get(e["episode_id"], {})
        taxonomy_label, tier = resolve_outcome_label(heuristic_taxonomy, override.get("human_outcome"), override.get("final_adjudicated_outcome"))
        if tier == "heuristic":
            any_heuristic_used = True
        category = OUTCOME_TAXONOMY_TO_CATEGORY.get(taxonomy_label)  # None if excluded (UNCERTAIN or unmapped)
        known_time = outcome_known_time(e["end_time"], e.get("outcome_confirmation_time"), taxonomy_label)
        events.append((known_time, e["team"], category))
    events.sort(key=lambda x: x[0])
    return events, any_heuristic_used


def cumulative_outcome_counts(outcome_events, cur_time):
    """Counts per team, restricted to `known_time <= cur_time` (causal,
    never future-leaking) -- returns `(counts_team_a, counts_team_b,
    excluded_count)`, where `counts_team_{a,b}` are `{category: int}`
    dicts covering exactly `OUTCOME_CATEGORIES` (0 where no real
    episode has resolved to that category yet) and `excluded_count` is
    the number of excluded (`UNCERTAIN`/unmapped) episodes already
    known by `cur_time`."""
    counts_a = {c: 0 for c in OUTCOME_CATEGORIES}
    counts_b = {c: 0 for c in OUTCOME_CATEGORIES}
    excluded = 0
    for known_time, team, category in outcome_events:
        if known_time > cur_time + 1e-9:
            continue
        if category is None:
            excluded += 1
            continue
        (counts_a if team == 0 else counts_b)[category] += 1
    return counts_a, counts_b, excluded


# ---------------------------------------------------------------------------
# 2026-09-10 -- COACH-MODE Graph 2 OPTIONAL causal display smoothing
# (`--coach-intensity-view ema`, default remains `raw` -- see this
# round's own final report for the preview-QA comparison that decided
# the default). `causal_ema` is presentation-only: it consumes
# `full_i0`/`full_i1` (UNCHANGED, still passed to nothing else) and
# returns a SEPARATE series; the raw series is never mutated or
# replaced anywhere else in this module.
COACH_EMA_ALPHA = 0.2  # 2026-09-10: both feasibility audits recommend alpha=0.2 for the
# opp_ballw-based signals now driving Graphs 1/2 (previously 0.3, tuned for the
# earlier raw_score-based Graph 2 this round replaces) -- a standard, disclosed
# default, not tuned to this clip's own outcome.

# 2026-09-09 -- COACH-MODE Graph 1/2 rolling-window tick spacing, one
# entry per `--coach-graph-window-seconds` choice. 24s uses 4s (not 5s):
# 24 is evenly divisible by 4 (exactly 6 major ticks, landing on the
# same "4-6 ticks" convention used elsewhere in this project, e.g.
# `nice_bar_axis`) while 24/5=4.8 leaves an uneven, visually-wider gap
# between the last tick and the NOW cursor at the right edge -- chosen
# after generating and comparing both directly (see this round's own
# audit report). Never a dense 1s spacing.
COACH_GRAPH_WINDOW_TICK_SPACING = {60: 10, 30: 5, 24: 4}


def causal_ema(full_hist, alpha=COACH_EMA_ALPHA, max_gap_sec=0.5):
    """EMA_t = alpha*x_t + (1-alpha)*EMA_(t-1) -- STRICTLY CAUSAL (each
    output only ever depends on samples at or before its own timestamp,
    never a future one). The running average RESETS (starts fresh at
    the next real sample, exactly like a raw sample would) after any
    gap exceeding `max_gap_sec` -- a genuinely missing stretch is never
    bridged by decaying a stale average across it. A real, valid input
    of `0.0` is a normal EMA input like any other value (never treated
    as missing); a `None` input is an honest gap and stays `None` in
    the output too -- the zero/missing distinction is preserved
    end-to-end, not just in the raw series."""
    out = []
    ema, last_t = None, None
    for t, v in full_hist:
        gap = last_t is not None and (t - last_t) > max_gap_sec
        if gap:
            ema = None  # a real gap -- never carry a stale average across it
        if v is None:
            out.append((t, None))
        else:
            ema = float(v) if ema is None else alpha * float(v) + (1 - alpha) * ema
            out.append((t, ema))
        last_t = t
    return out


def _select_pressing_value(snap_f, key_path):
    """Honest single-line merge: exactly one team (the one NOT in
    possession) carries a real value for a presser-relative feature on
    any given frame -- the possessing team's own branch returns None by
    construction (see pressing_v4.frame_features)."""
    f0 = snap_f["pressing"][0]["features"].get(key_path)
    f1 = snap_f["pressing"][1]["features"].get(key_path)
    return f0 if f0 is not None else f1


def _full_series(snapshot, fn):
    return [(s["time_sec"], fn(s)) for s in snapshot]


def _signed_y_range(values, pad_frac=0.1, floor_zero=False):
    """Robust 1st-99th-percentile range. `floor_zero=True` for
    naturally non-negative quantities (distance, count); False (no
    floor) for genuinely SIGNED quantities (closing speed, support
    balance) -- `live_graphs.compute_stable_y_range`'s unconditional
    `max(0.0, ...)` floor would silently clip negative values off axis.
    Kept local rather than editing that shared function (still used by
    the untouched V4 correctness-preview scripts)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return (0.0, 1.0) if floor_zero else (-1.0, 1.0)
    lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * pad_frac
    lo_out = lo - pad
    if floor_zero:
        lo_out = max(0.0, lo_out)
    return (lo_out, hi + pad)


def _to_px(t, v, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph):
    x = x0 + int((t - t_lo) / max(1e-6, (t_hi - t_lo)) * pw)
    y = y0 + ph - int((v - y_lo) / max(1e-6, (y_hi - y_lo)) * ph)
    return x, y


def _chrome(w, h, title, subtitle, y_label, y_range, tick_fmt, t_lo, t_hi, tick_spacing, accent=ACCENT_CYAN):
    """Beginner-readable FULL chrome, shared by all graph variants:
    large title + interpretation subtitle, explicit Y-axis label (with
    units, human wording -- never an internal field name), explicit
    'Match Time (seconds)' X-axis label, readable absolute-second tick
    labels (never relative-window numbers), and a zero reference line
    on any axis whose range actually spans zero."""
    img = panel_frame(w, h)
    cv2.rectangle(img, (12, 8), (15, 24), accent, -1)
    cv2.putText(img, title, (22, 22), FONT, 0.50, TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (14, 40), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

    pad_l, pad_r, pad_t, pad_b = 54, 16, 60, 48
    x0, y0 = pad_l, pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    y_lo, y_hi = y_range
    fmt = tick_fmt

    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), BORDER_SOFT, 1)
    for frac in (0.25, 0.5, 0.75):
        gy = y0 + int(ph * (1 - frac))
        cv2.line(img, (x0, gy), (x0 + pw, gy), GRID, 1, cv2.LINE_AA)
        cv2.putText(img, fmt.format(y_lo + frac * (y_hi - y_lo)), (4, gy + 4), FONT, 0.30, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, fmt.format(y_hi), (4, y0 + 8), FONT, 0.30, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, fmt.format(y_lo), (4, y0 + ph + 4), FONT, 0.30, TEXT_DIM, 1, cv2.LINE_AA)

    if y_lo < 0 < y_hi:
        zy = y0 + int(ph * (1 - (0 - y_lo) / (y_hi - y_lo)))
        x = x0
        while x < x0 + pw:
            cv2.line(img, (x, zy), (min(x + 9, x0 + pw), zy), TEXT_WHITE, 2, cv2.LINE_AA)
            x += 15
        cv2.putText(img, "0", (4, zy + 4), FONT, 0.32, TEXT_WHITE, 1, cv2.LINE_AA)

    t = math.ceil(t_lo / tick_spacing) * tick_spacing
    while t <= t_hi + 1e-6:
        x = x0 + int((t - t_lo) / max(1e-6, (t_hi - t_lo)) * pw)
        cv2.line(img, (x, y0), (x, y0 + ph), GRID, 1, cv2.LINE_AA)
        label = f"{int(round(t))}s"
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.30, 1)
        cv2.putText(img, label, (max(0, x - tw // 2), y0 + ph + 17), FONT, 0.30, TEXT_DIM, 1, cv2.LINE_AA)
        t += tick_spacing

    cv2.putText(img, y_label, (pad_l, pad_t - 6), FONT, 0.33, TEXT_DIM, 1, cv2.LINE_AA)
    (xw, _), _ = cv2.getTextSize("Match Time (seconds)", FONT, 0.32, 1)
    cv2.putText(img, "Match Time (seconds)", (x0 + pw - xw, h - 8), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    return img, x0, y0, pw, ph, y_lo, y_hi


def _draw_legend(img, x0, pw, label_a, label_b, color_a, color_b):
    lx = x0 + pw - 6
    for lbl, color in ((label_b, color_b), (label_a, color_a)):
        tw = cv2.getTextSize(lbl, FONT, 0.33, 1)[0][0]
        lx -= tw
        cv2.putText(img, lbl, (lx, 20), FONT, 0.33, TEXT_WHITE, 1, cv2.LINE_AA)
        lx -= 16
        cv2.rectangle(img, (lx, 12), (lx + 10, 20), color, -1)
        lx -= 10


def _plot_line(img, hist, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color, max_gap_sec):
    history = [(t, v) for t, v in hist if t_lo - 1e-9 <= t <= cur_time + 1e-9]
    segment, last_t = [], None
    for t, v in history:
        gap = (last_t is not None) and (t - last_t > max_gap_sec)
        if v is None or gap:
            if len(segment) >= 2:
                cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, color, 3, cv2.LINE_AA)
            elif len(segment) == 1:
                cv2.circle(img, segment[0], 3, color, -1, cv2.LINE_AA)
            segment = []
        if v is not None:
            segment.append(_to_px(t, np.clip(v, y_lo, y_hi), t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph))
        last_t = t
    if len(segment) >= 2:
        cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, color, 3, cv2.LINE_AA)
    elif len(segment) == 1:
        cv2.circle(img, segment[0], 3, color, -1, cv2.LINE_AA)


def _plot_step(img, hist, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color, max_gap_sec):
    history = [(t, v) for t, v in hist if t_lo - 1e-9 <= t <= cur_time + 1e-9]
    prev_pt, last_t = None, None
    for t, v in history:
        gap = (last_t is not None) and (t - last_t > max_gap_sec)
        if v is None or gap:
            prev_pt = None
        else:
            cx, cy = _to_px(t, np.clip(v, y_lo, y_hi), t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
            if prev_pt is not None:
                px, py = prev_pt
                cv2.line(img, (px, py), (cx, py), color, 3, cv2.LINE_AA)
                cv2.line(img, (cx, py), (cx, cy), color, 3, cv2.LINE_AA)
            else:
                cv2.circle(img, (cx, cy), 3, color, -1, cv2.LINE_AA)
            prev_pt = (cx, cy)
        last_t = t


def draw_graph(w, h, full_hist, cur_time, t_lo, t_hi, tick_spacing, title, subtitle, y_label, y_range, tick_fmt,
               color, accent=ACCENT_CYAN, max_gap_sec=0.5):
    img, x0, y0, pw, ph, y_lo, y_hi = _chrome(w, h, title, subtitle, y_label, y_range, tick_fmt, t_lo, t_hi, tick_spacing, accent)
    _plot_line(img, full_hist, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color, max_gap_sec)
    cx, _ = _to_px(cur_time, y_lo, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
    cv2.line(img, (cx, y0), (cx, y0 + ph), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (cx, y0), (cx, y0 + ph), TEXT_WHITE, 2, cv2.LINE_AA)
    return img


def draw_dual_graph(w, h, full_a, full_b, cur_time, t_lo, t_hi, tick_spacing, title, subtitle, y_label, y_range,
                     tick_fmt, label_a, label_b, color_a, color_b, accent=ACCENT_CYAN, max_gap_sec=0.5):
    img, x0, y0, pw, ph, y_lo, y_hi = _chrome(w, h, title, subtitle, y_label, y_range, tick_fmt, t_lo, t_hi, tick_spacing, accent)
    _plot_line(img, full_a, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color_a, max_gap_sec)
    _plot_line(img, full_b, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color_b, max_gap_sec)
    cx, _ = _to_px(cur_time, y_lo, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
    cv2.line(img, (cx, y0), (cx, y0 + ph), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (cx, y0), (cx, y0 + ph), TEXT_WHITE, 2, cv2.LINE_AA)
    _draw_legend(img, x0, pw, label_a, label_b, color_a, color_b)
    return img


# ---------------------------------------------------------------------------
# 2026-09-10 -- COACH-MODE Graph 2 ONLY. AUDIT FINDING: `_plot_line`
# (above, UNCHANGED, still used by `draw_dual_graph` for research mode)
# already draws a real `raw_score == 0.0` sample correctly -- verified
# directly (the pickled snapshot's zero/None counts match a fresh
# `build_pressing_v4` run exactly, and a real zero-containing window
# rendered through the UNMODIFIED `_plot_line` does paint pixels all
# the way down to the y=0 gridline). There is NO data-suppression bug.
# The real problem is purely visual: a flat run of zeros is a bare 3px
# line sitting exactly on the panel's own bottom border/gridline row,
# easy to mis-read as "nothing plotted here" rather than "a real
# measured zero". `_plot_line_with_markers` adds a small filled circle
# at EVERY real sample (valid zeros included) -- purely a rendering
# change, computed from the exact same gap-aware, causal filtering
# `_plot_line` already uses (never re-derived differently) -- so a
# zero-run reads as a visible row of dots instead of a hairline. Used
# ONLY by `draw_dual_graph_coach` below; `_plot_line`/`draw_dual_graph`
# (research mode) are completely untouched.
def _plot_line_with_markers(img, hist, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color, max_gap_sec, marker_radius=3):
    history = [(t, v) for t, v in hist if t_lo - 1e-9 <= t <= cur_time + 1e-9]
    segment, last_t = [], None
    for t, v in history:
        gap = (last_t is not None) and (t - last_t > max_gap_sec)
        if v is None or gap:
            if len(segment) >= 2:
                cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, color, 3, cv2.LINE_AA)
            for pt in segment:
                cv2.circle(img, pt, marker_radius, color, -1, cv2.LINE_AA)
            segment = []
        if v is not None:
            segment.append(_to_px(t, np.clip(v, y_lo, y_hi), t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph))
        last_t = t
    if len(segment) >= 2:
        cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, color, 3, cv2.LINE_AA)
    for pt in segment:
        cv2.circle(img, pt, marker_radius, color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 2026-09-10 (presentation polish, post-approval) -- EMA is now the coach-
# mode DEFAULT display for Graphs 1/2 (raw stays available via CLI). The
# original `_plot_line_with_markers` (dense per-sample dots, kept
# UNCHANGED above, still used for `--coach-intensity-view raw`) was
# designed to make a flat run of RAW valid-zeros legible against the
# panel's own bottom border. Under EMA that dense-dot rendering instead
# reads as a "cloud" -- `_plot_line_sparse` keeps the SAME real series
# (no measurement is altered, removed, or interpolated) but (a) draws
# each contiguous run as ONE smooth `cv2.polylines` call over a
# DECIMATED vertex subset (keeps every `decimate_n`-th real sample, ALSO
# always the run's first/last point so a genuine gap boundary is never
# shifted or hidden) instead of many short independent segments, and (b)
# places small markers only every `marker_every_sec` seconds (plus at
# the start of every run, so resumption-after-a-gap is still visibly
# marked) instead of on every single sample. `decimate_n=1` reproduces
# the full, undecimated series (used nowhere by default, kept for tests/
# comparison).
def _plot_line_sparse(img, hist, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color, max_gap_sec,
                       decimate_n=3, marker_every_sec=1.0, marker_radius=2, line_thickness=2):
    history = [(t, v) for t, v in hist if t_lo - 1e-9 <= t <= cur_time + 1e-9]

    def flush(run):
        if not run:
            return
        if len(run) == 1:
            cv2.circle(img, run[0][1], marker_radius, color, -1, cv2.LINE_AA)
            return
        kept_idx = sorted(set(list(range(0, len(run), max(1, decimate_n))) + [0, len(run) - 1]))
        pts = [run[i][1] for i in kept_idx]
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, line_thickness, cv2.LINE_AA)
        last_marker_t = None
        for i in kept_idx:
            t_i, px_i = run[i]
            if last_marker_t is None or (t_i - last_marker_t) >= marker_every_sec:
                cv2.circle(img, px_i, marker_radius, color, -1, cv2.LINE_AA)
                last_marker_t = t_i

    run, last_t = [], None
    for t, v in history:
        gap = (last_t is not None) and (t - last_t > max_gap_sec)
        if v is None or gap:
            flush(run)
            run = []
        if v is not None:
            px = _to_px(t, np.clip(v, y_lo, y_hi), t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
            run.append((t, px))
        last_t = t
    flush(run)


def draw_dual_graph_coach(w, h, full_a, full_b, cur_time, t_lo, t_hi, tick_spacing, title, subtitle, y_label, y_range,
                           tick_fmt, label_a, label_b, color_a, color_b, accent=ACCENT_CYAN, max_gap_sec=0.5,
                           marker_style="dense", decimate_n=3):
    """Coach-mode Graph 2 -- identical chrome/causality/gap rules to
    `draw_dual_graph`, only the per-sample marker rendering differs.
    `marker_style="dense"` (default, unchanged from before) uses
    `_plot_line_with_markers` (a marker at every real sample -- used for
    `--coach-intensity-view raw`); `"sparse"` uses `_plot_line_sparse`
    (decimated smooth curve + sparse markers -- used for the new EMA
    default, see its own docstring). `decimate_n` (sparse mode only):
    2026-09-09 -- since `--coach-graph-display-hz` now pre-decimates the
    series BEFORE it ever reaches this function (see main()'s own
    comment), the caller passes `decimate_n=1` in that case to avoid a
    second, redundant thinning pass; `decimate_n=3` (this function's own
    old default) remains available for `--coach-graph-display-hz 30`
    (full-density/debug mode) so that mode's own line still reads as a
    curve rather than a dense dot-cloud."""
    img, x0, y0, pw, ph, y_lo, y_hi = _chrome(w, h, title, subtitle, y_label, y_range, tick_fmt, t_lo, t_hi, tick_spacing, accent)
    if marker_style == "sparse":
        _plot_line_sparse(img, full_a, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color_a, max_gap_sec, decimate_n=decimate_n)
        _plot_line_sparse(img, full_b, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color_b, max_gap_sec, decimate_n=decimate_n)
    else:
        _plot_line_with_markers(img, full_a, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color_a, max_gap_sec)
        _plot_line_with_markers(img, full_b, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color_b, max_gap_sec)
    cx, _ = _to_px(cur_time, y_lo, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
    cv2.line(img, (cx, y0), (cx, y0 + ph), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (cx, y0), (cx, y0 + ph), TEXT_WHITE, 2, cv2.LINE_AA)
    _draw_legend(img, x0, pw, label_a, label_b, color_a, color_b)
    return img


# ---------------------------------------------------------------------------
# 2026-09-10 -- COACH-MODE Graph 1 (Pressure Dominance Over Time),
# REPLACING the prior round's categorical episode timeline per the two
# completed feasibility audits. `full_d`: `[(t, v), ...]` where `v` is a
# real float in [-1,1], the sentinel string `coach_pressure_signal.
# LOW_PRESSURE` (an EXPERIMENTAL, disclosed low-pressure gate result --
# see `gate_dominance`), or `None` (a genuine gap -- never bridged,
# never filled, never treated as 0). Color convention: this dashboard's
# OWN established Team A=green / Team B=gold mapping is used for the
# fill/line (positive=Team-A-stronger=green, negative=Team-B-stronger=
# gold) rather than the literal "positive=gold" wording of the
# reference mockup -- kept consistent with the Team A/B legend colors
# already used by Graph 2 and Graph 3, so the same team reads as the
# same color across all three panels (see this round's own report for
# the explicit note on this deviation).
def _plot_dominance(img, hist, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi,
                     color_pos, color_neg, color_gated, max_gap_sec, fill_alpha=0.25, decimate_n=1):
    """`decimate_n=1` (default) draws every real consecutive-frame pair
    as its own short segment -- the exact original, undecimated
    rendering (used for `--coach-dominance-view raw`). `decimate_n>1`
    (used for the new EMA default -- see this round's presentation-
    polish notes) keeps every Nth real sample within each contiguous
    run as a polyline VERTEX (always including that run's first/last
    point, so a genuine gap boundary is never shifted) -- this only
    reduces how many drawn vertices connect the SAME real measurements;
    no value is altered, removed, or interpolated, and gap detection
    below always uses the full, undecimated per-frame timestamps."""
    history = [(t, v) for t, v in hist if t_lo - 1e-9 <= t <= cur_time + 1e-9]
    zero_y = y0 + int(ph * (1 - (0.0 - y_lo) / (y_hi - y_lo)))

    def numeric(v):
        return 0.0 if v == LOW_PRESSURE else v

    def color_for_pair(v0, v1):
        if v0 == LOW_PRESSURE or v1 == LOW_PRESSURE:
            return color_gated
        return color_pos if (v0 + v1) >= 0 else color_neg

    runs = []
    run, last_t = [], None
    for t, v in history:
        gap = last_t is not None and (t - last_t > max_gap_sec)
        if v is None or gap:
            if run:
                runs.append(run)
            run = []
        if v is not None:
            run.append((t, v))
        last_t = t
    if run:
        runs.append(run)

    marker_radius = 2
    line_thickness = 2 if decimate_n > 1 else 3
    all_segments = []
    for run in runs:
        kept_idx = sorted(set(list(range(0, len(run), max(1, decimate_n))) + [0, len(run) - 1]))
        kept = [run[i] for i in kept_idx]
        if len(kept) == 1:
            t0v, v0v = kept[0]
            px0 = _to_px(t0v, np.clip(numeric(v0v), y_lo, y_hi), t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
            cv2.circle(img, px0, marker_radius, color_for_pair(v0v, v0v), -1, cv2.LINE_AA)
            continue
        for i in range(len(kept) - 1):
            t0v, v0v = kept[i]
            t1v, v1v = kept[i + 1]
            px0 = _to_px(t0v, np.clip(numeric(v0v), y_lo, y_hi), t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
            px1 = _to_px(t1v, np.clip(numeric(v1v), y_lo, y_hi), t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
            all_segments.append((px0, px1, color_for_pair(v0v, v1v)))

    if all_segments:
        overlay = img.copy()
        for px0, px1, col in all_segments:
            quad = np.array([[px0[0], zero_y], [px0[0], px0[1]], [px1[0], px1[1]], [px1[0], zero_y]], dtype=np.int32)
            cv2.fillPoly(overlay, [quad], col)
        cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)
        for px0, px1, col in all_segments:
            cv2.line(img, px0, px1, col, line_thickness, cv2.LINE_AA)


def draw_dominance_graph(w, h, full_d, cur_time, t_lo, t_hi, tick_spacing, gate_threshold=None,
                          accent=ACCENT_CYAN, max_gap_sec=0.5, decimate_n=1):
    """Coach-mode Graph 1. Fixed y_range=(-1,1) (D_raw = I_A - I_B, both
    I_A/I_B in [0,1] -> D naturally in [-1,1]). `gate_threshold=None`
    (default) renders the ungated real dominance line; a float renders
    whatever gating `gate_dominance` already applied to `full_d` before
    calling this function (this function only draws -- it never computes
    the gate itself), plus a clearly-labeled "Experimental low-pressure
    gate" note (never presented as a validated universal constant).
    `decimate_n`: see `_plot_dominance`'s own docstring -- 1 (default)
    for the exact original raw-mode rendering, >1 for the EMA
    presentation polish (this round's approved default calls with 3)."""
    subtitle = "Relative team pressure advantage over time"
    img, x0, y0, pw, ph, y_lo, y_hi = _chrome(w, h, "PRESSURE DOMINANCE OVER TIME", subtitle,
                                               "Dominance (Team A - Team B)", (-1.0, 1.0), "{:+.1f}",
                                               t_lo, t_hi, tick_spacing, accent)
    _plot_dominance(img, full_d, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi,
                     TEAM_COLOR[0], TEAM_COLOR[1], (130, 130, 130), max_gap_sec, decimate_n=decimate_n)
    for val, label in ((0.90, "Team A stronger"), (0.0, "Balanced / Neutral"), (-0.90, "Team B stronger")):
        _, ly = _to_px(t_lo, val, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
        cv2.putText(img, label, (x0 + 6, ly - 4 if val >= 0 else ly + 12), FONT, 0.24, TEXT_DIM, 1, cv2.LINE_AA)
    cx, _ = _to_px(cur_time, 0.0, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
    cv2.line(img, (cx, y0), (cx, y0 + ph), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (cx, y0), (cx, y0 + ph), TEXT_WHITE, 2, cv2.LINE_AA)
    _draw_legend(img, x0, pw, "Team A", "Team B", TEAM_COLOR[0], TEAM_COLOR[1])
    if gate_threshold is not None:
        cv2.putText(img, f"EXPERIMENTAL low-pressure gate: max(I_A,I_B) < {gate_threshold:.2f}",
                    (14, h - 8), FONT, 0.26, (140, 200, 235), 1, cv2.LINE_AA)
    return img


def draw_step_dual_graph(w, h, full_a, full_b, cur_time, t_lo, t_hi, tick_spacing, title, subtitle, y_label,
                          y_range, tick_fmt, label_a, label_b, color_a, color_b, accent=ACCENT_CYAN, max_gap_sec=0.5):
    """STEP-STYLE (post-step) variant for a small SIGNED INTEGER count --
    a horizontal hold at the previous value then a vertical jump at the
    sample time, never a sloped line implying fractional interpolation
    between counts."""
    img, x0, y0, pw, ph, y_lo, y_hi = _chrome(w, h, title, subtitle, y_label, y_range, tick_fmt, t_lo, t_hi, tick_spacing, accent)
    _plot_step(img, full_a, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color_a, max_gap_sec)
    _plot_step(img, full_b, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color_b, max_gap_sec)
    cx, _ = _to_px(cur_time, y_lo, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
    cv2.line(img, (cx, y0), (cx, y0 + ph), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (cx, y0), (cx, y0 + ph), TEXT_WHITE, 2, cv2.LINE_AA)
    _draw_legend(img, x0, pw, label_a, label_b, color_a, color_b)
    return img


# ---------------------------------------------------------------------------
# 2026-09-09 -- COACH-MODE BOTTOM ANALYTICS (Part of Graph 1: Pressing
# Team Timeline). A dedicated categorical chrome (exactly 3 fixed
# y-levels with TEXT labels, never numeric ticks) -- deliberately
# separate from `_chrome` (which is for continuous/numeric axes) rather
# than overloading it with a text-label special case.
def _chrome_categorical(w, h, title, subtitle, level_labels, t_lo, t_hi, tick_spacing, accent=ACCENT_CYAN):
    img = panel_frame(w, h)
    cv2.rectangle(img, (12, 8), (15, 24), accent, -1)
    cv2.putText(img, title, (22, 22), FONT, 0.50, TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (14, 40), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

    pad_l, pad_r, pad_t, pad_b = 130, 16, 60, 48
    x0, y0 = pad_l, pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    y_lo, y_hi = -1.0, 1.0

    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), BORDER_SOFT, 1)
    for val, label in zip((1.0, 0.0, -1.0), level_labels):
        gy = y0 + int(ph * (1 - (val - y_lo) / (y_hi - y_lo)))
        cv2.line(img, (x0, gy), (x0 + pw, gy), GRID, 1, cv2.LINE_AA)
        (tw_, _), _ = cv2.getTextSize(label, FONT, 0.34, 1)
        cv2.putText(img, label, (x0 - tw_ - 10, gy + 5), FONT, 0.34, TEXT_WHITE, 1, cv2.LINE_AA)

    t = math.ceil(t_lo / tick_spacing) * tick_spacing
    while t <= t_hi + 1e-6:
        x = x0 + int((t - t_lo) / max(1e-6, (t_hi - t_lo)) * pw)
        cv2.line(img, (x, y0), (x, y0 + ph), GRID, 1, cv2.LINE_AA)
        label = f"{int(round(t))}s"
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.30, 1)
        cv2.putText(img, label, (max(0, x - tw // 2), y0 + ph + 17), FONT, 0.30, TEXT_DIM, 1, cv2.LINE_AA)
        t += tick_spacing

    (xw, _), _ = cv2.getTextSize("Match Time (seconds)", FONT, 0.32, 1)
    cv2.putText(img, "Match Time (seconds)", (x0 + pw - xw, h - 8), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    return img, x0, y0, pw, ph, y_lo, y_hi


def _plot_categorical_step(img, hist, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi,
                            color_pos, color_zero, color_neg, max_gap_sec):
    """Post-step plot where each HORIZONTAL HOLD segment is colored by
    its OWN value (never two fixed per-series colors) -- exactly
    'Team A segments: yellow/gold, Neutral: light gray, Team B
    segments: green'. A small white ring marks every genuine level
    transition (never drawn on a plain hold)."""
    def color_for(v):
        return color_pos if v > 0 else (color_neg if v < 0 else color_zero)

    history = [(t, v) for t, v in hist if t_lo - 1e-9 <= t <= cur_time + 1e-9]
    prev_px, prev_v, last_t = None, None, None
    for t, v in history:
        gap = (last_t is not None) and (t - last_t > max_gap_sec)
        if v is None or gap:
            prev_px, prev_v = None, None
        else:
            cx, cy = _to_px(t, v, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
            if prev_px is not None:
                px, py = prev_px
                cv2.line(img, (px, py), (cx, py), color_for(prev_v), 4, cv2.LINE_AA)  # hold, in the OLD value's color
                if v != prev_v:
                    cv2.line(img, (cx, py), (cx, cy), color_for(v), 2, cv2.LINE_AA)   # the jump itself, in the NEW value's color
                    cv2.circle(img, (cx, cy), 4, TEXT_WHITE, 1, cv2.LINE_AA)           # small transition marker
            else:
                cv2.circle(img, (cx, cy), 3, color_for(v), -1, cv2.LINE_AA)
            prev_px, prev_v = (cx, cy), v
        last_t = t
    if prev_px is not None:
        cx, _ = _to_px(cur_time, prev_v, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
        cv2.line(img, prev_px, (cx, prev_px[1]), color_for(prev_v), 4, cv2.LINE_AA)  # hold through to "now"


def draw_pressing_timeline(w, h, full_signal, cur_time, t_lo, t_hi, tick_spacing, accent=ACCENT_CYAN):
    img, x0, y0, pw, ph, y_lo, y_hi = _chrome_categorical(
        w, h, "PRESSING TEAM TIMELINE", "Which team is applying the press over time",
        PRESSING_STATE_LEVELS, t_lo, t_hi, tick_spacing, accent)
    _plot_categorical_step(img, full_signal, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi,
                            TEAM_COLOR[0], (170, 170, 170), TEAM_COLOR[1], max_gap_sec=0.5)
    cx, _ = _to_px(cur_time, 0.0, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
    cv2.line(img, (cx, y0), (cx, y0 + ph), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (cx, y0), (cx, y0 + ph), TEXT_WHITE, 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# 2026-09-09 -- COACH-MODE BOTTOM ANALYTICS (Graph 3: Press Outcome
# Summary). Grouped bar chart, 4 categories x 2 teams = 8 bars, adaptive
# y-axis (never a hard-coded 0-50 range -- this dataset's real counts
# are small and that must stay visible, not padded out).
def _shadow_text(img, text, org, scale, color):
    """Count/percentage labels sit right at a bar's own top edge for a
    full-height (max-count) bar, where a plain light color loses
    contrast against a light (gold) bar fill -- a dark shadow guarantees
    legibility regardless of what's directly behind the text."""
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


SHORT_OUTCOME_LABEL = {"Regain": "Regain", "Forced Backward": "Forced Back", "Forced Lateral": "Forced Lateral", "Press Escaped": "Escaped"}


def draw_outcome_bars(w, h, counts_a, counts_b, excluded_count, any_heuristic_used=True, accent=ACCENT_CYAN, y_hi=None):
    """2026-09-10 (revision): SIDE-BY-SIDE grouped bars BY CATEGORY (4
    groups, each an [A][B] pair) -- supersedes the prior round's
    team-block layout (all 4 Team A bars, then all 4 Team B bars),
    which made cross-team comparison within one outcome category
    require scanning across the whole panel. ONE compact legend
    (top-right) replaces the prior per-group "TEAM A"/"TEAM B" text
    labels underneath every group. Percentage-within-team labels are
    dropped this round (count is the primary, uncluttered label, per
    this round's own explicit instruction). `any_heuristic_used`: a
    clip-wide flag (computed once from every real episode, not
    recomputed per-frame from only-what's-cumulatively-displayed-so-far)
    -- True as long as ANY episode anywhere in the clip resolves via the
    heuristic tier (see `resolve_outcome_label`'s evidence priority).
    Only False once real human/adjudicated labels cover EVERY episode --
    never shown once this clip is fully human-labeled.

    2026-09-10 (this round): `y_hi` is now an OPTIONAL caller-supplied
    axis ceiling (the audited `nice_bar_axis(max_count)` -- see
    `coach_pressure_signal.py`); when omitted, the exact prior
    behavior (`max(1, max_count)`) is unchanged, preserving old-caller
    compatibility."""
    img = panel_frame(w, h)
    cv2.rectangle(img, (12, 8), (15, 24), accent, -1)
    cv2.putText(img, "PRESS OUTCOME SUMMARY", (22, 22), FONT, 0.50, TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "Cumulative outcomes up to current time" + ("" if any_heuristic_used else " --"),
                (14, 40), FONT, 0.30, TEXT_DIM, 1, cv2.LINE_AA)
    if any_heuristic_used:
        cv2.putText(img, "heuristic outcomes, UNVALIDATED (no human labels yet)",
                    (14, 53), FONT, 0.30, (120, 130, 235), 1, cv2.LINE_AA)

    # one small legend, top-right -- replaces repeated per-group team labels
    lx = w - 16
    for label, color in (("Team B", TEAM_COLOR[1]), ("Team A", TEAM_COLOR[0])):
        tw_ = cv2.getTextSize(label, FONT, 0.30, 1)[0][0]
        lx -= tw_
        cv2.putText(img, label, (lx, 53), FONT, 0.30, TEXT_WHITE, 1, cv2.LINE_AA)
        lx -= 15
        cv2.rectangle(img, (lx, 45), (lx + 10, 53), color, -1)
        lx -= 10

    pad_l, pad_r, pad_t, pad_b = 44, 16, 68, 46
    x0, y0 = pad_l, pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    max_count = max([0] + list(counts_a.values()) + list(counts_b.values()))
    if y_hi is None:
        y_hi = max(1, max_count)  # adaptive -- never a hard-coded range; handles max=0 (still shows a 0-1 scale) and any real max>=1
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), BORDER_SOFT, 1)
    step = max(1, math.ceil(y_hi / 5))
    yv = 0
    while yv <= y_hi:
        gy = y0 + int(ph * (1 - yv / y_hi))
        cv2.line(img, (x0, gy), (x0 + pw, gy), GRID, 1, cv2.LINE_AA)
        cv2.putText(img, str(yv), (6, gy + 4), FONT, 0.28, TEXT_DIM, 1, cv2.LINE_AA)
        yv += step

    n_groups = len(OUTCOME_CATEGORIES)
    group_gap = 22
    group_w = (pw - group_gap * (n_groups - 1)) / n_groups
    bar_w = group_w / 2.0 - 4
    for gi, cat in enumerate(OUTCOME_CATEGORIES):
        group_x0 = x0 + gi * (group_w + group_gap)
        for team_idx, (counts, color) in enumerate(((counts_a, TEAM_COLOR[0]), (counts_b, TEAM_COLOR[1]))):
            c = counts[cat]
            bar_h = int(ph * (c / y_hi)) if y_hi > 0 else 0
            bx0 = int(group_x0 + team_idx * (bar_w + 4))
            bx1 = int(bx0 + bar_w)
            by1 = y0 + ph
            by0 = by1 - bar_h
            if c > 0:
                cv2.rectangle(img, (bx0, by0), (bx1, by1), color, -1)
            else:
                cv2.rectangle(img, (bx0, by1 - 2), (bx1, by1), color, -1)  # a visible sliver at 0 -- honest "real zero", not an absent bar
            _shadow_text(img, str(c), (bx0, max(y0 + 10, by0 - 6)), 0.30, TEXT_WHITE)
        lbl = SHORT_OUTCOME_LABEL[cat]
        (tw_, _), _ = cv2.getTextSize(lbl, FONT, 0.26, 1)
        cv2.putText(img, lbl, (int(group_x0 + group_w / 2 - tw_ / 2), y0 + ph + 16), FONT, 0.26, TEXT_DIM, 1, cv2.LINE_AA)
        if gi > 0:
            dx = int(group_x0 - group_gap / 2)
            cv2.line(img, (dx, y0), (dx, y0 + ph), (40, 46, 60), 1, cv2.LINE_AA)

    cv2.putText(img, f"Other/uncertain excluded: {excluded_count}", (14, h - 8), FONT, 0.26, TEXT_DIM, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_video_path", default=os.path.join(PROJECT_ROOT, "data/source_video/testVideo1_120s.mp4"))
    ap.add_argument("--tracking_dir", default=os.path.join(PROJECT_ROOT, "data/tracking/testVideo1_120s"))
    ap.add_argument("--analytics_dir", default=os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3"))
    ap.add_argument("--snapshot_pkl", default=os.path.join(PROJECT_ROOT, "data/snapshots/v4_snapshot.pkl"))
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "outputs/correctness_v4/dashboard_previews_simple"))
    ap.add_argument("--frame", type=int, action="append", default=None, help="explicit frame(s) to render (preview mode)")
    ap.add_argument("--name", action="append", default=None, help="output filename stem per --frame (preview mode)")
    ap.add_argument("--time_mode", choices=["rolling", "paper"], default="rolling",
                     help="rolling = trailing 20s window with absolute-time ticks (what --out_path renders); "
                          "paper = full 0-duration static timeline, for paper figures/reports")
    ap.add_argument("--out_path", default=None, help="if set, render the FULL clip to this .mp4 path (always rolling-20s mode)")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=None, help="inclusive; defaults to the last snapshot frame")
    ap.add_argument("--debug", action="store_true",
                     help="(P4) research/debug mode only: overlay 'Pressure at carrier: 0.XX' text on the radar. "
                          "OFF by default -- not part of the clean final video.")
    ap.add_argument("--radar-view", choices=["flat", "perspective"], default="flat",
                     help="Radar rendering mode: 'flat' (unchanged existing top-down radar) or 'perspective' "
                          "(NEW tilted tactical-board camera, same underlying pitch geometry/analytics).")
    ap.add_argument("--dashboard-mode", choices=["coach", "research"], default="coach",
                     help="'coach' (NEW default, 2026-09-09): 3 large bottom panels -- Pressing Team Timeline, "
                          "Pressure Intensity Trend, Press Outcome Summary. 'research': the original 4 detailed "
                          "scientific graphs (Press Intensity, Nearest Opponent Distance, Relative Closing Speed, "
                          "Numerical Support) -- code path UNCHANGED, kept for research/debug use.")
    ap.add_argument("--coach-intensity-view", choices=["raw", "ema"], default="ema",
                     help="Coach-mode Graph 2 (Pressure Intensity Trend) display only: 'ema' (2026-09-10 "
                          "APPROVED DEFAULT after visual QA -- a strictly causal, presentation-only exponential "
                          "moving average, alpha=%.1f, see `causal_ema`'s own docstring; the underlying opp_ballw "
                          "values are never changed either way) or 'raw' (exact per-frame opp_ballw, unmodified -- "
                          "still available for research/debug comparison)." % COACH_EMA_ALPHA)
    ap.add_argument("--coach-dominance-view", choices=["raw", "ema"], default="ema",
                     help="Coach-mode Graph 1 (Pressure Dominance Over Time) display only: 'ema' (2026-09-10 "
                          "APPROVED DEFAULT after visual QA -- strictly causal display smoothing applied directly "
                          "to D_raw=I_A-I_B, same `causal_ema` helper Graph 2 uses, alpha=%.1f) or 'raw' (still "
                          "available for research/debug comparison)." % COACH_EMA_ALPHA)
    ap.add_argument("--coach-dominance-min-pressure", type=float, default=None,
                     help="EXPERIMENTAL low-pressure gate for Graph 1: frames where max(I_A,I_B) is below this "
                          "threshold display as LOW PRESSURE / NEUTRAL instead of a real dominance reading. "
                          "Default None = gate DISABLED (ungated dominance). This is NOT a validated universal "
                          "constant -- audited only against the one clip this project has today (see "
                          "audit_reports/2026-09-09_continuous_pressure_signal_audit/, threshold=0.633 on that "
                          "data); pass it explicitly per use, never assume it transfers to a new match unchanged.")
    ap.add_argument("--coach-graph-display-hz", type=int, choices=[5, 10, 15, 30], default=10,
                     help="Coach-mode Graph 1/2 DISPLAY-ONLY sampling rate (2026-09-09 presentation fix): the "
                          "underlying opp_ballw/EMA analytics remain full 30fps resolution always -- this only "
                          "controls how many of those already-computed points get DRAWN, via deterministic "
                          "floor(t/(1/hz)) time buckets (never every-Nth-index), keeping the LAST causal sample "
                          "per bucket. 10 (default, APPROVED after A/B preview QA) keeps the panel readable as "
                          "the rolling 60s window fills with real data; 30 reproduces the full-density/debug "
                          "rendering. Applied AFTER causal EMA, never before (decimating first would change what "
                          "the EMA itself smooths over). See `coach_pressure_signal.decimate_for_display`.")
    ap.add_argument("--coach-graph-window-seconds", type=int, choices=[24, 30, 60], default=24,
                     help="Coach-mode Graph 1/2 rolling X-AXIS WINDOW WIDTH (2026-09-09 window-size audit): "
                          "x_min=max(0,cur_time-W), x_max=cur_time, identical W for BOTH graphs always (never "
                          "allowed to drift apart). 24 (default, APPROVED after visual A/B QA -- ~2.5x horizontal "
                          "detail vs the original 60s) with 4s major ticks; 30 and 60 remain available for "
                          "comparison/debug. Tick spacing is chosen per-window automatically (see "
                          "COACH_GRAPH_WINDOW_TICK_SPACING), never dense 1s ticks. Graph 3 and research mode are "
                          "entirely unaffected by this flag.")
    ap.add_argument("--camera-elevation-deg", type=float, default=None, help="override RadarCameraConfig.elevation_deg")
    ap.add_argument("--camera-fov-deg", type=float, default=None, help="override RadarCameraConfig.fov_deg")
    ap.add_argument("--camera-distance-scale", type=float, default=None, help="override RadarCameraConfig.distance_scale")
    args = ap.parse_args()

    camera_kwargs = {}
    if args.camera_elevation_deg is not None:
        camera_kwargs["elevation_deg"] = args.camera_elevation_deg
    if args.camera_fov_deg is not None:
        camera_kwargs["fov_deg"] = args.camera_fov_deg
    if args.camera_distance_scale is not None:
        camera_kwargs["distance_scale"] = args.camera_distance_scale
    radar_camera = RadarCameraConfig(**camera_kwargs) if camera_kwargs else DEFAULT_RADAR_CAMERA
    RADAR_H = compute_radar_homography(DEFAULT_PITCH, radar_camera, 1300, 800) if args.radar_view == "perspective" else None

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.snapshot_pkl, "rb") as fh:
        snapshot = pickle.load(fh)

    tracking = pl.read_parquet(os.path.join(args.tracking_dir, "tracking.parquet"))
    cleaned = build_cleaned_view(tracking)
    tracking_by_frame, cleaned_by_frame = {}, {}
    for r in tracking.to_dicts():
        tracking_by_frame.setdefault(r["frame"], []).append(r)
    for r in cleaned.to_dicts():
        cleaned_by_frame.setdefault(r["frame"], []).append(r)
    ball_by_frame = {r["frame"]: r for r in load_ball_view(args.analytics_dir).to_dicts()}
    with open(os.path.join(args.analytics_dir, "homography_transformers.pkl"), "rb") as fh:
        transformers = pickle.load(fh)
    cap = cv2.VideoCapture(args.source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = len(snapshot) / fps

    # ---- FULL-CLIP series, precomputed ONCE (not per frame) ----
    full_i0 = _full_series(snapshot, lambda s: s["pressing"][0]["raw_score"])
    full_i1 = _full_series(snapshot, lambda s: s["pressing"][1]["raw_score"])

    full_dist_cm = _full_series(snapshot, lambda s: _select_pressing_value(s, "nearest_opponent_distance_cm"))
    full_dist_m = [(t, v / 100.0 if v is not None else None) for t, v in full_dist_cm]
    y_dist = _signed_y_range([v for _, v in full_dist_m], floor_zero=True)

    full_close_cms = _full_series(snapshot, lambda s: _select_pressing_value(s, "relative_closing_cm_s"))
    full_close_ms = [(t, v / 100.0 if v is not None else None) for t, v in full_close_cms]
    y_close = _signed_y_range([v for _, v in full_close_ms])

    full_sup0 = _full_series(snapshot, lambda s: s["pressing"][0]["features"].get("support_minus_opponents"))
    full_sup1 = _full_series(snapshot, lambda s: s["pressing"][1]["features"].get("support_minus_opponents"))
    y_sup = _signed_y_range([v for _, v in full_sup0 + full_sup1])

    # (2A) full-pitch grid, built ONCE and reused every frame -- moved
    # earlier (was previously built just before `render_frame`) so the
    # coach-mode opp_ballw series below can reuse it too.
    gx, gy = pitch_grid(DEFAULT_PITCH)

    # 2026-09-10 -- COACH-MODE Graph 1 (Pressure Dominance) + Graph 2
    # (Pressure Intensity Trend) + Graph 3 (Outcomes) data, all derived
    # from ONE real `build_pressing_v4` run (never duplicated).
    # `full_ia`/`full_ib` (the audited `opp_ballw` signal) are the SAME
    # underlying series consumed by BOTH Graph 1 (as D=I_A-I_B) and
    # Graph 2 (plotted directly) -- computed exactly once, per this
    # round's explicit "do not duplicate formulas" requirement. Only
    # computed in coach mode (research mode needs none of this; its own
    # Graph 1 still uses the untouched `full_i0`/`full_i1` raw_score
    # series above).
    full_ia, full_ib = [], []
    modes_a, modes_b = [], []
    outcome_events = []
    any_heuristic_outcome_used = False
    if args.dashboard_mode == "coach":
        quality = build_quality_view(tracking, fps=fps, pitch=DEFAULT_PITCH)
        quality_by_frame = {}
        for r in quality.to_dicts():
            quality_by_frame.setdefault(r["frame"], []).append(r)
        roles = current_roles(quality_by_frame, ball_by_frame, len(snapshot), fps=fps, pitch=DEFAULT_PITCH)
        passes = pl.read_parquet(os.path.join(args.analytics_dir, "passes.parquet")).to_dicts()
        turnovers = pl.read_parquet(os.path.join(args.analytics_dir, "turnovers.parquet")).to_dicts()
        pv4 = build_pressing_v4(quality_by_frame, roles, passes, turnovers, fps=fps, pitch=DEFAULT_PITCH)

        frame_times = [(i, s["time_sec"]) for i, s in enumerate(snapshot)]
        ballw_series = compute_opp_ballw_series(quality_by_frame, ball_by_frame, frame_times, gx, gy,
                                                 sigma_cm=BALL_PROXIMITY_SIGMA_CM)
        full_ia = ballw_series[0]["values"]
        full_ib = ballw_series[1]["values"]
        modes_a = ballw_series[0]["modes"]
        modes_b = ballw_series[1]["modes"]

        # Frame-coverage audit, printed every run (per this round's own
        # "STOP and investigate if these differ substantially [from the
        # audit's 68.2% finding]" requirement) -- real counts, not an
        # assumption.
        n_total = len(snapshot)
        n_a_valid = sum(1 for _, v in full_ia if v is not None)
        n_b_valid = sum(1 for _, v in full_ib if v is not None)
        n_ball_weighted = sum(1 for m in modes_a + modes_b if m == MODE_BALL_WEIGHTED)
        n_fallback = sum(1 for m in modes_a + modes_b if m == MODE_UNWEIGHTED_FALLBACK)
        n_missing = sum(1 for m in modes_a + modes_b if m == MODE_MISSING)
        print(f"[coach opp_ballw coverage] Team A valid={n_a_valid}/{n_total} ({n_a_valid/n_total*100:.1f}%)  "
              f"Team B valid={n_b_valid}/{n_total} ({n_b_valid/n_total*100:.1f}%)  "
              f"BALL_WEIGHTED={n_ball_weighted}  UNWEIGHTED_FALLBACK={n_fallback}  MISSING={n_missing}  "
              f"(both teams combined, out of {2*n_total})")

        # Evidence priority (final_adjudicated_outcome > human_outcome >
        # heuristic_outcome): loaded from the real research-phase export
        # if it exists, optional/non-fatal otherwise -- both override
        # columns are `None` for every real episode today (0 human
        # labels exist), so this changes nothing observable yet, but is
        # the correct, forward-looking wiring per this round's own spec.
        label_overrides = {}
        research_parquet = os.path.join(PROJECT_ROOT, "outputs/research_dataset/pressing_episode_features.parquet")
        if os.path.exists(research_parquet):
            for row in pl.read_parquet(research_parquet).to_dicts():
                label_overrides[row["episode_id"]] = {
                    "human_outcome": row.get("human_outcome"),
                    "final_adjudicated_outcome": row.get("final_adjudicated_outcome"),
                }

        episodes_with_outcome = attribute_outcomes_all_episodes(pv4["episodes"], passes, turnovers, DEFAULT_PITCH, fps, quality_by_frame)
        outcome_events, any_heuristic_outcome_used = build_outcome_events(episodes_with_outcome, label_overrides)

    # 2026-09-10 -- COACH-MODE Graph 1/2 optional causal display
    # smoothing (`--coach-intensity-view`/`--coach-dominance-view
    # {raw,ema}`, independently selectable). `full_ia`/`full_ib`
    # (unchanged, the SAME series both graphs are built from) are what
    # get smoothed for DISPLAY; the raw series itself is never
    # modified/replaced. Dominance is computed from the RAW (never
    # EMA'd) I_A/I_B first, then optionally EMA'd itself as its OWN
    # display transform -- so `--coach-dominance-view ema` and
    # `--coach-intensity-view ema` are independent, composable choices,
    # per this round's explicit spec.
    if args.dashboard_mode == "coach" and args.coach_intensity_view == "ema":
        plot_ia = causal_ema(full_ia, alpha=COACH_EMA_ALPHA, max_gap_sec=0.5)
        plot_ib = causal_ema(full_ib, alpha=COACH_EMA_ALPHA, max_gap_sec=0.5)
    else:
        plot_ia, plot_ib = full_ia, full_ib

    full_d, full_max_intensity, plot_d, plot_d_gated = [], [], [], []
    if args.dashboard_mode == "coach":
        full_d = compute_dominance(full_ia, full_ib)
        full_max_intensity = compute_max_intensity(full_ia, full_ib)
        plot_d = causal_ema(full_d, alpha=COACH_EMA_ALPHA, max_gap_sec=0.5) if args.coach_dominance_view == "ema" else full_d
        plot_d_gated = gate_dominance(plot_d, full_max_intensity, args.coach_dominance_min_pressure)

    # 2026-09-09 -- COACH-MODE Graph 1/2 DISPLAY-DENSITY decimation
    # (`--coach-graph-display-hz`, default 10). Applied LAST, after EMA
    # and gating -- never before (see the flag's own help text and
    # `coach_pressure_signal.decimate_for_display`'s docstring for why
    # the order matters). This is presentation-only: `full_ia`/`full_ib`/
    # `full_d`/`plot_ia`/`plot_ib`/`plot_d`/`plot_d_gated` above are all
    # UNCHANGED by this step -- only the NEW `disp_*` series (what
    # actually gets drawn) are decimated.
    disp_ia, disp_ib, disp_d_gated = plot_ia, plot_ib, plot_d_gated
    if args.dashboard_mode == "coach":
        disp_ia = decimate_for_display(plot_ia, display_hz=args.coach_graph_display_hz, max_gap_sec=0.5)
        disp_ib = decimate_for_display(plot_ib, display_hz=args.coach_graph_display_hz, max_gap_sec=0.5)
        disp_d_gated = decimate_for_display(plot_d_gated, display_hz=args.coach_graph_display_hz, max_gap_sec=0.5)

        n_full_valid = sum(1 for _, v in plot_ia if v is not None) + sum(1 for _, v in plot_ib if v is not None)
        n_disp_valid = sum(1 for _, v in disp_ia if v is not None) + sum(1 for _, v in disp_ib if v is not None)
        n_missing_before = count_missing_segments(plot_ia) + count_missing_segments(plot_ib)
        n_missing_after = count_missing_segments(disp_ia) + count_missing_segments(disp_ib)
        reduction_pct = (1 - n_disp_valid / n_full_valid) * 100 if n_full_valid else 0.0
        print(f"[coach display decimation] hz={args.coach_graph_display_hz}  "
              f"full-res plotted samples={n_full_valid}  display samples={n_disp_valid}  "
              f"reduction={reduction_pct:.1f}%  missing segments before={n_missing_before}  "
              f"after={n_missing_after}  (both teams combined)")

    def render_frame(f, seek=True, time_mode="rolling"):
        snap_f = snapshot[f]
        if seek:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            return None
        transformer = _nearest_transformer(transformers, f)
        players_this_frame = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None]
        ball_row = ball_by_frame.get(f)
        ball_xy = (ball_row["x_pitch"], ball_row["y_pitch"]) if ball_row and ball_row.get("is_observed") and ball_row.get("x_pitch") is not None else None

        # Carrier identity: the EXACT track_id current_roles() assigned
        # this frame (snap_f["current_carrier_track"], now passed through
        # by tactical_shared.snapshot -- 2026-09-07 V5 pass), looked up
        # by identity, never approximated. None on frames the analytics
        # itself does not know who has the ball -- an honest gap.
        carrier = _lookup_by_track(players_this_frame, snap_f.get("current_carrier_track"))
        presser, presser_team, participant_ids, balance_text, closing_color = None, None, [], None, None
        field, field_b, dual_field_mode = None, None, False
        pressure_alpha_scale = PRESSURE_STATE_ALPHA_SCALE[PRESSURE_STATE_DEFAULT]
        pressure_contour_thickness = PRESSURE_STATE_CONTOUR_THICKNESS[PRESSURE_STATE_DEFAULT]
        pressure_palette_mode = PRESSURE_STATE_PALETTE_MODE[PRESSURE_STATE_DEFAULT]

        # (2026-09-08 round G) CONTINUOUS FIELD: team attribution for the
        # base spatial-influence field no longer requires the resolved
        # carrier -- `_select_pressure_field_team` implements the full
        # evidence hierarchy (instant carrier -> trailing-4s historical
        # hold -> None). This is computed BEFORE, and independently of,
        # the carrier-SPECIFIC overlay block below.
        field_team = _select_pressure_field_team(snap_f)
        if field_team is not None:
            # CASE 1/2 -- a single team is attributed as "defending" --
            # same math as before this round, just no longer requiring
            # `carrier is not None` to reach it. (2A-2D) the field is a
            # real, continuous function of CURRENT positions/velocities,
            # not gated on the FSM's PRESS_FORMING/ACTIVE_PRESS state,
            # which additionally requires SUSTAINED evidence over time --
            # a genuinely different, honest question: "how quickly could
            # this team close down this area right now".
            field_team_players = [p for p in players_this_frame
                                   if p["display_team_id"] == field_team and p["display_object_type"] in ("player", "goalkeeper")]
            if field_team_players:  # insufficient geometry for THIS team -> honest gap, field stays None (never borrow the other team's field under this team's label)
                field = team_pressure_field(field_team_players, gx, gy)
                pressure_state = snap_f["pressing"][field_team]["state"]
                pressure_alpha_scale = PRESSURE_STATE_ALPHA_SCALE.get(pressure_state, PRESSURE_STATE_ALPHA_SCALE[PRESSURE_STATE_DEFAULT])
                pressure_contour_thickness = PRESSURE_STATE_CONTOUR_THICKNESS.get(pressure_state, PRESSURE_STATE_CONTOUR_THICKNESS[PRESSURE_STATE_DEFAULT])
                pressure_palette_mode = PRESSURE_STATE_PALETTE_MODE.get(pressure_state, PRESSURE_STATE_PALETTE_MODE[PRESSURE_STATE_DEFAULT])
        else:
            # CASE 3 -- neither the instant carrier nor the historical
            # hold gives a defending-team attribution. NEVER guess Team
            # A/B: compute BOTH teams' OWN fields independently (each
            # None on its own if that team lacks geometry) and let
            # `draw_left_feed`/`draw_radar` render them in the separate,
            # extra-subdued, team-tinted `dual_field_mode`.
            team_a_players = [p for p in players_this_frame
                               if p["display_team_id"] == 0 and p["display_object_type"] in ("player", "goalkeeper")]
            team_b_players = [p for p in players_this_frame
                               if p["display_team_id"] == 1 and p["display_object_type"] in ("player", "goalkeeper")]
            if team_a_players or team_b_players:
                field = team_pressure_field(team_a_players, gx, gy) if team_a_players else None
                field_b = team_pressure_field(team_b_players, gx, gy) if team_b_players else None
                dual_field_mode = True
            # else: neither team has ANY valid geometry this frame --
            # field/field_b/dual_field_mode all stay at their "no field
            # at all" defaults (insufficient geometry, honestly gapped).

        if carrier is not None:
            # Carrier-SPECIFIC overlays only (escape arrows, nearest
            # presser, local numerical balance, participant links) --
            # the base field above is ALREADY computed/selected
            # independently and is NOT recomputed here.
            presser_team = 1 - carrier["display_team_id"]
            feat = snap_f["pressing"][presser_team]["features"]
            participant_ids = feat.get("participant_ids") or []
            # (2)/(3) nearest presser + closing direction -- EXACT
            # identity via nearest_opponent_track_id (2026-09-07 V5
            # pass: this field previously did not exist; the connector/
            # arrow were withheld pending it -- now implemented).
            presser = _lookup_by_track(players_this_frame, feat.get("nearest_opponent_track_id"))
            closing = feat.get("relative_closing_cm_s")
            if closing is not None:
                if closing > CLOSING_NEUTRAL_THRESHOLD_CM_S:
                    closing_color = CLOSING_COLOR
                elif closing < -CLOSING_NEUTRAL_THRESHOLD_CM_S:
                    closing_color = ESCAPING_COLOR
                else:
                    closing_color = NEUTRAL_CLOSING_COLOR
            bal = feat.get("support_minus_opponents")
            if bal is not None:
                carrier_team = carrier["display_team_id"]
                if bal > 0:
                    balance_text = f"{TEAM_LETTER[carrier_team]} +{bal:.0f}"
                elif bal < 0:
                    balance_text = f"{TEAM_LETTER[presser_team]} +{-bal:.0f}"
                else:
                    balance_text = "Even"

        # (P4) DEBUG/RESEARCH-ONLY: pressure field sampled exactly at the
        # carrier's own position. Deliberately NOT merged with the
        # existing Press Intensity FSM metric -- shown side by side only
        # when --debug is passed, so the two can be visually compared.
        # None whenever carrier or field is unknown -- an honest gap,
        # never defaulted to 0.
        carrier_field_pressure = None
        if carrier is not None and field is not None:
            carrier_field_pressure = sample_field(field, gx, gy, carrier["x_pitch"], carrier["y_pitch"])

        # (B3/B4) escape directions + (B5/B6) passing lanes -- both
        # derived from the SAME `field` array, computed once and shared
        # by radar and video.
        escape_points, lanes, carrier_xy = None, None, None
        if carrier is not None and field is not None:
            carrier_xy = (carrier["x_pitch"], carrier["y_pitch"])
            escape_points = escape_directions(field, gx, gy, carrier_xy, DEFAULT_PITCH, n_select=N_ESCAPE_ARROWS)
            teammates = [p for p in players_this_frame
                         if p["display_team_id"] == carrier["display_team_id"] and p["track_id"] != carrier["track_id"]
                         and p["display_object_type"] in ("player", "goalkeeper")]
            opponents = [p for p in players_this_frame if p["display_team_id"] == presser_team]
            lane_metrics = [(tm, passing_lane_metrics(field, gx, gy, carrier, tm, opponents)) for tm in teammates]
            lane_metrics.sort(key=lambda lm: lm[1]["mean_pressure"])
            open_lanes = lane_metrics[:N_OPEN_LANES]
            constrained_lanes = lane_metrics[-N_CONSTRAINED_LANES:] if len(lane_metrics) > N_OPEN_LANES else []
            open_colors = [LANE_OPEN_COLOR, LANE_OPEN_SECOND_COLOR]
            open_thickness = [3, 1]
            lanes = ([(carrier_xy, (tm["x_pitch"], tm["y_pitch"]), open_colors[i], open_thickness[i], False)
                      for i, (tm, _) in enumerate(open_lanes)] +
                     [(carrier_xy, (tm["x_pitch"], tm["y_pitch"]), LANE_CONSTRAINED_COLOR, 2, True)
                      for tm, _ in constrained_lanes])

        left = cv2.resize(draw_left_feed(frame, tracking_by_frame.get(f, []), snap_f, transformer,
                                          carrier["track_id"] if carrier is not None else None,
                                          presser["track_id"] if presser is not None else None,
                                          closing_color, field, carrier_xy, escape_points, lanes,
                                          pressure_alpha_scale=pressure_alpha_scale,
                                          pressure_contour_thickness=pressure_contour_thickness,
                                          pressure_palette_mode=pressure_palette_mode,
                                          field_b=field_b, dual_field_mode=dual_field_mode), (LEFT_W, TOP_ROW_H))
        radar = draw_radar(players_this_frame, ball_xy, snap_f, RIGHT_W, TOP_ROW_H, carrier, presser, presser_team,
                            participant_ids, balance_text, closing_color, field, escape_points, lanes,
                            debug=args.debug, carrier_field_pressure=carrier_field_pressure,
                            radar_view=args.radar_view, H=RADAR_H,
                            pressure_alpha_scale=pressure_alpha_scale,
                            pressure_contour_thickness=pressure_contour_thickness,
                            pressure_palette_mode=pressure_palette_mode,
                            field_b=field_b, dual_field_mode=dual_field_mode)
        top_row = np.hstack([left, radar])

        # 2026-09-08 (research-phase kickoff) -- public-facing title no
        # longer exposes the internal "V4" version number (internal
        # filenames/code/version history are unaffected -- this is a
        # presentation-only string).
        header = draw_header(CANVAS_W, HEADER_H, "PRESSING STRUCTURE ANALYTICS",
                              (f"t = {snap_f['time_sec']:.1f}s", TEXT_DIM), accent=ACCENT_CYAN)

        cur_time = snap_f["time_sec"]
        if time_mode == "rolling":
            t_lo, t_hi, tick_spacing = max(0.0, cur_time - 20.0), cur_time, 4
        else:
            t_lo, t_hi, tick_spacing = 0.0, duration_sec, 30

        if args.dashboard_mode == "coach":
            # 2026-09-09 -- COACH MODE (NEW default): 3 large panels,
            # ~35% / 32.5% / 32.5% of width. Graphs 1-2 use their OWN
            # rolling window (`--coach-graph-window-seconds`, default 60,
            # see COACH_GRAPH_WINDOW_TICK_SPACING below for the exact
            # per-window tick spacing), DELIBERATELY separate from the
            # 20s window `t_lo`/`t_hi` research mode still uses below --
            # never shared, so research mode's own behavior stays
            # byte-identical to before. Graph 1 and Graph 2 ALWAYS use
            # the exact same window (never allowed to drift apart).
            coach_window_sec = args.coach_graph_window_seconds
            t_lo_win = max(0.0, cur_time - coach_window_sec)
            t_hi_win = cur_time
            tick_win = COACH_GRAPH_WINDOW_TICK_SPACING[coach_window_sec]
            gw1 = int(CANVAS_W * 0.35) - GAP
            gw2 = int(CANVAS_W * 0.325) - GAP
            gw3 = CANVAS_W - gw1 - gw2 - 2 * GAP
            # 2026-09-09 (display-density fix): Graphs 1/2 now draw the
            # `disp_*` series -- already display-decimated to
            # `--coach-graph-display-hz` (default 10) -- never `plot_*`
            # directly. `decimate_n=1` below (no FURTHER internal vertex
            # thinning) whenever real pre-decimation already happened
            # (hz<30); at hz=30 (full-density/debug) the old internal
            # decimate_n=3-for-EMA behavior is kept so that mode doesn't
            # regress to a dot-cloud either.
            inner_decimate_n = 1 if args.coach_graph_display_hz < 30 else (3 if args.coach_dominance_view == "ema" else 1)
            g1 = draw_dominance_graph(gw1, GRAPH_ROW_H, disp_d_gated, cur_time, t_lo_win, t_hi_win, tick_win,
                                       gate_threshold=args.coach_dominance_min_pressure,
                                       decimate_n=inner_decimate_n)
            intensity_subtitle = "Continuous relevant pressure intensity for both teams" + (
                " (causal EMA display, raw values unchanged)" if args.coach_intensity_view == "ema" else " (raw per-frame values)")
            g2 = draw_dual_graph_coach(gw2, GRAPH_ROW_H, disp_ia, disp_ib, cur_time, t_lo_win, t_hi_win, tick_win,
                                        "PRESSURE INTENSITY TREND", intensity_subtitle,
                                        "Pressure Intensity (0 to 1)", (0.0, 1.0), "{:.2f}", "Team A", "Team B",
                                        TEAM_COLOR[0], TEAM_COLOR[1],
                                        marker_style=("sparse" if args.coach_intensity_view == "ema" else "dense"),
                                        decimate_n=(1 if args.coach_graph_display_hz < 30 else 3))
            counts_a, counts_b, excluded = cumulative_outcome_counts(outcome_events, cur_time)
            # nice_bar_axis is monotonic BY CONSTRUCTION here: cumulative_outcome_counts
            # never decreases as cur_time grows (see coach_pressure_signal.
            # monotonic_nice_axis's own docstring for the defensive, stateful
            # fallback this equals today), so re-deriving y_hi fresh from the
            # current max_count each frame is already a never-shrinks axis --
            # no cross-frame mutable state needed, and safe to call with
            # frames in any order (preview mode is not always chronological).
            outcome_y_hi = nice_bar_axis(max([0] + list(counts_a.values()) + list(counts_b.values())))
            g3 = draw_outcome_bars(gw3, GRAPH_ROW_H, counts_a, counts_b, excluded,
                                    any_heuristic_used=any_heuristic_outcome_used, y_hi=outcome_y_hi)
            gap_col = np.full((GRAPH_ROW_H, GAP, 3), BG, dtype=np.uint8)
            graph_row = np.hstack([g1, gap_col, g2, gap_col, g3])
        else:
            # research mode -- UNCHANGED code path, exactly as before this round.
            graph_w = (CANVAS_W - 5 * GAP) // 4
            g1 = draw_dual_graph(graph_w, GRAPH_ROW_H, full_i0, full_i1, cur_time, t_lo, t_hi, tick_spacing,
                                  "Press Intensity -- Team A vs Team B",
                                  "Higher value = stronger evidence of pressure around the ball carrier",
                                  "Press Intensity (0 to 1)", (0.0, 1.0), "{:.2f}", "Team A", "Team B",
                                  TEAM_COLOR[0], TEAM_COLOR[1])
            g2 = draw_graph(graph_w, GRAPH_ROW_H, full_dist_m, cur_time, t_lo, t_hi, tick_spacing,
                             "Nearest Opponent Distance to Ball Carrier",
                             "Lower value = nearest defender is closer to the ball carrier",
                             "Distance to Ball Carrier (metres)", y_dist, "{:.0f}", LINE_COLOR)
            g3 = draw_graph(graph_w, GRAPH_ROW_H, full_close_ms, cur_time, t_lo, t_hi, tick_spacing,
                             "Relative Closing Speed",
                             "Positive = defender closing the carrier; Negative = gap opening",
                             "Closing Speed (metres/second)", y_close, "{:.1f}", LINE_COLOR)
            g4 = draw_step_dual_graph(graph_w, GRAPH_ROW_H, full_sup0, full_sup1, cur_time, t_lo, t_hi, tick_spacing,
                                       "Numerical Support Around Ball Carrier",
                                       "Positive = ball carrier's side has more nearby support; Negative = the pressing team has more",
                                       "Local Numerical Balance (players)", y_sup, "{:.0f}", "Team A", "Team B",
                                       TEAM_COLOR[0], TEAM_COLOR[1])
            gap_col = np.full((GRAPH_ROW_H, GAP, 3), BG, dtype=np.uint8)
            graph_row = np.hstack([g1, gap_col, g2, gap_col, g3, gap_col, g4])
        if graph_row.shape[1] != CANVAS_W:
            graph_row = cv2.resize(graph_row, (CANVAS_W, GRAPH_ROW_H))

        row_gap = np.full((GAP, CANVAS_W, 3), BG, dtype=np.uint8)
        canvas = np.vstack([header, top_row, row_gap, graph_row])
        return canvas

    if args.out_path:
        f_start = args.start_frame
        f_end = args.end_frame if args.end_frame is not None else len(snapshot) - 1
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_path, fourcc, fps, (CANVAS_W, 1204))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)  # seek ONCE, then read sequentially
        n_written = 0
        t0 = time.time()
        for f in range(f_start, f_end + 1):
            canvas = render_frame(f, seek=False, time_mode="rolling")
            if canvas is None:
                print(f"stopped at frame {f}: source video ended early")
                break
            writer.write(canvas)
            n_written += 1
            if n_written % 300 == 0:
                elapsed = time.time() - t0
                print(f"...{n_written} frames written ({f/fps:.1f}s of {(f_end-f_start+1)/fps:.1f}s) "
                      f"in {elapsed:.1f}s ({n_written/max(elapsed,1e-6):.1f} fps)", flush=True)
        writer.release()
        cap.release()
        print(f"Wrote {args.out_path}: {n_written} frames, canvas {CANVAS_W}x1204, {time.time()-t0:.1f}s total")
        return

    names = args.name or [f"frame_{f}" for f in (args.frame or [])]
    for f, name in zip(args.frame or [], names):
        canvas = render_frame(f, time_mode=args.time_mode)
        if canvas is None:
            print(f"frame {f}: could not read video frame")
            continue
        out_path = os.path.join(args.out_dir, f"{name}.png")
        cv2.imwrite(out_path, canvas)
        print(f"wrote {out_path} ({canvas.shape[1]}x{canvas.shape[0]})")
    cap.release()


if __name__ == "__main__":
    main()
