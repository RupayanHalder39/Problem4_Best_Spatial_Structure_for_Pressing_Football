"""
2026-09-08 -- shared PERSPECTIVE RADAR camera/projection helper, used by
BOTH `pressing_structure` and `offside_break` dashboards. VISUALIZATION
ONLY: no analytics, no thresholds, no event/score computation lives
here -- this module's only job is turning a canonical pitch-plane point
(x_pitch_cm, y_pitch_cm, z=0 ALWAYS) into a pixel on a tilted tactical-
board canvas, via a real, principled pinhole-camera projection, not a
hand-fit screen-space approximation.

CORE PRINCIPLE (do not violate): every football object -- players,
lines, ball, pressure/threat rasters, pass vectors, escape arrows --
sits on the SAME z=0 pitch plane and is projected through the SAME
`compute_radar_homography()` matrix. Because every point lies on one
flat plane, a full 3D pinhole-camera projection collapses to a single
3x3 planar HOMOGRAPHY (the same mathematical object every other part of
this project already uses for pitch<->broadcast-image projection via
`cv2.perspectiveTransform`/`cv2.warpPerspective` -- not a new technique,
the same one, aimed at a virtual tactical-board camera instead of the
real broadcast camera).

MATH (see `compute_radar_homography` for the implementation):
  1. A virtual pinhole camera is placed above and behind the pitch,
     looking at the pitch centre. Its position is parameterized by
     `elevation_deg` (angle above the pitch plane), `yaw_deg` (rotation
     about the vertical axis -- 0 keeps the camera centred behind the
     y=0 touchline), and `distance_scale` (camera distance from the
     pitch centre, as a multiple of the pitch length).
  2. A standard look-at rotation R and translation t give the world
     (pitch-plane, z=0) -> camera-space transform. Because z=0 for every
     point, the camera's 3rd rotation column never contributes, leaving
     a 3x3 "reduced" extrinsic matrix [r_x | r_y | t].
  3. A pinhole intrinsic matrix K (built from `fov_deg` and the target
     canvas size) completes the projection: H0 = K @ [r_x | r_y | t].
  4. H0 alone does not necessarily fill the canvas nicely, so its own
     projection of the four real pitch corners is used to fit ONE
     similarity transform (uniform scale + 2D translate -- never a
     second, independent per-axis stretch, which would distort the
     pitch's real proportions) that centres and pads that quadrilateral
     inside the canvas. The final H = (fit) @ H0.

`RadarCameraConfig` is intentionally small and reusable so both
dashboards share one camera model -- tune it once, both radars change
together.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RadarCameraConfig:
    elevation_deg: float = 46.0   # angle of the camera above the pitch plane (90 = straight overhead/flat-equivalent; lower = more oblique)
    yaw_deg: float = 0.0          # rotation about the vertical axis; 0 = camera centred behind the y=0 touchline, looking toward y=width
    fov_deg: float = 30.0         # vertical field of view -- the PRIMARY perspective-strength knob (larger = more dramatic foreshortening)
    distance_scale: float = 1.6   # camera distance from the pitch centre, as a multiple of pitch length_cm
    canvas_padding_px: int = 36


# FROZEN FINAL DEFAULT (2026-09-08, final-prototype round): a moderate,
# broadcast-tactical-board-style tilt. `elevation_deg=46` was chosen
# after a final A/B/C comparison at 42/46/48 (yaw/fov/distance_scale
# held fixed) across 4 representative frames (pressing: strongest-
# pressure, clear escape/lane; offside: pass/runner, strong behind-line
# threat): 42 left a visibly excess dark-green margin at the near/far
# edges of the radar panel on all 4 frames; 46 filled the panel
# efficiently while every far-side player/line/contour stayed exactly
# as readable; 48 offered no further meaningful improvement over 46.
# Sits well short of the user's own originally-suggested 25-35 range,
# consistent with the explicit priority "TACTICAL READABILITY > 3D
# LOOK": at 25-35 the far touchline compresses enough to make far-side
# players hard to separate on a ~120x70m pitch. `--radar-view
# perspective` still accepts camera overrides for further tuning
# without code changes.
DEFAULT_RADAR_CAMERA = RadarCameraConfig()


def _camera_position_and_target(pitch, camera):
    target = np.array([pitch.length_cm / 2.0, pitch.width_cm / 2.0, 0.0])
    elevation = math.radians(camera.elevation_deg)
    yaw = math.radians(camera.yaw_deg)
    distance = camera.distance_scale * pitch.length_cm
    horiz = distance * math.cos(elevation)
    cam = target + np.array([
        horiz * math.sin(yaw),
        -horiz * math.cos(yaw),
        distance * math.sin(elevation),
    ])
    return cam, target


def _look_at_rotation(cam_pos, target, world_up=np.array([0.0, 0.0, 1.0])):
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        world_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm
    # camera "down" (image v increases downward, matching standard
    # image-coordinate convention) so the pitch renders right-side-up.
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=0)


def _intrinsics(fov_deg, canvas_w, canvas_h):
    f = (canvas_h / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[f, 0.0, canvas_w / 2.0],
                      [0.0, f, canvas_h / 2.0],
                      [0.0, 0.0, 1.0]])


def _apply_homography(H, pts_xy):
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1))
    hom = np.hstack([pts, ones]) @ H.T
    w = hom[:, 2:3]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return hom[:, :2] / w


def compute_radar_homography(pitch, camera=DEFAULT_RADAR_CAMERA, canvas_w=1300, canvas_h=800, padding_px=None):
    """THE single shared pitch(cm) -> canvas(px) transform. Every radar
    object this round (players, lines, ball, pressure/threat rasters,
    pass vectors, escape arrows, pitch markings) must be projected
    through this SAME matrix -- never independently repositioned."""
    padding = camera.canvas_padding_px if padding_px is None else padding_px
    cam_pos, target = _camera_position_and_target(pitch, camera)
    R = _look_at_rotation(cam_pos, target)
    t = -R @ cam_pos
    m_reduced = np.column_stack([R[:, 0], R[:, 1], t])  # world (X,Y,1) -> camera-space (x,y,z), Z=0 term dropped
    K = _intrinsics(camera.fov_deg, canvas_w, canvas_h)
    h0 = K @ m_reduced

    corners = [(0.0, 0.0), (pitch.length_cm, 0.0), (pitch.length_cm, pitch.width_cm), (0.0, pitch.width_cm)]
    proj = _apply_homography(h0, corners)
    min_xy, max_xy = proj.min(axis=0), proj.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    avail_w, avail_h = canvas_w - 2 * padding, canvas_h - 2 * padding
    scale = min(avail_w / span[0], avail_h / span[1])
    offset = np.array([canvas_w / 2.0, canvas_h / 2.0]) - scale * (min_xy + max_xy) / 2.0
    fit = np.array([[scale, 0.0, offset[0]], [0.0, scale, offset[1]], [0.0, 0.0, 1.0]])
    return fit @ h0


def project_points(H, pts_cm):
    """(N,2) pitch-cm points -> (N,2) canvas-px points, via H."""
    return _apply_homography(H, pts_cm)


def project_point(H, x, y):
    return tuple(project_points(H, [(x, y)])[0])


def project_point_int(H, x, y):
    px, py = project_point(H, x, y)
    return int(round(px)), int(round(py))


def homogeneous_w(H, x, y):
    """The raw homogeneous denominator at (x,y) -- proportional to true
    camera-space depth for a real pinhole projection; used only for
    BOUNDED marker-size scaling, never for anything analytic."""
    v = H @ np.array([x, y, 1.0])
    return float(v[2])


def marker_scale_factor(H, x, y, pitch, min_scale=0.72, max_scale=1.28):
    """Bounded perspective size scaling (near = slightly larger, far =
    slightly smaller) -- clipped so far players never disappear and
    near players never dominate the canvas."""
    w_ref = homogeneous_w(H, pitch.length_cm / 2.0, pitch.width_cm / 2.0)
    w_pt = homogeneous_w(H, x, y)
    raw = w_ref / max(abs(w_pt), 1e-6)
    return float(np.clip(raw, min_scale, max_scale))


def draw_pitch_markings_perspective(H, pitch_config, canvas_w, canvas_h,
                                     background_bgr=(34, 100, 34), line_color=(230, 230, 230),
                                     line_thickness=2, point_radius=4, circle_samples=64):
    """Draws touchlines/halfway line/penalty+goal boxes/centre circle/
    penalty spots by projecting the SAME vertex/edge data
    `sports.annotators.soccer.draw_pitch` (the flat renderer) already
    uses -- never a hand-drawn screen-space approximation."""
    img = np.full((canvas_h, canvas_w, 3), background_bgr, dtype=np.uint8)
    verts_px = project_points(H, pitch_config.vertices)
    for a, b in pitch_config.edges:
        p0 = tuple(np.round(verts_px[a - 1]).astype(int))
        p1 = tuple(np.round(verts_px[b - 1]).astype(int))
        cv2.line(img, p0, p1, line_color, line_thickness, cv2.LINE_AA)
    cx, cy = pitch_config.length / 2.0, pitch_config.width / 2.0
    circle_pts = [(cx + pitch_config.centre_circle_radius * math.cos(t),
                   cy + pitch_config.centre_circle_radius * math.sin(t))
                  for t in np.linspace(0, 2 * math.pi, circle_samples)]
    circ_px = np.round(project_points(H, circle_pts)).astype(np.int32)
    cv2.polylines(img, [circ_px], True, line_color, line_thickness, cv2.LINE_AA)
    for spot in ((pitch_config.penalty_spot_distance, pitch_config.width / 2.0),
                 (pitch_config.length - pitch_config.penalty_spot_distance, pitch_config.width / 2.0)):
        p = project_point_int(H, *spot)
        cv2.circle(img, p, point_radius, line_color, -1, cv2.LINE_AA)
    return img


def warp_raster_perspective(field_bgr_u8, alpha_f32, H, grid_step_cm, canvas_w, canvas_h):
    """Warps a GRID-ARRAY-space raster (e.g. a pressure/threat field's
    own color+alpha arrays) directly to canvas pixels via the SAME
    composition pattern already used for broadcast-video pitch overlays
    elsewhere in this project (grid-index -> pitch-cm -> image-px, one
    matrix product, one `cv2.warpPerspective` -- never a circular/
    floating approximation)."""
    grid_to_pitch = np.array([[grid_step_cm, 0.0, 0.0], [0.0, grid_step_cm, 0.0], [0.0, 0.0, 1.0]])
    m = H @ grid_to_pitch
    warped_color = cv2.warpPerspective(field_bgr_u8, m, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_alpha = cv2.warpPerspective(alpha_f32, m, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return warped_color, warped_alpha


def warp_scalar_field_perspective(field_f32, H, grid_step_cm, canvas_w, canvas_h):
    """Same composed-homography warp as `warp_raster_perspective`, for a
    single-channel scalar field (e.g. for contour thresholding directly
    in target-canvas space, mirroring this project's own existing
    `draw_contours_video`-style pattern -- warp the raw field, THEN
    threshold+findContours in pixel space, never pre-render contours in
    grid space and try to reproject a polyline approximation)."""
    grid_to_pitch = np.array([[grid_step_cm, 0.0, 0.0], [0.0, grid_step_cm, 0.0], [0.0, 0.0, 1.0]])
    m = H @ grid_to_pitch
    return cv2.warpPerspective(field_f32.astype(np.float32), m, (canvas_w, canvas_h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def composite_wash(base_img_u8, warped_color_u8, warped_alpha_f32):
    a = warped_alpha_f32[..., None] if warped_alpha_f32.ndim == 2 else warped_alpha_f32
    out = base_img_u8.astype(np.float32) * (1 - a) + warped_color_u8.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)
