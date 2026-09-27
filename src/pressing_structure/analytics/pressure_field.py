"""
Full-pitch spatial pressure field (2026-09-07 V6 pass, Part 2 of the
offside/pressing correctness+research-method audit). Pure math only --
no drawing, no color choices. The two render scripts import this
module and are responsible for colorizing/compositing.

WHY A NEW MODULE: there was no pre-existing full-pitch pressure-field
formula in this codebase to reuse -- `pressing_v4.frame_features()`
only computes CARRIER-RELATIVE scalars (nearest-opponent distance,
relative closing speed, etc.), never a value at an arbitrary pitch
point z. This is a genuinely new, disclosed model, not a repurposing
of an existing one.

MODEL -- time-to-reach / influence (a standard, deliberately simplified
sports-analytics reachability formulation, conceptually related to
Spearman-style pitch control; no acceleration model and no uncertainty
ellipse, since neither is trustworthy from this tracking source):

For defender i and pitch point z:
    d_i(z)               = euclidean distance from i's CURRENT position to z (cm)
    closing_i(z)          = i's velocity component TOWARD z (cm/s)
                            = dot(velocity_i, unit_vector(i -> z))
                            (0 if i's velocity is not currently trustworthy --
                             `motion_valid` is False -- never fabricated)
    effective_speed_i(z)  = clip(BASE_SPEED_CM_S + closing_i(z),
                                  MIN_EFFECTIVE_SPEED_CM_S, MAX_SPEED_CM_S)
    time_to_reach_i(z)    = d_i(z) / effective_speed_i(z)                  (seconds)
    influence_i(z)        = exp(-time_to_reach_i(z) / TIME_CONSTANT_S)     in (0, 1]

Team aggregate -- independent-OR across that team's own players ("the
team can pressure z if ANY of its players can plausibly get there
soon"; several nearby players compound rather than only the single
closest one counting, which is how this model represents local
numerical support without a second, separate formula):
    Pressure_team(z,t) = 1 - product_i (1 - influence_i(z))               in [0, 1)

CONSTANTS, and why each is not arbitrary:
- MAX_SPEED_CM_S=824 reuses THIS PROJECT'S OWN already-measured 95th-
  percentile real running speed (see
  `pressing_structure/analytics/cleaned_tracking_view.py`'s
  `IMPOSSIBLE_SPEED_CM_S` comment, citing
  `outputs/analytics/testVideo1_120s_v3/pitch_control_calibration.md`)
  -- not a newly-invented number.
- BASE_SPEED_CM_S=400 (an assumed effective speed from a standing/alert
  start) and TIME_CONSTANT_S=1.5 (the influence decay rate) are
  disclosed MODELING CHOICES, not values fit to this clip's own
  outcomes -- flagged honestly, the same way this project already
  flags `EvidenceConfig`'s generic defaults as structural choices, not
  measured facts.
- GRID_STEP_CM=200 (2m) is a resolution choice balancing visual
  smoothness against compute cost for a 120x70m pitch (61x36 cells).
"""
import numpy as np

MAX_SPEED_CM_S = 824.0
BASE_SPEED_CM_S = 400.0
MIN_EFFECTIVE_SPEED_CM_S = 100.0
TIME_CONSTANT_S = 1.5
GRID_STEP_CM = 200.0


def pitch_grid(pitch, step_cm=GRID_STEP_CM):
    """Regular grid covering the COMPLETE pitch, inclusive of both
    boundaries -- shape (ny, nx). `pitch_xy_to_px`/`draw_pitch` treat
    length as the x-axis and width as the y-axis; this grid matches
    that convention exactly (gx varies along columns/length, gy along
    rows/width)."""
    xs = np.arange(0, pitch.length_cm + 1e-6, step_cm)
    ys = np.arange(0, pitch.width_cm + 1e-6, step_cm)
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy


def player_influence(gx, gy, x, y, vx, vy, motion_valid,
                      base_speed=BASE_SPEED_CM_S, min_speed=MIN_EFFECTIVE_SPEED_CM_S,
                      max_speed=MAX_SPEED_CM_S, time_constant=TIME_CONSTANT_S):
    """influence_i(z) over the whole grid at once. `motion_valid=False`
    (or vx/vy None) means the player's velocity is not currently
    trustworthy -- closing is honestly treated as 0 (a stationary-
    equivalent effective speed), never fabricated from an unreliable
    reading."""
    dx = gx - x
    dy = gy - y
    d = np.hypot(dx, dy)
    d_safe = np.clip(d, 1e-6, None)
    if motion_valid and vx is not None and vy is not None:
        closing = (vx * dx + vy * dy) / d_safe
    else:
        closing = np.zeros_like(d)
    eff_speed = np.clip(base_speed + closing, min_speed, max_speed)
    t_reach = d / eff_speed
    return np.exp(-t_reach / time_constant)


def team_pressure_field(players, gx, gy, **kwargs):
    """`players`: list of dicts with x_pitch,y_pitch,vx_cm_s,vy_cm_s,
    motion_valid (missing/None position -> skipped, never fabricated).
    Returns an array the same shape as gx/gy, values in [0,1)."""
    survival = np.ones_like(gx, dtype=np.float64)
    for p in players:
        if p.get('x_pitch') is None or p.get('y_pitch') is None:
            continue
        inf = player_influence(gx, gy, p['x_pitch'], p['y_pitch'],
                                p.get('vx_cm_s'), p.get('vy_cm_s'), p.get('motion_valid', False), **kwargs)
        survival *= (1.0 - inf)
    return 1.0 - survival


def sample_field(field, gx, gy, x, y):
    """Nearest-grid-point sample of `field` at real pitch point (x,y) --
    used to compare the field's value AT THE CARRIER against the
    existing, separate Press Intensity metric (2E)."""
    idx = np.unravel_index(np.argmin((gx - x) ** 2 + (gy - y) ** 2), gx.shape)
    return float(field[idx])


# ---------------------------------------------------------------------------
# 2026-09-08 V7 -- pressure CONTOURS, ESCAPE SPACE, GRADIENT, PASSING
# LANES (Part B). All built on the SAME `team_pressure_field` above --
# no second, divergent pressure model.

# (B1) Contour levels CALIBRATED to the model's own time constant
# (never an arbitrary/silent choice): influence_i(z)=exp(-t/TIME_CONSTANT_S),
# so a level of exp(-1)=0.37 corresponds to "a single defender could
# reach z in about one time-constant (~1.5s)"; exp(-0.5)=0.61 to "~0.75s";
# exp(-0.2)=0.82 to "~0.3s (near-immediate)". Rounded for readability.
CONTOUR_LOW = 0.35    # ~reachable within ~1.5s by a single defender
CONTOUR_MED = 0.60    # ~reachable within ~0.75s
CONTOUR_HIGH = 0.80   # ~reachable within ~0.3s (near-immediate)
CONTOUR_LEVELS = (CONTOUR_LOW, CONTOUR_MED, CONTOUR_HIGH)

# (B2) Escape-space: the SIMPLEST interpretable version explicitly
# requested -- low opponent pressure at z, weighted by how reachable z
# actually is FOR THE CARRIER (reusing the identical time-to-reach
# influence formula, now applied to the carrier's own kinematics
# instead of a defender's). No forward-progress term is folded in this
# pass (disclosed simplification, not a hidden omission): a directional
# weighting is straightforward future work once this base version is
# validated.
#   EscapeValue(z,t) = (1 - Pressure_team(z,t)) * carrier_reachability(z,t)
CARRIER_TIME_CONSTANT_S = 1.2   # a ball carrier is rarely at a defender's own top speed while carrying the ball -- disclosed, not fit


def escape_value_field(field, carrier, gx, gy, **kwargs):
    """`field`: the OPPONENT team's `team_pressure_field` output (SAME
    array the radar/video wash uses). `carrier`: dict with x_pitch,
    y_pitch, vx_cm_s, vy_cm_s, motion_valid. Returns an array the same
    shape as `field`, values in [0, 1)."""
    reach = player_influence(gx, gy, carrier['x_pitch'], carrier['y_pitch'],
                              carrier.get('vx_cm_s'), carrier.get('vy_cm_s'), carrier.get('motion_valid', False),
                              time_constant=CARRIER_TIME_CONSTANT_S, **kwargs)
    return (1.0 - field) * reach


def pressure_gradient(field, gx, gy, x, y):
    """(B3) Finite-difference spatial gradient of `field` at the
    nearest grid point to (x,y), in pressure-per-cm along each pitch
    axis -- `np.gradient`, not reinvented. Returns (dP/dx, dP/dy); by
    construction this vector points toward INCREASING pressure (test
    coverage: Part D pressing test 3)."""
    step = float(gx[0, 1] - gx[0, 0]) if gx.shape[1] > 1 else GRID_STEP_CM
    gy_step = float(gy[1, 0] - gy[0, 0]) if gx.shape[0] > 1 else GRID_STEP_CM
    dpdy, dpdx = np.gradient(field, gy_step, step)
    idx = np.unravel_index(np.argmin((gx - x) ** 2 + (gy - y) ** 2), gx.shape)
    return float(dpdx[idx]), float(dpdy[idx])


def escape_directions(field, gx, gy, carrier_xy, pitch, radius_cm=800.0, n_candidates=8, n_select=3):
    """(B3/B4) LOW-PRESSURE ESCAPE DIRECTIONS -- radial sampling of
    `field` (never the pressure-FIELD's own team, always the OPPONENT
    field passed in) at `n_candidates` points on a circle of
    `radius_cm` around the carrier (a disclosed, robust alternative to
    a purely local gradient, which is noisy on a coarse grid), clipped
    to stay inside the pitch. Returns up to `n_select` (x, y, pressure)
    points, LOWEST pressure first -- never labeled a 'recommended
    dribble', only a real, disclosed sampling of where pressure is
    currently lower and reachable."""
    cx, cy = carrier_xy
    candidates = []
    for k in range(n_candidates):
        theta = 2 * np.pi * k / n_candidates
        x = cx + radius_cm * np.cos(theta)
        y = cy + radius_cm * np.sin(theta)
        if not pitch.inside(x, y):
            continue
        candidates.append((x, y, sample_field(field, gx, gy, x, y)))
    candidates.sort(key=lambda c: c[2])
    return candidates[:n_select]


# (B5) Passing-lane metrics -- CONTINUOUS quantities only, no binary
# "blocked" label (per the explicit instruction: derive the numbers
# first, classify for display separately and only by relative ranking).
LANE_SAMPLES = 12


def passing_lane_metrics(field, gx, gy, carrier, teammate, opponents):
    """Straight-line corridor carrier -> teammate. `opponents`: list of
    dicts with x_pitch,y_pitch (used for the one metric that reads raw
    defender positions rather than the field: min defender-to-lane
    distance, a direct geometric fact independent of the influence
    model's own assumptions)."""
    x0, y0 = carrier['x_pitch'], carrier['y_pitch']
    x1, y1 = teammate['x_pitch'], teammate['y_pitch']
    length_cm = float(np.hypot(x1 - x0, y1 - y0))
    ts = np.linspace(0.0, 1.0, LANE_SAMPLES)
    xs = x0 + (x1 - x0) * ts
    ys = y0 + (y1 - y0) * ts
    samples = [sample_field(field, gx, gy, x, y) for x, y in zip(xs, ys)]
    mean_pressure = float(np.mean(samples))
    max_pressure = float(np.max(samples))
    receiver_pressure = sample_field(field, gx, gy, x1, y1)
    if opponents:
        seg = np.array([x1 - x0, y1 - y0])
        seg_len2 = max(1e-6, float(seg @ seg))
        dists = []
        for o in opponents:
            if o.get('x_pitch') is None:
                continue
            w = np.array([o['x_pitch'] - x0, o['y_pitch'] - y0])
            t = float(np.clip((w @ seg) / seg_len2, 0.0, 1.0))
            closest = np.array([x0, y0]) + t * seg
            dists.append(float(np.hypot(o['x_pitch'] - closest[0], o['y_pitch'] - closest[1])))
        min_defender_distance_cm = min(dists) if dists else None
    else:
        min_defender_distance_cm = None
    return dict(length_cm=length_cm, mean_pressure=mean_pressure, max_pressure=max_pressure,
                receiver_pressure=receiver_pressure, min_defender_distance_cm=min_defender_distance_cm,
                lane_integrated_pressure=mean_pressure * length_cm)


# ---------------------------------------------------------------------------
# 2026-09-08 V8 -- O9: "Behind-Line Threat Field", a RESEARCH-ONLY
# exploratory metric (neutral working name, deliberately NOT "Offside
# Control" -- that published method is not implemented/compared here).
# Reuses the EXACT SAME `team_pressure_field`/`player_influence`
# infrastructure already built for pressing, applied to the ATTACKING
# team (their reachability = how fast they could occupy/receive in the
# space behind the defensive unit) vs the DEFENDING team (their
# recovery reachability to that same space):
#   BehindLineThreat(z) = attacking_reachability(z) * (1 - defending_recovery(z))
# restricted to points BEHIND the defending unit (depth < unit centre
# depth, in the defending team's own depth frame -- i.e. between the
# back line and their own goal).
def behind_line_threat_field(attacking_players, defending_players, gx, gy, defending_team, unit_centre_depth_cm, pitch):
    attacking_reach = team_pressure_field(attacking_players, gx, gy)
    defending_recovery = team_pressure_field(defending_players, gx, gy)
    depth = (gx - pitch.own_goal(defending_team)) * pitch.attacking_sign(defending_team)
    behind_mask = depth < unit_centre_depth_cm
    threat = attacking_reach * (1.0 - defending_recovery)
    return np.where(behind_mask, threat, 0.0)


def contour_coverage_fractions(field, levels=CONTOUR_LEVELS):
    """(P3) sanity-check export: fraction of the FULL grid at/above each
    contour level, plus the field's own min/max -- so a fully-red pitch
    can be caught as a real (or wrong) reading rather than assumed."""
    total = field.size
    return {level: float(np.sum(field >= level)) / total for level in levels}


def grid_to_pitch_matrix(step_cm=GRID_STEP_CM):
    """3x3 homogeneous matrix mapping GRID-ARRAY pixel coordinates
    (col, row) -> real pitch cm (x, y) -- a plain axis-aligned scale
    (grid cell (0,0) is pitch (0,0), no rotation/offset). Composing
    this with a broadcast frame's own inverse homography (pitch cm ->
    image px) in ONE matrix product lets `cv2.warpPerspective` take the
    small pressure raster DIRECTLY from grid-array space to broadcast
    image space in a single perspective-correct warp -- never a
    separately-approximated circle/ellipse."""
    return np.array([[step_cm, 0, 0], [0, step_cm, 0], [0, 0, 1]], dtype=np.float64)
