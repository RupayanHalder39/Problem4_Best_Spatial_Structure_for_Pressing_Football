"""
Ball-perception Stage 2/3/4/5, hardened after a user visual review of
`dashboard_demo_ballfixed.mp4` found remaining abrupt ball teleports. A
real, lightweight, single-object TEMPORAL ball tracker.

IMPORTANT DISTINCTION (do not confuse these two): `sports/common/
ball.py::BallTracker` (upstream) is NOT a temporal tracker despite the
name -- it only disambiguates BETWEEN CANDIDATES WITHIN THE SAME FRAME
by picking whichever is closest to a running centroid of past picks; it
has no prediction step, no missing-frame handling, and no concept of
"reject this frame's only candidate because it's implausible." This
module is the actual temporal tracker: it predicts where the ball should
be, scores each frame's raw candidates against that prediction, and
explicitly marks frames with no plausible candidate as missing rather
than guessing.

ROOT CAUSE OF THE REMAINING TELEPORTS (found via `scripts/
diagnose_ball_jumps.py` + a direct homography cross-check, documented in
full in `outputs/analytics/testVideo1_20s/ball_tracking_summary.md` Part
2/3): the ORIGINAL tracker scored candidate plausibility purely in IMAGE
SPACE (pixels). A fixed-point homography cross-check on the worst
transitions ruled out homography discontinuity as the cause (re-applying
frame N's transform vs. frame N+1's transform to the SAME image point
differed by only 1-31cm in every case checked) -- the transforms
themselves are fine. The real problem is PERSPECTIVE: the same pixel
displacement corresponds to a wildly different real ground distance
depending on WHERE in the image it happens (near the camera vs. near the
far touchline/goal), so a single fixed `max_displacement_px_per_frame`
threshold that is appropriately strict near the bottom of the frame is
far too lenient near the top -- letting through candidates 500cm+ apart
in a single 1/30s frame (implying 100+ m/s, physically impossible for a
real ball). Fix: score plausibility in PITCH SPACE (a genuine physical
speed bound, position-independent) whenever a per-frame homography is
available, falling back to the image-space check only for the ~29% of
frames where it is not (see `analytics/homography_replay.py`).

Two-pass design (unchanged from the original Stage 2/3/4/5 build):
  Pass 1 (this module): STRICTLY CAUSAL -- frame-by-frame, only ever
    looks at past accepted positions to predict the current frame, never
    at future frames.
  Pass 2 (`analytics/temporal_utils.py::fill_short_gaps`, reused as-is):
    fills short gaps between two ACCEPTED observations with linear
    interpolation.

Never fabricates a position for a genuinely long gap, and every output
row is tagged `is_observed`/`track_state` so provenance is never hidden.
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import polars as pl


class TrackState(str, Enum):
    TRACKED = "TRACKED"                    # observed this frame or the frame just before
    TEMPORARILY_MISSING = "TEMPORARILY_MISSING"  # short gap, still predicting from velocity
    REACQUIRING = "REACQUIRING"            # longer gap -- requires 2 consistent candidates before resuming
    RESET = "RESET"                        # gap exceeded the reset threshold -- momentum discarded


@dataclass
class BallTrackerConfig:
    max_displacement_px_per_frame: float = 90.0
    # ^ IMAGE-SPACE fallback bound, used only for frames where no
    # per-frame homography is available to score in pitch space (see
    # module docstring -- this alone is known to be an imperfect,
    # perspective-blind proxy, kept only as a fallback).
    max_speed_cm_s: float = 3500.0
    # ^ PITCH-SPACE bound: 35 m/s, a generous physical ceiling for a
    # struck football (real sustained speeds in this project's own
    # cleanest, manually-verified pass segments average 9-16 m/s; even
    # world-record strikes are well under 40 m/s). This is the PRIMARY
    # plausibility gate whenever a homography is available for both the
    # reference position and the candidate -- derived from measured
    # evidence in `scripts/diagnose_ball_jumps.py`'s output, not
    # guessed: the worst rejected transitions on testVideo1_20s implied
    # 82-156 m/s, nowhere close to this bound, while every manually-
    # confirmed real trajectory segment stayed under it.
    min_confidence: float = 0.20
    min_confidence_to_seed: float = 0.50
    # ^ see original rationale (Part 4 of ball_tracking_summary.md): a
    # low-confidence lone detection must not seed the track anchor when
    # there is no prediction yet to check its position against.
    velocity_smoothing_alpha: float = 0.5
    max_missing_frames_before_reacquire_check: int = 3
    # ^ TEMPORARILY_MISSING -> REACQUIRING transition point. Below this,
    # a returning candidate near the velocity-based prediction is
    # accepted immediately (normal short-gap continuation). At or above
    # it, the prediction is no longer trusted enough to gate a single
    # candidate -- see `reacquire_confirmation_radius_cm` below.
    reacquire_confirmation_radius_cm: float = 300.0
    # ^ 3m: after a long gap, TWO CONSECUTIVE candidate frames must agree
    # within this pitch-space radius (or, if pitch space is unavailable,
    # an equivalent image-space radius) before the track resumes. This
    # is what stops a single isolated false positive from teleporting
    # the ball after a gap (Part 5/6's explicit requirement) -- one
    # detection is never enough evidence on its own to re-anchor.
    reacquire_confirmation_radius_px: float = 60.0
    max_missing_frames_before_reset: int = 15
    direction_change_penalty_deg: float = 90.0
    # ^ soft tie-breaker only (never a hard reject): among candidates
    # that already pass the physical speed gate, one implying a travel
    # direction within this many degrees of the recent velocity is
    # preferred over one implying a sharp reversal, when confidence is
    # otherwise close. Real ball trajectories rarely reverse instantly.


@dataclass
class _TrackState:
    x: float = None
    y: float = None
    vx: float = 0.0
    vy: float = 0.0
    last_observed_frame: int = None
    consecutive_missing: int = 0
    recent_hits: list = field(default_factory=list)
    pending: dict = None  # a not-yet-confirmed reacquisition candidate, or None


def _pitch_xy(transformers: dict, frame: int, x: float, y: float):
    if transformers is None or x is None:
        return None
    transformer = transformers.get(frame)
    if transformer is None:
        return None
    pt = np.array([[x, y]], dtype=np.float32)
    p = transformer.transform_points(pt)[0]
    return float(p[0]), float(p[1])


def _direction_deg(dx, dy):
    if dx is None or (dx == 0 and dy == 0):
        return None
    return float(np.degrees(np.arctan2(dy, dx)))


def _angle_diff(a, b):
    if a is None or b is None:
        return 0.0
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _score_candidates(cands: list[dict], frame: int, ref_frame: int, ref_x, ref_y,
                       pred_x, pred_y, dt_frames: float, cfg: BallTrackerConfig,
                       transformers: dict, prev_vx: float, prev_vy: float) -> dict | None:
    """Returns the best plausible candidate (with 'distance_px' and
    'pitch_ok' keys) or None. Plausibility is judged in PITCH SPACE
    (a physical max-speed bound) whenever a homography is available for
    both the reference frame and the candidate frame; otherwise falls
    back to the image-space displacement bound. See module docstring for
    why pitch space is the primary, more correct check."""
    if not cands:
        return None

    if pred_x is None:  # first sighting / post-reset: no position to check against
        seedable = [c for c in cands if c["confidence"] >= cfg.min_confidence_to_seed]
        if not seedable:
            return None
        best = max(seedable, key=lambda c: c["confidence"])
        return {**best, "distance_px": 0.0, "pitch_ok": True}

    ref_pitch = _pitch_xy(transformers, ref_frame, ref_x, ref_y)
    dt_sec = dt_frames / 30.0  # nominal; exact fps doesn't materially change a 35 m/s bound
    scored = []
    for c in cands:
        d_px = float(np.hypot(c["x_image"] - pred_x, c["y_image"] - pred_y))
        cand_pitch = _pitch_xy(transformers, frame, c["x_image"], c["y_image"])
        if ref_pitch is not None and cand_pitch is not None:
            d_cm = float(np.hypot(cand_pitch[0] - ref_pitch[0], cand_pitch[1] - ref_pitch[1]))
            speed_cm_s = d_cm / dt_sec if dt_sec > 0 else d_cm
            if speed_cm_s > cfg.max_speed_cm_s:
                continue  # physically implausible -- reject regardless of image-space distance
            scored.append({**c, "distance_px": d_px, "pitch_ok": True, "pitch_speed_cm_s": speed_cm_s})
        else:
            # no homography for one/both points this frame -- fall back
            # to the (known-imperfect) image-space bound, scaled for gap length
            if d_px <= cfg.max_displacement_px_per_frame * dt_frames:
                scored.append({**c, "distance_px": d_px, "pitch_ok": False, "pitch_speed_cm_s": None})

    if not scored:
        return None

    prev_dir = _direction_deg(prev_vx, prev_vy) if (abs(prev_vx) + abs(prev_vy)) > 0.5 else None

    def sort_key(c):
        cand_dir = _direction_deg(c["x_image"] - ref_x, c["y_image"] - ref_y)
        turn = _angle_diff(prev_dir, cand_dir) if prev_dir is not None else 0.0
        penalty = 1 if turn > cfg.direction_change_penalty_deg else 0
        return (penalty, -c["confidence"], c["distance_px"])

    scored.sort(key=sort_key)
    return scored[0]


def track_ball_image_space(candidates_df: pl.DataFrame, frame_min: int, frame_max: int,
                            config: BallTrackerConfig = None, transformers: dict = None) -> pl.DataFrame:
    """candidates_df: raw per-frame candidate detections, columns frame,
    x_image, y_image, confidence. `transformers`: optional {frame:
    ViewTransformer or None} from `analytics/homography_replay.py` --
    strongly recommended; without it the tracker falls back entirely to
    the known-imperfect image-space bound. Returns one row per frame in
    [frame_min, frame_max]: frame, x_image, y_image, confidence,
    is_observed, track_state, track_quality, n_candidates_this_frame,
    distance_from_prediction, pitch_gated (whether the accepted point
    was validated in pitch space or only by the image-space fallback)."""
    cfg = config or BallTrackerConfig()
    by_frame: dict[int, list[dict]] = {}
    for r in candidates_df.filter(pl.col("confidence") >= cfg.min_confidence).to_dicts():
        by_frame.setdefault(r["frame"], []).append(
            {"x_image": r["x_image"], "y_image": r["y_image"], "confidence": r["confidence"]})

    state = _TrackState()
    rows = []

    def accept(f, cand, track_state, is_restart=False):
        if state.x is not None and not is_restart:
            dt_frames = max(f - (state.last_observed_frame or f), 1)
            new_vx = (cand["x_image"] - state.x) / dt_frames
            new_vy = (cand["y_image"] - state.y) / dt_frames
            state.vx = cfg.velocity_smoothing_alpha * new_vx + (1 - cfg.velocity_smoothing_alpha) * state.vx
            state.vy = cfg.velocity_smoothing_alpha * new_vy + (1 - cfg.velocity_smoothing_alpha) * state.vy
        else:
            state.vx = state.vy = 0.0
        state.x, state.y = cand["x_image"], cand["y_image"]
        state.last_observed_frame = f
        state.consecutive_missing = 0
        state.pending = None
        state.recent_hits.append(1)
        return {"frame": f, "x_image": state.x, "y_image": state.y, "confidence": cand["confidence"],
                "is_observed": True, "track_state": track_state.value,
                "n_candidates_this_frame": len(by_frame.get(f, [])),
                "distance_from_prediction": cand.get("distance_px", 0.0),
                "pitch_gated": cand.get("pitch_ok", False),
                # True whenever this row is the FIRST point of a fresh
                # anchor (a genuine track discontinuity) rather than a
                # smooth continuation of the previous position -- used
                # downstream (analytics/ball_motion.py) to force a new
                # trajectory segment here regardless of how small the raw
                # frame gap happens to be (Part 6's explicit requirement:
                # never draw a line from the old position to a freshly
                # reacquired one, even if the gap is short in frame-count
                # terms).
                "is_track_restart": is_restart}

    for f in range(frame_min, frame_max + 1):
        cands = by_frame.get(f, [])

        if state.x is None:
            track_state = TrackState.RESET
        elif state.consecutive_missing == 0:
            track_state = TrackState.TRACKED
        elif state.consecutive_missing < cfg.max_missing_frames_before_reacquire_check:
            track_state = TrackState.TEMPORARILY_MISSING
        elif state.consecutive_missing <= cfg.max_missing_frames_before_reset:
            track_state = TrackState.REACQUIRING
        else:
            track_state = TrackState.RESET

        if track_state in (TrackState.TRACKED, TrackState.TEMPORARILY_MISSING):
            dt_frames = state.consecutive_missing + 1
            pred_x = state.x + state.vx * dt_frames
            pred_y = state.y + state.vy * dt_frames
            best = _score_candidates(cands, f, state.last_observed_frame, state.x, state.y,
                                      pred_x, pred_y, dt_frames, cfg, transformers, state.vx, state.vy)
            if best is not None:
                rows.append(accept(f, best, TrackState.TRACKED))
                continue

        elif track_state == TrackState.RESET:
            # No prior position to predict from (very first sighting, or
            # momentum discarded after a very long absence) -- position
            # can't reject a bad candidate, so a high confidence bar does
            # instead (same rationale as the original min_confidence_to_seed
            # fix: see module/config docstrings).
            best = _score_candidates(cands, f, f, None, None, None, None, 1, cfg, transformers, 0.0, 0.0)
            if best is not None:
                rows.append(accept(f, best, TrackState.RESET, is_restart=True))
                continue

        elif track_state == TrackState.REACQUIRING:
            # Require 2 consecutive, mutually consistent candidate frames
            # before resuming the track -- one isolated detection after a
            # long gap is not enough evidence (Part 5/6's explicit rule).
            # No velocity-based prediction is trusted this far out; a
            # candidate qualifies purely on confidence, then must be
            # confirmed by the NEXT frame's candidate landing within
            # `reacquire_confirmation_radius_*` of it.
            best_this_frame = max(cands, key=lambda c: c["confidence"]) if cands else None
            if best_this_frame is not None and best_this_frame["confidence"] < cfg.min_confidence_to_seed:
                best_this_frame = None

            if state.pending is not None and best_this_frame is not None:
                p = state.pending
                dt_confirm = max(f - p["frame"], 1) / 30.0
                d_px = float(np.hypot(best_this_frame["x_image"] - p["x_image"],
                                       best_this_frame["y_image"] - p["y_image"]))
                p_pitch = _pitch_xy(transformers, p["frame"], p["x_image"], p["y_image"])
                c_pitch = _pitch_xy(transformers, f, best_this_frame["x_image"], best_this_frame["y_image"])
                confirmed = False
                if p_pitch is not None and c_pitch is not None:
                    d_cm = float(np.hypot(c_pitch[0] - p_pitch[0], c_pitch[1] - p_pitch[1]))
                    # Confirmation must satisfy BOTH a fixed spatial
                    # radius (two candidates for "the same real ball"
                    # shouldn't be far apart at all) AND the same physical
                    # speed bound used everywhere else -- a fixed radius
                    # alone is not time-aware, so if the two confirming
                    # frames happen to be only 1 frame apart, 300cm in
                    # 1/30s would itself imply an impossible ~90 m/s.
                    confirmed = (d_cm <= cfg.reacquire_confirmation_radius_cm and
                                 d_cm / dt_confirm <= cfg.max_speed_cm_s)
                else:
                    confirmed = d_px <= cfg.reacquire_confirmation_radius_px
                if confirmed:
                    # Retroactively accept the pending frame too, then this one.
                    # Both are marked is_track_restart on the FIRST point only --
                    # that's where the new segment must begin; the second point
                    # is a normal continuation of the freshly-established anchor.
                    rows.append({"frame": p["frame"], "x_image": p["x_image"], "y_image": p["y_image"],
                                 "confidence": p["confidence"], "is_observed": True,
                                 "track_state": TrackState.REACQUIRING.value,
                                 "n_candidates_this_frame": len(by_frame.get(p["frame"], [])),
                                 "distance_from_prediction": None, "pitch_gated": p_pitch is not None,
                                 "is_track_restart": True})
                    state.x, state.y = p["x_image"], p["y_image"]
                    state.last_observed_frame = p["frame"]
                    state.consecutive_missing = 0
                    state.vx = state.vy = 0.0
                    rows.append(accept(f, best_this_frame, TrackState.REACQUIRING, is_restart=False))
                    continue
            state.pending = {**best_this_frame, "frame": f} if best_this_frame is not None else None

        state.consecutive_missing += 1
        state.recent_hits.append(0)
        rows.append({"frame": f, "x_image": None, "y_image": None, "confidence": None,
                     "is_observed": False, "track_state": track_state.value,
                     "n_candidates_this_frame": len(cands), "distance_from_prediction": None,
                     "pitch_gated": False, "is_track_restart": False})

    quality_window = []
    for r in rows:
        quality_window.append(1 if r["is_observed"] else 0)
        quality_window = quality_window[-20:]
        r["track_quality"] = round(sum(quality_window) / len(quality_window), 2)

    return pl.DataFrame(rows)
