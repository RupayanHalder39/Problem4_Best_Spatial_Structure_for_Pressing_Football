"""
Ball-perception Stage 7: pitch-mapping support that reuses the EXACT SAME
per-frame homography decision process `scripts/export_tracking.py` used
to produce `tracking.parquet`'s `x_pitch`/`y_pitch` columns for players --
never a second, independently-fit homography.

Why a "replay" is needed at all: no per-frame homography matrix is
persisted anywhere in the export (only the resulting x_pitch/y_pitch for
the single anchor actually used is saved -- see Phase 7's `scripts/
diagnose_anchor.py` docstring for the same situation solved the same way
for player anchors). The per-frame transformer depends ONLY on that
frame's pitch-keypoint detection plus the gate's rolling internal state
(itself only a function of the sequence of candidate transformers seen so
far) -- NOT on player detections, team classification, or ByteTrack. So
re-running just the pitch-keypoint model, `build_transformer()`, and
`HomographyConsistencyGate` in the same order with the same settings
reproduces the identical accept/reject sequence and transform
coefficients. This was verified exactly reproducible (0cm delta across
every tested frame/track) in Phase 7's `diagnose_anchor.py` work.

This module loads ONLY the pitch-keypoint model (`data/football-pitch-
detection.pt`) -- never the player detector, ball detector, or team
classifier -- so it is much cheaper than a full export re-run.
"""
import os
import sys

import cv2
import numpy as np
import polars as pl
import supervision as sv
from ultralytics import YOLO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
SOCCER_EXAMPLE_DIR = os.path.join(REPO_ROOT, "external", "sports", "examples", "soccer")
for p in (SCRIPTS_DIR, SOCCER_EXAMPLE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


def replay_transformers(source_video_path: str, device: str = "mps",
                         frame_min: int = 0, frame_max: int = None,
                         min_keypoint_confidence: float = None) -> dict:
    """Returns {frame_no: ViewTransformer or None}, reproducing
    export_tracking.py's exact per-frame accept/reject/gate-reuse
    sequence. Defaults (min_keypoint_confidence, gate on/off,
    gate_on_reject) match export_tracking.py's own defaults, which is
    what tracking.parquet was actually built with (confirmed against
    outputs/tracking/<video>/metadata.json's homography_fix block --
    pass min_keypoint_confidence explicitly if a given export used a
    non-default value)."""
    import main as radar_main
    from export_tracking import (DEFAULT_MIN_KEYPOINT_CONFIDENCE, HomographyConsistencyGate,
                                  build_transformer)

    min_conf = min_keypoint_confidence if min_keypoint_confidence is not None else DEFAULT_MIN_KEYPOINT_CONFIDENCE

    pitch_model = YOLO(radar_main.PITCH_DETECTION_MODEL_PATH).to(device=device)

    cap = cv2.VideoCapture(source_video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_max = frame_max if frame_max is not None else total_frames - 1

    test_points = np.array([
        [x, y]
        for x in (w * 0.05, w * 0.25, w * 0.5, w * 0.75, w * 0.95)
        for y in (h * 0.05, h * 0.3, h * 0.5, h * 0.7, h * 0.95)
    ], dtype=np.float32)
    gate = HomographyConsistencyGate(test_points=test_points, on_reject="null")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    transformers = {}
    f = 0
    while f <= frame_max:
        ret, frame = cap.read()
        if not ret:
            break
        # Always run pitch detection + advance the gate's rolling state
        # from frame 0, even if frame_min > 0 -- the gate's accept/reject
        # decision for frame_min depends on the transform history built
        # up before it, so skipping ahead would NOT reproduce the same
        # sequence export_tracking.py originally saw.
        pitch_result = pitch_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
        candidate = build_transformer(keypoints, min_confidence=min_conf)
        transformer, _was_rejected = gate.evaluate(candidate)
        if f >= frame_min:
            transformers[f] = transformer
        f += 1
    cap.release()
    return transformers


def apply_transformers_to_points(transformers: dict, frame_xy: list) -> list:
    """frame_xy: list of (frame, x_image, y_image) (x_image/y_image may
    be None -- passed through as (None, None)). Returns a list of
    (x_pitch, y_pitch), null for frames with no accepted transformer or
    no input point -- never fabricated."""
    out = []
    for frame, x, y in frame_xy:
        if x is None or y is None:
            out.append((None, None))
            continue
        transformer = transformers.get(frame)
        if transformer is None:
            out.append((None, None))
            continue
        pt = np.array([[x, y]], dtype=np.float32)
        pitch_xy = transformer.transform_points(pt)[0]
        out.append((float(pitch_xy[0]), float(pitch_xy[1])))
    return out


def verify_against_export(transformers: dict, tracking_parquet_path: str, n_check: int = 200) -> dict:
    """Sanity check: re-transform a sample of already-exported player
    BOTTOM_CENTER rows using the replayed transformers and compare
    against the export's own saved x_pitch/y_pitch. Should match within
    noise (Phase 7 found exactly 0cm delta) -- if it doesn't, something
    about the replay's settings has drifted from the original export and
    must be fixed before trusting any ball pitch-mapping built on it."""
    df = pl.read_parquet(tracking_parquet_path)
    sample = df.filter(pl.col("x_pitch").is_not_null()).sample(n=min(n_check, df.height), seed=0)
    deltas = []
    for r in sample.to_dicts():
        transformer = transformers.get(r["frame"])
        if transformer is None:
            continue
        pt = np.array([[r["x_image"], r["y_image"]]], dtype=np.float32)
        recomputed = transformer.transform_points(pt)[0]
        d = float(np.hypot(recomputed[0] - r["x_pitch"], recomputed[1] - r["y_pitch"]))
        deltas.append(d)
    deltas = np.array(deltas) if deltas else np.array([0.0])
    return {"n_checked": len(deltas), "mean_delta_cm": float(deltas.mean()),
            "max_delta_cm": float(deltas.max())}
