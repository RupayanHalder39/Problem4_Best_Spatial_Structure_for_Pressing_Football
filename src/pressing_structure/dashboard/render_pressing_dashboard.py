"""
Pressing-structure dashboard renderer (Project 1). NEW script, NEW
project folder -- does not touch, import-modify, or overwrite anything
in the existing PhD `scripts/`/`visualization/` packages; it IMPORS
existing, validated pieces (draw_pitch, ViewTransformer inverse-warp
technique already used by `visualization/voronoi.py`, `analytics.
pitch_control`) rather than re-deriving them.

Usage:
    external/sports/.venv/bin/python pressing_structure/dashboard/render_pressing_dashboard.py \\
        --start_sec 0 --end_sec 25 \\
        --out_path pressing_structure/outputs/pressing_dashboard.mp4
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import polars as pl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PARENT = os.path.dirname(PROJECT_ROOT)
sys.path[:0] = [PROJECT_ROOT, PROJECT_PARENT]

from sports.annotators.soccer import draw_pitch  # noqa: E402
from sports.configs.soccer import SoccerPitchConfiguration  # noqa: E402

from legacy_analytics.pitch_control import compute_pitch_control_grid  # noqa: E402
from pressing_structure.analytics.cleaned_tracking_view import build_cleaned_view, load_ball_view  # noqa: E402
from pressing_structure.analytics.pressing import (LANE_BLOCK_DISTANCE_CM, PRESS_TRIGGER_CLOSING_SPEED_CM_S,  # noqa: E402
                                                     PRESS_TRIGGER_RADIUS_CM, PRESSER_RADII_CM, SUPPORT_RADIUS_CM,
                                                     _passing_lane_open, build_possession, build_pressing_features,
                                                     classify_press_outcomes, detect_pressing_events)

CONFIG = SoccerPitchConfiguration()
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEAM_COLOR = {0: (147, 20, 255), 1: (255, 191, 0)}
CARRIER_COLOR = (0, 255, 255)
PRESSER_COLOR = (0, 0, 255)


def _pitch_xy_to_px(x, y, scale=0.1, padding=50):
    return int(x * scale) + padding, int(y * scale) + padding


def _pitch_circle_points(cx, cy, r_cm, n=32):
    ang = np.linspace(0, 2 * np.pi, n)
    return [(cx + r_cm * np.cos(a), cy + r_cm * np.sin(a)) for a in ang]


def _draw_pitch_circle_on_image(frame, transformer, cx, cy, r_cm, color, thickness=2):
    if transformer is None:
        return frame
    try:
        inv_m = np.linalg.inv(transformer.m)
    except np.linalg.LinAlgError:
        return frame
    pts = np.array(_pitch_circle_points(cx, cy, r_cm), dtype=np.float32).reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(pts, inv_m).reshape(-1, 2)
    img_pts = np.round(img_pts).astype(np.int32)
    cv2.polylines(frame, [img_pts], True, color, thickness, cv2.LINE_AA)
    return frame


def _nearest_transformer(transformers, f, window=10):
    if transformers.get(f) is not None:
        return transformers[f]
    for d in range(1, window + 1):
        if transformers.get(f - d) is not None:
            return transformers[f - d]
        if transformers.get(f + d) is not None:
            return transformers[f + d]
    return None


def draw_left_panel(frame, tracking_frame_rows, carrier_id, presser_ids, transformer,
                     carrier_xy=None, press_active=False):
    vis = frame.copy()
    for r in tracking_frame_rows:
        if r["object_type"] not in ("player", "goalkeeper"):
            continue
        color = TEAM_COLOR.get(r["team_id"], (200, 200, 200))
        thickness = 2
        label = None
        if r["track_id"] == carrier_id:
            color, thickness, label = CARRIER_COLOR, 3, "CARRIER"
        elif r["track_id"] in presser_ids:
            color, thickness, label = PRESSER_COLOR, 3, "PRESSER"
        x1, y1, x2, y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if label == "CARRIER":
            cv2.putText(vis, label, (x1, max(12, y1 - 6)), FONT, 0.45, color, 2, cv2.LINE_AA)
        elif label == "PRESSER":
            # below the box, not above -- avoids colliding with an
            # adjacent CARRIER label when the two are right next to
            # each other (the whole point of a press)
            cv2.putText(vis, label, (x1, min(vis.shape[0] - 4, y2 + 16)), FONT, 0.45, color, 2, cv2.LINE_AA)
    if carrier_xy is not None and press_active:
        vis = _draw_pitch_circle_on_image(vis, transformer, carrier_xy[0], carrier_xy[1],
                                           PRESS_TRIGGER_RADIUS_CM, (0, 0, 255), 2)
    state_txt = "PRESS ACTIVE" if press_active else "NO PRESS"
    state_color = (0, 0, 255) if press_active else (200, 200, 200)
    cv2.rectangle(vis, (0, 0), (260, 34), (0, 0, 0), -1)
    cv2.putText(vis, state_txt, (10, 24), FONT, 0.7, state_color, 2, cv2.LINE_AA)
    return vis


def draw_right_top(players_this_frame, ball_xy, carrier_id, presser_ids, feat):
    team_a = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0}
              for p in players_this_frame if p["display_team_id"] == 0]
    team_b = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0}
              for p in players_this_frame if p["display_team_id"] == 1]
    pc = compute_pitch_control_grid(team_a, team_b) if (team_a and team_b) else {"valid": False}

    pitch = draw_pitch(config=CONFIG)
    if pc.get("valid"):
        h, w = pitch.shape[:2]
        up = cv2.resize(pc["grid"].astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        up = np.clip(up, 0, 1)[..., None]
        color_field = (up * np.array(TEAM_COLOR[0]) + (1 - up) * np.array(TEAM_COLOR[1])).astype(np.uint8)
        pitch = cv2.addWeighted(color_field, 0.45, pitch, 0.55, 0)

    carrier = next((p for p in players_this_frame if p["track_id"] == carrier_id), None)
    if feat is not None and carrier is not None:
        support = [p for p in players_this_frame
                   if p["display_team_id"] == feat["ball_carrier_team_id"] and p["track_id"] != carrier_id
                   and np.hypot(p["x_pitch"] - carrier["x_pitch"], p["y_pitch"] - carrier["y_pitch"]) <= SUPPORT_RADIUS_CM]
        defenders = [p for p in players_this_frame
                     if p["display_team_id"] is not None and p["display_team_id"] != feat["ball_carrier_team_id"]]
        for s in support:
            open_lane = _passing_lane_open((carrier["x_pitch"], carrier["y_pitch"]),
                                            (s["x_pitch"], s["y_pitch"]), defenders)
            p0 = _pitch_xy_to_px(carrier["x_pitch"], carrier["y_pitch"])
            p1 = _pitch_xy_to_px(s["x_pitch"], s["y_pitch"])
            cv2.line(pitch, p0, p1, (0, 200, 0) if open_lane else (0, 0, 200), 2, cv2.LINE_AA)
        cv2.circle(pitch, _pitch_xy_to_px(carrier["x_pitch"], carrier["y_pitch"]), 9, CARRIER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(pitch, _pitch_xy_to_px(carrier["x_pitch"], carrier["y_pitch"]), 9, (0, 0, 0), 2, cv2.LINE_AA)

    for p in players_this_frame:
        if p["track_id"] == carrier_id:
            continue
        px, py = _pitch_xy_to_px(p["x_pitch"], p["y_pitch"])
        color = TEAM_COLOR.get(p["display_team_id"], (150, 150, 150))
        if p["track_id"] in presser_ids:
            cv2.circle(pitch, (px, py), 10, PRESSER_COLOR, 2, cv2.LINE_AA)
        cv2.circle(pitch, (px, py), 6, color, -1, cv2.LINE_AA)

    if ball_xy is not None:
        bx, by = _pitch_xy_to_px(*ball_xy)
        cv2.circle(pitch, (bx, by), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(pitch, (bx, by), 6, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.putText(pitch, "PRESSING RADAR", (16, 30), FONT, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(pitch, "Background: pitch-control | lines: green=open lane, red=blocked", (16, 54),
                FONT, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
    return pitch


def draw_right_bottom(feat, outcome_text, w=900, h=460):
    panel = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "PRESSING METRICS", (16, 34), FONT, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    if feat is None:
        cv2.putText(panel, "No confident ball carrier this frame", (16, 70), FONT, 0.55, (100, 100, 100), 1, cv2.LINE_AA)
        return panel
    lines = [
        f"Press Status: {'ACTIVE' if outcome_text.get('active') else 'inactive'}",
        f"Pressers (<=5m): {feat['n_pressers_5m']}  (<=3m: {feat['n_pressers_3m']}, <=8m: {feat['n_pressers_8m']})",
        f"Nearest Presser Distance: {feat['nearest_defender_distance_cm']/100:.1f} m",
        f"Closing Speed: {(feat['nearest_defender_closing_speed_cm_s'] or 0)/100:.1f} m/s",
        f"Local Numbers (10m): attack {feat['n_attack_support_local']} vs def {feat['n_defenders_local']} "
        f"(balance {feat['local_numerical_balance']:+d})",
        f"Local Compactness: {feat['local_compactness_cm']/100:.1f} m" if feat['local_compactness_cm'] else "Local Compactness: n/a",
        f"Passing Options: {feat['n_passing_options']} ({feat['n_open_lanes']} open / {feat['n_blocked_lanes']} blocked)",
        f"Escape Space (proxy = nearest defender dist): {feat['escape_space_score_cm']/100:.1f} m",
        f"Pitch Control (carrier's team, at ball): "
        f"{feat['pitch_control_for_carrier_team']*100:.0f}%" if feat['pitch_control_for_carrier_team'] is not None else "Pitch Control: n/a",
    ]
    if outcome_text.get("outcome"):
        lines.append(f"Press Outcome: {outcome_text['outcome']}")
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (16, 64 + i * 28), FONT, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return panel


def draw_timeline(events_in_window, cur_time_sec, window_start, window_end, w=900, h=60):
    bar = np.full((h, w, 3), 230, dtype=np.uint8)
    span = window_end - window_start
    for e in events_in_window:
        x0 = int((e["start_time_sec"] - window_start) / span * w)
        x1 = int((e["end_time_sec"] - window_start) / span * w)
        x1 = max(x1, x0 + 3)
        color = {"SUCCESS": (0, 160, 0), "FAILED": (0, 0, 200), "UNCERTAIN": (140, 140, 140)}.get(e["outcome"], (140, 140, 140))
        cv2.rectangle(bar, (x0, 10), (x1, h - 10), color, -1)
    cx = int((cur_time_sec - window_start) / span * w)
    cv2.line(bar, (cx, 0), (cx, h), (0, 0, 0), 2)
    cv2.putText(bar, "PRESS TIMELINE (green=success red=failed grey=uncertain)", (8, h - 4),
                FONT, 0.4, (40, 40, 40), 1, cv2.LINE_AA)
    return bar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_video_path", type=str,
                         default=os.path.join(PROJECT_ROOT, "data/source_video/testVideo1_120s.mp4"))
    parser.add_argument("--tracking_dir", type=str, default=os.path.join(PROJECT_ROOT, "data/tracking/testVideo1_120s"))
    parser.add_argument("--analytics_dir", type=str, default=os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3"))
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--end_sec", type=float, default=25.0)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    tracking = pl.read_parquet(os.path.join(args.tracking_dir, "tracking.parquet"))
    cleaned = build_cleaned_view(tracking)
    ball = load_ball_view(args.analytics_dir)
    poss = build_possession(cleaned, ball)
    features = build_pressing_features(cleaned, ball, poss)
    events = detect_pressing_events(features)
    passes = pl.read_parquet(os.path.join(args.analytics_dir, "passes.parquet"))
    turnovers = pl.read_parquet(os.path.join(args.analytics_dir, "turnovers.parquet"))
    events_out = classify_press_outcomes(events, passes, turnovers) if events.height else events
    events_list = events_out.to_dicts() if events_out.height else []

    with open(os.path.join(args.analytics_dir, "homography_transformers.pkl"), "rb") as fh:
        transformers = pickle.load(fh)

    poss_by_frame = {r["frame"]: r for r in poss.to_dicts()}
    feat_by_frame = {r["frame"]: r for r in features.to_dicts()}
    tracking_by_frame = {}
    for r in tracking.to_dicts():
        tracking_by_frame.setdefault(r["frame"], []).append(r)
    cleaned_by_frame = {}
    for r in cleaned.to_dicts():
        cleaned_by_frame.setdefault(r["frame"], []).append(r)
    ball_by_frame = {r["frame"]: r for r in ball.to_dicts()}

    cap = cv2.VideoCapture(args.source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    f_start = int(args.start_sec * fps)
    f_end = int(args.end_sec * fps)

    right_w = 900
    left_w, left_h = 1144, 714
    # right_top (the radar) keeps the pitch utility's own native aspect
    # ratio at right_w wide -- never stretched/squashed -- and
    # right_bottom is sized to comfortably fit its text (measured: 9
    # metric lines + header need ~320px at this font), so canvas_h is
    # DERIVED from the right column's real needs, not forced to match
    # the left video. The left video is vertically centered in the
    # taller canvas on a plain dark background rather than left glued
    # to one edge with dead black space below it.
    sample_pitch = draw_pitch(config=CONFIG)
    pitch_aspect_h_over_w = sample_pitch.shape[0] / sample_pitch.shape[1]
    right_top_h = int(round(right_w * pitch_aspect_h_over_w))
    right_bottom_h = 320
    timeline_h = 46
    canvas_w = left_w + right_w
    canvas_h = right_top_h + right_bottom_h + timeline_h
    left_y0 = (canvas_h - left_h) // 2

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_path, fourcc, fps, (canvas_w, canvas_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    n_written = 0
    for f in range(f_start, f_end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        poss_row = poss_by_frame.get(f)
        feat = feat_by_frame.get(f)
        # Only highlight a carrier when possession is CONTROLLED (the
        # same bar `build_pressing_features` itself uses) -- an
        # UNCERTAIN-state track_id is a lower-confidence guess and must
        # NOT be drawn with the same firm "CARRIER" label (caught during
        # visual QA: a frame showed CARRIER highlighted on the video
        # while the metrics panel correctly said "no confident carrier").
        carrier_id = (poss_row["possessing_track_id"]
                      if poss_row and poss_row["possession_state"] == "CONTROLLED" else None)
        presser_ids = set()
        if feat is not None:
            players = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None]
            carrier_row = next((p for p in players if p["track_id"] == carrier_id), None)
            if carrier_row is not None:
                for p in players:
                    if (p["display_team_id"] is not None and p["display_team_id"] != feat["ball_carrier_team_id"]):
                        d = np.hypot(p["x_pitch"] - carrier_row["x_pitch"], p["y_pitch"] - carrier_row["y_pitch"])
                        closing = None
                        if p.get("vx_cm_s") is not None and d > 0:
                            dx, dy = (carrier_row["x_pitch"] - p["x_pitch"]) / d, (carrier_row["y_pitch"] - p["y_pitch"]) / d
                            closing = p["vx_cm_s"] * dx + p["vy_cm_s"] * dy
                        if d <= PRESS_TRIGGER_RADIUS_CM and (closing or 0) >= PRESS_TRIGGER_CLOSING_SPEED_CM_S:
                            presser_ids.add(p["track_id"])

        press_active = len(presser_ids) > 0
        transformer = _nearest_transformer(transformers, f)
        carrier_xy = (feat["ball_carrier_x"], feat["ball_carrier_y"]) if feat else None

        left = draw_left_panel(frame, tracking_by_frame.get(f, []), carrier_id, presser_ids,
                                transformer, carrier_xy, press_active)
        left = cv2.resize(left, (left_w, left_h))

        players_this_frame = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None]
        ball_row = ball_by_frame.get(f)
        ball_xy = (ball_row["x_pitch"], ball_row["y_pitch"]) if ball_row and ball_row.get("is_observed") and ball_row.get("x_pitch") is not None else None
        right_top = draw_right_top(players_this_frame, ball_xy, carrier_id, presser_ids, feat)
        right_top = cv2.resize(right_top, (right_w, right_top_h))

        cur_events = [e for e in events_list if e["start_frame"] <= f <= e["end_frame"]]
        outcome_info = {"active": press_active}
        if cur_events:
            outcome_info["outcome"] = cur_events[0]["outcome"]
        right_bottom = draw_right_bottom(feat, outcome_info, w=right_w, h=right_bottom_h)

        timeline = draw_timeline(events_list, f / fps, args.start_sec, args.end_sec, w=right_w, h=timeline_h)

        right_col = np.vstack([right_top, right_bottom, timeline])
        canvas = np.full((canvas_h, canvas_w, 3), 35, dtype=np.uint8)
        canvas[left_y0:left_y0 + left_h, 0:left_w] = left
        canvas[0:right_col.shape[0], left_w:left_w + right_w] = right_col
        cv2.line(canvas, (left_w, 0), (left_w, canvas_h), (80, 80, 80), 2)
        writer.write(canvas)
        n_written += 1

    writer.release()
    cap.release()
    print(f"Wrote {args.out_path}: {n_written} frames, canvas {canvas_w}x{canvas_h}")
    print(f"n_events_in_window={len([e for e in events_list if e['start_time_sec'] <= args.end_sec])}")


if __name__ == "__main__":
    main()
