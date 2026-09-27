"""
Pressing-structure V2 dashboard renderer -- full 120s, refined outcome
taxonomy, live scrolling graphs, richer timeline. NEW script; v1's
`render_pressing_dashboard.py` and its output are UNCHANGED and remain
reproducible exactly as before.

Usage:
    external/sports/.venv/bin/python pressing_structure/dashboard/render_pressing_dashboard_v2.py \\
        --out_path pressing_structure/outputs/v2_120s/pressing_dashboard_120s_v2.mp4
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
from pressing_structure.analytics.pressing import (PRESS_TRIGGER_CLOSING_SPEED_CM_S, PRESS_TRIGGER_RADIUS_CM,  # noqa: E402
                                                     SUPPORT_RADIUS_CM, _passing_lane_open, build_possession,
                                                     build_pressing_features, classify_press_outcomes_v2,
                                                     detect_pressing_events)
from pressing_structure.dashboard.live_graphs import compute_stable_y_range, draw_scrolling_graph  # noqa: E402
from pressing_structure.dashboard.render_pressing_dashboard import (_draw_pitch_circle_on_image,  # noqa: E402
                                                                       _nearest_transformer)

CONFIG = SoccerPitchConfiguration()
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEAM_COLOR = {0: (147, 20, 255), 1: (255, 191, 0)}
CARRIER_COLOR = (0, 255, 255)
PRESSER_COLOR = (0, 0, 255)
OUTCOME_COLOR = {"BALL_REGAIN": (0, 150, 0), "FORCED_BACKWARD": (0, 100, 220),
                 "FORCED_LATERAL": (0, 200, 200), "ESCAPED_PRESS": (0, 0, 220),
                 "UNCERTAIN": (140, 140, 140)}
GRAPH_WINDOW_SEC = 8.0


def _pitch_xy_to_px(x, y, scale=0.1, padding=50):
    return int(x * scale) + padding, int(y * scale) + padding


def draw_left_panel(frame, tracking_rows, carrier_id, presser_ids, transformer, carrier_xy, press_active):
    vis = frame.copy()
    for r in tracking_rows:
        if r["object_type"] not in ("player", "goalkeeper"):
            continue
        color = TEAM_COLOR.get(r["team_id"], (200, 200, 200))
        thickness, label = 2, None
        if r["track_id"] == carrier_id:
            color, thickness, label = CARRIER_COLOR, 3, "CARRIER"
        elif r["track_id"] in presser_ids:
            color, thickness, label = PRESSER_COLOR, 3, "PRESSER"
        x1, y1, x2, y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if label == "CARRIER":
            cv2.putText(vis, label, (x1, max(12, y1 - 6)), FONT, 0.42, color, 2, cv2.LINE_AA)
        elif label == "PRESSER":
            cv2.putText(vis, label, (x1, min(vis.shape[0] - 4, y2 + 16)), FONT, 0.42, color, 2, cv2.LINE_AA)
    if carrier_xy is not None and press_active:
        vis = _draw_pitch_circle_on_image(vis, transformer, carrier_xy[0], carrier_xy[1], PRESS_TRIGGER_RADIUS_CM, (0, 0, 255), 2)
    state_txt = "PRESS ACTIVE" if press_active else "NO PRESS"
    state_color = (0, 0, 255) if press_active else (200, 200, 200)
    cv2.rectangle(vis, (0, 0), (260, 34), (0, 0, 0), -1)
    cv2.putText(vis, state_txt, (10, 24), FONT, 0.7, state_color, 2, cv2.LINE_AA)
    return vis


def draw_right_top(players_this_frame, ball_xy, carrier_id, presser_ids, feat, w, h):
    team_a = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0} for p in players_this_frame if p["display_team_id"] == 0]
    team_b = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0} for p in players_this_frame if p["display_team_id"] == 1]
    pc = compute_pitch_control_grid(team_a, team_b) if (team_a and team_b) else {"valid": False}
    pitch = draw_pitch(config=CONFIG)
    if pc.get("valid"):
        ph, pw = pitch.shape[:2]
        up = cv2.resize(pc["grid"].astype(np.float32), (pw, ph), interpolation=cv2.INTER_CUBIC)
        up = np.clip(up, 0, 1)[..., None]
        color_field = (up * np.array(TEAM_COLOR[0]) + (1 - up) * np.array(TEAM_COLOR[1])).astype(np.uint8)
        pitch = cv2.addWeighted(color_field, 0.45, pitch, 0.55, 0)

    carrier = next((p for p in players_this_frame if p["track_id"] == carrier_id), None)
    if feat is not None and carrier is not None:
        support = [p for p in players_this_frame if p["display_team_id"] == feat["ball_carrier_team_id"]
                   and p["track_id"] != carrier_id
                   and np.hypot(p["x_pitch"] - carrier["x_pitch"], p["y_pitch"] - carrier["y_pitch"]) <= SUPPORT_RADIUS_CM]
        defenders = [p for p in players_this_frame if p["display_team_id"] is not None and p["display_team_id"] != feat["ball_carrier_team_id"]]
        for s in support:
            open_lane = _passing_lane_open((carrier["x_pitch"], carrier["y_pitch"]), (s["x_pitch"], s["y_pitch"]), defenders)
            cv2.line(pitch, _pitch_xy_to_px(carrier["x_pitch"], carrier["y_pitch"]), _pitch_xy_to_px(s["x_pitch"], s["y_pitch"]),
                     (0, 200, 0) if open_lane else (0, 0, 200), 2, cv2.LINE_AA)
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
    return cv2.resize(pitch, (w, h))


def draw_metrics_panel(feat, cur_events, w, h):
    panel = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "PRESSING METRICS", (16, 30), FONT, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
    if feat is None:
        cv2.putText(panel, "No confident ball carrier this frame", (16, 64), FONT, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        return panel
    lines = [
        f"Pressers: <=3m {feat['n_pressers_3m']}  <=5m {feat['n_pressers_5m']}  <=8m {feat['n_pressers_8m']}",
        f"Local Numbers (10m): atk {feat['n_attack_support_local']} vs def {feat['n_defenders_local']} (balance {feat['local_numerical_balance']:+d})",
        f"Local Compactness: {feat['local_compactness_cm']/100:.1f} m" if feat['local_compactness_cm'] else "Local Compactness: n/a",
        f"Passing Options: {feat['n_passing_options']} ({feat['n_open_lanes']} open / {feat['n_blocked_lanes']} blocked)",
        f"Escape Space (proxy): {feat['escape_space_score_cm']/100:.1f} m",
    ]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (16, 60 + i * 26), FONT, 0.46, (30, 30, 30), 1, cv2.LINE_AA)
    y = 60 + len(lines) * 26 + 12
    cv2.line(panel, (16, y), (w - 16, y), (210, 210, 210), 1)
    y += 26
    cv2.putText(panel, "EPISODE / OUTCOME", (16, y), FONT, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    y += 26
    if cur_events:
        e = cur_events[0]
        outcome = e.get("outcome_v2", "ACTIVE")
        color = OUTCOME_COLOR.get(outcome, (0, 0, 0))
        cv2.putText(panel, f"Team {e['carrier_team_id']} pressed -> {outcome}", (16, y), FONT, 0.5, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(panel, "(no episode active)", (16, y), FONT, 0.48, (120, 120, 120), 1, cv2.LINE_AA)
    return panel


def draw_timeline(events_list, cur_time, fps, w, h):
    bar = np.full((h, w, 3), 230, dtype=np.uint8)
    total_dur = events_list[-1]["end_time_sec"] + 10 if events_list else max(cur_time, 1)
    span = max(total_dur, cur_time)
    for e in events_list:
        x0 = int(e["start_time_sec"] / span * w)
        x1 = max(int(e["end_time_sec"] / span * w), x0 + 2)
        color = OUTCOME_COLOR.get(e.get("outcome_v2", "UNCERTAIN"), (140, 140, 140))
        cv2.rectangle(bar, (x0, 6), (x1, h - 18), color, -1)
    cx = int(cur_time / span * w)
    cv2.line(bar, (cx, 0), (cx, h), (0, 0, 0), 2)
    cv2.putText(bar, "TIMELINE", (8, h - 4), FONT, 0.34, (40, 40, 40), 1, cv2.LINE_AA)
    lx = 90
    for k, color in OUTCOME_COLOR.items():
        cv2.rectangle(bar, (lx, h - 14), (lx + 10, h - 4), color, -1)
        cv2.putText(bar, k, (lx + 14, h - 4), FONT, 0.32, (40, 40, 40), 1, cv2.LINE_AA)
        lx += 20 + 8 * len(k)
    return bar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_video_path", type=str, default=os.path.join(PROJECT_ROOT, "data/source_video/testVideo1_120s.mp4"))
    parser.add_argument("--tracking_dir", type=str, default=os.path.join(PROJECT_ROOT, "data/tracking/testVideo1_120s"))
    parser.add_argument("--analytics_dir", type=str, default=os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3"))
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--end_sec", type=float, default=None)
    args = parser.parse_args()

    tracking = pl.read_parquet(os.path.join(args.tracking_dir, "tracking.parquet"))
    cleaned = build_cleaned_view(tracking)
    ball = load_ball_view(args.analytics_dir)
    poss = build_possession(cleaned, ball)
    features = build_pressing_features(cleaned, ball, poss)
    events = detect_pressing_events(features)
    passes = pl.read_parquet(os.path.join(args.analytics_dir, "passes.parquet"))
    turnovers = pl.read_parquet(os.path.join(args.analytics_dir, "turnovers.parquet"))
    events_out = classify_press_outcomes_v2(events, passes, turnovers) if events.height else events
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
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_start = int(args.start_sec * fps)
    f_end = int(args.end_sec * fps) if args.end_sec is not None else n_total - 1

    # --- precompute full-clip metric histories for the live graphs, with a STABLE y-axis ---
    all_frames = list(range(0, n_total))
    hist_dist = [(f / fps, (feat_by_frame[f]["nearest_defender_distance_cm"] / 100.0) if f in feat_by_frame else None) for f in all_frames]
    hist_closing = [(f / fps, ((feat_by_frame[f]["nearest_defender_closing_speed_cm_s"] or 0) / 100.0) if f in feat_by_frame else None) for f in all_frames]
    hist_pc = [(f / fps, (feat_by_frame[f]["pitch_control_for_carrier_team"] * 100.0) if (f in feat_by_frame and feat_by_frame[f]["pitch_control_for_carrier_team"] is not None) else None) for f in all_frames]
    y_dist = compute_stable_y_range([v for _, v in hist_dist])
    y_closing = compute_stable_y_range([v for _, v in hist_closing])
    y_pc = (0.0, 100.0)

    left_w, left_h = 1144, 714
    sample_pitch = draw_pitch(config=CONFIG)
    right_w = int(round(left_h * sample_pitch.shape[1] / sample_pitch.shape[0]))
    graph_h = 150
    bottom_h = graph_h * 3
    timeline_h = 40
    canvas_w = left_w + right_w
    canvas_h = left_h + bottom_h + timeline_h

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_path, fourcc, fps, (canvas_w, canvas_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    n_written = 0
    for f in range(f_start, f_end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        cur_time = f / fps
        poss_row = poss_by_frame.get(f)
        feat = feat_by_frame.get(f)
        carrier_id = (poss_row["possessing_track_id"] if poss_row and poss_row["possession_state"] == "CONTROLLED" else None)

        presser_ids = set()
        if feat is not None:
            players = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None]
            carrier_row = next((p for p in players if p["track_id"] == carrier_id), None)
            if carrier_row is not None:
                for p in players:
                    if p["display_team_id"] is not None and p["display_team_id"] != feat["ball_carrier_team_id"]:
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

        left = cv2.resize(draw_left_panel(frame, tracking_by_frame.get(f, []), carrier_id, presser_ids, transformer, carrier_xy, press_active), (left_w, left_h))
        players_this_frame = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None]
        ball_row = ball_by_frame.get(f)
        ball_xy = (ball_row["x_pitch"], ball_row["y_pitch"]) if ball_row and ball_row.get("is_observed") and ball_row.get("x_pitch") is not None else None
        right_top = draw_right_top(players_this_frame, ball_xy, carrier_id, presser_ids, feat, right_w, left_h)

        g1 = draw_scrolling_graph(hist_dist, cur_time, GRAPH_WINDOW_SEC, left_w, graph_h,
                                   "Nearest Defender Distance", f"last {GRAPH_WINDOW_SEC:.0f}s", y_dist, "m", (0, 100, 220))
        g2 = draw_scrolling_graph(hist_closing, cur_time, GRAPH_WINDOW_SEC, left_w, graph_h,
                                   "Nearest Defender Closing Speed", f"last {GRAPH_WINDOW_SEC:.0f}s", y_closing, "m/s", (220, 100, 0))
        g3 = draw_scrolling_graph(hist_pc, cur_time, GRAPH_WINDOW_SEC, left_w, graph_h,
                                   "Pitch Control at Carrier", f"last {GRAPH_WINDOW_SEC:.0f}s", y_pc, "%", (0, 150, 0))
        bottom_left = np.vstack([g1, g2, g3])

        cur_events = [e for e in events_list if e["start_frame"] <= f <= e["end_frame"]]
        bottom_right = draw_metrics_panel(feat, cur_events, right_w, bottom_h)

        timeline = draw_timeline(events_list, cur_time, fps, canvas_w, timeline_h)

        top = np.hstack([left, right_top])
        bottom = np.hstack([bottom_left, bottom_right])
        canvas = np.vstack([top, bottom, timeline])
        writer.write(canvas)
        n_written += 1
        if n_written % 600 == 0:
            print(f"...{n_written} frames written ({cur_time:.1f}s)", flush=True)

    writer.release()
    cap.release()
    print(f"Wrote {args.out_path}: {n_written} frames, canvas {canvas_w}x{canvas_h}")


if __name__ == "__main__":
    main()
