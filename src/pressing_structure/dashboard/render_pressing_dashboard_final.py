"""
Pressing "final" dashboard -- same V2 analytics (unchanged import, no
logic touched), NEW dark-card visual language modeled on a locked
reference mockup (layout/hierarchy/color-treatment only -- no values,
labels, or event categories from the mockup are used; every number
shown here comes from `pressing_structure.analytics.pressing`'s real
V2 outputs).

Canvas is fixed at 2304x1204 (unchanged). Top row (video + radar)
occupies >60% of the canvas height per the locked layout rule.

Usage:
    external/sports/.venv/bin/python pressing_structure/dashboard/render_pressing_dashboard_final.py \\
        --out_path pressing_structure/outputs/final_dashboard/pressing_dashboard_120s_final.mp4
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
from pressing_structure.dashboard.dashboard_style import (ACCENT_CYAN, ACCENT_MAGENTA, BG_DARK, BORDER_COLOR,  # noqa: E402
                                                            CARD_DARK, FONT, PANEL_DARK, TEXT_DIM, TEXT_WHITE,
                                                            draw_card, draw_dark_scrolling_graph, draw_event_timeline_dark,
                                                            draw_header_strip, draw_metric_tile, rounded_rect)
from pressing_structure.dashboard.live_graphs import compute_stable_y_range  # noqa: E402
from pressing_structure.dashboard.render_pressing_dashboard import _draw_pitch_circle_on_image, _nearest_transformer  # noqa: E402

CONFIG = SoccerPitchConfiguration()
TEAM_A_COLOR = (255, 200, 60)     # BGR cyan-ish (defending-style accent for team 0)
TEAM_B_COLOR = (200, 60, 220)     # BGR magenta accent for team 1
CARRIER_COLOR = (0, 230, 255)
PRESSER_COLOR = (60, 60, 255)
OUTCOME_COLOR = {"BALL_REGAIN": (110, 220, 90), "FORCED_BACKWARD": (60, 165, 255),
                  "FORCED_LATERAL": (60, 220, 220), "ESCAPED_PRESS": (70, 70, 235),
                  "UNCERTAIN": (150, 150, 150)}
GRAPH_WINDOW_SEC = 8.0
HEADER_H = 64
TIMELINE_H = 50
TILE_ROW_H = 92
GRAPH_ROW_H = 268  # sized so HEADER_H + top_row_h(730) + TILE_ROW_H + GRAPH_ROW_H + TIMELINE_H == 1204 exactly
GAP = 10


def _pitch_xy_to_px(x, y, scale=0.1, padding=50):
    return int(x * scale) + padding, int(y * scale) + padding


def draw_left_feed(frame, tracking_rows, carrier_id, presser_ids, transformer, carrier_xy, press_active):
    vis = frame.copy()
    for r in tracking_rows:
        if r["object_type"] not in ("player", "goalkeeper"):
            continue
        color = TEAM_A_COLOR if r["team_id"] == 0 else TEAM_B_COLOR if r["team_id"] == 1 else (180, 180, 180)
        thickness, label = 2, None
        if r["track_id"] == carrier_id:
            color, thickness, label = CARRIER_COLOR, 3, "BALL CARRIER"
        elif r["track_id"] in presser_ids:
            color, thickness, label = PRESSER_COLOR, 3, "PRESSER"
        x1, y1, x2, y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if label == "BALL CARRIER":
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.42, 2)
            cv2.rectangle(vis, (x1, y1 - th - 12), (x1 + tw + 8, y1 - 4), (0, 0, 0), -1)
            cv2.putText(vis, label, (x1 + 4, y1 - 8), FONT, 0.42, color, 2, cv2.LINE_AA)
        elif label == "PRESSER":
            cv2.putText(vis, label, (x1, min(vis.shape[0] - 4, y2 + 16)), FONT, 0.4, color, 2, cv2.LINE_AA)
    if carrier_xy is not None and press_active:
        vis = _draw_pitch_circle_on_image(vis, transformer, carrier_xy[0], carrier_xy[1], PRESS_TRIGGER_RADIUS_CM, (60, 60, 255), 2)
    # dark label chip, top-left, matching reference "TACTICAL MATCH FEED" style
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (330, 40), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, dst=vis)
    state_txt = "PRESS ACTIVE" if press_active else "NO PRESS"
    state_color = (60, 60, 255) if press_active else (170, 170, 170)
    cv2.putText(vis, state_txt, (12, 27), FONT, 0.62, state_color, 2, cv2.LINE_AA)
    return vis


def draw_radar(players_this_frame, ball_xy, carrier_id, presser_ids, feat, w, h):
    team_a = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0} for p in players_this_frame if p["display_team_id"] == 0]
    team_b = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0} for p in players_this_frame if p["display_team_id"] == 1]
    pc = compute_pitch_control_grid(team_a, team_b) if (team_a and team_b) else {"valid": False}
    pitch = draw_pitch(config=CONFIG)
    # darken the base pitch slightly to match the dark theme, then wash in a
    # vivid pitch-control gradient (reference mockups use a strong, clearly
    # visible cyan/magenta split, not a muted tint)
    pitch = cv2.addWeighted(pitch, 0.7, np.zeros_like(pitch), 0.3, 0)
    if pc.get("valid"):
        ph, pw = pitch.shape[:2]
        up = cv2.resize(pc["grid"].astype(np.float32), (pw, ph), interpolation=cv2.INTER_CUBIC)
        up = np.clip(up, 0, 1)[..., None]
        color_field = (up * np.array(TEAM_A_COLOR) + (1 - up) * np.array(TEAM_B_COLOR)).astype(np.uint8)
        pitch = cv2.addWeighted(color_field, 0.65, pitch, 0.35, 0)

    carrier = next((p for p in players_this_frame if p["track_id"] == carrier_id), None)
    if feat is not None and carrier is not None:
        support = [p for p in players_this_frame if p["display_team_id"] == feat["ball_carrier_team_id"]
                   and p["track_id"] != carrier_id
                   and np.hypot(p["x_pitch"] - carrier["x_pitch"], p["y_pitch"] - carrier["y_pitch"]) <= SUPPORT_RADIUS_CM]
        defenders = [p for p in players_this_frame if p["display_team_id"] is not None and p["display_team_id"] != feat["ball_carrier_team_id"]]
        for s in support:
            open_lane = _passing_lane_open((carrier["x_pitch"], carrier["y_pitch"]), (s["x_pitch"], s["y_pitch"]), defenders)
            cv2.line(pitch, _pitch_xy_to_px(carrier["x_pitch"], carrier["y_pitch"]), _pitch_xy_to_px(s["x_pitch"], s["y_pitch"]),
                     (90, 235, 90) if open_lane else (60, 60, 230), 2, cv2.LINE_AA)
        cv2.circle(pitch, _pitch_xy_to_px(carrier["x_pitch"], carrier["y_pitch"]), 10, CARRIER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(pitch, _pitch_xy_to_px(carrier["x_pitch"], carrier["y_pitch"]), 10, (255, 255, 255), 2, cv2.LINE_AA)
    for p in players_this_frame:
        if p["track_id"] == carrier_id:
            continue
        px, py = _pitch_xy_to_px(p["x_pitch"], p["y_pitch"])
        color = TEAM_A_COLOR if p["display_team_id"] == 0 else TEAM_B_COLOR if p["display_team_id"] == 1 else (150, 150, 150)
        if p["track_id"] in presser_ids:
            cv2.circle(pitch, (px, py), 11, PRESSER_COLOR, 2, cv2.LINE_AA)
        cv2.circle(pitch, (px, py), 7, color, -1, cv2.LINE_AA)
        cv2.circle(pitch, (px, py), 7, (20, 20, 20), 1, cv2.LINE_AA)
    if ball_xy is not None:
        bx, by = _pitch_xy_to_px(*ball_xy)
        cv2.circle(pitch, (bx, by), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(pitch, (bx, by), 6, (20, 20, 20), 1, cv2.LINE_AA)
    overlay = pitch.copy()
    cv2.rectangle(overlay, (0, 0), (300, 38), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, pitch, 0.45, 0, dst=pitch)
    cv2.putText(pitch, "PRESSING RADAR", (14, 26), FONT, 0.6, TEXT_WHITE, 2, cv2.LINE_AA)
    return cv2.resize(pitch, (w, h))


def draw_tiles_row(feat, cur_events, width, height):
    n = 4
    tile_w = (width - (n + 1) * GAP) // n
    img = np.full((height, width, 3), BG_DARK, dtype=np.uint8)
    values = []
    if feat is not None:
        values = [
            ("PRESSERS <=5M", str(feat["n_pressers_5m"]), f"<=3m {feat['n_pressers_3m']} / <=8m {feat['n_pressers_8m']}"),
            ("NEAREST DEFENDER", f"{feat['nearest_defender_distance_cm']/100:.1f} m", "closing" if (feat.get('nearest_defender_closing_speed_cm_s') or 0) > 0 else "steady"),
            ("PITCH CONTROL (CARRIER)", f"{(feat['pitch_control_for_carrier_team'] or 0)*100:.0f}%" if feat.get("pitch_control_for_carrier_team") is not None else "n/a", "of ball-carrier team"),
            ("LOCAL COMPACTNESS", f"{feat['local_compactness_cm']/100:.1f} m" if feat.get("local_compactness_cm") is not None else "n/a", f"balance {feat['local_numerical_balance']:+d}"),
        ]
    else:
        values = [("PRESSERS <=5M", "n/a", "no confident carrier"), ("NEAREST DEFENDER", "n/a", ""),
                  ("PITCH CONTROL (CARRIER)", "n/a", ""), ("LOCAL COMPACTNESS", "n/a", "")]
    x = GAP
    for label, val, sub in values:
        tile = draw_metric_tile(tile_w, height, label, val, sub, accent=ACCENT_CYAN)
        img[0:height, x:x + tile_w] = tile
        x += tile_w + GAP
    return img


def render_event_count_bg(events_list, width, height, span_sec, bucket_sec=10.0):
    """Static bar histogram (pressing events per time bucket) rendered
    ONCE -- the current-time marker is drawn per-frame on a copy."""
    img = draw_card(width, height, bg=PANEL_DARK)
    n_buckets = int(np.ceil(span_sec / bucket_sec))
    counts = np.zeros(n_buckets, dtype=int)
    for e in events_list:
        b = min(n_buckets - 1, int(e["start_time_sec"] // bucket_sec))
        counts[b] += 1
    pad_l, pad_r, pad_t, pad_b = 54, 14, 28, 22
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    max_c = max(1, counts.max())
    bar_w = plot_w / n_buckets
    for i, c in enumerate(counts):
        x0 = int(pad_l + i * bar_w) + 2
        x1 = int(pad_l + (i + 1) * bar_w) - 2
        bar_h = int((c / max_c) * plot_h)
        cv2.rectangle(img, (x0, pad_t + plot_h - bar_h), (max(x1, x0 + 2), pad_t + plot_h), ACCENT_MAGENTA, -1)
    cv2.rectangle(img, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), BORDER_COLOR, 1)
    cv2.putText(img, f"Press Events per {bucket_sec:.0f}s (count)", (pad_l, 18), FONT, 0.38, TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"{max_c}", (4, pad_t + 8), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, "0", (4, pad_t + plot_h + 4), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
    return img, pad_l, plot_w, pad_t, plot_h


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
    span_sec = n_total / fps

    all_frames = list(range(0, n_total))
    hist_dist = [(f / fps, (feat_by_frame[f]["nearest_defender_distance_cm"] / 100.0) if f in feat_by_frame else None) for f in all_frames]
    hist_closing = [(f / fps, ((feat_by_frame[f]["nearest_defender_closing_speed_cm_s"] or 0) / 100.0) if f in feat_by_frame else None) for f in all_frames]
    hist_pc = [(f / fps, (feat_by_frame[f]["pitch_control_for_carrier_team"] * 100.0) if (f in feat_by_frame and feat_by_frame[f]["pitch_control_for_carrier_team"] is not None) else None) for f in all_frames]
    hist_compact = [(f / fps, (feat_by_frame[f]["local_compactness_cm"] / 100.0) if (f in feat_by_frame and feat_by_frame[f].get("local_compactness_cm") is not None) else None) for f in all_frames]
    y_dist = compute_stable_y_range([v for _, v in hist_dist])
    y_closing = compute_stable_y_range([v for _, v in hist_closing])
    y_pc = (0.0, 100.0)
    y_compact = compute_stable_y_range([v for _, v in hist_compact])

    canvas_w = 2304
    top_row_h = 730
    sample_pitch = draw_pitch(config=CONFIG)
    right_w = int(round(top_row_h * sample_pitch.shape[1] / sample_pitch.shape[0]))
    left_w = canvas_w - right_w
    canvas_h = HEADER_H + top_row_h + TILE_ROW_H + GRAPH_ROW_H + TIMELINE_H
    n_graphs = 5
    graph_w = (canvas_w - (n_graphs + 1) * GAP) // n_graphs

    event_bg, ebl, ebw, ebt, ebh = render_event_count_bg(events_list, graph_w, GRAPH_ROW_H, span_sec)

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

        left = cv2.resize(draw_left_feed(frame, tracking_by_frame.get(f, []), carrier_id, presser_ids, transformer, carrier_xy, press_active), (left_w, top_row_h))
        players_this_frame = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None]
        ball_row = ball_by_frame.get(f)
        ball_xy = (ball_row["x_pitch"], ball_row["y_pitch"]) if ball_row and ball_row.get("is_observed") and ball_row.get("x_pitch") is not None else None
        radar = draw_radar(players_this_frame, ball_xy, carrier_id, presser_ids, feat, right_w, top_row_h)

        top_row = np.hstack([left, radar])

        cur_events = [e for e in events_list if e["start_frame"] <= f <= e["end_frame"]]
        outcome_label = f"Team {cur_events[0]['carrier_team_id']} pressed -> {cur_events[0].get('outcome_v2', 'ACTIVE')}" if cur_events else "(no episode active)"
        right_top_lines = [(outcome_label, OUTCOME_COLOR.get(cur_events[0].get("outcome_v2"), TEXT_WHITE) if cur_events else TEXT_DIM),
                            (f"t={cur_time:5.1f}s", TEXT_DIM)]
        header = draw_header_strip(canvas_w, HEADER_H, "PRESSING ANALYTICS", "rule-based | V2 taxonomy", right_top_lines, accent=ACCENT_CYAN)

        tiles = draw_tiles_row(feat, cur_events, canvas_w, TILE_ROW_H)

        g1 = draw_dark_scrolling_graph(hist_dist, cur_time, GRAPH_WINDOW_SEC, graph_w, GRAPH_ROW_H,
                                         "Nearest Defender Distance", f"last {GRAPH_WINDOW_SEC:.0f}s", y_dist, "m", ACCENT_CYAN)
        g2 = draw_dark_scrolling_graph(hist_closing, cur_time, GRAPH_WINDOW_SEC, graph_w, GRAPH_ROW_H,
                                         "Closing Speed", f"last {GRAPH_WINDOW_SEC:.0f}s", y_closing, "m/s", ACCENT_MAGENTA)
        g3 = draw_dark_scrolling_graph(hist_pc, cur_time, GRAPH_WINDOW_SEC, graph_w, GRAPH_ROW_H,
                                         "Pitch Control (Carrier)", f"last {GRAPH_WINDOW_SEC:.0f}s", y_pc, "%", (110, 220, 90))
        g4 = draw_dark_scrolling_graph(hist_compact, cur_time, GRAPH_WINDOW_SEC, graph_w, GRAPH_ROW_H,
                                         "Local Compactness", f"last {GRAPH_WINDOW_SEC:.0f}s", y_compact, "m", (60, 165, 255))
        g5 = event_bg.copy()
        mx = int(ebl + (cur_time / span_sec) * ebw)
        cv2.line(g5, (mx, ebt), (mx, ebt + ebh), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(g5, (mx, ebt), (mx, ebt + ebh), TEXT_WHITE, 1, cv2.LINE_AA)
        gap_col = np.full((GRAPH_ROW_H, GAP, 3), BG_DARK, dtype=np.uint8)
        graphs_row = np.hstack([gap_col, g1, gap_col, g2, gap_col, g3, gap_col, g4, gap_col, g5, gap_col])
        if graphs_row.shape[1] != canvas_w:
            graphs_row = cv2.resize(graphs_row, (canvas_w, GRAPH_ROW_H))

        segments = [(e["start_time_sec"], e["end_time_sec"], OUTCOME_COLOR.get(e.get("outcome_v2"), (150, 150, 150))) for e in events_list]
        legend = [(k, v) for k, v in OUTCOME_COLOR.items()]
        timeline = draw_event_timeline_dark(canvas_w, TIMELINE_H, span_sec, cur_time, segments, [], legend, title="PRESSING EVENT TIMELINE")

        canvas = np.vstack([header, top_row, tiles, graphs_row, timeline])
        writer.write(canvas)
        n_written += 1
        if n_written % 600 == 0:
            print(f"...{n_written} frames written ({cur_time:.1f}s)", flush=True)

    writer.release()
    cap.release()
    print(f"Wrote {args.out_path}: {n_written} frames, canvas {canvas_w}x{canvas_h}")


if __name__ == "__main__":
    main()
