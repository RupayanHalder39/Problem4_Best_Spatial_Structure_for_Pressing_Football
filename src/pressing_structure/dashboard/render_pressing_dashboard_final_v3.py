"""
Pressing dashboard -- SECOND-generation "final" layout, now with
broadcast-style polish (icon KPI row, team-comparison table/bars,
data-derived insights) after feedback that the structural rebuild
still read as a cleaned-up debug screen. Same V2 analytics, UNCHANGED:
`build_possession`, `build_pressing_features`, `detect_pressing_events`,
`classify_press_outcomes_v2` from `pressing_structure/analytics/
pressing.py` are imported read-only -- every number on screen is
either a real per-frame value or a real whole-clip aggregate computed
here from those outputs, never invented. This script and its
dark-theme primitives (`dashboard_style_v2.py`) are NEW files; the
first final attempt (`render_pressing_dashboard_final.py`,
`dashboard_style.py`, `outputs/final_dashboard/`) is untouched.

Team-aware, not hardcoded: the pressing team for any given frame/
episode is always `1 - ball_carrier_team_id` (whichever team does NOT
have the ball is the candidate presser) -- this was already how
`detect_pressing_events`/`build_pressing_features` work; this script
now also surfaces it explicitly as an on-screen "Presser: Team X" /
"Presser: Uncertain" label and as real per-team aggregates in the
comparison table.

Usage (preview -- ONE frame, no video written):
    external/sports/.venv/bin/python pressing_structure/dashboard/render_pressing_dashboard_final_v2.py \\
        --preview_frame_sec 5.1 --out_path /tmp/pressing_preview.png

Usage (full render, only after preview approval):
    external/sports/.venv/bin/python pressing_structure/dashboard/render_pressing_dashboard_final_v2.py \\
        --out_path pressing_structure/outputs/final_dashboard_v2/pressing_dashboard_120s_final_v2.mp4
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
                                                     SUCCESSFUL_PRESS_STATES, SUPPORT_RADIUS_CM, _passing_lane_open,
                                                     build_possession, build_pressing_features,
                                                     classify_press_outcomes_v2, detect_pressing_events)
from pressing_structure.analytics.pressing_v3 import (PRESS_FSM_CONFIG, build_dual_team_press_states,  # noqa: E402
                                                         decide_dominant_state)
from pressing_structure.dashboard.dashboard_style_v2 import (ACCENT_CYAN, ACCENT_GREEN, ACCENT_MAGENTA,  # noqa: E402
                                                               ACCENT_ORANGE, ACCENT_RED, BG, BORDER_SOFT, FONT,
                                                               TEAM_PRESS_GLOW_STOPS, TEXT_DIM, add_team_pressure_glow,
                                                               apply_color_wash_preserve_lines, composite_state_badge,
                                                               draw_bar_comparison, draw_big_dual_graph, draw_big_graph,
                                                               draw_header, draw_icon_kpi_card, draw_insights_panel,
                                                               draw_radar_legend_box, draw_team_table,
                                                               draw_timeline_v2, panel_frame, panel_title)
from pressing_structure.dashboard.live_graphs import compute_stable_y_range  # noqa: E402
from pressing_structure.dashboard.render_pressing_dashboard import _draw_pitch_circle_on_image, _nearest_transformer  # noqa: E402

STATE_COLOR = {
    PRESS_FSM_CONFIG.low_state: (110, 110, 110), PRESS_FSM_CONFIG.forming_state: ACCENT_ORANGE,
    PRESS_FSM_CONFIG.active_state: ACCENT_RED, PRESS_FSM_CONFIG.ending_state: (60, 200, 220),
    PRESS_FSM_CONFIG.uncertain_state: (90, 90, 90), "CONTESTED": (200, 60, 220),
}

CONFIG = SoccerPitchConfiguration()
TEAM_A_COLOR = ACCENT_CYAN
TEAM_B_COLOR = ACCENT_MAGENTA
CARRIER_COLOR = (0, 235, 255)
PRESSER_COLOR = (60, 60, 255)
OUTCOME_COLOR = {"BALL_REGAIN": ACCENT_GREEN, "FORCED_BACKWARD": ACCENT_ORANGE,
                  "FORCED_LATERAL": (60, 220, 220), "ESCAPED_PRESS": ACCENT_RED,
                  "UNCERTAIN": (150, 150, 150)}
GRAPH_WINDOW_SEC = 14.0  # widened from 8s for readability -- display-only, no analytics change

# --- layout constants (2304x1204 total; top row >60% of canvas height) ---
# Geometry locked to the reference's spatial distribution: header ~5%,
# top row ~50%, lower analytics (KPI+graphs+row3+timeline) ~45% -- see
# the layout-bounding-box report printed by --preview_frame_sec.
HEADER_H = 60          # 5.0% of 1204
TOP_ROW_H = 602         # 50.0% of 1204
TILE_ROW_H = 88
GRAPH_ROW_H = 216       # tall enough for title+axis+lines+marker to read clearly
ROW_C_H = 154
TIMELINE_H = 60
GAP = 6
CANVAS_W = 2304
LEFT_W = round(CANVAS_W * 0.52)
RIGHT_W = CANVAS_W - LEFT_W
assert HEADER_H + TOP_ROW_H + TILE_ROW_H + GRAPH_ROW_H + ROW_C_H + TIMELINE_H + 4 * GAP == 1204


def _pitch_xy_to_px(x, y, scale=0.1, padding=50):
    return int(x * scale) + padding, int(y * scale) + padding


def compute_team_aggregates(events_list):
    """Real, whole-clip per-team aggregates (the presser team for an
    episode is 1 - carrier_team_id) -- used by the KPI row, the
    team-comparison table/bars, and the insights panel. Returns a dict
    keyed by team id."""
    agg = {}
    for team in (0, 1):
        team_eps = [e for e in events_list if (1 - e["carrier_team_id"]) == team]
        n = len(team_eps)
        uncertain = sum(1 for e in team_eps if e.get("outcome_v2") == "UNCERTAIN")
        succ = sum(1 for e in team_eps if e.get("outcome_v2") in SUCCESSFUL_PRESS_STATES)
        denom = n - uncertain
        dists = [e["min_nearest_defender_distance_cm"] for e in team_eps if e.get("min_nearest_defender_distance_cm") is not None]
        pressers = [e["mean_pressers_5m"] for e in team_eps if e.get("mean_pressers_5m") is not None]
        agg[team] = {
            "n_episodes": n, "n_uncertain": uncertain, "n_successful": succ,
            "success_pct": (succ / denom * 100.0) if denom > 0 else None,
            "mean_min_dist_cm": float(np.mean(dists)) if dists else None,
            "mean_pressers_5m": float(np.mean(pressers)) if pressers else None,
        }
    return agg


def draw_left_feed(frame, tracking_rows, carrier_id, presser_ids, transformer, carrier_xy, banner_text, banner_color, presser_team_text):
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
    if carrier_xy is not None and presser_ids:
        vis = _draw_pitch_circle_on_image(vis, transformer, carrier_xy[0], carrier_xy[1], PRESS_TRIGGER_RADIUS_CM, (60, 60, 255), 2)

    # compact, rounded, translucent state badge -- real per-frame state only
    vis = composite_state_badge(vis, 10, 10, banner_text, banner_color, presser_team_text)

    # title bar above the feed, matching the radar panel's title-bar
    # treatment (the locked reference gives both top-row panels a
    # matching dark title strip -- the video previously had none)
    out = np.full((vis.shape[0] + 32, vis.shape[1], 3), BG, dtype=np.uint8)
    out[32:, :] = vis
    panel_title(out, "TACTICAL MATCH FEED", sub="pressers  •  ball carrier  •  press radius", accent=ACCENT_CYAN)
    return out


def draw_radar(players_this_frame, ball_xy, carrier_id, presser_ids, feat, w, h, dominant_team, dominant_score):
    team_a = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0} for p in players_this_frame if p["display_team_id"] == 0]
    team_b = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0} for p in players_this_frame if p["display_team_id"] == 1]
    pc = compute_pitch_control_grid(team_a, team_b) if (team_a and team_b) else {"valid": False}
    pitch = draw_pitch(config=CONFIG)
    pitch_dark = cv2.addWeighted(pitch, 0.7, np.zeros_like(pitch), 0.3, 0)
    if pc.get("valid"):
        ph, pw = pitch.shape[:2]
        up = cv2.resize(pc["grid"].astype(np.float32), (pw, ph), interpolation=cv2.INTER_CUBIC)
        up = np.clip(up, 0, 1)[..., None]
        color_field = (up * np.array(TEAM_A_COLOR) + (1 - up) * np.array(TEAM_B_COLOR)).astype(np.uint8)
        pitch_dark = apply_color_wash_preserve_lines(pitch_dark, color_field, alpha=0.88)

    carrier = next((p for p in players_this_frame if p["track_id"] == carrier_id), None)
    if feat is not None and carrier is not None:
        # Team-colored pressure hotspot: amber/orange for Team 0
        # pressing, lime/green for Team 1 pressing (Goal 3) -- driven
        # by the REAL, already-smoothed dual-team press score, not the
        # old single-team n_pressers_8m proxy.
        intensity = float(np.clip(dominant_score or 0.0, 0.0, 1.0))
        glow_stops = TEAM_PRESS_GLOW_STOPS.get(dominant_team, TEAM_PRESS_GLOW_STOPS[0])
        if dominant_team is not None and intensity > 0:
            pitch_dark = add_team_pressure_glow(pitch_dark, _pitch_xy_to_px(carrier["x_pitch"], carrier["y_pitch"]), intensity, stops=glow_stops)
        support = [p for p in players_this_frame if p["display_team_id"] == feat["ball_carrier_team_id"]
                   and p["track_id"] != carrier_id
                   and np.hypot(p["x_pitch"] - carrier["x_pitch"], p["y_pitch"] - carrier["y_pitch"]) <= SUPPORT_RADIUS_CM]
        defenders = [p for p in players_this_frame if p["display_team_id"] is not None and p["display_team_id"] != feat["ball_carrier_team_id"]]
        for s in support:
            open_lane = _passing_lane_open((carrier["x_pitch"], carrier["y_pitch"]), (s["x_pitch"], s["y_pitch"]), defenders)
            cv2.line(pitch_dark, _pitch_xy_to_px(carrier["x_pitch"], carrier["y_pitch"]), _pitch_xy_to_px(s["x_pitch"], s["y_pitch"]),
                     (90, 235, 90) if open_lane else (60, 60, 230), 2, cv2.LINE_AA)
        cv2.circle(pitch_dark, _pitch_xy_to_px(carrier["x_pitch"], carrier["y_pitch"]), 11, CARRIER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(pitch_dark, _pitch_xy_to_px(carrier["x_pitch"], carrier["y_pitch"]), 11, (255, 255, 255), 2, cv2.LINE_AA)
    for p in players_this_frame:
        if p["track_id"] == carrier_id:
            continue
        px, py = _pitch_xy_to_px(p["x_pitch"], p["y_pitch"])
        color = TEAM_A_COLOR if p["display_team_id"] == 0 else TEAM_B_COLOR if p["display_team_id"] == 1 else (150, 150, 150)
        if p["track_id"] in presser_ids:
            cv2.circle(pitch_dark, (px, py), 12, PRESSER_COLOR, 2, cv2.LINE_AA)
        cv2.circle(pitch_dark, (px, py), 8, color, -1, cv2.LINE_AA)
        cv2.circle(pitch_dark, (px, py), 8, (10, 10, 10), 2, cv2.LINE_AA)
    if ball_xy is not None:
        bx, by = _pitch_xy_to_px(*ball_xy)
        cv2.circle(pitch_dark, (bx, by), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(pitch_dark, (bx, by), 6, (15, 15, 15), 1, cv2.LINE_AA)

    out = np.full((h, w, 3), BG, dtype=np.uint8)
    resized = cv2.resize(pitch_dark, (w, h - 34))
    out[34:h, 0:w] = resized
    panel_title(out, "PRESSING RADAR", sub="pitch control  •  pressure hotspot  •  passing lanes")
    draw_radar_legend_box(out, w - 172, 40, 160, 210,
                           [("Team 0", TEAM_A_COLOR), ("Team 1", TEAM_B_COLOR),
                            ("Ball", (255, 255, 255)), ("Presser", PRESSER_COLOR),
                            ("T0 Pressing", (0, 170, 255)), ("T1 Pressing", (60, 220, 90))],
                           gradient=(TEAM_A_COLOR, TEAM_B_COLOR),
                           gradient_labels=("Team 0", "Control", "Team 1"))
    return out


def draw_tiles_row(state0, state1, dominant_state, dominant_team, dual_stats, width, height):
    """Goal 3/4: explicit simultaneous per-team state + dominant-team
    call, always computed from BOTH teams' independent FSM states --
    never a single-team-centric label."""
    n = 4
    tile_w = (width - (n + 1) * GAP) // n
    img = np.full((height, width, 3), BG, dtype=np.uint8)
    dominant_label = ("No Press" if dominant_state == PRESS_FSM_CONFIG.low_state else
                       "Uncertain" if dominant_state == PRESS_FSM_CONFIG.uncertain_state else
                       "Contested" if dominant_state == "CONTESTED" else
                       f"Team {dominant_team}" if dominant_team is not None else dominant_state.replace("_", " ").title())
    values = [
        ("Team 0 Press State", state0.replace("_", " ").title(), f"score-based, live", STATE_COLOR.get(state0, ACCENT_CYAN), "shield"),
        ("Team 1 Press State", state1.replace("_", " ").title(), f"score-based, live", STATE_COLOR.get(state1, ACCENT_MAGENTA), "shield"),
        ("Dominant This Frame", dominant_label, "both teams checked every frame", STATE_COLOR.get(dominant_state, ACCENT_ORANGE), "target"),
        ("FSM Episodes (T0 / T1)", f"{dual_stats[0]['n_active_episodes']} / {dual_stats[1]['n_active_episodes']}", "ACTIVE_PRESS segments, full clip", ACCENT_GREEN, "cluster"),
    ]
    x = GAP
    for label, val, sub, accent, glyph in values:
        tile = draw_icon_kpi_card(tile_w, height, label, val, sub, accent=accent, glyph=glyph)
        img[0:height, x:x + tile_w] = tile
        x += tile_w + GAP
    return img


def render_dots_bg(events_list):
    return [((e["start_time_sec"] + e["end_time_sec"]) / 2, OUTCOME_COLOR.get(e.get("outcome_v2"), (150, 150, 150)), 7)
            for e in events_list]


def render_event_count_panel(events_list, width, height, span_sec, bucket_sec=10.0):
    """5th graph panel (reference layout): a static per-bucket press-
    event-count bar chart, styled like the other graph cards. Static
    because the bucket counts don't change across the clip -- only the
    current-time marker moves, drawn fresh per frame on a copy."""
    img = panel_frame(width, height)
    panel_title(img, "Press Events per 10s", sub=f"{span_sec:.0f}s clip  •  count", accent=ACCENT_MAGENTA)
    pad_l, pad_r, pad_t, pad_b = 46, 16, 42, 26
    x0, y0 = pad_l, pad_t
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    n_buckets = int(np.ceil(span_sec / bucket_sec))
    counts = np.zeros(n_buckets, dtype=int)
    for e in events_list:
        b = min(n_buckets - 1, int(e["start_time_sec"] // bucket_sec))
        counts[b] += 1
    max_c = max(1, counts.max())
    bar_w = pw / n_buckets
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), BORDER_SOFT, 1)
    for i, c in enumerate(counts):
        bx0 = int(x0 + i * bar_w) + 2
        bx1 = int(x0 + (i + 1) * bar_w) - 2
        bar_h = int((c / max_c) * ph)
        cv2.rectangle(img, (bx0, y0 + ph - bar_h), (max(bx1, bx0 + 2), y0 + ph), ACCENT_MAGENTA, -1)
    cv2.putText(img, f"{max_c}", (6, y0 + 8), FONT, 0.3, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, "0", (6, y0 + ph + 4), FONT, 0.3, TEXT_DIM, 1, cv2.LINE_AA)
    return img, x0, y0, pw, ph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_video_path", type=str, default=os.path.join(PROJECT_ROOT, "data/source_video/testVideo1_120s.mp4"))
    parser.add_argument("--tracking_dir", type=str, default=os.path.join(PROJECT_ROOT, "data/tracking/testVideo1_120s"))
    parser.add_argument("--analytics_dir", type=str, default=os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3"))
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--end_sec", type=float, default=None)
    parser.add_argument("--preview_frame_sec", type=float, default=None,
                         help="If set, render exactly ONE frame at this timestamp to a PNG and exit (no video written).")
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
    agg = compute_team_aggregates(events_list)

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
    span_sec = n_total / fps

    # --- Goal 1/2: simultaneous dual-team press scores + hysteresis FSM ---
    print("Computing dual-team press states (both teams, every frame)...", flush=True)
    dual = build_dual_team_press_states(features, n_total, fps=fps)
    dominant_state_by_frame = [None] * n_total
    dominant_team_by_frame = [None] * n_total
    for f in range(n_total):
        d, team = decide_dominant_state(dual["team0"]["states"][f], dual["team1"]["states"][f])
        dominant_state_by_frame[f] = d
        dominant_team_by_frame[f] = team
    dual_stats = {}
    for i, team_key in enumerate(("team0", "team1")):
        raws = [v for v in dual[team_key]["raw"] if v is not None and v > 0]
        dual_stats[i] = {
            "n_active_episodes": sum(1 for s in dual[team_key]["segments"] if s["state"] == PRESS_FSM_CONFIG.active_state),
            "mean_score": float(np.mean(raws)) if raws else None,
        }
    hist_score0 = [(f / fps, dual["team0"]["smoothed"][f]) for f in range(n_total)]
    hist_score1 = [(f / fps, dual["team1"]["smoothed"][f]) for f in range(n_total)]
    y_score = (0.0, 1.0)

    all_frames = list(range(0, n_total))
    hist_closing = [(f / fps, ((feat_by_frame[f]["nearest_defender_closing_speed_cm_s"] or 0) / 100.0) if f in feat_by_frame else None) for f in all_frames]
    hist_pc = [(f / fps, (feat_by_frame[f]["pitch_control_for_carrier_team"] * 100.0) if (f in feat_by_frame and feat_by_frame[f]["pitch_control_for_carrier_team"] is not None) else None) for f in all_frames]
    hist_compact = [(f / fps, (feat_by_frame[f]["local_compactness_cm"] / 100.0) if (f in feat_by_frame and feat_by_frame[f].get("local_compactness_cm") is not None) else None) for f in all_frames]
    y_closing = compute_stable_y_range([v for _, v in hist_closing])
    y_pc = (0.0, 100.0)
    y_compact = compute_stable_y_range([v for _, v in hist_compact])

    dots = render_dots_bg(events_list)
    legend = [(k.replace("_", " ").title(), v) for k, v in OUTCOME_COLOR.items()]
    graph_w = (CANVAS_W - 6 * GAP) // 5
    event_bg, ebx0, eby0, ebw, ebh = render_event_count_panel(events_list, graph_w, GRAPH_ROW_H, span_sec)

    def find_banner(cur_time, f):
        """V3: the banner now reflects the STABLE, HYSTERESIS-SMOOTHED
        dual-team FSM state (`pressing_v3.decide_dominant_state`),
        computed independently for both teams every frame and combined
        with the brief's exact decision rule -- this replaces V2's
        approach of scanning discrete outcome-classified episodes
        (which needed a manual two-pass fix for overlapping hold
        windows; the FSM's own cooldown/hold counters make that whole
        class of bug structurally impossible here, since there is only
        ever one authoritative state per team per frame, precomputed
        once for the whole clip)."""
        state = dominant_state_by_frame[f]
        team = dominant_team_by_frame[f]
        color = STATE_COLOR.get(state, (150, 150, 150))
        if state == "CONTESTED":
            text, role = "CONTESTED", "Both teams pressing signals active"
        elif team is not None:
            text = state.replace("_", " ").title()
            role = f"Presser: Team {team}"
        elif state == PRESS_FSM_CONFIG.uncertain_state:
            text, role = "UNCERTAIN", "Insufficient evidence this window"
        else:
            text, role = "NO PRESS", "Presser: n/a"
        return text, color, role

    def render_frame(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            return None
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
        transformer = _nearest_transformer(transformers, f)
        carrier_xy = (feat["ball_carrier_x"], feat["ball_carrier_y"]) if feat else None
        banner_text, banner_color, presser_text = find_banner(cur_time, f)
        state0, state1 = dual["team0"]["states"][f], dual["team1"]["states"][f]
        dominant_state, dominant_team = dominant_state_by_frame[f], dominant_team_by_frame[f]
        dominant_score = (dual["team0"]["smoothed"][f] if dominant_team == 0 else
                           dual["team1"]["smoothed"][f] if dominant_team == 1 else None)

        left = cv2.resize(draw_left_feed(frame, tracking_by_frame.get(f, []), carrier_id, presser_ids, transformer,
                                          carrier_xy, banner_text, banner_color, presser_text), (LEFT_W, TOP_ROW_H))
        players_this_frame = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None]
        ball_row = ball_by_frame.get(f)
        ball_xy = (ball_row["x_pitch"], ball_row["y_pitch"]) if ball_row and ball_row.get("is_observed") and ball_row.get("x_pitch") is not None else None
        radar = draw_radar(players_this_frame, ball_xy, carrier_id, presser_ids, feat, RIGHT_W, TOP_ROW_H, dominant_team, dominant_score)
        top_row = np.hstack([left, radar])

        header = draw_header(CANVAS_W, HEADER_H, "PRESSING ANALYTICS", (f"t = {cur_time:.1f}s", TEXT_DIM), accent=ACCENT_CYAN)
        tiles = draw_tiles_row(state0, state1, dominant_state, dominant_team, dual_stats, CANVAS_W, TILE_ROW_H)

        g1 = draw_big_dual_graph(graph_w, GRAPH_ROW_H, hist_score0, hist_score1, cur_time, GRAPH_WINDOW_SEC,
                                   "Press Score: Team 0 vs Team 1", y_score, "Team 0", "Team 1", "0-1",
                                   color_a=ACCENT_CYAN, color_b=ACCENT_MAGENTA)
        g2 = draw_big_graph(graph_w, GRAPH_ROW_H, hist_closing, cur_time, GRAPH_WINDOW_SEC, "Closing Speed", y_closing, "m/s", ACCENT_MAGENTA)
        g3 = draw_big_graph(graph_w, GRAPH_ROW_H, hist_pc, cur_time, GRAPH_WINDOW_SEC, "Pitch Control at Carrier", y_pc, "%", ACCENT_GREEN)
        g4 = draw_big_graph(graph_w, GRAPH_ROW_H, hist_compact, cur_time, GRAPH_WINDOW_SEC, "Local Compactness", y_compact, "m", ACCENT_ORANGE)
        g5 = event_bg.copy()
        mx = int(ebx0 + np.clip(cur_time / span_sec, 0, 1) * ebw)
        cv2.line(g5, (mx, eby0), (mx, eby0 + ebh), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(g5, (mx, eby0), (mx, eby0 + ebh), (245, 245, 245), 2, cv2.LINE_AA)
        gap_col_b = np.full((GRAPH_ROW_H, GAP, 3), BG, dtype=np.uint8)
        graph_row = np.hstack([gap_col_b, g1, gap_col_b, g2, gap_col_b, g3, gap_col_b, g4, gap_col_b, g5, gap_col_b])
        if graph_row.shape[1] != CANVAS_W:
            graph_row = cv2.resize(graph_row, (CANVAS_W, GRAPH_ROW_H))

        # Row C: team-wide table (wide) + metric comparison + insights
        col_w = (CANVAS_W - 4 * GAP) // 3
        wide_w = col_w + col_w // 2
        narrow_w = (CANVAS_W - 4 * GAP - wide_w) // 2
        table = draw_team_table(wide_w, ROW_C_H, "TEAM-WIDE PRESSING INDICATORS (V2 OUTCOME vs V3 FSM)",
                                 ["V2 Episodes", "Successful %", "V3 Mean Score", "V3 FSM Episodes"],
                                 [("Team 0", [agg[0]["n_episodes"],
                                              f"{agg[0]['success_pct']:.0f}%" if agg[0]["success_pct"] is not None else "n/a",
                                              f"{dual_stats[0]['mean_score']:.2f}" if dual_stats[0]["mean_score"] is not None else "n/a",
                                              dual_stats[0]["n_active_episodes"]]),
                                  ("Team 1", [agg[1]["n_episodes"],
                                              f"{agg[1]['success_pct']:.0f}%" if agg[1]["success_pct"] is not None else "n/a",
                                              f"{dual_stats[1]['mean_score']:.2f}" if dual_stats[1]["mean_score"] is not None else "n/a",
                                              dual_stats[1]["n_active_episodes"]])],
                                 [TEAM_A_COLOR, TEAM_B_COLOR])
        bars = draw_bar_comparison(narrow_w, ROW_C_H, "METRIC COMPARISON (V3)",
                                    [("Mean Press Score", dual_stats[0]["mean_score"], dual_stats[1]["mean_score"], "", 1.0),
                                     ("FSM Active Episodes", float(dual_stats[0]["n_active_episodes"]), float(dual_stats[1]["n_active_episodes"]), "",
                                      max(1, max(dual_stats[0]["n_active_episodes"], dual_stats[1]["n_active_episodes"])))],
                                    TEAM_A_COLOR, TEAM_B_COLOR)
        insight_lines = [f"V3 FSM: {dual_stats[0]['n_active_episodes']} Team 0 + {dual_stats[1]['n_active_episodes']} Team 1 stable ACTIVE_PRESS episodes (hysteresis-smoothed).",
                          f"V2 outcome taxonomy (unchanged): {len(events_list)} episodes total, Team 0 pressed {agg[0]['n_episodes']}x, Team 1 {agg[1]['n_episodes']}x.",
                          f"Current dominant state: {dominant_state_by_frame[f].replace('_',' ').title()}" + (f" (Team {dominant_team_by_frame[f]})" if dominant_team_by_frame[f] is not None else ""),
                          "Both teams' press scores are computed every frame, independently -- a team scores 0 (not None) whenever it holds the ball, since it cannot press itself."]
        insights = draw_insights_panel(narrow_w, ROW_C_H, "KEY FACTS (DATA-DERIVED)", insight_lines, accent=ACCENT_ORANGE)
        gap_col_c = np.full((ROW_C_H, GAP, 3), BG, dtype=np.uint8)
        row_c = np.hstack([gap_col_c, table, gap_col_c, bars, gap_col_c, insights, gap_col_c])
        if row_c.shape[1] != CANVAS_W:
            row_c = cv2.resize(row_c, (CANVAS_W, ROW_C_H))

        timeline = draw_timeline_v2(CANVAS_W, TIMELINE_H, span_sec, cur_time, dots, legend, title="PRESSING EVENTS")

        row_gap = np.full((GAP, CANVAS_W, 3), BG, dtype=np.uint8)
        canvas = np.vstack([header, top_row, row_gap, tiles, row_gap, graph_row, row_gap, row_c, row_gap, timeline])
        return canvas

    if args.preview_frame_sec is not None:
        f = int(args.preview_frame_sec * fps)
        canvas = render_frame(f)
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        cv2.imwrite(args.out_path, canvas)
        print(f"Wrote preview frame {args.out_path}: dims {canvas.shape[1]}x{canvas.shape[0]}")
        y = 0
        for name, h in [("header", HEADER_H), ("top_row(video+radar)", TOP_ROW_H), ("gap", GAP),
                         ("kpi_row", TILE_ROW_H), ("gap", GAP), ("graph_row(5 panels)", GRAPH_ROW_H),
                         ("gap", GAP), ("row3(table+bars+insights)", ROW_C_H), ("gap", GAP),
                         ("timeline", TIMELINE_H)]:
            print(f"  {name:28s} y={y:4d} h={h:4d}  ({h/1204:.1%} of canvas height)")
            y += h
        print(f"  video x=0 w={LEFT_W} ({LEFT_W/CANVAS_W:.1%})  |  radar x={LEFT_W} w={RIGHT_W} ({RIGHT_W/CANVAS_W:.1%})")
        print(f"  canvas total height check: {y} (must be 1204)")
        cap.release()
        return

    f_start = int(args.start_sec * fps)
    f_end = int(args.end_sec * fps) if args.end_sec is not None else n_total - 1
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_path, fourcc, fps, (CANVAS_W, 1204))
    n_written = 0
    for f in range(f_start, f_end + 1):
        canvas = render_frame(f)
        if canvas is None:
            break
        writer.write(canvas)
        n_written += 1
        if n_written % 600 == 0:
            print(f"...{n_written} frames written ({f/fps:.1f}s)", flush=True)
    writer.release()
    cap.release()
    print(f"Wrote {args.out_path}: {n_written} frames, canvas {CANVAS_W}x1204")


if __name__ == "__main__":
    main()
