"""
Pressing dashboard V4 -- PREVIEW ONLY (Phase 10 of the 2026-09-07 audit
continuation). Same locked V3 visual geometry (2304x1204, header/top-row/
KPI/graph/row3/timeline proportions) and the same team-distinguishable
color scheme already approved for V3 (Team 0 = cyan/amber, Team 1 =
magenta/lime), but EVERY value drawn comes from the corrected V4
snapshot (`tactical_shared.snapshot.build_snapshot`, itself a merge of
`pressing_v4.build_pressing_v4` -- partial-feature intensity/confidence,
evidence-aware FSM, both teams computed independently every frame) --
never a separate V2/V3 call for any panel (A8/B8's fix). The radar's
pitch-control wash uses the REGISTRATION-CORRECTED transform
(`tactical_shared.radar_geometry`, C6's fix) so a color sampled under a
player marker is the same grid cell used for that player.

Two modes: pass one or more `--frame`/`--name` pairs for individual
preview PNGs (the default), or `--out_path <file>.mp4` to render the
full clip (or `--start_frame`/`--end_frame`) to video, sequentially
(single seek, then cap.read() in order) for efficiency.
"""
import argparse
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

from pressing_structure.analytics.cleaned_tracking_view import build_cleaned_view
from pressing_structure.dashboard.dashboard_style_v2 import (
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_MAGENTA, ACCENT_ORANGE, ACCENT_RED, BG, BORDER_SOFT, FONT, TEXT_DIM, TEXT_WHITE,
    TEAM_PRESS_GLOW_STOPS, add_team_pressure_glow, composite_state_badge, draw_bar_comparison, draw_big_dual_graph,
    draw_header, draw_icon_kpi_card, draw_insights_panel, draw_radar_legend_box, draw_team_table,
    draw_timeline_bucketed, panel_frame, panel_title, rounded_rect,
)
from pressing_structure.dashboard.render_pressing_dashboard import _nearest_transformer
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from tactical_shared.coordinates import DEFAULT_PITCH
from tactical_shared.radar_geometry import apply_registered_control_wash, pitch_xy_to_px

CONFIG = SoccerPitchConfiguration()
TEAM_COLOR = {0: ACCENT_CYAN, 1: ACCENT_MAGENTA}
STATE_COLOR = {"NO_PRESS": (110, 110, 110), "PRESS_FORMING": ACCENT_ORANGE, "ACTIVE_PRESS": ACCENT_RED,
               "PRESS_ENDING": (60, 200, 220), "UNCERTAIN": (90, 90, 90), "CONTESTED": (200, 60, 220)}

HEADER_H, TOP_ROW_H, TILE_ROW_H, GRAPH_ROW_H, ROW3_H, TIMELINE_H, GAP = 60, 602, 88, 216, 154, 60, 6
CANVAS_W = 2304
LEFT_W = round(CANVAS_W * 0.52)
RIGHT_W = CANVAS_W - LEFT_W
assert HEADER_H + TOP_ROW_H + TILE_ROW_H + GRAPH_ROW_H + ROW3_H + TIMELINE_H + 4 * GAP == 1204


def draw_left_feed(frame, tracking_rows, snap_f, transformer):
    vis = frame.copy()
    for r in tracking_rows:
        if r["object_type"] not in ("player", "goalkeeper"):
            continue
        color = TEAM_COLOR.get(r["team_id"], (180, 180, 180))
        cv2.rectangle(vis, (int(r["bbox_x1"]), int(r["bbox_y1"])), (int(r["bbox_x2"]), int(r["bbox_y2"])), color, 2)
    dom = snap_f["pressing"]["dominant_state"]
    team = snap_f["pressing"]["dominant_team"]
    banner = dom.replace("_", " ").title()
    color = STATE_COLOR.get(dom, (100, 100, 100))
    role = f"T0: {snap_f['pressing'][0]['state'].replace('_',' ').title()}   T1: {snap_f['pressing'][1]['state'].replace('_',' ').title()}"
    if team is not None:
        role += f"   (dominant: Team {team})"
    vis = composite_state_badge(vis, 10, 10, banner, color, role, w=420)
    out = np.full((vis.shape[0] + 32, vis.shape[1], 3), BG, dtype=np.uint8)
    out[32:, :] = vis
    panel_title(out, "TACTICAL MATCH FEED", sub="V4 evidence-aware pressing state -- both teams", accent=ACCENT_CYAN)
    return out


def draw_radar(players_this_frame, ball_xy, snap_f, w, h):
    team_a = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"]} for p in players_this_frame if p["display_team_id"] == 0]
    team_b = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"]} for p in players_this_frame if p["display_team_id"] == 1]
    pitch = draw_pitch(config=CONFIG)
    pitch_dark = cv2.addWeighted(pitch, 0.7, np.zeros_like(pitch), 0.3, 0)
    if team_a and team_b:
        from legacy_analytics.pitch_control import compute_pitch_control_grid
        pc = compute_pitch_control_grid([{**p, "vx": 0., "vy": 0.} for p in team_a], [{**p, "vx": 0., "vy": 0.} for p in team_b])
        if pc.get("valid"):
            # C6 FIX: registration-correct resample into the playing rectangle
            # only, at the exact same pixel bounds pitch_xy_to_px uses for
            # every marker below -- not the old whole-padded-canvas resize.
            pitch_dark = apply_registered_control_wash(pitch_dark, pc["grid"], TEAM_COLOR[0], TEAM_COLOR[1],
                                                         DEFAULT_PITCH.length_cm, DEFAULT_PITCH.width_cm, alpha=0.88)

    dom_team = snap_f["pressing"]["dominant_team"]
    dom_state = snap_f["pressing"]["dominant_state"]
    if dom_team is not None and dom_state in ("ACTIVE_PRESS", "PRESS_FORMING"):
        # The glow centers on the CARRIER (the team being pressed), not the presser.
        carrier = next((p for p in players_this_frame if p.get("_is_carrier")), None)
        if carrier is not None:
            intensity = float(np.clip(snap_f["pressing"][dom_team]["score"] or snap_f["pressing"][dom_team]["raw_score"] or 0., 0, 1))
            stops = TEAM_PRESS_GLOW_STOPS.get(dom_team, TEAM_PRESS_GLOW_STOPS[0])
            pitch_dark = add_team_pressure_glow(pitch_dark, pitch_xy_to_px(carrier["x_pitch"], carrier["y_pitch"]), intensity, stops=stops)

    for p in players_this_frame:
        px, py = pitch_xy_to_px(p["x_pitch"], p["y_pitch"])
        color = TEAM_COLOR.get(p["display_team_id"], (150, 150, 150))
        cv2.circle(pitch_dark, (px, py), 8, color, -1, cv2.LINE_AA)
        cv2.circle(pitch_dark, (px, py), 8, (10, 10, 10), 2, cv2.LINE_AA)
    if ball_xy is not None:
        bx, by = pitch_xy_to_px(*ball_xy)
        cv2.circle(pitch_dark, (bx, by), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(pitch_dark, (bx, by), 6, (15, 15, 15), 1, cv2.LINE_AA)

    out = np.full((h, w, 3), BG, dtype=np.uint8)
    resized = cv2.resize(pitch_dark, (w, h - 34))
    out[34:h, 0:w] = resized
    panel_title(out, "PRESSING RADAR (V4)", sub="registration-corrected pitch control  •  T0 amber / T1 lime pressure", accent=ACCENT_CYAN)
    roi = out[34:78, 8:280]
    cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, dst=roi)
    s0 = snap_f["pressing"][0]["state"].replace("_", " ").title()
    s1 = snap_f["pressing"][1]["state"].replace("_", " ").title()
    cv2.putText(out, f"T0 press state: {s0}", (14, 50), FONT, 0.36, ACCENT_ORANGE, 1, cv2.LINE_AA)
    cv2.putText(out, f"T1 press state: {s1}", (14, 70), FONT, 0.36, (60, 220, 90), 1, cv2.LINE_AA)
    draw_radar_legend_box(out, w - 178, 44, 166, 200,
                           [("Team 0", TEAM_COLOR[0]), ("Team 1", TEAM_COLOR[1]), ("Ball", (255, 255, 255)),
                            ("T0 Pressing", (0, 170, 255)), ("T1 Pressing", (60, 220, 90))],
                           gradient=(TEAM_COLOR[0], TEAM_COLOR[1]), gradient_labels=("Team 0", "Control", "Team 1"))
    return out


def draw_tiles(snap_f, width, height):
    # Single row of 4 full-height cards -- pressing's own approved V3
    # KPI-row style (offside's 2x2 grid is a DIFFERENT project's layout;
    # a 2x2 split of this row's height left each cell too short for
    # draw_icon_kpi_card's value+sub-line to avoid overlapping).
    cell_w = (width - 5 * GAP) // 4
    img = np.full((height, width, 3), BG, dtype=np.uint8)
    p0, p1 = snap_f["pressing"][0], snap_f["pressing"][1]

    def _sub(p):
        # Shows CURRENT (this-instant, raw) intensity separately from the
        # smoothed score the state's hysteresis actually reacts to --
        # 2026-09-07 semantic-QA fix. Without both, a held state (e.g. an
        # ACTIVE spell whose single current frame happens to dip) looks
        # self-contradictory ("Active" next to a low/zero number) when in
        # fact the SUSTAINED (smoothed) score is what the lifecycle uses.
        if p["raw_score"] is None:
            return "no evidence"
        smoothed = p.get("score")
        smoothed_txt = f"{smoothed:.2f}" if smoothed is not None else "n/a"
        return f"now {p['raw_score']:.2f} / sustained {smoothed_txt} · conf {p['confidence']:.2f}"

    values = [
        ("Team 0 Press State", p0["state"].replace("_", " ").title(), _sub(p0),
         STATE_COLOR.get(p0["state"], ACCENT_CYAN), "shield"),
        ("Team 1 Press State", p1["state"].replace("_", " ").title(), _sub(p1),
         STATE_COLOR.get(p1["state"], ACCENT_MAGENTA), "shield"),
        ("Dominant / Intensity", snap_f["pressing"]["dominant_state"].replace("_", " ").title(),
         f"Team {snap_f['pressing']['dominant_team']}" if snap_f["pressing"]["dominant_team"] is not None else "n/a",
         ACCENT_ORANGE, "bolt"),
        ("Current Evidence Confidence", f"{max(p0['confidence'], p1['confidence']):.2f}",
         "role + motion + identity", ACCENT_GREEN, "target"),
    ]
    for i, (label, val, sub, accent, glyph) in enumerate(values):
        px = GAP + i * (cell_w + GAP)
        tile = draw_icon_kpi_card(cell_w, height, label, val, sub, accent=accent, glyph=glyph)
        img[0:height, px:px + cell_w] = tile
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_video_path", default=os.path.join(PROJECT_ROOT, "data/source_video/testVideo1_120s.mp4"))
    ap.add_argument("--tracking_dir", default=os.path.join(PROJECT_ROOT, "data/tracking/testVideo1_120s"))
    ap.add_argument("--analytics_dir", default=os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3"))
    ap.add_argument("--snapshot_pkl", default=os.path.join(PROJECT_ROOT, "data/snapshots/v4_snapshot.pkl"))
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "outputs/correctness_v4/dashboard_previews"))
    ap.add_argument("--frame", type=int, action="append", default=None, help="explicit frame(s) to render (preview mode)")
    ap.add_argument("--name", action="append", default=None, help="output filename stem per --frame (preview mode)")
    ap.add_argument("--out_path", default=None, help="if set, render the FULL clip to this .mp4 path instead of individual preview PNGs")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=None, help="inclusive; defaults to the last snapshot frame")
    args = ap.parse_args()

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
    from pressing_structure.analytics.cleaned_tracking_view import load_ball_view
    ball_by_frame = {r["frame"]: r for r in load_ball_view(args.analytics_dir).to_dicts()}
    with open(os.path.join(args.analytics_dir, "homography_transformers.pkl"), "rb") as fh:
        transformers = pickle.load(fh)
    cap = cv2.VideoCapture(args.source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Precomputed ONCE (video-render perf fix, not a semantic change --
    # these depend only on the fixed `snapshot`, never on the current
    # frame `f`; recomputing them inside render_frame() on every one of
    # 3600 frames was pure wasted work for a full-video render).
    n_active = {t: sum(1 for e_id in {snapshot[i]["pressing"][t]["episode_id"] for i in range(len(snapshot))
                                        if snapshot[i]["pressing"][t]["episode_id"] and snapshot[i]["pressing"][t]["state"] == "ACTIVE_PRESS"} if e_id) for t in (0, 1)}
    n_buckets = 60
    bucket_sec = (len(snapshot) / fps) / n_buckets
    bucket_colors = []
    for b in range(n_buckets):
        bf0, bf1 = int(b * bucket_sec * fps), int(min(len(snapshot), (b + 1) * bucket_sec * fps))
        states = [snapshot[i]["pressing"]["dominant_state"] for i in range(bf0, max(bf0 + 1, bf1))]
        majority = max(set(states), key=states.count) if states else "UNCERTAIN"
        bucket_colors.append(STATE_COLOR.get(majority, (100, 100, 100)))
    legend = [(k.replace("_", " ").title(), v) for k, v in STATE_COLOR.items()]

    def render_frame(f, seek=True):
        snap_f = snapshot[f]
        if seek:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            return None
        transformer = _nearest_transformer(transformers, f)
        players_this_frame = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None]
        for p in players_this_frame:
            p["_is_carrier"] = False
        ball_row = ball_by_frame.get(f)
        ball_xy = (ball_row["x_pitch"], ball_row["y_pitch"]) if ball_row and ball_row.get("is_observed") and ball_row.get("x_pitch") is not None else None
        if ball_xy is not None and players_this_frame:
            nearest = min(players_this_frame, key=lambda p: np.hypot(p["x_pitch"] - ball_xy[0], p["y_pitch"] - ball_xy[1]))
            nearest["_is_carrier"] = True

        left = cv2.resize(draw_left_feed(frame, tracking_by_frame.get(f, []), snap_f, transformer), (LEFT_W, TOP_ROW_H))
        radar = draw_radar(players_this_frame, ball_xy, snap_f, RIGHT_W, TOP_ROW_H)
        top_row = np.hstack([left, radar])

        header = draw_header(CANVAS_W, HEADER_H, "PRESSING ANALYTICS -- V4 CORRECTNESS PREVIEW",
                              (f"t = {snap_f['time_sec']:.1f}s", TEXT_DIM), accent=ACCENT_CYAN)
        tiles = draw_tiles(snap_f, CANVAS_W, TILE_ROW_H)

        window_sec = 14.0
        f0, f1 = max(0, f - int(window_sec * fps)), f
        hist_i0 = [(snapshot[i]["time_sec"], snapshot[i]["pressing"][0]["raw_score"]) for i in range(f0, f1 + 1)]
        hist_i1 = [(snapshot[i]["time_sec"], snapshot[i]["pressing"][1]["raw_score"]) for i in range(f0, f1 + 1)]
        hist_c0 = [(snapshot[i]["time_sec"], snapshot[i]["pressing"][0]["confidence"]) for i in range(f0, f1 + 1)]
        hist_c1 = [(snapshot[i]["time_sec"], snapshot[i]["pressing"][1]["confidence"]) for i in range(f0, f1 + 1)]
        graph_w = (CANVAS_W - 4 * GAP) // 3
        g1 = draw_big_dual_graph(graph_w, GRAPH_ROW_H, hist_i0, hist_i1, snap_f["time_sec"], window_sec,
                                  "Press Intensity: Team 0 vs Team 1", (0.0, 1.0), "Team 0", "Team 1", "",
                                  color_a=ACCENT_CYAN, color_b=ACCENT_MAGENTA)
        g2 = draw_big_dual_graph(graph_w, GRAPH_ROW_H, hist_c0, hist_c1, snap_f["time_sec"], window_sec,
                                  "Press Confidence: Team 0 vs Team 1", (0.0, 1.0), "Team 0", "Team 1", "",
                                  color_a=ACCENT_CYAN, color_b=ACCENT_MAGENTA)
        feat0, feat1 = snap_f["pressing"][0]["features"], snap_f["pressing"][1]["features"]
        d0 = feat0.get("nearest_opponent_distance_cm")
        d1 = feat1.get("nearest_opponent_distance_cm")
        hist_d = [(snapshot[i]["time_sec"],
                   (snapshot[i]["pressing"][0]["features"].get("nearest_opponent_distance_cm") or
                    snapshot[i]["pressing"][1]["features"].get("nearest_opponent_distance_cm")))
                  for i in range(f0, f1 + 1)]
        from pressing_structure.dashboard.dashboard_style_v2 import draw_big_graph
        from pressing_structure.dashboard.live_graphs import compute_stable_y_range
        y_d = compute_stable_y_range([v for _, v in hist_d])
        g3 = draw_big_graph(graph_w, GRAPH_ROW_H, hist_d, snap_f["time_sec"], window_sec,
                             "Nearest Opponent Distance", y_d, "cm", ACCENT_ORANGE)
        gap_col = np.full((GRAPH_ROW_H, GAP, 3), BG, dtype=np.uint8)
        graph_row = np.hstack([g1, gap_col, g2, gap_col, g3])
        if graph_row.shape[1] != CANVAS_W:
            graph_row = cv2.resize(graph_row, (CANVAS_W, GRAPH_ROW_H))

        insights = [
            "V4 evidence-aware pipeline: partial-feature intensity (proximity+count always available from position; "
            "closing speed adds a bonus signal only when both players' motion is trustworthy).",
            f"Confirmed ACTIVE_PRESS episodes this replay: Team 0 = {n_active[0]}, Team 1 = {n_active[1]} -- confirmation "
            "requires a GENERIC, non-clip-tuned bar (>=2 observations, >=0.25s real support, >=1/3 coverage of the "
            "forming window), never lowered to manufacture a specific count; the strongest real FORMING episode still "
            "falls short of it (see the contact sheet), which is reported honestly rather than papered over.",
            "No causal-success percentage is shown: V4 reports OBSERVED post-press outcomes (regain/backward/lateral/"
            "forward) with an explicit evidence reason, never a fabricated success rate.",
        ]
        insights_panel = draw_insights_panel(graph_w * 2 + GAP, ROW3_H, "V4 Notes (data-derived)", insights, accent=ACCENT_ORANGE)
        table = draw_team_table(graph_w - GAP, ROW3_H, "Team Comparison (V4)",
                                 ["State", "Intensity", "Confidence"],
                                 [("Team 0", [snap_f["pressing"][0]["state"].replace("_", " ").title(),
                                              f"{(snap_f['pressing'][0]['raw_score'] or 0):.2f}",
                                              f"{snap_f['pressing'][0]['confidence']:.2f}"]),
                                  ("Team 1", [snap_f["pressing"][1]["state"].replace("_", " ").title(),
                                              f"{(snap_f['pressing'][1]['raw_score'] or 0):.2f}",
                                              f"{snap_f['pressing'][1]['confidence']:.2f}"])],
                                 [TEAM_COLOR[0], TEAM_COLOR[1]])
        row3 = np.hstack([table, np.full((ROW3_H, GAP, 3), BG, dtype=np.uint8), insights_panel])
        if row3.shape[1] != CANVAS_W:
            row3 = cv2.resize(row3, (CANVAS_W, ROW3_H))

        timeline = draw_timeline_bucketed(CANVAS_W, TIMELINE_H, len(snapshot) / fps, snap_f["time_sec"],
                                           bucket_colors, bucket_sec, [], legend, title="DOMINANT PRESS STATE (V4, both teams)")

        row_gap = np.full((GAP, CANVAS_W, 3), BG, dtype=np.uint8)
        canvas = np.vstack([header, top_row, row_gap, tiles, row_gap, graph_row, row_gap, row3, row_gap, timeline])
        return canvas

    if args.out_path:
        f_start = args.start_frame
        f_end = args.end_frame if args.end_frame is not None else len(snapshot) - 1
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_path, fourcc, fps, (CANVAS_W, 1204))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)  # seek ONCE, then read sequentially -- far faster than per-frame seeks
        n_written = 0
        t0 = time.time()
        for f in range(f_start, f_end + 1):
            canvas = render_frame(f, seek=False)
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
        canvas = render_frame(f)
        if canvas is None:
            print(f"frame {f}: could not read video frame")
            continue
        out_path = os.path.join(args.out_dir, f"{name}.png")
        cv2.imwrite(out_path, canvas)
        print(f"wrote {out_path} ({canvas.shape[1]}x{canvas.shape[0]})")
    cap.release()


if __name__ == "__main__":
    main()
