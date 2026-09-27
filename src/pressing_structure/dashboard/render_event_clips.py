"""
Event-centered clips (Part 5 of the V2 brief): for each detected
pressing episode, cut [episode_start - 5s, episode_end + 5s] (clamped
to the clip's own bounds) out of the ALREADY-RENDERED full V2 dashboard
(`pressing_dashboard_120s_v2.mp4`) -- no new analytics or re-rendering,
just frame-accurate extraction via OpenCV so panel/graph/timeline
context is preserved exactly as QA'd. Writes to
`pressing_structure/outputs/event_clips/`.
"""
import os
import sys

import cv2
import polars as pl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PARENT = os.path.dirname(PROJECT_ROOT)
sys.path[:0] = [PROJECT_ROOT, PROJECT_PARENT]

from pressing_structure.analytics.cleaned_tracking_view import build_cleaned_view, load_ball_view  # noqa: E402
from pressing_structure.analytics.pressing import (build_possession, build_pressing_features,  # noqa: E402
                                                     classify_press_outcomes_v2, detect_pressing_events)

SOURCE = os.path.join(PROJECT_ROOT, "outputs/v2_120s/pressing_dashboard_120s_v2.mp4")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/event_clips")
PAD_SEC = 5.0


def main():
    tracking = pl.read_parquet(os.path.join(PROJECT_ROOT, "data/tracking/testVideo1_120s/tracking.parquet"))
    cleaned = build_cleaned_view(tracking)
    ball = load_ball_view(os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3"))
    poss = build_possession(cleaned, ball)
    features = build_pressing_features(cleaned, ball, poss)
    events = detect_pressing_events(features)
    passes = pl.read_parquet(os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3/passes.parquet"))
    turnovers = pl.read_parquet(os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3/turnovers.parquet"))
    events_v2 = classify_press_outcomes_v2(events, passes, turnovers).to_dicts()

    cap = cv2.VideoCapture(SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = []
    for e in events_v2:
        f0 = max(0, e["start_frame"] - int(PAD_SEC * fps))
        f1 = min(n_total - 1, e["end_frame"] + int(PAD_SEC * fps))
        out_path = os.path.join(OUT_DIR, f"episode_{e['press_event_id']:02d}_{e['outcome_v2']}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for f in range(f0, f1 + 1):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()
        manifest.append({"episode_id": e["press_event_id"], "outcome_v2": e["outcome_v2"],
                          "episode_start_sec": e["start_time_sec"], "episode_end_sec": e["end_time_sec"],
                          "clip_start_sec": f0 / fps, "clip_end_sec": f1 / fps, "file": os.path.basename(out_path)})
        print(f"wrote {out_path} ({f0}-{f1})")
    cap.release()

    with open(os.path.join(OUT_DIR, "README.md"), "w") as fh:
        fh.write("# Pressing event-centered clips\n\n")
        fh.write(f"Cut from the QA'd full-120s V2 dashboard "
                 f"(`{os.path.relpath(SOURCE, PROJECT_ROOT)}`), not re-rendered -- same panels/graphs/"
                 f"timeline as the full dashboard. Each clip = [episode_start - {PAD_SEC:.0f}s, "
                 f"episode_end + {PAD_SEC:.0f}s], clamped to the clip's own [0, 120s] bounds. "
                 "All 11 detected pressing episodes are included (a manageable number; no sampling needed).\n\n")
        fh.write("| Episode | Outcome (v2) | Episode window (s) | Clip window (s) | File |\n|---|---|---|---|---|\n")
        for m in manifest:
            fh.write(f"| {m['episode_id']} | {m['outcome_v2']} | "
                     f"{m['episode_start_sec']:.1f}-{m['episode_end_sec']:.1f} | "
                     f"{m['clip_start_sec']:.1f}-{m['clip_end_sec']:.1f} | {m['file']} |\n")
    print(f"\n{len(manifest)} clips written to {OUT_DIR}")


if __name__ == "__main__":
    main()
