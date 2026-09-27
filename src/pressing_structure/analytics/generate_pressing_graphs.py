"""
Static, full-120s-clip summary graphs for the pressing V2 report
(Part 2 of the V2 brief). Reads the SAME analytics already computed for
the dashboard (recomputed here fresh, since it's cheap -- ~2s -- rather
than depending on the dashboard render having been run first). Writes
PNGs to `pressing_structure/outputs/v2_120s/graphs/`.

**Sample-size honesty**: this clip has only 11 detected pressing
episodes. Every "by outcome" plot below is EXPLICITLY labeled
"descriptive/exploratory (n=11, NOT statistical evidence)" in its own
title -- these are illustrative summaries of what happened in this one
120-second window, not a claim generalizable beyond it.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PARENT = os.path.dirname(PROJECT_ROOT)
sys.path[:0] = [PROJECT_ROOT, PROJECT_PARENT]

from pressing_structure.analytics.cleaned_tracking_view import build_cleaned_view, load_ball_view  # noqa: E402
from pressing_structure.analytics.pressing import (build_possession, build_pressing_features,  # noqa: E402
                                                     classify_press_outcomes_v2, detect_pressing_events)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "v2_120s", "graphs")
OUTCOME_COLOR = {"BALL_REGAIN": "#2e7d32", "FORCED_BACKWARD": "#ef6c00", "FORCED_LATERAL": "#f9a825",
                  "ESCAPED_PRESS": "#c62828", "UNCERTAIN": "#757575"}
EXPLORATORY_NOTE = "descriptive/exploratory only -- NOT statistical evidence"


def _savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_episodes_over_time(events: list, n: int):
    fig, ax = plt.subplots(figsize=(11, 2.6))
    for e in events:
        ax.barh(0, e["end_time_sec"] - e["start_time_sec"] + 0.15, left=e["start_time_sec"],
                color=OUTCOME_COLOR.get(e["outcome_v2"], "#999"), height=0.6)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, 120)
    ax.set_title(f"Pressing episodes over the full 120s (n={n})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in OUTCOME_COLOR.values()]
    ax.legend(handles, OUTCOME_COLOR.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=5, fontsize=8)
    _savefig(fig, "01_episodes_over_time.png")


def plot_duration_hist(events: list, n: int):
    durations = [e["duration_sec"] for e in events]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(durations, bins=min(8, max(3, n)), color="#455a64", edgecolor="white")
    ax.set_xlabel("Episode duration (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Episode duration distribution ({EXPLORATORY_NOTE}, n={n})")
    _savefig(fig, "02_duration_distribution.png")


def _strip_by_outcome(events: list, key: str, ylabel: str, title: str, fname: str, scale: float = 1.0):
    by_outcome = {}
    for e in events:
        by_outcome.setdefault(e["outcome_v2"], []).append(e[key] * scale)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs, labels = [], []
    for i, (outcome, vals) in enumerate(by_outcome.items()):
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=OUTCOME_COLOR.get(outcome, "#999"), s=60, zorder=3)
        ax.scatter([i], [np.mean(vals)], color="black", marker="_", s=400, zorder=4)
        xs.append(i)
        labels.append(f"{outcome}\n(n={len(vals)})")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n({EXPLORATORY_NOTE})", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, fname)


def main():
    tracking = pl.read_parquet(os.path.join(PROJECT_ROOT, "data/tracking/testVideo1_120s/tracking.parquet"))
    cleaned = build_cleaned_view(tracking)
    ball = load_ball_view(os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3"))
    poss = build_possession(cleaned, ball)
    features = build_pressing_features(cleaned, ball, poss)
    events = detect_pressing_events(features)
    passes = pl.read_parquet(os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3/passes.parquet"))
    turnovers = pl.read_parquet(os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3/turnovers.parquet"))
    events_v2 = classify_press_outcomes_v2(events, passes, turnovers)
    events_list = events_v2.to_dicts()
    n = len(events_list)

    plot_episodes_over_time(events_list, n)
    plot_duration_hist(events_list, n)
    _strip_by_outcome(events_list, "min_nearest_defender_distance_cm", "Min nearest-defender distance (cm)",
                       "Nearest defender distance by outcome", "03_nearest_defender_distance_by_outcome.png")
    _strip_by_outcome(events_list, "mean_pressers_5m", "Mean pressers within 5m",
                       "Number of pressers by outcome", "05_n_pressers_by_outcome.png")
    _strip_by_outcome(events_list, "mean_local_numerical_balance", "Mean local numerical balance",
                       "Local numerical balance by outcome", "06_numerical_balance_by_outcome.png")

    # closing speed / compactness / open-lanes-by-outcome need per-frame
    # feature values re-joined to episodes (not stored on the event row
    # itself) -- compute per-episode means directly from `features`.
    feat_by_frame = {r["frame"]: r for r in features.to_dicts()}
    for e in events_list:
        seg = [feat_by_frame[f] for f in range(e["start_frame"], e["end_frame"] + 1) if f in feat_by_frame]
        e["mean_closing_speed_cm_s"] = float(np.mean([s["nearest_defender_closing_speed_cm_s"] or 0 for s in seg])) if seg else None
        e["mean_compactness_cm"] = float(np.mean([s["local_compactness_cm"] for s in seg if s["local_compactness_cm"] is not None])) if seg else None
        e["mean_open_lanes"] = float(np.mean([s["n_open_lanes"] for s in seg])) if seg else None
        e["mean_pitch_control"] = float(np.mean([s["pitch_control_for_carrier_team"] for s in seg if s["pitch_control_for_carrier_team"] is not None])) if seg else None

    _strip_by_outcome(events_list, "mean_closing_speed_cm_s", "Mean closing speed (cm/s)",
                       "Closing speed by outcome", "04_closing_speed_by_outcome.png")
    _strip_by_outcome(events_list, "mean_compactness_cm", "Mean local compactness (cm)",
                       "Compactness by outcome", "06b_compactness_by_outcome.png")
    _strip_by_outcome(events_list, "mean_open_lanes", "Mean open passing lanes",
                       "Open passing lanes by outcome", "07_open_lanes_by_outcome.png")
    _strip_by_outcome(events_list, "mean_pitch_control", "Mean pitch control for pressed team (0-1)",
                       "Pitch control at carrier by outcome", "08_pitch_control_by_outcome.png")

    print(f"\n{n} episodes, {len(OUT_DIR and os.listdir(OUT_DIR))} graphs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
