"""
Research-phase export script (Part 3): runs the real V4 pressing
pipeline against the real tracking data, extracts one feature row per
real episode via `pressing_features.extract_episode_features`, attaches
`heuristic_outcome` via `attribute_outcomes_all_episodes`, and writes

    pressing_structure/outputs/research_dataset/pressing_episode_features.parquet

No fabricated rows -- an episode whose onset frame lacks resolvable
carrier/defender positions is skipped (with a printed count), never
padded with placeholder values.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PARENT = os.path.dirname(PROJECT_ROOT)
sys.path[:0] = [PROJECT_ROOT, PROJECT_PARENT]

import polars as pl  # noqa: E402

from tactical_shared.coordinates import DEFAULT_PITCH  # noqa: E402
from tactical_shared.tracking import build_quality_view, current_roles  # noqa: E402
from pressing_structure.analytics.cleaned_tracking_view import load_ball_view  # noqa: E402
from pressing_structure.analytics.pressing_v4 import build_pressing_v4  # noqa: E402
from pressing_structure.analytics.pressure_field import pitch_grid  # noqa: E402
from pressing_structure.analytics.pressing_features import (  # noqa: E402
    extract_episode_features, attribute_outcomes_all_episodes, HEURISTIC_TO_TAXONOMY,
)

FPS = 30.0
MATCH_ID = "testVideo1_120s"
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/research_dataset")


def main():
    tracking = pl.read_parquet(os.path.join(PROJECT_ROOT, "data/tracking/testVideo1_120s/tracking.parquet"))
    n_total = int(tracking["frame"].max()) + 1
    quality = build_quality_view(tracking, fps=FPS, pitch=DEFAULT_PITCH)
    players_by_frame = {}
    for r in quality.to_dicts():
        players_by_frame.setdefault(r["frame"], []).append(r)
    ball_by_frame = {r["frame"]: r for r in load_ball_view(os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3")).to_dicts()}
    roles = current_roles(players_by_frame, ball_by_frame, n_total, fps=FPS, pitch=DEFAULT_PITCH)
    passes = pl.read_parquet(os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3/passes.parquet")).to_dicts()
    turnovers = pl.read_parquet(os.path.join(PROJECT_ROOT, "data/analytics/testVideo1_120s_v3/turnovers.parquet")).to_dicts()

    pv4 = build_pressing_v4(players_by_frame, roles, passes, turnovers, fps=FPS, pitch=DEFAULT_PITCH)
    episodes = pv4["episodes"]
    episodes_with_outcome = attribute_outcomes_all_episodes(episodes, passes, turnovers, DEFAULT_PITCH, FPS, players_by_frame)
    outcome_by_id = {e["episode_id"]: e for e in episodes_with_outcome}

    gx, gy = pitch_grid(DEFAULT_PITCH)
    rows, skipped = [], 0
    for ep in episodes:
        row = extract_episode_features(ep, players_by_frame, roles, gx, gy, DEFAULT_PITCH)
        if row is None:
            skipped += 1
            continue
        oc = outcome_by_id[ep["episode_id"]]
        row.update(match_id=MATCH_ID,
                    onset_frame=ep["onset_frame"], onset_time=ep["onset_time"],
                    active_start=ep["active_start"], end_frame=ep["end_frame"], end_time=ep["end_time"],
                    termination_reason=ep["termination_reason"],
                    supported_duration_sec=ep["supported_duration_sec"], observations=ep["observations"],
                    max_state_reached="ACTIVE_PRESS" if ep["active_start"] is not None else "PRESS_FORMING",
                    heuristic_outcome_raw=oc["heuristic_outcome"],
                    heuristic_outcome=HEURISTIC_TO_TAXONOMY.get(oc["heuristic_outcome"], "UNCERTAIN"),
                    outcome_evidence_id=oc["outcome_evidence_id"], forward_progress_cm=oc["forward_progress_cm"],
                    human_outcome=None, final_adjudicated_outcome=None)
        rows.append(row)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "pressing_episode_features.parquet")
    df = pl.DataFrame(rows)
    df.write_parquet(out_path)
    print(f"wrote {out_path}: {len(rows)} episode rows, {skipped} skipped (unresolvable onset-frame evidence), {len(df.columns)} columns")
    print("heuristic_outcome distribution:", dict(df["heuristic_outcome"].value_counts().sort("heuristic_outcome").rows()))
    return df


if __name__ == "__main__":
    main()
