#!/usr/bin/env python3
"""Export one verified feature row per detected press episode."""

import argparse
from pathlib import Path

import polars as pl

from tactical_shared.coordinates import DEFAULT_PITCH
from tactical_shared.tracking import build_quality_view, current_roles
from pressing_structure.analytics.cleaned_tracking_view import load_ball_view
from pressing_structure.analytics.pressing_features import (
    HEURISTIC_TO_TAXONOMY,
    attribute_outcomes_all_episodes,
    extract_episode_features,
)
from pressing_structure.analytics.pressing_v4 import build_pressing_v4
from pressing_structure.analytics.pressure_field import pitch_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", required=True, type=Path)
    parser.add_argument("--analytics-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--match-id", default="testVideo1_120s")
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    required = [
        args.tracking,
        args.analytics_dir / "ball_trajectory.parquet",
        args.analytics_dir / "passes.parquet",
        args.analytics_dir / "turnovers.parquet",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required private input(s): " + ", ".join(missing))

    tracking = pl.read_parquet(args.tracking)
    n_total = int(tracking["frame"].max()) + 1
    quality = build_quality_view(tracking, fps=args.fps, pitch=DEFAULT_PITCH)
    players_by_frame = {}
    for row in quality.to_dicts():
        players_by_frame.setdefault(row["frame"], []).append(row)

    ball_rows = load_ball_view(str(args.analytics_dir)).to_dicts()
    ball_by_frame = {row["frame"]: row for row in ball_rows}
    roles = current_roles(players_by_frame, ball_by_frame, n_total, fps=args.fps, pitch=DEFAULT_PITCH)
    passes = pl.read_parquet(args.analytics_dir / "passes.parquet").to_dicts()
    turnovers = pl.read_parquet(args.analytics_dir / "turnovers.parquet").to_dicts()

    detected = build_pressing_v4(players_by_frame, roles, passes, turnovers, fps=args.fps, pitch=DEFAULT_PITCH)
    episodes = detected["episodes"]
    attributed = attribute_outcomes_all_episodes(
        episodes, passes, turnovers, DEFAULT_PITCH, args.fps, players_by_frame
    )
    outcomes = {row["episode_id"]: row for row in attributed}
    gx, gy = pitch_grid(DEFAULT_PITCH)

    rows = []
    skipped = 0
    for episode in episodes:
        row = extract_episode_features(episode, players_by_frame, roles, gx, gy, DEFAULT_PITCH)
        if row is None:
            skipped += 1
            continue
        outcome = outcomes[episode["episode_id"]]
        raw = outcome["heuristic_outcome"]
        row.update(
            match_id=args.match_id,
            onset_frame=episode["onset_frame"],
            onset_time=episode["onset_time"],
            active_start=episode["active_start"],
            end_frame=episode["end_frame"],
            end_time=episode["end_time"],
            termination_reason=episode["termination_reason"],
            supported_duration_sec=episode["supported_duration_sec"],
            observations=episode["observations"],
            max_state_reached="ACTIVE_PRESS" if episode["active_start"] is not None else "PRESS_FORMING",
            heuristic_outcome_raw=raw,
            heuristic_outcome=HEURISTIC_TO_TAXONOMY.get(raw, "UNCERTAIN"),
            outcome_evidence_id=outcome.get("outcome_evidence_id"),
            forward_progress_cm=outcome.get("forward_progress_cm"),
            human_outcome=None,
            final_adjudicated_outcome=None,
        )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(args.output)
    distribution = {}
    for row in rows:
        label = row["heuristic_outcome"]
        distribution[label] = distribution.get(label, 0) + 1
    print(f"wrote {args.output}: {len(rows)} rows, {skipped} skipped, {len(rows[0]) if rows else 0} columns")
    print(f"heuristic_outcome distribution: {distribution}")


if __name__ == "__main__":
    main()
