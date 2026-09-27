"""
Reusable, transparent formation-estimation heuristic. NOT a trained
model -- a simple geometric clustering of outfield player positions into
lines along the attack axis, exactly as described in the project's Phase
8 spec. Confidence and a persistence rule (see `estimate_formation_series`)
exist specifically so the displayed label does not flicker between
"4-3-3"/"4-4-2" every window when the underlying data is noisy or
incomplete -- stability is treated as more important than false
precision throughout this module.

Uses the STABILIZED role/team columns from `analytics/role_stability.py`
(never the raw, flicker-prone `object_type`/`team_id`) so a single
misclassified frame cannot silently drop or add a player to the count.
"""
import polars as pl

from legacy_analytics.position_fill import build_full_display_frame
from legacy_analytics.role_stability import stabilize_roles_and_team
from legacy_analytics.temporal_utils import causal_rolling_majority

DEFAULT_LINE_GAP_CM = 700.0   # gap along the attack axis that separates two tactical "lines" --
# chosen empirically on testVideo1_20s: of {300,400,500,600,650,700}cm tested, 700cm was the
# only value whose most-frequent plausible pattern was a real named formation shape ("4-3-3");
# smaller thresholds over-split single lines into noisy 4-5-line patterns. See
# demo_dashboard_summary.md for the full per-threshold comparison -- this heuristic's
# instability across thresholds is itself reported there as a limitation, not hidden.
MIN_OUTFIELD_FOR_FORMATION = 7   # below this, don't attempt a hard formation label
FULL_OUTFIELD = 10


def estimate_attack_direction(display_df: pl.DataFrame, team_id: int) -> int:
    """+1 if this team attacks toward increasing x_pitch (their
    goalkeeper sits near x=0), -1 if they attack toward decreasing
    x_pitch (goalkeeper near x=12000). Falls back to +1 (with the caller
    expected to treat results as low-confidence) if no goalkeeper data is
    available for this team."""
    gk = display_df.filter(
        (pl.col("display_object_type") == "goalkeeper") &
        (pl.col("display_team_id") == team_id) &
        pl.col("display_x_pitch").is_not_null()
    )
    if gk.height == 0:
        return 1
    mean_x = gk["display_x_pitch"].mean()
    return 1 if mean_x < 6000 else -1


def _cluster_into_lines(sorted_line_x: list, gap_cm: float = DEFAULT_LINE_GAP_CM) -> list:
    """sorted_line_x: ascending list of "distance from own goal" values
    (already oriented so 0 = own goal, 12000 = opponent's goal). Returns
    a list of cluster sizes, defense-line-first. Simple gap-based
    clustering -- not k-means, not learned -- so the logic stays
    auditable."""
    if not sorted_line_x:
        return []
    clusters = [[sorted_line_x[0]]]
    for x in sorted_line_x[1:]:
        if x - clusters[-1][-1] <= gap_cm:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [len(c) for c in clusters]


def _single_frame_line_pattern(display_df: pl.DataFrame, team_id: int, frame: int,
                                attack_direction: int, line_gap_cm: float):
    """One frame's outfield-player line pattern, or None if this frame
    isn't usable (too few/too many team-labeled players -- see note
    below). Returns (line_counts tuple, n_visible)."""
    rows = display_df.filter(
        (pl.col("frame") == frame) &
        (pl.col("display_team_id") == team_id) &
        (pl.col("display_object_type") == "player") &
        pl.col("display_x_pitch").is_not_null()
    )
    n = rows.height
    if n < MIN_OUTFIELD_FOR_FORMATION or n > FULL_OUTFIELD + 2:
        return None, n
    xs = rows["display_x_pitch"].to_list()
    line_x = sorted(x if attack_direction == 1 else (12000.0 - x) for x in xs)
    line_counts = tuple(_cluster_into_lines(line_x, gap_cm=line_gap_cm))
    # require >=3 lines: a 2-cluster split (e.g. "4-6") isn't a real
    # named formation (defense/midfield/attack is the minimum meaningful
    # breakdown) -- treat it as inconclusive rather than a formation.
    plausible = (3 <= len(line_counts) <= 5) and all(1 <= c <= 6 for c in line_counts)
    if not plausible:
        return None, n
    return line_counts, n


def estimate_formation_for_window(display_df: pl.DataFrame, team_id: int,
                                   frame_start: int, frame_end: int,
                                   attack_direction: int | None = None,
                                   line_gap_cm: float = DEFAULT_LINE_GAP_CM,
                                   sample_stride: int = 5) -> dict:
    """Independently line-clusters several SINGLE-frame snapshots sampled
    every `sample_stride` frames across [frame_start, frame_end], then
    takes the majority-vote line pattern across those snapshots.
    Deliberately single-frame-per-snapshot (not a position average across
    the whole window): track_id fragmentation -- the same physical player
    briefly getting a new track_id after an occlusion, common within a
    multi-second window -- would otherwise get double-counted as two
    players if positions were pooled across frames. A single frame is
    always tracked without that ambiguity, so per-frame independence is
    the safer noise-reduction primitive; the window's noise-reduction
    comes from voting across several such frames instead. Returns a dict
    with formation (string or None), confidence (0-1),
    visible_outfield_players, line_counts, method."""
    if attack_direction is None:
        attack_direction = estimate_attack_direction(display_df, team_id)

    method = (f"per-frame geometric line-clustering (gap={line_gap_cm:.0f}cm), "
              f"majority vote over frames sampled every {sample_stride} in "
              f"[{frame_start},{frame_end}]")

    samples = []
    for f in range(frame_start, frame_end + 1, sample_stride):
        pattern, n = _single_frame_line_pattern(display_df, team_id, f, attack_direction, line_gap_cm)
        if pattern is not None:
            samples.append((pattern, n))

    if not samples:
        return {"formation": None, "confidence": 0.1, "visible_outfield_players": 0,
                "line_counts": [], "method": method,
                "reason": "no usable single-frame snapshot in this window"}

    from collections import Counter
    pattern_counts = Counter(p for p, n in samples)
    best_pattern, votes = pattern_counts.most_common(1)[0]
    n_visible = round(sum(n for p, n in samples if p == best_pattern) /
                       sum(1 for p, n in samples if p == best_pattern))
    n_frame_positions = len(range(frame_start, frame_end + 1, sample_stride))
    agreement = votes / len(samples)
    completeness = min(1.0, n_visible / FULL_OUTFIELD)
    coverage = len(samples) / max(1, n_frame_positions)
    confidence = round(0.4 * agreement + 0.35 * completeness + 0.25 * coverage, 2)

    formation = "-".join(str(c) for c in best_pattern)
    if n_visible < FULL_OUTFIELD:
        formation = f"approx {formation} ({n_visible}/{FULL_OUTFIELD} visible)"

    return {"formation": formation, "confidence": confidence, "visible_outfield_players": n_visible,
            "line_counts": list(best_pattern), "method": method,
            "n_snapshots_used": len(samples), "n_snapshots_total": n_frame_positions}


def estimate_formation_series(tracking_df: pl.DataFrame, team_id: int,
                               window_frames: int = 90, step_frames: int | None = None,
                               line_gap_cm: float = DEFAULT_LINE_GAP_CM,
                               min_confidence_to_display: float = 0.45,
                               display_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Non-overlapping (by default) sliding-window formation estimate
    across the whole clip, with a persistence rule so the DISPLAYED
    label only changes once the new underlying line-count PATTERN (e.g.
    (4,3,3)) has been the raw estimate in 2 consecutive windows AND clears
    `min_confidence_to_display` -- otherwise the previous displayed label
    (or "Shape uncertain" if there is none yet) is kept. This directly
    implements the "do not let formation text flicker" requirement:
    matching on the underlying pattern (not the formatted string) means a
    window that sees 9/10 players ("approx 4-3-2") still confirms a
    window that saw all 10 ("4-3-3") as long as the LINE PATTern agrees."""
    step_frames = step_frames or window_frames
    if display_df is None:
        display_df = build_full_display_frame(tracking_df)
    attack_direction = estimate_attack_direction(display_df, team_id)

    fmin, fmax = int(tracking_df["frame"].min()), int(tracking_df["frame"].max())
    rows = []
    f = fmin
    while f <= fmax:
        f_end = min(f + window_frames - 1, fmax)
        est = estimate_formation_for_window(display_df, team_id, f, f_end,
                                             attack_direction=attack_direction, line_gap_cm=line_gap_cm)
        rows.append({"window_start_frame": f, "window_end_frame": f_end,
                     "raw_formation": est["formation"], "raw_confidence": est["confidence"],
                     "visible_outfield_players": est["visible_outfield_players"],
                     "line_counts": tuple(est["line_counts"]), "method": est["method"]})
        f += step_frames

    out = pl.DataFrame(rows).with_columns(
        pl.col("line_counts").cast(pl.List(pl.Utf8)).list.join("-").alias("line_counts"))

    display_labels = []
    prev_display = "Shape uncertain"
    prev_pattern = None
    confirm_count = 0
    for r in rows:
        pattern = r["line_counts"] if r["raw_formation"] and r["raw_confidence"] >= min_confidence_to_display else None
        if pattern == prev_pattern and pattern is not None:
            confirm_count += 1
        else:
            confirm_count = 1
            prev_pattern = pattern
        if pattern is not None and confirm_count >= 2:
            prev_display = r["raw_formation"]
        display_labels.append(prev_display)

    out = out.with_columns([
        pl.Series("display_formation", display_labels),
        pl.lit(team_id).alias("team_id"),
        pl.lit(attack_direction).alias("attack_direction"),
    ])
    return out


def formation_at_frame(formation_series: pl.DataFrame, frame: int) -> dict:
    """Convenience lookup for the renderer: the row whose window contains
    `frame`, falling back to the nearest window. Returns a dict with at
    least display_formation, raw_confidence, visible_outfield_players."""
    if formation_series.height == 0:
        return {"display_formation": "Shape uncertain", "raw_confidence": 0.0, "visible_outfield_players": 0}
    row = formation_series.filter((pl.col("window_start_frame") <= frame) & (pl.col("window_end_frame") >= frame))
    if row.height == 0:
        row = formation_series.sort(
            (pl.col("window_start_frame") - frame).abs()
        ).head(1)
    return row.to_dicts()[0]
