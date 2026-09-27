"""
Part 6: short-horizon outcome prediction. Defines the horizon-based
target construction (never using data past the observation cutoff as
input) and reuses `models.baselines`'s existing fitting/evaluation
machinery for the actual modeling -- no new model class, only a new
LABEL CONSTRUCTION layer, per Part 6's own framing ("define a
short-horizon prediction target").

HORIZONS: 1s, 2s, 3s, 5s (fixed, as specified) -- a horizon is applied
at an episode's `end_frame` (the last real observed evidence frame, NOT
`onset_frame` -- using onset as the cutoff would leak the episode's own
still-unfolding evidence into "future"). Only pass/turnover evidence
whose `start_frame` falls within `[end_frame, end_frame + horizon_frames]`
is used to build the label -- evidence starting after the horizon is
excluded, never peeked at.

SUCCESSFUL-PRESSURE GROUPING (documented explicitly, per the research-
phase brief): reuses PRESS_LABELING_PROTOCOL.md section 1's grouping --
`{BALL_REGAIN, FORCED_BACKWARD, FORCED_LATERAL, FORCED_LONG_CLEARANCE,
REDUCED_FORWARD_PROGRESSION, FORCED_LOW_VALUE_PASS}` = successful;
`{PRESS_ESCAPED, DANGEROUS_FORWARD_PROGRESSION}` = escape/progression;
`UNCERTAIN` excluded from the binary target entirely.
"""
from typing import Dict, List, Optional, Sequence

from pressing_structure.analytics.pressing_v4 import evidence_quality
from pressing_structure.analytics.pressing_features import HEURISTIC_TO_TAXONOMY
from pressing_structure.models.baselines import fit_eval_grouped, MIN_ROWS_PER_CLASS_TO_FIT
from tactical_shared.coordinates import DEFAULT_PITCH

HORIZONS_SEC = (1.0, 2.0, 3.0, 5.0)

SUCCESSFUL_PRESSURE_OUTCOMES = frozenset({
    "BALL_REGAIN", "FORCED_BACKWARD", "FORCED_LATERAL", "FORCED_LONG_CLEARANCE",
    "REDUCED_FORWARD_PROGRESSION", "FORCED_LOW_VALUE_PASS",
})
ESCAPE_PROGRESSION_OUTCOMES = frozenset({"PRESS_ESCAPED", "DANGEROUS_FORWARD_PROGRESSION"})


def horizon_multiclass_label(episode: Dict, passes: Sequence[Dict], turnovers: Sequence[Dict],
                              horizon_sec: float, pitch=DEFAULT_PITCH, fps: float = 30.0,
                              players_by_frame: Optional[Dict] = None) -> str:
    """The SAME evidence-matching principle as `pressing_features.
    attribute_outcomes_all_episodes`, restricted to evidence starting
    within `end_frame + horizon_sec*fps` (never beyond it -- the
    horizon IS the cutoff, not merely a suggestion). Returns a
    `heuristic_outcome_raw`-style label (`BALL_REGAIN`/`BACKWARD_ACTION`/
    `LATERAL_ACTION`/`FORWARD_PROGRESSION`/`UNCERTAIN`), mapped by the
    caller via `HEURISTIC_TO_TAXONOMY` into the full taxonomy."""
    horizon_frames = horizon_sec * fps
    end_frame = episode["end_frame"]
    team = episode["team"]
    evidence = [("pass", p) for p in passes] + [("turnover", p) for p in turnovers]
    cands = []
    for kind, p in evidence:
        source = p.get("source_team_id") if kind == "turnover" else p.get("team_id")
        if 1 - team != source:
            continue
        if not (end_frame <= p["start_frame"] <= end_frame + horizon_frames):
            continue
        if pitch.period(end_frame) != pitch.period(p["start_frame"]):
            continue
        cands.append((kind, p))
    if not cands:
        return "UNCERTAIN"
    kind, p = min(cands, key=lambda kp: kp[1]["start_frame"])  # earliest evidence within the horizon, never the latest (would peek further than necessary)
    valid, _ = evidence_quality(p, kind, players_by_frame)
    if not valid:
        return "UNCERTAIN"
    if kind == "turnover":
        return "BALL_REGAIN" if p.get("receiver_team_id") == team else "UNCERTAIN"
    progress = pitch.progress(p.get("team_id"), p["start_x"], p["end_x"], p["start_frame"])
    return "BACKWARD_ACTION" if progress <= -300 else "FORWARD_PROGRESSION" if progress >= 300 else "LATERAL_ACTION"


def binary_successful_pressure_label(taxonomy_label: str) -> Optional[bool]:
    """True/False for the two documented groupings, None for `UNCERTAIN`
    (excluded from the binary target entirely, never coerced to either
    side)."""
    if taxonomy_label in SUCCESSFUL_PRESSURE_OUTCOMES:
        return True
    if taxonomy_label in ESCAPE_PROGRESSION_OUTCOMES:
        return False
    return None


def build_horizon_dataset(episode_feature_rows: Sequence[Dict], episodes_by_id: Dict[str, Dict],
                           passes: Sequence[Dict], turnovers: Sequence[Dict], horizon_sec: float,
                           pitch=DEFAULT_PITCH, fps: float = 30.0, players_by_frame: Optional[Dict] = None) -> List[Dict]:
    """One row per episode-feature-row, with `horizon_{h}s_multiclass`
    and `horizon_{h}s_binary` columns added -- built ONLY from each
    row's own already-computed features (the observation, at
    `onset_frame`, strictly before `end_frame`) plus the horizon label
    (strictly after `end_frame`). No feature is recomputed using
    post-cutoff frames."""
    out = []
    for row in episode_feature_rows:
        ep = episodes_by_id.get(row["episode_id"])
        if ep is None:
            continue
        raw = horizon_multiclass_label(ep, passes, turnovers, horizon_sec, pitch, fps, players_by_frame)
        taxonomy = HEURISTIC_TO_TAXONOMY.get(raw, "UNCERTAIN")
        out.append({**row,
                    f"horizon_{horizon_sec:g}s_multiclass": taxonomy,
                    f"horizon_{horizon_sec:g}s_binary": binary_successful_pressure_label(taxonomy)})
    return out


def evaluate_horizon_binary(rows: Sequence[Dict], horizon_sec: float, baseline_name: str = "D") -> Dict:
    """Match-grouped evaluation of the binary "successful pressure"
    target at one horizon, reusing `models.baselines.fit_eval_grouped`
    unchanged (never a second, divergent CV implementation)."""
    col = f"horizon_{horizon_sec:g}s_binary"
    labeled = [r for r in rows if r.get(col) is not None]
    if not labeled:
        return dict(status="AWAITING_REAL_MANUAL_LABELS_OR_MORE_EVIDENCE", horizon_sec=horizon_sec, n_rows=len(rows))
    y = [r[col] for r in labeled]
    groups = [r.get("match_id") for r in labeled]
    class_counts = {c: y.count(c) for c in set(y)}
    if len(set(groups)) < 2 or any(v < MIN_ROWS_PER_CLASS_TO_FIT for v in class_counts.values()):
        return dict(status="INSUFFICIENT_LABELED_DATA", horizon_sec=horizon_sec, n_rows=len(labeled), class_counts=class_counts)
    return {"status": "EVALUATED", "horizon_sec": horizon_sec, **fit_eval_grouped(baseline_name, labeled, y, groups)}
