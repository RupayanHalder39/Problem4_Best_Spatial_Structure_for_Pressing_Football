"""
Part 4 validation framework: outcome-classification metrics
(accuracy, macro F1, confusion matrix) and inter-annotator reliability
(Cohen's kappa), both against real human labels once they exist.
Uses `sklearn.metrics` for the standard metrics (never reimplemented)
-- a declared project dependency (`requirements-working.txt`,
scikit-learn 1.9.0), same as `models/baselines.py`.
"""
from typing import Dict, List, Optional

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

# The 9-label taxonomy from PRESS_LABELING_PROTOCOL.md section 1 --
# duplicated here (not imported from a shared module) only as a
# convenience default for `labels=`; callers may always pass their own.
OUTCOME_LABELS = [
    "BALL_REGAIN", "FORCED_BACKWARD", "FORCED_LATERAL", "FORCED_LONG_CLEARANCE",
    "REDUCED_FORWARD_PROGRESSION", "FORCED_LOW_VALUE_PASS", "PRESS_ESCAPED",
    "DANGEROUS_FORWARD_PROGRESSION", "UNCERTAIN",
]


def evaluate_outcomes(predicted: List[str], true: List[str], labels: Optional[List[str]] = None) -> Dict:
    """`predicted`/`true`: parallel lists of outcome-taxonomy strings
    (PRESS_LABELING_PROTOCOL.md section 1), same length, no None
    entries (filter those out before calling -- a missing label is not
    a UNCERTAIN prediction, it's an absent ground-truth row)."""
    if len(predicted) != len(true):
        raise ValueError("predicted and true must be the same length")
    if not predicted:
        return dict(status="NO_LABELED_EPISODES", n=0)
    labels = labels or sorted(set(true) | set(predicted))
    return dict(status="EVALUATED", n=len(predicted),
                accuracy=float(accuracy_score(true, predicted)),
                macro_f1=float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)),
                labels=labels,
                confusion_matrix=confusion_matrix(true, predicted, labels=labels).tolist())


def inter_annotator_kappa(annotator_a: List[str], annotator_b: List[str], labels: Optional[List[str]] = None) -> Dict:
    """Cohen's kappa between two independent annotators' `human_outcome`
    labels on the SAME set of episodes (parallel lists, same order,
    same episodes -- caller's responsibility to align them by
    `episode_id` before calling)."""
    if len(annotator_a) != len(annotator_b):
        raise ValueError("annotator_a and annotator_b must be the same length")
    if not annotator_a:
        return dict(status="AWAITING_SECOND_ANNOTATOR", n=0, kappa=None)
    labels = labels or sorted(set(annotator_a) | set(annotator_b))
    return dict(status="EVALUATED", n=len(annotator_a), kappa=float(cohen_kappa_score(annotator_a, annotator_b, labels=labels)))


def run_on_real_pipeline_output(episode_rows: List[Dict]) -> Dict:
    """`episode_rows`: the real `pressing_episode_features.parquet` rows
    (as dicts). Uses `final_adjudicated_outcome` as ground truth when
    present, else `human_outcome`, else reports honestly that no real
    labels exist yet -- `heuristic_outcome` alone is NEVER compared
    against itself to fabricate a metric."""
    pairs = [(r["heuristic_outcome"], r.get("final_adjudicated_outcome") or r.get("human_outcome"))
             for r in episode_rows]
    labeled = [(p, t) for p, t in pairs if t is not None]
    if not labeled:
        return dict(status="AWAITING_REAL_MANUAL_LABELS", n_episodes=len(episode_rows), n_labeled=0)
    predicted, true = zip(*labeled)
    return evaluate_outcomes(list(predicted), list(true))
