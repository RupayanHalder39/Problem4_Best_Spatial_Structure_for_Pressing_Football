"""
Part 4 validation framework: participant-identification metrics
(primary presser accuracy, participant set precision/recall/F1)
against human labels. Pure evaluation code, plain dicts/sets only.
"""
from typing import Dict, List, Optional


def participant_set_metrics(predicted_ids: List[int], true_ids: List[int]) -> Dict:
    """Set-overlap precision/recall/F1 on participant track ids for one episode."""
    p, t = set(predicted_ids or []), set(true_ids or [])
    tp = len(p & t)
    fp = len(p - t)
    fn = len(t - p)
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not t else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not p else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def primary_presser_correct(predicted_primary: Optional[int], true_primary: Optional[int]) -> Optional[bool]:
    """None (not a scoreable case) if either side is None (a human
    annotator may genuinely record `primary_presser=null` for a
    simultaneous double-team -- that is not a wrong prediction, it's an
    excluded case, per PRESS_LABELING_PROTOCOL.md."""
    if predicted_primary is None or true_primary is None:
        return None
    return predicted_primary == true_primary


def evaluate_participants(matched_episodes: List[Dict]) -> Dict:
    """`matched_episodes`: list of dicts, each with `predicted_participant_ids`,
    `true_participant_ids`, `predicted_primary_presser`,
    `true_primary_presser` -- typically built by joining
    `evaluate_press_detection.match_episodes`'s matched pairs to the
    real FSM `participant_ids` and the human `pressing_players`/
    `primary_presser` labels. Aggregates per-episode set metrics into a
    single mean (macro, not micro -- every episode weighted equally
    regardless of how many participants it has) plus overall primary-
    presser accuracy over only the scoreable cases."""
    if not matched_episodes:
        return dict(status="NO_MATCHED_EPISODES", n_episodes=0)
    per_episode = [participant_set_metrics(e["predicted_participant_ids"], e["true_participant_ids"]) for e in matched_episodes]
    mean_precision = sum(m["precision"] for m in per_episode) / len(per_episode)
    mean_recall = sum(m["recall"] for m in per_episode) / len(per_episode)
    mean_f1 = sum(m["f1"] for m in per_episode) / len(per_episode)

    primary_results = [primary_presser_correct(e.get("predicted_primary_presser"), e.get("true_primary_presser")) for e in matched_episodes]
    scoreable = [r for r in primary_results if r is not None]
    primary_accuracy = sum(scoreable) / len(scoreable) if scoreable else None

    return dict(status="EVALUATED", n_episodes=len(matched_episodes),
                participant_precision=mean_precision, participant_recall=mean_recall, participant_f1=mean_f1,
                primary_presser_accuracy=primary_accuracy, primary_presser_scoreable_n=len(scoreable))


def run_on_real_pipeline_output(episodes_with_participants: List[Dict], human_labels: Optional[List[Dict]]) -> Dict:
    """Entry point for real data (mirrors `evaluate_press_detection.
    run_on_real_pipeline_output`'s honesty contract): reports
    `AWAITING_REAL_MANUAL_LABELS` rather than a fabricated metric when
    no human labels exist yet."""
    if not human_labels:
        return dict(status="AWAITING_REAL_MANUAL_LABELS", n_episodes=len(episodes_with_participants))
    return evaluate_participants(human_labels)
