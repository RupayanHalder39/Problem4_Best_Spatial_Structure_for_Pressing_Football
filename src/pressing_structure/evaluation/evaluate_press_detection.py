"""
Part 4 validation framework: event-detection metrics for pressing
episodes against human labels. Pure evaluation code -- does not touch
`pressing_v4.py` or the FSM. Every function here works on plain dicts
(`{start_frame, end_frame, team}` at minimum) so it can be tested with
synthetic fixtures without any tracking data.

Matching rule: a predicted episode matches a ground-truth episode iff
they share the same `team` AND their temporal IoU (frame overlap /
frame union) is >= `iou_threshold` (default 0.3, a standard,
disclosed choice for event-detection tasks -- not tuned to this
project's own data). Each ground-truth episode may be matched by at
most one predicted episode (greedy, highest-IoU-first), and vice versa
-- avoids double-counting.
"""
from typing import Dict, List, Optional, Tuple


def temporal_iou(a: Dict, b: Dict) -> float:
    """Frame-range intersection-over-union of two {start_frame,end_frame} dicts."""
    lo = max(a["start_frame"], b["start_frame"])
    hi = min(a["end_frame"], b["end_frame"])
    inter = max(0, hi - lo + 1)
    union = (a["end_frame"] - a["start_frame"] + 1) + (b["end_frame"] - b["start_frame"] + 1) - inter
    return inter / union if union > 0 else 0.0


def match_episodes(predicted: List[Dict], ground_truth: List[Dict], iou_threshold: float = 0.3) -> List[Tuple[int, int, float]]:
    """Greedy highest-IoU-first matching, restricted to same-team pairs.
    Returns a list of (predicted_index, ground_truth_index, iou) for
    every accepted match."""
    candidates = []
    for pi, p in enumerate(predicted):
        for gi, g in enumerate(ground_truth):
            if p.get("team") != g.get("team"):
                continue
            iou = temporal_iou(p, g)
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))
    candidates.sort(key=lambda c: -c[0])
    used_p, used_g, matches = set(), set(), []
    for iou, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, iou))
    return matches


def evaluate_detection(predicted: List[Dict], ground_truth: List[Dict], iou_threshold: float = 0.3, fps: float = 30.0) -> Dict:
    """precision/recall/F1 (event-level, via `match_episodes`), mean
    event-level temporal IoU of matched pairs, and mean absolute
    start/end timing error (seconds) of matched pairs. Returns
    `n_predicted`/`n_ground_truth` alongside so a 0-ground-truth call
    (no labels yet) is self-evidently not a real evaluation rather than
    a silently misleading 0/0 metric."""
    matches = match_episodes(predicted, ground_truth, iou_threshold)
    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else None

    if matches:
        mean_iou = sum(m[2] for m in matches) / len(matches)
        start_errs = [abs(predicted[pi]["start_frame"] - ground_truth[gi]["start_frame"]) / fps for pi, gi, _ in matches]
        end_errs = [abs(predicted[pi]["end_frame"] - ground_truth[gi]["end_frame"]) / fps for pi, gi, _ in matches]
        mean_start_err_sec = sum(start_errs) / len(start_errs)
        mean_end_err_sec = sum(end_errs) / len(end_errs)
    else:
        mean_iou = mean_start_err_sec = mean_end_err_sec = None

    return dict(n_predicted=len(predicted), n_ground_truth=len(ground_truth), n_matched=tp,
                precision=precision, recall=recall, f1=f1, mean_temporal_iou=mean_iou,
                mean_start_timing_error_sec=mean_start_err_sec, mean_end_timing_error_sec=mean_end_err_sec,
                iou_threshold=iou_threshold)


def run_on_real_pipeline_output(episodes: List[Dict], human_labeled_episodes: Optional[List[Dict]]) -> Dict:
    """Entry point for real data: `episodes` from `pressing_v4.
    build_pressing_v4`, `human_labeled_episodes` from the labeling tool's
    export (Part 12) or None. Converts `pressing_v4` episodes'
    `onset_frame`/`end_frame` into the `{start_frame,end_frame,team}`
    shape this module expects. Reports honestly rather than fabricating
    a result when no human labels exist yet."""
    predicted = [{"start_frame": e["onset_frame"], "end_frame": e["end_frame"], "team": e["team"]} for e in episodes]
    if not human_labeled_episodes:
        return dict(status="AWAITING_REAL_MANUAL_LABELS", n_predicted=len(predicted), n_ground_truth=0)
    return {"status": "EVALUATED", **evaluate_detection(predicted, human_labeled_episodes)}
