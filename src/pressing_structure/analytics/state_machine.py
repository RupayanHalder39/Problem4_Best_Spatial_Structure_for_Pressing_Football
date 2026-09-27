"""Legacy FSM compatibility API; V4 evidence semantics override old documentation."""
from dataclasses import dataclass


@dataclass
class FSMConfig:
    enter_th: float = 0.5
    exit_th: float = 0.3
    min_activate_frames: int = 8     # sustained >=enter_th before FORMING -> ACTIVE
    min_hold_frames: int = 10        # ACTIVE cannot leave before this many frames
    ending_frames: int = 10          # ENDING lasts this long before -> low state
    cooldown_frames: int = 15        # grace period after ACTIVE/ENDING before re-arming
    low_state: str = "NO_PRESS"
    forming_state: str = "PRESS_FORMING"
    active_state: str = "ACTIVE_PRESS"
    ending_state: str = "PRESS_ENDING"
    uncertain_state: str = "UNCERTAIN"


def smooth_scores(raw_scores: list, window_frames: int) -> list:
    """`raw_scores`: list of float-or-None, one per frame (already in
    frame order). Returns a same-length list of float-or-None: the
    mean of non-None values in the trailing `window_frames` window, or
    None if that window has no evidence at all."""
    out = []
    window = []  # trailing (frame_index, value) for non-None entries only
    for i, v in enumerate(raw_scores):
        if v is not None:
            window.append((i, v))
        while window and i - window[0][0] >= window_frames:
            window.pop(0)
        out.append(sum(x[1] for x in window) / len(window) if window else None)
    return out


def run_fsm(smoothed_scores: list, cfg: FSMConfig, evidence_present=None, raw_scores=None, eligibility=None) -> list:
    """Compatibility adapter. Direct callers must supply one current observation
    per non-null score. Prefer EvidenceFSM for timestamp/episode/quality records."""
    from tactical_shared.temporal import EvidenceFSM, EvidenceConfig
    c = EvidenceConfig(enter=cfg.enter_th, exit=cfg.exit_th,
        min_observations=cfg.min_activate_frames, min_supported_sec=cfg.min_activate_frames/30.,
        min_hold_sec=cfg.min_hold_frames/30., ending_sec=cfg.ending_frames/30.,
        cooldown_sec=cfg.cooldown_frames/30., low=cfg.low_state, forming=cfg.forming_state,
        active=cfg.active_state, ending=cfg.ending_state)
    machine = EvidenceFSM(0, c)
    raws = raw_scores if raw_scores is not None else smoothed_scores
    return [machine.step(f, f/30., raw, (evidence_present[f] if evidence_present is not None else raw is not None),
                         eligibility[f] if eligibility is not None else (True if raw is not None else None))["state"]
            for f, raw in enumerate(raws)]


def raw_to_stable_states(raw_scores: list, window_frames: int, cfg: FSMConfig):
    """Legacy array API, corrected to forbid held averages as fresh evidence.
    Missing output scores remain None; V4 exports evidence ages explicitly."""
    smoothed = smooth_scores(raw_scores, window_frames)
    scores = [v if raw is not None else None for v, raw in zip(smoothed, raw_scores)]
    states = run_fsm(scores, cfg, evidence_present=[v is not None for v in raw_scores], raw_scores=raw_scores)
    return scores, states


def states_to_segments(states: list, fps: float = 30.0) -> list:
    """Run-length-encode a per-frame state list into
    [{state, start_frame, end_frame, start_time_sec, end_time_sec,
    duration_sec}] segments -- the 'stable temporal segments' the brief
    asks for, built by simply merging consecutive identical states
    (no smoothing here; smoothing already happened upstream)."""
    if not states:
        return []
    segs = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            segs.append({
                "state": states[start], "start_frame": start, "end_frame": i - 1,
                "start_time_sec": start / fps, "end_time_sec": (i - 1) / fps,
                "duration_sec": (i - start) / fps,
            })
            start = i
    return segs
