"""Evidence-aware causal lifecycle. Scores are not probabilities of correctness.
Missing observations suspend display, retain an episode for at most max_gap_sec,
and never contribute support. All deadlines use timestamps, not output labels.
"""
from dataclasses import dataclass
from collections import deque
import math

@dataclass(frozen=True)
class EvidenceConfig:
    enter: float = .4
    exit: float = .25
    # min_observations/min_supported_sec/min_coverage_ratio TOGETHER define
    # "sustained enough to confirm" (2026-09-07 semantic-QA pass). Previously
    # min_observations/min_supported_sec were recalibrated to the exact
    # MAXIMUM observation count/duration a specific clip's best real episode
    # achieved -- on inspection this is Case B (clip-specific tuning) even
    # though it wasn't chosen to hit a desired event count: a bar set to
    # "whatever this clip's ceiling is" will always confirm at least one
    # episode by construction, which is not a property a GENERIC evidence
    # requirement should have. Redesigned to be independent of any specific
    # clip's outcome:
    #   - min_observations=2: the generic structural floor from A2 ("one
    #     observation cannot activate a supposedly sustained state") --
    #     requires at least a second, independent, time-separated reading.
    #     Not derived from any clip's achieved count.
    #   - min_supported_sec=0.25 = HALF of `smooth_sec` (0.5s) below: the
    #     score being thresholded is itself a rolling mean over up to
    #     smooth_sec of evidence, so requiring real support for at least
    #     half of that averaging window is the same round, structural
    #     ("half") fraction already used elsewhere in this codebase for
    #     confidence blending (frame_features' motion_coverage term uses a
    #     0.5/0.5 split) -- not picked to admit or exclude any specific
    #     episode.
    #   - min_coverage_ratio=1/3: of the elapsed onset-to-now FORMING
    #     window, at least a third must be actual measured support -- this
    #     is what actually distinguishes "coordinated and dense" from "a
    #     few scattered spikes stretched across a long, mostly-absent
    #     window" (min_supported_sec alone cannot catch that pattern, since
    #     it only counts real support time regardless of how sparse it is
    #     relative to the window it's spread across).
    # These three numbers are IDENTICAL for PRESS_CONFIG and TRAP_CONFIG
    # (see pressing_v4.py / offside_v4.py) precisely because they are
    # generic properties of the shared FSM engine, not project- or
    # clip-specific calibration -- only `enter`/`exit` (the SCORE's own
    # measured scale) differ per project, as documented where those are set.
    min_observations: int = 2
    min_supported_sec: float = .25
    min_coverage_ratio: float = 1/3
    max_gap_sec: float = 3/30
    min_hold_sec: float = 10/30
    ending_sec: float = 10/30
    cooldown_sec: float = .5
    smooth_sec: float = .5
    # NEW (2026-09-07 continuation, Phase 2): an observation must ALSO
    # clear this confidence bar to count toward confirming ACTIVE --
    # "sustained high intensity with sufficient confidence", not
    # intensity alone. Default 0.0 preserves the exact old
    # confidence-blind behavior for every existing caller (the legacy
    # state_machine.py compat path never passes a real confidence and
    # must not be affected). A LOW-confidence observation still counts
    # as real, fresh evidence (updates the smoothed score, can still
    # show FORMING) -- it just cannot, by itself, confirm ACTIVE. This
    # is deliberately NOT a freshness/UNCERTAIN gate: moderate evidence
    # must not be converted into UNCERTAIN, only kept from prematurely
    # confirming a sustained-and-confident state.
    min_confidence: float = 0.0
    low: str = 'NO_PRESS'
    forming: str = 'PRESS_FORMING'
    active: str = 'ACTIVE_PRESS'
    ending: str = 'PRESS_ENDING'

class EvidenceFSM:
    def __init__(self,team,cfg=EvidenceConfig(),fps=30.):
        self.team,self.cfg,self.fps=team,cfg,fps
        self.state=cfg.low;self.episode=None;self.episodes=[];self.window=deque()
        self.last_time=None;self.last_fresh=None;self.last_support=None
        self.support=0.;self.observations=0;self.active_time=None;self.ending_time=None
        self.cooldown_until=-math.inf;self.context=None

    def terminate(self,frame,time,reason):
        if self.episode:
            e=self.episode;e.update(termination_frame=frame,termination_time=time,termination_reason=reason,
                end_frame=e['last_observed_frame'],end_time=e['last_observed_frame']/self.fps,
                supported_duration_sec=self.support,observations=self.observations)
            self.episodes.append(e)
        self.episode=None;self.state=self.cfg.low;self.window.clear()
        self.support=0.;self.observations=0;self.last_support=None;self.active_time=None;self.ending_time=None

    def step(self,frame,time,raw_score,evidence_present,eligibility,context=None,confidence=0.,evidence_age=0.):
        if self.last_time is not None and time<=self.last_time:raise ValueError('Distinct increasing timestamps required')
        self.last_time=time;cfg=self.cfg
        if self.context is not None and context is not None and context!=self.context:
            self.terminate(frame,time,'CONTEXT_CHANGED');self.last_fresh=None
        if context is not None:self.context=context
        age=None if self.last_fresh is None else time-self.last_fresh
        if age is not None and age>cfg.max_gap_sec+1e-9:
            self.terminate(frame,time,'EVIDENCE_EXPIRED');self.last_fresh=None
        fresh=bool(evidence_present and evidence_age<=1e-9 and raw_score is not None and math.isfinite(raw_score) and eligibility is True)
        if eligibility is False:
            self.terminate(frame,time,'ROLE_INELIGIBLE');self.last_fresh=None
            return self.record(frame,time,raw_score,False,False,0.,confidence,cfg.low)
        if not fresh:
            if self.episode:
                gaps=self.episode['observability_gaps']
                if gaps and gaps[-1]['end_frame']==frame-1:gaps[-1]['end_frame']=frame
                else:gaps.append(dict(start_frame=frame,end_frame=frame))
            return self.record(frame,time,None,False,eligibility,age,0.,'UNCERTAIN')
        self.last_fresh=time
        self.window.append((time,float(raw_score)))
        while self.window and time-self.window[0][0]>cfg.smooth_sec:self.window.popleft()
        score=sum(v for _,v in self.window)/len(self.window)
        if self.state==cfg.low:
            if time>=self.cooldown_until and raw_score>=cfg.enter and score>=cfg.enter:
                self.state=cfg.forming
                self.episode=dict(episode_id=f'T{self.team}-{frame}',team=self.team,onset_frame=frame,onset_time=time,
                    active_start=None,active_end=None,confirmation_time=None,last_observed_frame=frame,
                    observability_gaps=[],outcome_evidence_id=None,outcome_evidence_time=None)
        if self.episode:self.episode['last_observed_frame']=frame
        if self.state==cfg.forming:
            strong=raw_score>=cfg.enter and score>=cfg.enter
            if strong and confidence>=cfg.min_confidence:
                self.observations+=1
                # Only measured intervals count, not absent evidence between them.
                self.support+=1/self.fps if self.last_support is None else min(1/self.fps,time-self.last_support)
                self.last_support=time
                elapsed=max(time-self.episode['onset_time'],1e-9)
                coverage=self.support/elapsed
                if (self.observations>=cfg.min_observations and self.support+1e-9>=cfg.min_supported_sec
                        and coverage>=cfg.min_coverage_ratio):
                    self.state=cfg.active;self.active_time=time
                    self.episode.update(active_start=frame,active_end=frame,confirmation_time=time,
                        confirmation_coverage_ratio=coverage,confirmation_observations=self.observations,
                        confirmation_supported_sec=self.support)
            elif strong:
                # Real, elevated intensity, but not yet confident enough to
                # COUNT toward confirming ACTIVE -- neither progress nor a
                # reset. A momentary low-confidence reading must not erase
                # otherwise-good accumulated observations (that would be
                # "moderate evidence punished like absent evidence").
                pass
            elif score<cfg.exit:
                self.terminate(frame,time,'FORMING_ABORTED')
            else:
                self.support=0.;self.observations=0;self.last_support=None
        elif self.state==cfg.active:
            self.episode['active_end']=frame
            if raw_score<cfg.exit and score<cfg.exit and time-self.active_time>=cfg.min_hold_sec:
                self.state=cfg.ending;self.ending_time=time
        elif self.state==cfg.ending:
            if raw_score>=cfg.enter and score>=cfg.enter:
                self.state=cfg.active;self.active_time=time;self.episode['active_end']=frame
            elif time-self.ending_time+1e-9>=cfg.ending_sec:
                self.terminate(frame,time,'SCORE_ENDED');self.cooldown_until=time+cfg.cooldown_sec
        return self.record(frame,time,raw_score,True,True,0.,confidence,self.state,score)

    def record(self,frame,time,raw,fresh,eligible,age,confidence,state,score=None):
        return dict(frame=frame,time_sec=time,team=self.team,state=state,raw_score=raw,score=score,
                    evidence_present=fresh,evidence_age_sec=age,eligible=eligible,
                    confidence=confidence if fresh or eligible is False else 0.,
                    episode_id=self.episode['episode_id'] if self.episode else None,
                    confirmation_time=self.episode['confirmation_time'] if self.episode else None)

    def finish(self,frame,time):
        self.terminate(frame,time,'CLIP_END')
        return self.episodes

def dominant(records,low='NO_PRESS'):
    rank={'ACTIVE_PRESS':4,'TRAP_ACTIVE':4,'TRAP_BREAK_THREAT':5,'TRAP_BROKEN':6,'PRESS_FORMING':3,'TRAP_FORMING':3,'PRESS_ENDING':1,'TRAP_ENDING':1}
    viable=[r for r in records if r['state'] in rank]
    if viable:
        top=max(rank[r['state']] for r in viable);leaders=[r for r in viable if rank[r['state']]==top]
        return (leaders[0]['state'],leaders[0]['team']) if len(leaders)==1 else ('CONTESTED',None)
    return (low,None) if all(r['state']==low for r in records) else ('UNCERTAIN',None)
