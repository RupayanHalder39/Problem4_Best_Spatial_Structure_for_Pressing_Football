"""Causal V4 quality view; raw/canonical files are never written.
Identity confidence is empirical label agreement, not a learned probability.
Motion gates combine history, gaps, observed homography, regression residual,
speed and acceleration. Bounds are conservative reference-scale plausibility
limits, not validated performance measurements. No missing position is filled.
"""
from collections import deque,Counter
import numpy as np
import polars as pl
from tactical_shared.coordinates import DEFAULT_PITCH

MAX_SPEED_CM_S=1200.
# MEASURED, not assumed (2026-09-07 continuation): a "physically real"
# human acceleration limit (~1200 cm/s^2 = 12 m/s^2) is NOT a usable
# frame-to-frame consistency gate for this data. Directly measuring the
# apparent |delta-velocity|/dt between consecutive same-track causal
# velocity estimates on the real clip (53,022 samples, position-only,
# no team/role gating) gives p50=2216, p75=4184, p90=7030, p95=9477,
# p99=15790, p99.9=27842, max=42871 cm/s^2 -- i.e. ordinary homography/
# detection position jitter at 30fps already produces a MEDIAN apparent
# acceleration nearly double the old 1200 threshold, which was silently
# rejecting the majority of legitimate estimates (Case B: an
# evidence requirement that was impossible given real measurement
# noise, not Case A: genuinely absent motion). Recalibrated to sit
# just above the measured 99.9th percentile -- comfortably clear of
# routine noise, while still catching the audit's documented one-frame
# artifacts (its own cited raw line-selection jumps were 238-507 m/s,
# i.e. 23,800-50,700 cm/s^2 divided by a single frame's dt, an order of
# magnitude beyond even this bound).
MAX_ACCEL_CM_S2=30000.
MAX_RESIDUAL_CM=75.
MIN_MOTION_SAMPLES=2         # causal minimum -- a real 2-point finite difference, not a fabricated value
PREFERRED_MOTION_SAMPLES=5   # smoothed regression fit used when this many gap-free samples are available
MAX_OBSERVATION_GAP_SEC=.10

class TeamIdentity:
    def __init__(self):self.labels=deque(maxlen=15);self.team=None;self.last_frame=None;self.epoch=0
    def update(self,frame,label):
        if self.last_frame is not None and frame-self.last_frame>3:
            self.labels.clear();self.team=None;self.epoch+=1
        self.last_frame=frame
        if label not in (0,1):return None,0.,self.epoch
        self.labels.append(label);counts=Counter(self.labels);best=max(counts.values());winners=[t for t,c in counts.items() if c==best]
        candidate=winners[0] if len(winners)==1 else None
        confidence=best/len(self.labels)
        eligible=len(self.labels)>=8 and confidence>=.8 and candidate==label
        if not eligible:return None,confidence,self.epoch
        if self.team is not None and candidate!=self.team:self.epoch+=1
        self.team=candidate
        return candidate,confidence,self.epoch

def motion_estimate(history,fps=30.):
    """Causal velocity from a SHORT recent window, not a fixed 5-sample
    regression. Audit finding (2026-09-07 continuation): the original
    5-sample/0.15s-span requirement rejected ~28.6% of ALL player rows
    purely on warm-up grounds (REACQUISITION + SHORT_HISTORY) even
    though 99.5% of real consecutive-position frame gaps in this clip
    are already <=3 frames -- i.e. the raw data supports much denser
    velocity estimation than the old gate allowed (measured Case B:
    overly strict evidence requirements, not Case A: genuine absence).

    `history` is the CALLER's already gap-managed deque (build_quality_view
    clears it itself on any oversized gap/context change before appending),
    so a genuine internal gap here mostly matters for direct/defensive
    calls (e.g. tests): if the chosen trailing window contains ANY
    internal gap over `MAX_OBSERVATION_GAP_SEC`, the whole window is
    rejected outright (no partial-suffix salvage -- an old stale portion
    must not quietly contaminate a fresh one).

    Returns (velocity_or_None, reason_or_None, meta) -- meta (always a
    dict, even on rejection) carries `n_samples`/`span_sec`/`quality`/
    `residual_cm` so callers can treat a 2-point causal estimate as
    lower-quality than a smoothed 5-point fit without a separate query."""
    if len(history)<MIN_MOTION_SAMPLES:return None,'REACQUISITION',dict(n_samples=len(history))
    window=history[-PREFERRED_MOTION_SAMPLES:]
    f=np.array([p[0] for p in window]);xy=np.array([[p[1],p[2]] for p in window]);ts=(f-f[-1])/fps
    n=len(window)
    meta=dict(n_samples=n,span_sec=float(-ts[0]))
    if n>1 and np.max(np.diff(f))/fps>MAX_OBSERVATION_GAP_SEC:return None,'TRAJECTORY_GAP',meta
    if n==2:
        v=(xy[-1]-xy[0])/max(ts[-1]-ts[0],1e-9);residual=0.0;meta['quality']='TWO_POINT_CAUSAL'
    else:
        coef=np.linalg.lstsq(np.column_stack([ts,np.ones(n)]),xy,rcond=None)[0]
        residual=float(np.max(np.linalg.norm(xy-np.column_stack([ts,np.ones(n)])@coef,axis=1)))
        v=coef[0];meta['quality']='FIT'
    meta['residual_cm']=residual
    if np.linalg.norm(v)>MAX_SPEED_CM_S:return None,'SPEED',meta
    if n>=3 and residual>MAX_RESIDUAL_CM:return None,'HOMOGRAPHY_OR_TRACK_RESIDUAL',meta
    return tuple(float(x) for x in v),None,meta

def build_quality_view(tracking,fps=30.,pitch=DEFAULT_PITCH):
    outputs=[]
    for _,group in tracking.filter(pl.col('track_id').is_not_null()).group_by('track_id',maintain_order=True):
        identity=TeamIdentity();history=deque(maxlen=9);roles=deque(maxlen=15)
        segment=0;prev=None;prev_good_v=None;last_context=None
        for r in group.sort('frame').to_dicts():
            f=r['frame'];team,conf,epoch=identity.update(f,r.get('team_id'))
            if prev and f-prev['frame']>3:roles.clear()
            roles.append(r['object_type']);counts=Counter(roles);best=max(counts.values());winners=[x for x,n in counts.items() if n==best]
            role=winners[0] if len(winners)==1 else 'unknown'
            if r['object_type']!=role:role='unknown'
            context=(team,epoch,pitch.period(f),role)
            restart=prev is None or context!=last_context or f-prev['frame']>3
            if restart:segment+=1;history.clear();prev_good_v=None
            last_context=context
            x,y=r.get('x_pitch'),r.get('y_pitch');reason=None
            valid=pitch.inside(x,y) and np.isfinite(x) and np.isfinite(y)
            if not valid:reason='NO_VALID_CURRENT_HOMOGRAPHY_OR_BOUNDS'
            if valid and prev and pitch.inside(prev.get('x_pitch'),prev.get('y_pitch')):
                dt=(f-prev['frame'])/fps
                jump=np.hypot(x-prev['x_pitch'],y-prev['y_pitch'])/dt
                if dt<=MAX_OBSERVATION_GAP_SEC and jump>MAX_SPEED_CM_S:
                    valid=False;reason='POSITION_JUMP';segment+=1;restart=True;history.clear();prev_good_v=None
            velocity=None;meta={}
            if valid and team is not None and role in ('player','goalkeeper'):
                if history and (f-history[-1][0])/fps>MAX_OBSERVATION_GAP_SEC:
                    history.clear();prev_good_v=None;segment+=1;restart=True
                history.append((f,float(x),float(y)))
                velocity,reason,meta=motion_estimate(list(history),fps)
                if velocity and prev_good_v:
                    pf,pv=prev_good_v;dt=(f-pf)/fps
                    # A single inconsistent-looking velocity estimate is
                    # treated as a rejected ESTIMATE, not a track-identity
                    # discontinuity: keep the position trajectory (`history`)
                    # intact so the next frame can fall back to a more
                    # robust multi-point FIT instead of re-paying a full
                    # REACQUISITION warm-up for what may just be one noisy
                    # 2-point difference. `prev_good_v` also stays at its
                    # last real value (not cleared) so a single blip can't
                    # blind the very next frame's consistency check too.
                    if dt<=.3 and np.linalg.norm(np.array(velocity)-pv)/dt>MAX_ACCEL_CM_S2:
                        velocity=None;reason='ACCELERATION'
                if velocity:prev_good_v=(f,np.array(velocity))
            else:
                if not valid:prev_good_v=None
                if team is None:history.clear();reason='TEAM_UNCERTAIN'
            outputs.append(dict(frame=f,timestamp_sec=f/fps,track_id=r['track_id'],cleaned_segment_id=segment,
                is_track_restart=restart,display_team_id=team,team_confidence=conf,display_object_type=role,
                x_pitch=float(x) if valid else None,y_pitch=float(y) if valid else None,
                raw_x_pitch=x,raw_y_pitch=y,is_position_filled=False,position_valid=valid,
                vx_cm_s=velocity[0] if velocity else None,vy_cm_s=velocity[1] if velocity else None,
                speed_cm_s=float(np.hypot(*velocity)) if velocity else None,motion_valid=velocity is not None,
                motion_reason=reason,motion_quality=meta.get('quality'),motion_n_samples=meta.get('n_samples'),
                motion_residual_cm=meta.get('residual_cm'),
                position_provenance='CURRENT_RAW_HOMOGRAPHY' if valid else 'UNAVAILABLE',
                confirmation_frame=f,latency_frames=0,period=pitch.period(f)))
            prev=r
    return pl.DataFrame(outputs,infer_schema_length=None).sort(['frame','track_id'])

def current_roles(players_by_frame,ball_by_frame,n,fps=30.,pitch=DEFAULT_PITCH):
    """CONTROLLED/UNCERTAIN/FREE_BALL definition intentionally REUSES
    `analytics.possession.compute_possession`'s already-established,
    already-accepted radius+confidence formula (CONTROL_RADIUS_CM=200,
    nearest-player-wins, confidence scaled 1.0->0.7 by distance) instead
    of a fresh, stricter reimplementation.

    Audit finding (2026-09-07 continuation): a first version of this
    function additionally required `ball.confidence>=0.4` and rejected
    any frame where a second player of the OTHER team sat within 50cm of
    the nearest one ("ambiguous") -- neither restriction exists in
    `compute_possession`, which this project has already relied on and
    accepted elsewhere. That extra strictness was measured directly: it
    cut current-role-known frames from the established ~7% baseline down
    to 3.5%, entirely because of those two additional, unprecedented
    gates layered on top of the exact same 200cm radius -- a clear Case
    B (self-imposed over-strictness), not a genuine evidence gap. Ball
    quality/ambiguity are real considerations, but degrading confidence
    (as `compute_possession` already does for 200-400cm) is the right
    response, not an outright reject at 0-400cm.

    The one deliberate difference from `compute_possession` is TEMPORAL:
    this function reports a true CURRENT-frame estimate with no
    persistence/bridging (`current_role_estimate` is None the instant
    evidence is unavailable), plus a SEPARATELY labeled
    `historical_context_role` (trailing 4s majority) -- this is B7's
    actual fix (current truth vs. historical context must not be
    blended into one number), not a reason to also re-invent
    possession's spatial/confidence logic from scratch."""
    from legacy_analytics.possession import CONTROL_RADIUS_CM, EXTENDED_RADIUS_CM
    rows=[];recent=deque();last=None
    for f in range(n):
        if f and pitch.period(f)!=pitch.period(f-1):recent.clear();last=None
        b=ball_by_frame.get(f);players=[p for p in players_by_frame.get(f,[]) if p.get('x_pitch') is not None and p['display_object_type'] in ('player','goalkeeper')]
        carrier=None;confidence=0.;reason='BALL_UNAVAILABLE'
        if b and b.get('is_observed') and pitch.inside(b.get('x_pitch'),b.get('y_pitch')):
            if players:
                d,p=min(((np.hypot(pp['x_pitch']-b['x_pitch'],pp['y_pitch']-b['y_pitch']),pp) for pp in players),key=lambda t:t[0])
                if d<=CONTROL_RADIUS_CM and p['display_team_id'] is not None:
                    carrier=p;confidence=(1.0-0.3*d/CONTROL_RADIUS_CM)*p['team_confidence'];reason='CONTROLLED'
                elif d<=EXTENDED_RADIUS_CM:reason='EXTENDED_UNCERTAIN'
                else:reason='FREE_BALL'
            else:reason='NO_VISIBLE_PLAYERS'
        team=carrier['display_team_id'] if carrier else None
        if team is not None:recent.append((f,team));last=f
        while recent and (f-recent[0][0])/fps>4:recent.popleft()
        counts=Counter(t for _,t in recent);historical=None
        if counts:
            best=max(counts.values());winners=[t for t,c in counts.items() if c==best]
            if len(winners)==1:historical=1-winners[0]
        rows.append(dict(frame=f,time_sec=f/fps,carrier_team=team,carrier_track=carrier['track_id'] if carrier else None,
            carrier_segment=carrier['cleaned_segment_id'] if carrier else None,current_role_estimate=1-team if team is not None else None,
            role_confidence=confidence,evidence_age_sec=(f-last)/fps if last is not None else None,
            historical_context_role=historical,possession_state=reason,period=pitch.period(f)))
    return rows
