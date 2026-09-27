"""V4 observed carrier-pressure episodes, with independent team eligibility.
No causal-success label is emitted. Reference-scale metrics and confidence
(evidence/identity coverage, not intensity) are separate throughout.
"""
import numpy as np
from tactical_shared.coordinates import DEFAULT_PITCH
from tactical_shared.temporal import EvidenceFSM, EvidenceConfig, dominant

# min_observations/min_supported_sec/min_coverage_ratio: REVERTED
# (2026-09-07 semantic-QA pass) from an earlier recalibration to this
# clip's own measured ceiling (4 observations/0.133s) -- on reflection
# that was Case B (clip-specific tuning) even though it wasn't chosen to
# hit a desired count: a bar set to "whatever THIS clip's best episode
# achieves" is guaranteed to confirm at least one episode by
# construction, which a generic evidence requirement must not do. Left
# at `EvidenceConfig`'s own generic defaults (2 observations / 0.25s
# support / 1/3 coverage -- see that class's docstring for the
# structural, non-clip-derived justification of each number) so that
# whether any episode confirms ACTIVE on a given clip is a genuine,
# unmanipulated measurement outcome. `min_confidence` is kept.
PRESS_CONFIG=EvidenceConfig(min_confidence=.3)
OBSERVED_OUTCOMES=('BALL_REGAIN','BACKWARD_ACTION','LATERAL_ACTION','FORWARD_PROGRESSION','UNCERTAIN')

def relative_closing(defender,carrier):
    valid=lambda p:p.get('motion_valid',True) and p.get('vx_cm_s') is not None and p.get('vy_cm_s') is not None
    if not valid(defender) or not valid(carrier):return None
    delta=np.array([carrier['x_pitch']-defender['x_pitch'],carrier['y_pitch']-defender['y_pitch']]);distance=np.linalg.norm(delta)
    if distance<=1e-9:return None
    return float(np.dot(np.array([defender['vx_cm_s']-carrier['vx_cm_s'],defender['vy_cm_s']-carrier['vy_cm_s']]),delta/distance))

# Component weights for the PARTIAL-FEATURE press score (Phase 2 redesign,
# 2026-09-07 continuation). Audit finding: the prior version required
# carrier motion AND >=75% of nearby opponents' motion AND a computable
# relative closing rate to ALL be valid in the SAME frame before returning
# anything but None -- on this clip that joint (Case D) requirement left
# 0 nonzero press-intensity frames for the ENTIRE 120s clip, even though
# each individual signal (position-based proximity is available whenever
# POSITIONS are valid; closing speed, when computable, clears its own bar
# a comfortable majority of the time -- median measured 296 cm/s against
# a 100 cm/s bar). Proximity/count are POSITION-only (no velocity
# needed) and are near-always available once a carrier and opponents are
# visible; closing speed is a genuine bonus signal that contributes when
# (and only when) motion is trustworthy, WITHOUT gating the other two to
# None just because it's absent. `coverage` reports exactly which
# fraction of the full weight was actually available this frame -- this
# IS the intensity/confidence separation the brief asks for: `coverage`
# feeds `confidence`, never `intensity`.
PROXIMITY_WEIGHT=.35   # 1 - clip(distance/800cm) -- same 8m reference PRESSER_RADII_CM["8m"] already uses
COUNT_WEIGHT=.30       # clip(n_pressers_5m/4) -- same 5m radius pressing.py's own PRESS_TRIGGER_RADIUS_CM uses
CLOSING_WEIGHT=.35     # clip(relative_closing/300cm/s) -- only when both carrier+nearest defender motion is valid

def frame_features(players,role,team):
    # `nearest_opponent_track_id` (2026-09-07 V5 audit): purely ADDITIVE
    # -- exposes the SAME `nearest` player already selected below to
    # compute `nearest_opponent_distance_cm` (the identity was
    # previously a discarded local variable). No selection logic
    # changed; this is the exact opponent that distance/closing-speed
    # value already describes.
    base=dict(nearby_opponents_3m=None,nearby_opponents_5m=None,nearest_opponent_distance_cm=None,
        nearest_opponent_track_id=None,
        relative_closing_cm_s=None,defender_velocity_toward_carrier_cm_s=None,participant_ids=[],
        support_minus_opponents=None,local_visible_compactness_cm=None,
        coverage=0.,intensity=None,confidence=0.,components_used=[],
        raw_score=None)  # `raw_score` kept as an alias of `intensity` for callers/tests written against the old name
    if role['carrier_team'] is None:return base   # genuinely no possession evidence at all this frame
    if role['carrier_team']==team:
        # Real, defined zero: this team has the ball, cannot be pressing --
        # full coverage/confidence since this fact itself is certain.
        return {**base,'intensity':0.,'raw_score':0.,'confidence':role['role_confidence'],'coverage':1.,'components_used':['own_possession']}
    eligible=[p for p in players if p.get('x_pitch') is not None and p['display_object_type'] in ('player','goalkeeper')]
    carrier=next((p for p in eligible if p['track_id']==role['carrier_track']),None)
    defs=[p for p in eligible if p['display_team_id']==team]
    if carrier is None or not defs:return base   # no positional evidence for this team's shape right now
    distances=sorted([(float(np.hypot(d['x_pitch']-carrier['x_pitch'],d['y_pitch']-carrier['y_pitch'])),d) for d in defs],key=lambda x:x[0])
    distance,nearest=distances[0]
    local=[d for dist,d in distances if dist<=1500]
    n3=sum(dist<=300 for dist,_ in distances);n5=sum(dist<=500 for dist,_ in distances)
    closing=relative_closing(nearest,carrier)
    participants=[d['track_id'] for dist,d in distances if dist<=500 and (relative_closing(d,carrier) or 0)>=100]

    components={'proximity':(float(np.clip(1-distance/800,0,1)),PROXIMITY_WEIGHT),
                'count':(float(np.clip(n5/4,0,1)),COUNT_WEIGHT)}
    if closing is not None:
        components['closing']=(float(np.clip(closing/300,0,1)),CLOSING_WEIGHT)
    total_w=sum(w for _,w in components.values())
    intensity=sum(v*w for v,w in components.values())/total_w
    coverage=total_w/(PROXIMITY_WEIGHT+COUNT_WEIGHT+CLOSING_WEIGHT)

    nearby=[d for dist,d in distances if dist<=800]
    motion_coverage=sum(d.get('motion_valid',False) for d in nearby)/max(1,len(nearby)) if nearby else 1.0
    # Confidence is a SEPARATE, disclosed evidence-reliability estimate --
    # identity confidence for both carrier and nearest opponent, the
    # coverage above (did we get the bonus closing signal or only the
    # position-only components), and how much of the local opponent
    # population itself has trustworthy motion. It is NEVER folded into
    # `intensity`, and a team can genuinely have high intensity with low
    # confidence (a close, populous, but poorly-tracked press) or the
    # reverse (a confidently-measured but genuinely low-pressure moment).
    confidence=float(np.clip(role['role_confidence']*coverage*(0.5+0.5*motion_coverage)*nearest.get('team_confidence',1.)*carrier.get('team_confidence',1.),0,1))

    support=sum(p['display_team_id']==role['carrier_team'] and p['track_id']!=carrier['track_id'] and np.hypot(p['x_pitch']-carrier['x_pitch'],p['y_pitch']-carrier['y_pitch'])<=1000 for p in eligible)
    compact=float(np.mean([np.hypot(a['x_pitch']-b['x_pitch'],a['y_pitch']-b['y_pitch']) for i,a in enumerate(local) for b in local[i+1:]])) if len(local)>=2 else None
    toward=None
    if nearest.get('motion_valid') and distance>0:toward=float((nearest['vx_cm_s']*(carrier['x_pitch']-nearest['x_pitch'])+nearest['vy_cm_s']*(carrier['y_pitch']-nearest['y_pitch']))/distance)
    return dict(nearby_opponents_3m=n3,nearby_opponents_5m=n5,nearest_opponent_distance_cm=distance,
        nearest_opponent_track_id=nearest['track_id'],
        relative_closing_cm_s=closing,defender_velocity_toward_carrier_cm_s=toward,participant_ids=participants,
        support_minus_opponents=support-sum(d<=1000 for d,_ in distances),local_visible_compactness_cm=compact,
        coverage=coverage,intensity=intensity,raw_score=intensity,confidence=confidence,
        components_used=list(components.keys()))

def evidence_quality(e,kind,players_by_frame=None):
    if (e.get('confidence') or 0)<.7 or e.get('missing_ball_fraction') is None or e['missing_ball_fraction']>.25:return False,'LOW_EVIDENCE_QUALITY'
    if e.get('start_is_player_proxy') or e.get('end_is_player_proxy'):return False,'PLAYER_PROXY_ENDPOINT'
    if any(e.get(k) is None for k in ('start_x','end_x','source_track_id','target_track_id')):return False,'MISSING_ENDPOINT'
    if players_by_frame is not None:
        for endpoint,field,expected in [('start_frame','source_track_id',e.get('source_team_id') if kind=='turnover' else e.get('team_id')),('end_frame','target_track_id',e.get('receiver_team_id') if kind=='turnover' else e.get('team_id'))]:
            p=next((p for p in players_by_frame.get(e[endpoint],[]) if p['track_id']==e[field]),None)
            if not p or p['display_team_id']!=expected or p.get('x_pitch') is None:return False,'IDENTITY_OR_COORDINATE_UNCERTAIN'
    return True,None

def attribute_outcomes(episodes,passes,turnovers,pitch=DEFAULT_PITCH,fps=30.,players_by_frame=None):
    """Assign each chronological evidence item to latest eligible unassigned episode.
    Window: action release >= last supported episode frame; confirmation <= end+3s.
    Low-quality evidence is reserved once and yields UNCERTAIN, not skipped to
    cherry-pick a later favorable action. No causal success percentage is computed.
    """
    out=[{**e,'observed_outcome':'UNCERTAIN','outcome_reason':'NO_MATCHED_EVIDENCE','outcome_evidence_id':None,
          'outcome_evidence_time':None,'outcome_confirmation_time':None,'outcome_display_time':None,'forward_progress_cm':None} for e in episodes if e['active_start'] is not None]
    evidence=[('pass',p) for p in passes]+[('turnover',p) for p in turnovers]
    for kind,p in sorted(evidence,key=lambda kp:(kp[1]['end_frame'],kp[1]['start_frame'],kp[0],kp[1].get('pass_id',0))):
        source=p.get('source_team_id') if kind=='turnover' else p.get('team_id')
        cands=[e for e in out if e['outcome_evidence_id'] is None and 1-e['team']==source and e['end_frame']<=p['start_frame'] and p['end_frame']<=e['end_frame']+3*fps and pitch.period(e['end_frame'])==pitch.period(p['end_frame'])]
        if not cands:continue
        e=max(cands,key=lambda e:e['end_frame']);valid,reason=evidence_quality(p,kind,players_by_frame)
        label='UNCERTAIN';progress=None
        if valid:
            if kind=='turnover':label='BALL_REGAIN' if p.get('receiver_team_id')==e['team'] else 'UNCERTAIN'
            else:
                progress=pitch.progress(source,p['start_x'],p['end_x'],p['start_frame'])
                label='BACKWARD_ACTION' if progress<=-300 else 'FORWARD_PROGRESSION' if progress>=300 else 'LATERAL_ACTION'
        confirmation=p['end_frame']/fps
        e.update(observed_outcome=label,outcome_reason=reason or 'OBSERVED_POST_PRESS_ACTION_NOT_CAUSAL_SUCCESS',
            outcome_evidence_id=f'{kind}:{p.get("scene_id",0)}:{p["pass_id"]}',outcome_evidence_time=p['start_frame']/fps,
            outcome_confirmation_time=confirmation,outcome_display_time=max(confirmation,e['termination_time']),forward_progress_cm=progress)
    return out

def build_pressing_v4(players_by_frame,roles,passes,turnovers,fps=30.,pitch=DEFAULT_PITCH):
    machines={t:EvidenceFSM(t,PRESS_CONFIG,fps) for t in (0,1)};frames=[]
    for f,role in enumerate(roles):
        pair=[]
        for team in (0,1):
            feature=frame_features(players_by_frame.get(f,[]),role,team)
            eligibility=None if role['carrier_team'] is None else role['carrier_team']!=team
            context=(role['period'],role['carrier_track'],role['carrier_segment']) if role['carrier_team'] is not None else (role['period'],None,None) if f and role['period']!=roles[f-1]['period'] else None
            r=machines[team].step(f,f/fps,feature['raw_score'],feature['raw_score'] is not None,eligibility,context,feature['confidence'])
            pair.append({**r,'features':feature,'carrier_team':role['carrier_team'],'carrier_track':role['carrier_track']})
        state,team=dominant(pair)
        frames.append(dict(frame=f,time_sec=f/fps,teams=pair,dominant_state=state,dominant_team=team,role=role))
    episodes=[e for machine in machines.values() for e in machine.finish(len(roles),len(roles)/fps)]
    events=attribute_outcomes(episodes,passes,turnovers,pitch,fps,players_by_frame)
    return dict(frames=frames,episodes=episodes,events=events,config=PRESS_CONFIG.__dict__)
