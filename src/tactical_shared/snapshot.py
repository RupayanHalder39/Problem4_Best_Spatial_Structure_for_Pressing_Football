"""
Phase 8 (2026-09-07 continuation): ONE coherent per-frame V4 snapshot,
merging `pressing_v4.build_pressing_v4` and `offside_v4.build_offside_v4`
output so every dashboard panel (banner, radar, KPI, graph, timeline)
reads the SAME record for a given frame -- never a separate V2/V3 call
for the timeline and a V4 call for the banner (A8/B8's actual fix).

This module does not recompute anything -- it is a pure, disclosed
merge of the two pipelines' already-produced `frames` lists (which are
frame-index-aligned since both are built from the same `roles` list
over the same frame range).
"""


def build_snapshot(press_out, off_out, roles, fps=30.):
    """Returns a list of one dict per frame:
    {frame, time_sec, possession_state, current_carrier_team, historical_defender,
     pressing: {0: {...}, 1: {...}, dominant_state, dominant_team},
     offside:  {0: {...}, 1: {...}, dominant_state, dominant_team}}
    Team sub-dicts carry state/raw_score/score/confidence/episode_id for
    pressing, and state/raw_score/confidence/episode_id/line/line_velocity_cm_s/
    run_threat_score/break_state_as_attacker/runners for offside -- i.e.
    exactly the fields the two pipelines already compute, just addressed
    from one place."""
    n = len(roles)
    assert len(press_out["frames"]) == n and len(off_out["frames"]) == n, \
        "pressing_v4 and offside_v4 frame counts must match roles (same replay range)"
    snap = []
    for f in range(n):
        role = roles[f]
        pf = press_out["frames"][f]
        of = off_out["frames"][f]
        pressing = {t: {"state": pf["teams"][t]["state"], "raw_score": pf["teams"][t]["raw_score"],
                         "score": pf["teams"][t]["score"], "confidence": pf["teams"][t]["confidence"],
                         "episode_id": pf["teams"][t]["episode_id"], "eligible": pf["teams"][t]["eligible"],
                         "features": pf["teams"][t]["features"]} for t in (0, 1)}
        pressing["dominant_state"] = pf["dominant_state"]
        pressing["dominant_team"] = pf["dominant_team"]
        offside = {t: {"state": of["teams"][t]["state"], "raw_score": of["teams"][t]["raw_score"],
                        "score": of["teams"][t].get("score"),
                        "confidence": of["teams"][t]["confidence"], "episode_id": of["teams"][t]["episode_id"],
                        "line": of["teams"][t]["line"], "line_velocity_cm_s": of["teams"][t]["line_velocity_cm_s"],
                        "run_threat_score": of["teams"][t].get("run_threat_score"),
                        "break_state_as_attacker": of["teams"][t].get("break_state_as_attacker"),
                        "runners": of["teams"][t].get("runners", []),
                        "lifecycle_updates": of["teams"][t].get("lifecycle_updates", []),
                        "confirmed_trap_break_events": of["teams"][t].get("confirmed_trap_break_events", []),
                        # 2026-09-07 V5 audit: purely ADDITIVE passthrough of fields
                        # `build_offside_v4` already computes but this module previously
                        # dropped (found in the prior visualization-pass audit) --
                        # no analytics formula changed to add these.
                        "unit_ids": of["teams"][t].get("unit_ids", []),
                        "unit_motion_ids": of["teams"][t].get("unit_motion_ids", []),
                        "metrics": of["teams"][t].get("metrics", {}),
                        # 2026-09-07 V6: the new back-line-cluster front/deep
                        # fields (LINE 1 / LINE 2, distinct from `line`'s own
                        # LINE 3 reference) -- additive passthrough.
                        "back_line": of["teams"][t].get("back_line", {})}
                   for t in (0, 1)}
        offside["dominant_state"] = of["dominant_state"]
        offside["dominant_team"] = of["dominant_team"]
        snap.append(dict(frame=f, time_sec=f / fps, possession_state=role["possession_state"],
                          current_carrier_team=role["carrier_team"], current_role_confidence=role["role_confidence"],
                          current_carrier_track=role.get("carrier_track"),  # additive (2026-09-07 V5 audit)
                          historical_defender=role["historical_context_role"], period=role["period"],
                          pressing=pressing, offside=offside))
    return snap
