"""
Part 9: counterfactual pressing-structure repositioning. A framework,
operating on real static snapshots (an episode's `onset_frame`), NOT a
source of "optimal press" claims -- see module-level wording rules
below.

For one observed press:
  1. opponent (ball-carrying team) positions are held FIXED.
  2. exactly one pressing defender is perturbed within a bounded local
     grid of pitch-space offsets (default +/-1m/+/-2m/+/-3m on each
     axis, clipped to stay inside the pitch -- never a teleport).
  3. `team_pressure_field` (UNCHANGED formula, just re-evaluated on the
     perturbed defender list) and `PSV_proxy` (`pressing_structure_value.
     psv_proxy`, UNCHANGED formula) are recomputed for each candidate
     position.
  4. `Delta_PSV = PSV_proxy(counterfactual) - PSV_proxy(actual)`.

WORDING RULE (enforced by this module's own vocabulary, not just a
comment): results are described as a "candidate improved structure" or
"higher-valued counterfactual structure", NEVER "optimal press" --
the search here is a BOUNDED LOCAL grid over one defender at a time,
not an exhaustive multi-defender optimization, so optimality is not
proven and must not be claimed.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pressing_structure.analytics.pressure_field import team_pressure_field
from pressing_structure.analytics.pressing_structure_value import psv_proxy
from tactical_shared.coordinates import DEFAULT_PITCH

DEFAULT_OFFSETS_CM: Tuple[float, ...] = (-300.0, -200.0, -100.0, 0.0, 100.0, 200.0, 300.0)  # +/-1m/2m/3m local grid, as specified


def local_grid_offsets(offsets_cm: Sequence[float] = DEFAULT_OFFSETS_CM) -> List[Tuple[float, float]]:
    """All (dx, dy) combinations of the given 1D offsets -- a bounded
    local grid, never an unconstrained search."""
    return [(dx, dy) for dx in offsets_cm for dy in offsets_cm]


def perturb_defender(defender: Dict, dx_cm: float, dy_cm: float, pitch=DEFAULT_PITCH) -> Dict:
    """Returns a NEW defender dict (never mutates the input) with
    position shifted by `(dx_cm, dy_cm)`, clamped to stay strictly
    inside the pitch -- the "remain inside pitch" / "no teleportation"
    constraints, enforced here, not left to the caller."""
    new_x = float(np.clip(defender["x_pitch"] + dx_cm, 0.0, pitch.length_cm))
    new_y = float(np.clip(defender["y_pitch"] + dy_cm, 0.0, pitch.width_cm))
    return {**defender, "x_pitch": new_x, "y_pitch": new_y}


def counterfactual_players(presser_players: Sequence[Dict], target_track_id: int, dx_cm: float, dy_cm: float, pitch=DEFAULT_PITCH) -> List[Dict]:
    """The full presser-team player list with exactly ONE player
    (`target_track_id`) perturbed -- every other defender's REAL
    observed position is held fixed, per the framework's own rule."""
    return [perturb_defender(p, dx_cm, dy_cm, pitch) if p["track_id"] == target_track_id else p for p in presser_players]


def evaluate_counterfactual(carrier: Dict, carrier_team: int, presser_players: Sequence[Dict], gx, gy,
                             target_track_id: int, dx_cm: float, dy_cm: float, horizon_sec: float,
                             pitch=DEFAULT_PITCH) -> Dict:
    """One counterfactual evaluation: PSV_proxy of the perturbed
    structure, the actual (dx=dy=0) structure, and their difference."""
    actual = psv_proxy(carrier, carrier_team, presser_players, gx, gy, horizon_sec, pitch)
    cf_players = counterfactual_players(presser_players, target_track_id, dx_cm, dy_cm, pitch)
    counterfactual = psv_proxy(carrier, carrier_team, cf_players, gx, gy, horizon_sec, pitch)
    return dict(target_track_id=target_track_id, dx_cm=dx_cm, dy_cm=dy_cm,
                psv_proxy_actual=actual["psv_proxy"], psv_proxy_counterfactual=counterfactual["psv_proxy"],
                delta_psv=counterfactual["psv_proxy"] - actual["psv_proxy"])


def search_local_repositioning(carrier: Dict, carrier_team: int, presser_players: Sequence[Dict], gx, gy,
                                target_track_id: int, horizon_sec: float, pitch=DEFAULT_PITCH,
                                offsets_cm: Sequence[float] = DEFAULT_OFFSETS_CM) -> Dict:
    """Evaluate every offset in the bounded local grid for ONE defender,
    return every result plus the best-by-`delta_psv` one -- labeled
    `best_local_candidate`, deliberately NOT `optimal_position` (see
    module docstring's wording rule): this is the best of a bounded
    local grid around one player, not a proven global optimum, and not
    a joint multi-player search."""
    if not any(p["track_id"] == target_track_id for p in presser_players):
        return dict(status="TARGET_NOT_IN_PRESSER_PLAYERS", target_track_id=target_track_id)
    results = [evaluate_counterfactual(carrier, carrier_team, presser_players, gx, gy, target_track_id, dx, dy, horizon_sec, pitch)
               for dx, dy in local_grid_offsets(offsets_cm)]
    best = max(results, key=lambda r: r["delta_psv"])
    return dict(status="EVALUATED", target_track_id=target_track_id, n_candidates=len(results),
                results=results, best_local_candidate=best,
                interpretation=("candidate improved structure" if best["delta_psv"] > 0 else
                                 "no improving local candidate found — actual position already locally best within the searched grid"))


def search_local_repositioning_for_episode(episode: Dict, players_by_frame: Dict, roles, gx, gy,
                                            horizon_sec: float, target_track_id: Optional[int] = None,
                                            pitch=DEFAULT_PITCH, offsets_cm: Sequence[float] = DEFAULT_OFFSETS_CM) -> Optional[Dict]:
    """Episode-anchored convenience wrapper (same `onset_frame` anchor
    convention as `pressing_features`/`pressing_structure_value`).
    `target_track_id` defaults to the nearest presser to the carrier at
    onset (the single most consequential defender to ask "what if they
    stood somewhere else" about) if not given explicitly."""
    f = episode["onset_frame"]
    role = roles[f]
    if role.get("carrier_team") is None or role.get("carrier_track") is None:
        return None
    players = players_by_frame.get(f, [])
    eligible = [p for p in players if p.get("x_pitch") is not None and p["display_object_type"] in ("player", "goalkeeper")]
    carrier = next((p for p in eligible if p["track_id"] == role["carrier_track"]), None)
    if carrier is None:
        return None
    presser_players = [p for p in eligible if p["display_team_id"] == episode["team"]]
    if not presser_players:
        return None
    if target_track_id is None:
        target_track_id = min(presser_players, key=lambda d: np.hypot(d["x_pitch"] - carrier["x_pitch"], d["y_pitch"] - carrier["y_pitch"]))["track_id"]
    result = search_local_repositioning(carrier, role["carrier_team"], presser_players, gx, gy, target_track_id, horizon_sec, pitch, offsets_cm)
    result["episode_id"] = episode["episode_id"]
    return result
