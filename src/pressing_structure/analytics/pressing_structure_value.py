"""
Part 7: PRESSING STRUCTURE VALUE, initial proxy implementation.

Full definition documented in `pressing_structure/docs/
PRESSING_STRUCTURE_VALUE.md` -- summary:

    PSV_proxy_t = expected_forward_progress_without_pressure
                - observed_or_predicted_forward_progress_under_pressure

Named `PSV_proxy` everywhere (never bare `PSV`) because it is a
disclosed, temporary progression-based stand-in for a full xT/EPV-based
PSV, which this project does not yet implement (no xT/EPV model exists
in this codebase). Do NOT treat `PSV_proxy` as validated -- Experiment
6 (`EXPERIMENT_PLAN.md`) is the not-yet-run test of whether it actually
correlates with real short-horizon outcomes.

Formula (this module's own, disclosed, simple physical model -- not a
trained model, not claimed optimal):

    carrier_forward_velocity_cm_s = the carrier's REAL velocity
        component along their own attacking direction (0 if motion is
        untrustworthy -- never fabricated)
    naive_unimpeded_progress_cm = carrier_forward_velocity_cm_s * horizon_sec
        (a constant-velocity, zero-defensive-influence projection --
        the "expected ... without pressure" term)
    predicted_progress_under_pressure_cm = naive_unimpeded_progress_cm
        * (1 - pressure_at_carrier)
        (the SAME projection, damped by the REAL, already-computed
        `field_features.pressure_at_carrier` value at the observation
        frame -- 0 pressure -> undamped, pressure->1 -> fully suppressed)

    => PSV_proxy_t = naive_unimpeded_progress_cm - predicted_progress_under_pressure_cm
                    = naive_unimpeded_progress_cm * pressure_at_carrier

When REAL matched evidence exists within the horizon (from
`models.short_horizon_outcome`), the REAL observed forward progress is
ALSO reported alongside (`observed_progress_cm`) as a validation
signal -- never substituted into the PSV_proxy formula itself (mixing
a model term with an observed term inconsistently would make the
number impossible to interpret).
"""
from typing import Dict, Optional

from pressing_structure.analytics.pressure_field import team_pressure_field, sample_field
from tactical_shared.coordinates import DEFAULT_PITCH


def carrier_forward_velocity_cm_s(carrier: Dict, carrier_team: int, pitch=DEFAULT_PITCH) -> float:
    """0.0 (never fabricated, never None) if motion is untrustworthy --
    matches `pressure_field.player_influence`'s own convention for
    untrustworthy motion."""
    if not carrier.get("motion_valid") or carrier.get("vx_cm_s") is None or carrier.get("vy_cm_s") is None:
        return 0.0
    sign = pitch.attacking_sign(carrier_team)
    return float(carrier["vx_cm_s"] * sign)


def psv_proxy(carrier: Dict, carrier_team: int, presser_players, gx, gy, horizon_sec: float,
              pitch=DEFAULT_PITCH, observed_progress_cm: Optional[float] = None) -> Dict:
    """One PSV_proxy value for one (carrier, defending-team geometry,
    horizon) snapshot. `presser_players`: the defending team's eligible
    players this frame (SAME input `team_pressure_field` already takes
    -- no new geometry function). Returns every intermediate term,
    never just the final number, so the computation is auditable."""
    field = team_pressure_field(presser_players, gx, gy)
    pressure_at_carrier = sample_field(field, gx, gy, carrier["x_pitch"], carrier["y_pitch"])
    fwd_vel = carrier_forward_velocity_cm_s(carrier, carrier_team, pitch)
    naive_unimpeded_progress_cm = fwd_vel * horizon_sec
    predicted_progress_under_pressure_cm = naive_unimpeded_progress_cm * (1.0 - pressure_at_carrier)
    proxy = naive_unimpeded_progress_cm - predicted_progress_under_pressure_cm
    return dict(
        horizon_sec=horizon_sec,
        pressure_at_carrier=pressure_at_carrier,
        carrier_forward_velocity_cm_s=fwd_vel,
        naive_unimpeded_progress_cm=naive_unimpeded_progress_cm,
        predicted_progress_under_pressure_cm=predicted_progress_under_pressure_cm,
        psv_proxy=proxy,
        observed_progress_cm=observed_progress_cm,  # None unless real matched evidence supplied it -- never fabricated
    )


def psv_proxy_for_episode(episode: Dict, players_by_frame: Dict, roles, gx, gy, horizon_sec: float,
                           pitch=DEFAULT_PITCH, observed_progress_cm: Optional[float] = None) -> Optional[Dict]:
    """Convenience wrapper anchored at an episode's `onset_frame` (same
    anchor convention as `pressing_features.extract_episode_features`).
    Returns None (honest gap) if the anchor frame lacks resolvable
    carrier/defender positions."""
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
    result = psv_proxy(carrier, role["carrier_team"], presser_players, gx, gy, horizon_sec, pitch, observed_progress_cm)
    result["episode_id"] = episode["episode_id"]
    return result
