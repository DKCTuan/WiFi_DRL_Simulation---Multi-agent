"""PHY and pairwise power allocation for coordinated spatial reuse."""
import math

import config
from env.channel import calculate_channel_gain, calculate_distance
from env.mcs import sinr_to_mcs_rate_mbps


def channel_gain(ap, sta):
    return calculate_channel_gain(
        calculate_distance(ap, sta), sta.get("shadowing_db", 0.0)
    )


def scheduled_link_sinr(aps, serving_ap_id, sta, scheduled_ap_ids, powers):
    signal = powers[serving_ap_id] * channel_gain(aps[serving_ap_id], sta)
    interference = 0.0
    for other_ap_id in scheduled_ap_ids:
        if other_ap_id != serving_ap_id:
            interference += powers[other_ap_id] * channel_gain(aps[other_ap_id], sta)
    return signal / (config.NOISE_POWER + interference)


def group_phy_rates(aps, stas_by_id, group, powers):
    """Return {sta_id: discrete PHY rate Mbps} for a scheduled group."""
    scheduled_ap_ids = tuple(ap_id for ap_id, _ in group)
    return {
        sta_id: sinr_to_mcs_rate_mbps(
            scheduled_link_sinr(
                aps, ap_id, stas_by_id[sta_id], scheduled_ap_ids, powers
            )
        )
        for ap_id, sta_id in group
    }


def standalone_rate(aps, sta, ap_id, power=None):
    power = config.P_MAX if power is None else float(power)
    sinr = power * channel_gain(aps[ap_id], sta) / config.NOISE_POWER
    return sinr_to_mcs_rate_mbps(sinr)


def is_compatible_group(aps, stas_by_id, group, tolerance=None):
    """Check |G|*R_CoSR >= R_single for every scheduled STA.

    This is the no-throughput-loss compatibility rule used by the referenced
    DRL scheduling work.  Single-link groups are always allowed when their PHY
    rate is non-zero.
    """
    tolerance = (
        config.COSR_COMPATIBILITY_TOLERANCE if tolerance is None else tolerance
    )
    full_powers = {ap_id: config.P_MAX for ap_id, _ in group}
    concurrent = group_phy_rates(aps, stas_by_id, group, full_powers)
    group_size = len(group)
    for ap_id, sta_id in group:
        single = standalone_rate(aps, stas_by_id[sta_id], ap_id)
        if single <= 0.0 or group_size * concurrent[sta_id] + tolerance < single:
            return False
    return True


def allocate_group_power(aps, stas_by_id, group, mode="full"):
    """Allocate power for one scheduled group.

    ``pf_pair`` searches only the two Pareto-boundary edges where at least one
    AP uses P_MAX.  This follows the structural result in the Co-SR power paper
    while retaining the simulator's discrete power levels.
    """
    if mode == "full" or len(group) == 1:
        return {ap_id: config.P_MAX for ap_id, _ in group}
    if mode != "pf_pair":
        raise ValueError(f"Unknown Co-SR power mode: {mode}")
    if len(group) != 2:
        raise ValueError("pf_pair currently supports exactly two APs")

    ap_i, _ = group[0]
    ap_j, _ = group[1]
    levels = sorted(set(float(p) for p in config.COSR_POWER_LEVELS_W))
    candidates = []
    for p in levels:
        candidates.append({ap_i: config.P_MAX, ap_j: p})
        candidates.append({ap_i: p, ap_j: config.P_MAX})

    standalone = {
        sta_id: standalone_rate(aps, stas_by_id[sta_id], ap_id)
        for ap_id, sta_id in group
    }
    best_powers = None
    best_utility = -math.inf
    for powers in candidates:
        rates = group_phy_rates(aps, stas_by_id, group, powers)
        if any(rate <= 0.0 for rate in rates.values()):
            continue
        if any(
            len(group) * rates[sta_id] + config.COSR_COMPATIBILITY_TOLERANCE
            < standalone[sta_id]
            for _, sta_id in group
        ):
            continue
        utility = sum(math.log(rate) for rate in rates.values())
        if utility > best_utility:
            best_utility = utility
            best_powers = powers

    return best_powers or {ap_id: config.P_MAX for ap_id, _ in group}
