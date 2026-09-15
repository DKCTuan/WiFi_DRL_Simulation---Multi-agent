"""Legacy fixed-suppression coordination experiment.

Each helper AP can request one adjacent BSS on the circular topology. The
coordinator chooses the edge STA for which the helper link is strongest
relative to the serving link. The PHY effect is applied in ``env.metrics``.
This is preserved for reproducing historical runs. It is not Co-BF: the model
contains no antenna array, CSI, steering vector, or precoder. New Wi-Fi 8 work
should use ``env.cosr_env`` until a proper Co-BF PHY is implemented.
"""
from collections import defaultdict

import config
from env.channel import calculate_channel_gain, calculate_distance


def decode_cooperative_action(action):
    """Return ``(cca_index, target_offset)`` for a cooperative Hybrid action."""
    num_modes = len(config.COOPERATION_TARGET_OFFSETS)
    action = int(max(0, min(action, config.COOPERATIVE_HYBRID_ACTION_SIZE - 1)))
    cca_idx = action // num_modes
    mode_idx = action % num_modes
    return cca_idx, config.COOPERATION_TARGET_OFFSETS[mode_idx]


def _gain(ap, sta):
    return calculate_channel_gain(
        calculate_distance(ap, sta), sta.get("shadowing_db", 0.0)
    )


def helper_opportunity_score(helper_ap, target_ap, target_stas):
    """Best helper/serving gain ratio in a target BSS, normalized to [0, 1]."""
    best = 0.0
    for sta in target_stas:
        serving_gain = max(_gain(target_ap, sta), 1e-18)
        helper_gain = _gain(helper_ap, sta)
        best = max(best, helper_gain / (serving_gain + helper_gain))
    return float(best)


def neighbor_opportunity_scores(helper_id, aps, stas):
    """Scores for helping the left and right neighboring BSS, in that order."""
    scores = []
    num_aps = len(aps)
    for offset in (-1, 1):
        target_id = (helper_id + offset) % num_aps
        target_stas = [sta for sta in stas if sta["ap_id"] == target_id]
        scores.append(helper_opportunity_score(aps[helper_id], aps[target_id], target_stas))
    return scores


def build_cooperation_links(aps, stas, active_aps, target_requests):
    """Resolve AP requests into a bounded ``STA -> helper APs`` assignment."""
    active_ids = {ap["id"] for ap in active_aps}
    candidates = []

    for helper_id, target_id in target_requests.items():
        if target_id is None or helper_id == target_id:
            continue
        if helper_id not in active_ids or target_id not in active_ids:
            continue
        target_stas = [sta for sta in stas if sta["ap_id"] == target_id]
        for sta in target_stas:
            serving_gain = max(_gain(aps[target_id], sta), 1e-18)
            helper_gain = _gain(aps[helper_id], sta)
            ratio = helper_gain / (serving_gain + helper_gain)
            if ratio >= config.COORDINATION_MIN_HELPER_RATIO:
                candidates.append((ratio, helper_id, target_id, sta["id"]))

    candidates.sort(reverse=True)
    helpers_by_sta = defaultdict(list)
    used_helpers = set()
    details = []
    for ratio, helper_id, target_id, sta_id in candidates:
        if helper_id in used_helpers:
            continue
        if len(helpers_by_sta[sta_id]) >= config.COORDINATION_MAX_HELPERS_PER_STA:
            continue
        helpers_by_sta[sta_id].append(helper_id)
        used_helpers.add(helper_id)
        details.append({
            "sta_id": int(sta_id),
            "serving_ap": int(target_id),
            "helper_ap": int(helper_id),
            "helper_ratio": float(ratio),
        })

    links = {sta_id: tuple(helper_ids) for sta_id, helper_ids in helpers_by_sta.items()}
    return links, details
