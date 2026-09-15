# env/water_filling.py
"""
Water-filling power allocation — Section 4.5, công thức (11)-(12).
Ước lượng channel cost (11) cho từng AP active, rồi bisection tìm water
level µ (12). Mutates tx_power của các AP active in-place trên `aps` gốc.
"""
import numpy as np
import config
from env.channel import calculate_distance, calculate_channel_gain

BISECTION_ITERS = 30


def _channel_cost(ap, bss_stas, active_aps):
    """Công thức (11): normalized interference-plus-noise cost c_b."""
    if not bss_stas:
        return float("inf")

    gains = [
        calculate_channel_gain(calculate_distance(ap, sta), sta['shadowing_db'])
        for sta in bss_stas
    ]
    avg_gain = max(float(np.mean(gains)), 1e-18)

    interference = 0
    for other_ap in active_aps:
        if other_ap['id'] != ap['id']:
            link_losses = [
                calculate_channel_gain(calculate_distance(other_ap, sta), sta['shadowing_db'])
                for sta in bss_stas
            ]
            interference += other_ap['tx_power'] * float(np.mean(link_losses))

    return (interference + config.NOISE_POWER) / avg_gain


def apply_water_filling(aps, stas, active_aps):
    """Phân bổ công suất cho các AP trong `active_aps`, ghi thẳng vào
    aps[ap_id]['tx_power']."""
    if not active_aps:
        return

    channel_costs = []
    for ap in active_aps:
        bss_stas = [sta for sta in stas if sta['ap_id'] == ap['id']]
        channel_costs.append(_channel_cost(ap, bss_stas, active_aps))

    finite_costs = [cost for cost in channel_costs if np.isfinite(cost)]
    if not finite_costs:
        return

    finite_indices = [i for i, cost in enumerate(channel_costs) if np.isfinite(cost)]
    min_power = config.TX_POWER_LEVELS[0]
    budget = min(config.TOTAL_POWER_BUDGET_W, len(finite_indices) * config.P_MAX)
    min_required = len(finite_indices) * min_power
    if budget + 1e-12 < min_required:
        raise ValueError(
            "TOTAL_POWER_BUDGET_W is smaller than the minimum power required "
            "by the active APs"
        )

    # Bounded water-filling: enforce P_min inside the optimization. Clamping
    # each AP after bisection can silently violate the total power budget.
    mu_low = min(finite_costs) + min_power
    mu_high = max(finite_costs) + config.P_MAX
    mu_optimal = 0.0

    for _ in range(BISECTION_ITERS):
        mu_mid = (mu_low + mu_high) / 2
        total_allocated = 0
        for cost in channel_costs:
            power = 0.0 if not np.isfinite(cost) else np.clip(
                mu_mid - cost, min_power, config.P_MAX
            )
            total_allocated += power

        if total_allocated > budget:
            mu_high = mu_mid
        else:
            mu_low = mu_mid
            mu_optimal = mu_mid

    for i, ap in enumerate(active_aps):
        if not np.isfinite(channel_costs[i]):
            optimal_power = 0.0
        else:
            optimal_power = np.clip(
                mu_optimal - channel_costs[i], min_power, config.P_MAX
            )
        aps[ap['id']]['tx_power'] = float(optimal_power)

    allocated = sum(aps[active_aps[i]['id']]['tx_power'] for i in finite_indices)
    if allocated > budget + 1e-6:
        raise RuntimeError(
            f"Water-filling violated power budget: {allocated:.9f} > {budget:.9f}"
        )
