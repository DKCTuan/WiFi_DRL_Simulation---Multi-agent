"""Synthetic sustained application demand and causal per-user service status.

No packet queue, burst timing or latency guarantee is modeled here. Delivered
rate is limited by offered demand; unmet demand is a service deficit, not a
packet-loss estimate. Status used for action selection comes from prior slots.
"""
import numpy as np
import config


def initialize_users(stas):
    for sta in stas:
        # Independent per-user stream keeps assignments identical across arms.
        rng = np.random.default_rng(np.random.SeedSequence(
            [config.USER_DEMAND_SEED, int(sta['id'])]))
        app, demand = config.USER_APPLICATION_PROFILES[
            int(rng.integers(len(config.USER_APPLICATION_PROFILES)))]
        if not np.isfinite(demand) or demand <= 0:
            raise ValueError('Application demand must be finite and positive')
        sta.update(application=app, demand_mbps=float(demand),
                   average_delivered_mbps=0.0, satisfaction=0.0,
                   service_deficit=1.0, needs_rescue=False)


def update_users(stas, cumulative_rates, steps):
    rows = []
    for sta, total in zip(stas, cumulative_rates):
        average = float(total / steps)
        satisfaction = float(np.clip(average / sta['demand_mbps'], 0, 1))
        rescue = (steps >= config.USER_RESCUE_GRACE_STEPS
                  and satisfaction < config.USER_RESCUE_SATISFACTION_THRESHOLD)
        sta.update(average_delivered_mbps=average, satisfaction=satisfaction,
                   service_deficit=1-satisfaction, needs_rescue=bool(rescue))
        rows.append({key: sta[key] for key in (
            'id', 'ap_id', 'application', 'demand_mbps',
            'average_delivered_mbps', 'satisfaction', 'service_deficit', 'needs_rescue')})
    return rows


def local_demand_observation(stas):
    if not stas:
        return [0.0]*config.USER_DEMAND_OBS_SIZE
    demand = sum(sta['demand_mbps'] for sta in stas)
    return [min(demand / 200.0, 1.0),
            float(np.mean([sta['satisfaction'] for sta in stas])),
            float(np.mean([sta['needs_rescue'] for sta in stas])),
            max(sta['service_deficit'] for sta in stas)]
