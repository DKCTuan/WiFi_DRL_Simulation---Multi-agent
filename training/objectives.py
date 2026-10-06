"""Shared scalar objective for training rewards, checkpoint selection and diagnostics."""

import config


def throughput_temporal_episode_score(mean_throughput_norm, temporal_jfi,
                                      throughput_weight=None):
    """Episode objective with dimensionless throughput and temporal fairness."""
    weight = (config.ALIGNED_THROUGHPUT_WEIGHT if throughput_weight is None
              else float(throughput_weight))
    if not 0.0 <= weight <= 1.0:
        raise ValueError('Throughput weight must be in [0, 1]')
    return weight * float(mean_throughput_norm) + (1.0 - weight) * float(temporal_jfi)


def throughput_temporal_step_reward(throughput_norm, temporal_jfi,
                                    previous_temporal_jfi, episode_steps,
                                    throughput_weight=None):
    """Undiscounted sum equals 0.5*mean normalized throughput+0.5*final JFI.

    More generally, sum_t [w * throughput_norm_t / T + (1-w) * (J_t - J_(t-1))]
    = w * mean_t throughput_norm_t + (1-w) * J_T, with J_0=0.
    """
    if episode_steps <= 0:
        raise ValueError('Episode length must be positive')
    weight = (config.ALIGNED_THROUGHPUT_WEIGHT if throughput_weight is None
              else float(throughput_weight))
    if not 0.0 <= weight <= 1.0:
        raise ValueError('Throughput weight must be in [0, 1]')
    return (weight * float(throughput_norm) / episode_steps
            + (1.0 - weight) * (float(temporal_jfi) - float(previous_temporal_jfi)))


def validate_stationary_reward():
    """Reject legacy switches that silently misalign training and validation."""
    if config.REWARD_ACTIVE_AP_WARMUP_EPISODES != 0:
        raise ValueError('V5 requires REWARD_ACTIVE_AP_WARMUP_EPISODES=0 for a stationary reward')
    if config.REWARD_JFI_SMOOTHING_ENABLED:
        raise ValueError('V5 requires REWARD_JFI_SMOOTHING_ENABLED=False to match the utility shield')
    if config.REWARD_OBJECTIVE_MODE not in {'legacy', 'throughput_temporal_jfi'}:
        raise ValueError('Unknown reward objective mode')
    if not 0.0 <= config.ALIGNED_THROUGHPUT_WEIGHT <= 1.0:
        raise ValueError('Aligned throughput weight must be in [0, 1]')


def reward_components(throughput_mbps, jfi, per_ap_throughput_mbps, active_ratio,
                      active_ap_weight=None):
    """Return per-agent rewards and the components used to form team utility."""
    if active_ap_weight is None:
        active_ap_weight = config.REWARD_ACTIVE_AP_WEIGHT
    total_norm = min(max(float(throughput_mbps) / config.REWARD_TOTAL_THROUGHPUT_REF_MBPS, 0.0), 1.0)
    local_norms = [
        min(max(float(value) / config.LOCAL_THROUGHPUT_REF_MBPS, 0.0), 1.0)
        for value in per_ap_throughput_mbps
    ]
    shared = (
        config.REWARD_GLOBAL_WEIGHT * total_norm
        + config.REWARD_FAIRNESS_WEIGHT * float(jfi)
        + float(active_ap_weight) * (1.0 - float(active_ratio))
    )
    rewards = [
        min(max(shared + config.REWARD_LOCAL_WEIGHT * local, 0.0), 1.0)
        for local in local_norms
    ]
    components = {
        "total_thr_norm": total_norm,
        "mean_local_thr_norm": sum(local_norms) / max(len(local_norms), 1),
        "jfi": float(jfi),
        "active_ratio": float(active_ratio),
        "active_ap_weight": float(active_ap_weight),
    }
    return rewards, components


def team_utility(throughput_mbps, jfi, per_ap_throughput_mbps, active_ratio,
                 active_ap_weight=None):
    """Mean per-agent reward; used consistently for validation and diagnostics."""
    rewards, _ = reward_components(
        throughput_mbps, jfi, per_ap_throughput_mbps, active_ratio,
        active_ap_weight=active_ap_weight,
    )
    return sum(rewards) / max(len(rewards), 1)
