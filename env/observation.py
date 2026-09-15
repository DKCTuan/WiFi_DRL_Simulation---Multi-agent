# env/observation.py
"""
Xây dựng vector observation 7 chiều cho mỗi agent — Section 4.1.
"""
import numpy as np
import config
from env.channel import calculate_distance, calculate_channel_gain
from env.cooperation import neighbor_opportunity_scores


def build_observation(
    agent_idx, aps, stas, local_throughput, active_aps, ap_load_profile,
    cooperation_enabled=False,
):
    ap = aps[agent_idx]
    bss_stas = [s for s in stas if s['ap_id'] == agent_idx]
    active_ap_ids = {a['id'] for a in active_aps}

    # SINR trung bình của các STA thuộc AP này
    sinr_list = []
    for sta in bss_stas:
        dist = calculate_distance(ap, sta)
        gain = calculate_channel_gain(dist, sta['shadowing_db'])
        signal = ap['tx_power'] * gain
        interf = sum(
            o['tx_power'] * calculate_channel_gain(calculate_distance(o, sta), sta['shadowing_db'])
            for o in active_aps if o['id'] != agent_idx
        )
        sinr_list.append(signal / (interf + config.NOISE_POWER))

    avg_sinr = float(np.mean(sinr_list)) if sinr_list else 0.0
    avg_sinr_db = 10 * np.log10(avg_sinr + 1e-10)

    cca_idx = config.CCA_THRESHOLDS.index(ap['cca_threshold'])
    num_agents = len(aps)

    observation = [
        cca_idx / (len(config.CCA_THRESHOLDS) - 1),           # CCA norm [0,1]
        min(local_throughput / 150.0, 1.0),                    # Throughput norm
        ap['tx_power'] / config.P_MAX,                         # TX power norm
        np.clip((avg_sinr_db + 10) / 50, 0, 1),                # SINR norm
        len(bss_stas) / (max(ap_load_profile) * 1.5),          # Load norm
        float(ap['id'] in active_ap_ids),                      # AP này có đang active không
        len(active_ap_ids) / num_agents,                       # Tỷ lệ AP active toàn mạng
    ]
    if cooperation_enabled:
        observation.extend(neighbor_opportunity_scores(agent_idx, aps, stas))
    return np.asarray(observation, dtype=np.float32)
