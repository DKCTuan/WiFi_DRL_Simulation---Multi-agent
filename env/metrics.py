# env/metrics.py
"""
Rate model + performance metrics — Section 2.3, công thức (3)-(4):
  - per-STA achievable rate (Shannon capacity, chia đều bandwidth trong BSS)
  - aggregate network throughput
  - Jain's Fairness Index (JFI)

Đây cũng là nơi chứa cơ chế JFI-EMA dùng để làm mượt tín hiệu reward
(xem giải thích trong config.py, mục REWARD_JFI_SMOOTHING_ENABLED).
QUAN TRỌNG: EMA chỉ ảnh hưởng đến giá trị dùng làm REWARD, KHÔNG thay đổi
jain_index tức thời — số liệu report trong info["jfi"] (dùng để vẽ Hình 1-4,
tính eval) vẫn là jain_index gốc, không bị làm mượt hộ.
"""
import numpy as np
import config
from env.channel import calculate_distance, calculate_channel_gain


def _sinr(ap, sta, active_aps, noise_power, cooperation_links=None):
    dist_signal = calculate_distance(ap, sta)
    gain_signal = calculate_channel_gain(dist_signal, sta['shadowing_db'])
    signal_power = ap['tx_power'] * gain_signal

    helper_ids = set((cooperation_links or {}).get(sta["id"], ()))
    residual = 10 ** (-config.COBF_INTERFERENCE_SUPPRESSION_DB / 10.0)
    interference_power = 0
    for other_ap in active_aps:
        if other_ap['id'] != ap['id']:
            dist_interf = calculate_distance(other_ap, sta)
            gain_interf = calculate_channel_gain(dist_interf, sta['shadowing_db'])
            suppression = residual if other_ap['id'] in helper_ids else 1.0
            interference_power += suppression * other_ap['tx_power'] * gain_interf

    return signal_power / (interference_power + noise_power)


def _coordination_efficiency_by_ap(aps, stas, cooperation_links):
    """Airtime left after sounding/coordination overhead at each participant."""
    link_load = [0] * len(aps)
    sta_by_id = {sta["id"]: sta for sta in stas}
    for sta_id, helper_ids in (cooperation_links or {}).items():
        sta = sta_by_id.get(sta_id)
        if sta is None:
            continue
        link_load[sta["ap_id"]] += len(helper_ids)
        for helper_id in helper_ids:
            link_load[helper_id] += 1
    return [
        max(0.5, 1.0 - config.COORDINATION_OVERHEAD_PER_LINK * count)
        for count in link_load
    ]


def compute_individual_ap_throughput(
    aps, stas, active_aps, bandwidth, cooperation_links=None,
):
    """Throughput (Mbps) riêng của từng AP, index theo ap['id']."""
    num_agents = len(aps)
    throughputs = [0.0] * num_agents
    efficiencies = _coordination_efficiency_by_ap(aps, stas, cooperation_links)
    for ap in active_aps:
        bss_stas = [sta for sta in stas if sta['ap_id'] == ap['id']]
        if len(bss_stas) == 0:
            continue
        user_bandwidth = bandwidth * efficiencies[ap['id']] / len(bss_stas)

        ap_thr_bps = 0
        for sta in bss_stas:
            sinr = _sinr(ap, sta, active_aps, _noise_power(), cooperation_links)
            ap_thr_bps += user_bandwidth * np.log2(1 + sinr)
        throughputs[ap['id']] = ap_thr_bps / 1e6
    return throughputs


def compute_network_throughput_and_jfi(
    aps, stas, active_aps, bandwidth, cooperation_links=None,
):
    """
    Trả về (total_throughput_mbps, jain_index) — công thức (3)-(4).
    STA thuộc AP không active nhận rate = 0 (vẫn tính vào JFI, đúng ngữ
    nghĩa "được phục vụ công bằng" của bài).
    """
    total_throughput_bps = 0
    sta_throughputs = []
    active_ap_ids = {ap['id'] for ap in active_aps}
    noise_power = _noise_power()
    efficiencies = _coordination_efficiency_by_ap(aps, stas, cooperation_links)

    for ap in aps:
        bss_stas = [sta for sta in stas if sta['ap_id'] == ap['id']]
        if len(bss_stas) == 0:
            continue

        if ap['id'] not in active_ap_ids:
            sta_throughputs.extend([0.0] * len(bss_stas))
            continue

        user_bandwidth = bandwidth * efficiencies[ap['id']] / len(bss_stas)
        for sta in bss_stas:
            sinr = _sinr(ap, sta, active_aps, noise_power, cooperation_links)
            throughput = user_bandwidth * np.log2(1 + sinr)
            sta_throughputs.append(throughput)
            total_throughput_bps += throughput

    total_throughput_mbps = total_throughput_bps / 1e6

    if len(sta_throughputs) > 0 and sum(sta_throughputs) > 0:
        sum_thr = sum(sta_throughputs)
        sum_sq = sum(x ** 2 for x in sta_throughputs)
        jain_index = (sum_thr ** 2) / (len(sta_throughputs) * sum_sq)
    else:
        jain_index = 0.0

    return total_throughput_mbps, jain_index


def _noise_power():
    return config.NOISE_POWER


def update_jfi_ema(prev_ema, jain_index, alpha):
    """EMA đơn giản: ema_t = alpha * jfi_t + (1-alpha) * ema_{t-1}.
    prev_ema=None ở step đầu tiên (dùng luôn jain_index làm điểm khởi đầu)."""
    if prev_ema is None:
        return jain_index
    return alpha * jain_index + (1 - alpha) * prev_ema
