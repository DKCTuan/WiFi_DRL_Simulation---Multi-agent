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


def _sinr(ap, sta, active_aps, noise_power):
    dist_signal = calculate_distance(ap, sta)
    gain_signal = calculate_channel_gain(dist_signal, sta['shadowing_db'])
    signal_power = ap['tx_power'] * gain_signal

    interference_power = 0
    for other_ap in active_aps:
        if other_ap['id'] != ap['id']:
            dist_interf = calculate_distance(other_ap, sta)
            gain_interf = calculate_channel_gain(dist_interf, sta['shadowing_db'])
            # Coordination changes the AP's actual tx_power before this call;
            # there is no artificial fixed-dB interference suppression here.
            interference_power += other_ap['tx_power'] * gain_interf

    return signal_power / (interference_power + noise_power)


def compute_individual_ap_throughput(
    aps, stas, active_aps, bandwidth,
):
    """Throughput (Mbps) riêng của từng AP, index theo ap['id']."""
    return compute_network_metrics(
        aps, stas, active_aps, bandwidth,
    )["ap_throughputs_mbps"]


def compute_network_throughput_and_jfi(
    aps, stas, active_aps, bandwidth,
):
    """
    Trả về (total_throughput_mbps, jain_index) — công thức (3)-(4).
    STA thuộc AP không active nhận rate = 0 (vẫn tính vào JFI, đúng ngữ
    nghĩa "được phục vụ công bằng" của bài).
    """
    metrics = compute_network_metrics(
        aps, stas, active_aps, bandwidth,
    )
    return metrics["throughput_mbps"], metrics["jfi"]


def compute_channel_gain_matrix(aps, stas):
    """Return gains[ap_id, sta_index] once for matched counterfactuals."""
    return np.asarray([
        [
            calculate_channel_gain(
                calculate_distance(ap, sta), sta["shadowing_db"]
            )
            for sta in stas
        ]
        for ap in aps
    ],
        dtype=np.float64,
    )


def compute_network_metrics(
    aps, stas, active_aps, bandwidth,
    gain_matrix=None,
):
    """Compute total/Jain/per-AP/per-STA rates from one shared snapshot."""
    num_aps = len(aps)
    num_stas = len(stas)
    if gain_matrix is None:
        gain_matrix = compute_channel_gain_matrix(aps, stas)
    gains = np.asarray(gain_matrix, dtype=np.float64)
    if gains.shape != (num_aps, num_stas):
        raise ValueError(
            f"gain_matrix must have shape {(num_aps, num_stas)}, got {gains.shape}"
        )

    serving_ids = np.asarray([int(sta["ap_id"]) for sta in stas], dtype=np.int64)
    active_ids = {int(ap["id"]) for ap in active_aps}
    active_mask = np.asarray([ap_id in active_ids for ap_id in range(num_aps)])
    powers = np.zeros(num_aps, dtype=np.float64)
    for ap in active_aps:
        powers[int(ap["id"])] = float(ap["tx_power"])
    rx_power = gains.T * powers[None, :]
    if num_stas:
        sta_rows = np.arange(num_stas)
        desired = rx_power[sta_rows, serving_ids]
        interference = np.maximum(rx_power.sum(axis=1) - desired, 0.0)
        sinr = desired / (interference + _noise_power())
        counts = np.bincount(serving_ids, minlength=num_aps)
        airtime = np.ones(num_stas, dtype=np.float64)
        active_sta = active_mask[serving_ids]
        rates_bps = (
            bandwidth * airtime / np.maximum(counts[serving_ids], 1)
            * np.log2(1.0 + sinr)
        )
        rates_bps = np.where(active_sta, rates_bps, 0.0)
    else:
        rates_bps = np.zeros(0, dtype=np.float64)

    capacity_mbps = rates_bps / 1e6
    if config.USER_DEMAND_ENABLED:
        demand = np.asarray([sta['demand_mbps'] for sta in stas])
        rates_bps = np.minimum(rates_bps, demand * 1e6)
    total_mbps = float(rates_bps.sum() / 1e6)
    ap_mbps = np.bincount(
        serving_ids, weights=rates_bps, minlength=num_aps
    ) / 1e6
    sum_rates = float(rates_bps.sum())
    sum_squares = float(np.square(rates_bps).sum())
    jfi = (sum_rates ** 2 / (num_stas * sum_squares)) if num_stas and sum_squares else 0.0
    return {
        "throughput_mbps": total_mbps,
        "jfi": float(jfi),
        "ap_throughputs_mbps": [float(value) for value in ap_mbps],
        "sta_throughputs_mbps": [float(value / 1e6) for value in rates_bps],
        "sta_capacity_mbps": capacity_mbps.tolist(),
        "gain_matrix": gains,
    }


def _noise_power():
    return config.NOISE_POWER


def update_jfi_ema(prev_ema, jain_index, alpha):
    """EMA đơn giản: ema_t = alpha * jfi_t + (1-alpha) * ema_{t-1}.
    prev_ema=None ở step đầu tiên (dùng luôn jain_index làm điểm khởi đầu)."""
    if prev_ema is None:
        return jain_index
    return alpha * jain_index + (1 - alpha) * prev_ema
