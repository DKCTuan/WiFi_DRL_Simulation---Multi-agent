# env/channel.py
"""
Mô hình kênh truyền (channel model) — tương ứng Section 2.1 của bài báo:
dual-slope path-loss + log-normal shadowing, công thức (1).
Tách riêng ra khỏi wifi_env.py để dễ debug/unit-test độc lập với phần
RL/agent, vì mọi lỗi liên quan đến SINR/throughput bất thường thường bắt
nguồn từ đây.
"""
import numpy as np
import config


def calculate_distance(node1, node2):
    """node1, node2: dict có key 'x', 'y' (AP hoặc STA)."""
    return np.sqrt((node1['x'] - node2['x']) ** 2 + (node1['y'] - node2['y']) ** 2)


def calculate_channel_gain(distance, shadowing_db=0.0):
    """
    Dual-slope path-loss model, công thức (1) trong bài báo:
      PL(d) = L0 + 20log10(fc/2.4) + 20log10(min(d,dbp))
              + 35log10(d/dbp) * 1[d > dbp] + Lw + shadowing
    Trả về linear channel gain g = 10^(-PL/10).
    """
    d = max(distance, 0.1)
    term1 = 40.05
    term2 = 20 * np.log10(config.FREQ_C / 2.4)
    term3 = 20 * np.log10(min(d, config.D_BP))
    term4 = 35 * np.log10(d / config.D_BP) if d > config.D_BP else 0

    path_loss_db = term1 + term2 + term3 + term4 + config.L_W + shadowing_db
    return 10 ** (-path_loss_db / 10)
