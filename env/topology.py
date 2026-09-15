# env/topology.py
"""
Sinh topology mạng: vị trí AP (vòng tròn quanh tâm) và vị trí/seed của STA.
Tách riêng để dễ debug các vấn đề liên quan đến tính tái lập (reproducibility)
của topology qua các giá trị K khác nhau, độc lập với logic RL/step().
"""
import numpy as np
import config


def setup_aps():
    """Xếp AP đều theo vòng tròn quanh tâm vùng phủ sóng (100x100 m)."""
    aps = []
    side = config.AREA_SIZE
    for i in range(config.NUM_APS):
        angle = 2 * np.pi * i / config.NUM_APS
        radius = side * 0.35  # 35% kích thước vùng
        x = side / 2 + radius * np.cos(angle)
        y = side / 2 + radius * np.sin(angle)
        aps.append({
            'id': i, 'x': x, 'y': y,
            'cca_threshold': config.DEFAULT_CCA,
            'tx_power': config.TX_POWER_LEVELS[2],  # Mức công suất mặc định = 0.05W
        })
    return aps


def setup_stas(aps, ap_load_profile, base_seed):
    """
    Sinh STA đảm bảo tính SUPERSET HOÀN TOÀN qua mọi giá trị K:
      K=8 chứa đúng các STA của K=6, K=6 chứa đúng các STA của K=4, v.v.

    Cách hoạt động:
      - Mỗi AP dùng một seed riêng cố định (base_seed + ap_id).
      - Toàn bộ K STA được sinh tuần tự từ seed đó, không phụ thuộc
        vào RNG bên ngoài.
      - K=6 → sinh STA 0..5 từ sequence; K=8 → sinh STA 0..7 từ cùng
        sequence → STA 0..5 của K=8 giống hệt K=6.
      - RNG bên ngoài được lưu/phục hồi để không ảnh hưởng phần còn lại.
    """
    stas = []
    sta_id = 0

    # Lưu RNG bên ngoài để phục hồi sau
    outer_rng = np.random.get_state()

    for ap in aps:
        sta_count = ap_load_profile[ap['id']]
        radius_max = config.AP_STA_RADIUS_MAX[ap['id']]
        shadow_bias = config.AP_SHADOW_BIAS_DB[ap['id']]

        # Seed hoàn toàn cố định theo AP — mọi K đều bắt đầu từ đây
        np.random.seed(base_seed + ap['id'])

        for _ in range(sta_count):
            radius = np.random.uniform(1, radius_max)
            angle = np.random.uniform(0, 2 * np.pi)
            vx = np.random.uniform(-0.1, 0.1)
            vy = np.random.uniform(-0.1, 0.1)
            shadow = np.random.normal(shadow_bias, config.SHADOW_STD)
            stas.append({
                'id': sta_id, 'ap_id': ap['id'],
                'x': ap['x'] + radius * np.cos(angle),
                'y': ap['y'] + radius * np.sin(angle),
                'shadowing_db': shadow,
                'vx': vx, 'vy': vy,
            })
            sta_id += 1

    # Phục hồi RNG bên ngoài
    np.random.set_state(outer_rng)
    return stas
