# env/mobility.py
"""
Mô hình mobility (Section 2.2): bounded random-waypoint với velocity vector
mượt. Mutates `stas` in-place mỗi step.

Lưu ý: các hằng số vx/vy step (0.02), max_speed (0.3), và bán kính phản xạ
(22.0) hiện đang HARD-CODE ở đây — đây là phần "lý tưởng hóa" mà nếu sau
này làm mô hình thực tế hơn (theo hướng thầy giao) thì đây chính là chỗ cần
sửa đầu tiên (ví dụ: đổi sang mô hình group mobility, hoặc pha thời gian
đứng yên/di chuyển thực tế hơn).
"""
import numpy as np

STA_VELOCITY_JITTER = 0.02
STA_MAX_SPEED = 0.3
STA_REFLECT_RADIUS = 22.0


def update_mobility(stas, aps):
    """Cập nhật vị trí/vận tốc của toàn bộ STA theo 1 step (in-place)."""
    for sta in stas:
        associated_ap = aps[sta['ap_id']]

        sta['x'] += sta['vx']
        sta['y'] += sta['vy']

        sta['vx'] += np.random.uniform(-STA_VELOCITY_JITTER, STA_VELOCITY_JITTER)
        sta['vy'] += np.random.uniform(-STA_VELOCITY_JITTER, STA_VELOCITY_JITTER)

        speed = np.sqrt(sta['vx'] ** 2 + sta['vy'] ** 2)
        if speed > STA_MAX_SPEED:
            sta['vx'] *= STA_MAX_SPEED / speed
            sta['vy'] *= STA_MAX_SPEED / speed

        dist_to_ap = np.sqrt(
            (sta['x'] - associated_ap['x']) ** 2 + (sta['y'] - associated_ap['y']) ** 2
        )
        if dist_to_ap > STA_REFLECT_RADIUS:
            sta['vx'] *= -1
            sta['vy'] *= -1
