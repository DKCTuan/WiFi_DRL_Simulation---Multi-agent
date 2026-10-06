# env/mobility.py
"""
Mô hình mobility (Section 2.2): bounded random-waypoint với velocity vector
mượt. Mutates `stas` in-place mỗi step.

Velocity jitter and speed are simplified constants. Reflection uses each
BSS's configured placement radius and projects an overshoot back inside it.
The environment passes its own RNG, independent from agent exploration.
"""
import numpy as np
import config

STA_VELOCITY_JITTER = 0.02
STA_MAX_SPEED = 0.3


def update_mobility(stas, aps, rng=None):
    """Cập nhật vị trí/vận tốc của toàn bộ STA theo 1 step (in-place)."""
    rng = np.random if rng is None else rng
    for sta in stas:
        associated_ap = aps[sta['ap_id']]

        sta['x'] += sta['vx']
        sta['y'] += sta['vy']

        sta['vx'] += rng.uniform(-STA_VELOCITY_JITTER, STA_VELOCITY_JITTER)
        sta['vy'] += rng.uniform(-STA_VELOCITY_JITTER, STA_VELOCITY_JITTER)

        speed = np.sqrt(sta['vx'] ** 2 + sta['vy'] ** 2)
        if speed > STA_MAX_SPEED:
            sta['vx'] *= STA_MAX_SPEED / speed
            sta['vy'] *= STA_MAX_SPEED / speed

        dist_to_ap = np.sqrt(
            (sta['x'] - associated_ap['x']) ** 2 + (sta['y'] - associated_ap['y']) ** 2
        )
        radius = config.AP_STA_RADIUS_MAX[sta['ap_id']]
        if dist_to_ap > radius:
            nx = (sta['x'] - associated_ap['x']) / dist_to_ap
            ny = (sta['y'] - associated_ap['y']) / dist_to_ap
            sta['x'] = associated_ap['x'] + radius * nx
            sta['y'] = associated_ap['y'] + radius * ny
            outward_speed = sta['vx'] * nx + sta['vy'] * ny
            if outward_speed > 0:
                sta['vx'] -= 2 * outward_speed * nx
                sta['vy'] -= 2 * outward_speed * ny
