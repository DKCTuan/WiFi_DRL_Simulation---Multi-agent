# env/active_set.py
"""
Xác định tập AP active (Section 2.3, công thức (2) trong bài báo), có thêm
2 cơ chế làm mượt KHÔNG có trong công thức gốc của bài (engineering trick để
tránh active-set "nhấp nháy" liên tục giữa các step, vốn là một nguồn gây
JFI dao động mạnh vì mỗi lần 1 AP bật/tắt là toàn bộ throughput của các STA
thuộc AP đó nhảy 0 <-> giá trị dương):
  1. EWMA smoothing trên interference (dBm) trước khi so với ngưỡng CCA.
  2. Hysteresis: ngưỡng bật/tắt lệch nhau `margin` dB để tránh dao động khi
     interference dao động quanh đúng ngưỡng.

Toàn bộ state mượt (EWMA, active/inactive trước đó) được truyền vào dưới
dạng list và được MUTATE IN-PLACE (giữ đúng hành vi bản gốc, để không phá
kết quả đã có).
"""
import numpy as np
from env.channel import calculate_distance, calculate_channel_gain


def calculate_ap_interference_dbm(ap, aps, transmitting_ids=None):
    """Nhiễu (dBm) mà `ap` nhận được từ mọi AP khác (không tính shadowing,
    vì đây là sensing link AP-to-AP theo giả định đơn giản hóa của bài)."""
    interference_received = 0
    for other_ap in aps:
        if (
            other_ap['id'] != ap['id']
            and (transmitting_ids is None or other_ap['id'] in transmitting_ids)
        ):
            dist = calculate_distance(ap, other_ap)
            gain = calculate_channel_gain(dist, shadowing_db=0)
            interference_received += other_ap['tx_power'] * gain

    if interference_received > 0:
        return 10 * np.log10(interference_received / 1e-3)
    return -150.0


def get_active_aps(aps, ewma_dbm_state, active_state, ewma_alpha, hysteresis_db):
    """
    Trả về danh sách AP đang active, dùng EWMA + hysteresis để làm mượt.
    `ewma_dbm_state` và `active_state` là list[float/bool] theo ap_id,
    được cập nhật in-place (giữ trạng thái xuyên suốt các lần gọi step()).
    """
    active_aps = []
    # Snapshot the transmitters from the previous slot. Previously every AP,
    # including inactive APs, contributed interference despite receiving zero
    # throughput and consuming no transmit power in the metrics.
    transmitting_ids = {
        ap['id'] for ap in aps if active_state[ap['id']]
    }
    for ap in aps:
        ap_id = ap['id']
        interference_dbm = calculate_ap_interference_dbm(
            ap, aps, transmitting_ids=transmitting_ids
        )

        prev_ewma = ewma_dbm_state[ap_id]
        if prev_ewma is None:
            smoothed_interference_dbm = interference_dbm
        else:
            smoothed_interference_dbm = (
                ewma_alpha * interference_dbm + (1 - ewma_alpha) * prev_ewma
            )
        ewma_dbm_state[ap_id] = smoothed_interference_dbm

        threshold = ap['cca_threshold']
        margin = hysteresis_db
        was_active = active_state[ap_id]
        if was_active:
            is_active = smoothed_interference_dbm < threshold + margin
        else:
            is_active = smoothed_interference_dbm < threshold - margin

        active_state[ap_id] = is_active
        if is_active:
            active_aps.append(ap)

    if len(active_aps) == 0:
        # Tránh network-wide impasse: ép AP có ngưỡng CCA cao nhất active
        fallback_ap = max(aps, key=lambda a: a['cca_threshold'])
        active_state[fallback_ap['id']] = True
        active_aps.append(fallback_ap)

    return active_aps


def get_active_aps_instant(aps):
    """Phiên bản KHÔNG smoothing (ngưỡng nhị phân thuần công thức (2) gốc).
    Giữ lại làm baseline/tiện ích so sánh, không dùng trong vòng lặp step()
    chính (WiFiEnv dùng get_active_aps ở trên)."""
    active_aps = []
    for ap in aps:
        interference_received = 0
        for other_ap in aps:
            if other_ap['id'] != ap['id']:
                dist = calculate_distance(ap, other_ap)
                gain = calculate_channel_gain(dist, shadowing_db=0)
                interference_received += other_ap['tx_power'] * gain

        if interference_received > 0:
            interference_dbm = 10 * np.log10(interference_received / 1e-3)
        else:
            interference_dbm = -150.0

        if interference_dbm < ap['cca_threshold']:
            active_aps.append(ap)

    if len(active_aps) == 0:
        active_aps.append(max(aps, key=lambda a: a['cca_threshold']))

    return active_aps
