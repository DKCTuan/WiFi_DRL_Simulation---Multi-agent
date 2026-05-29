# config.py
import numpy as np

# 1. Cấu hình Mạng (Network Configuration)
NUM_APS = 3                 # Số lượng trạm phát (B = 3)
NUM_STAS_PER_AP = 5         # Số lượng người dùng mỗi AP (K = 5)
BANDWIDTH = 20e6            # Băng thông kênh (W = 20 MHz)
FREQ_C = 5.0                # Tần số sóng mang (fc = 5 GHz)
P_MAX = 0.1                 # Công suất phát tối đa (P_max = 0.1 W)
AREA_SIZE = 50.0            # Kích thước vùng phục vụ 50m x 50m

# 2. Cấu hình Kênh truyền (Channel & Propagation)
L_W = 7.0                   # Suy hao do vật cản (L_w = 7 dB)
D_BP = 10.0                 # Khoảng cách điểm gãy (d_bp = 10 m)
SHADOW_STD = 1.5            # Độ lệch chuẩn Shadowing (1.5 dB)
NOISE_POWER_DBM = -100      # Tạp âm nhiệt (dBm) - Giả định
NOISE_POWER = 10 ** (NOISE_POWER_DBM / 10) * 1e-3  # Đổi sang Watt

# 3. Cấu hình cho Trí tuệ nhân tạo (AI Settings)
CCA_THRESHOLDS = [-82, -78, -74, -70, -66, -62, -58] # Tập ngưỡng CCA (dBm)
DEFAULT_CCA = -82           # Ngưỡng CCA mặc định

# --- CÔNG SUẤT VÀ NĂNG LƯỢNG ---
MAX_TX_POWER_DBM = 20  # Công suất tối đa của 1 AP (100mW)
MAX_TX_POWER_W = 10 ** (MAX_TX_POWER_DBM / 10) * 1e-3

# Tổng ngân sách công suất của toàn hệ thống (Ví dụ: 3 AP thì ngân sách tối đa là 300mW)
# Water-filling sẽ chia sẻ ngân sách này dựa trên độ sạch của kênh truyền
TOTAL_POWER_BUDGET_W = MAX_TX_POWER_W * 3 

NOISE_POWER_DBM = -100
NOISE_POWER_W = 10 ** (NOISE_POWER_DBM / 10) * 1e-3