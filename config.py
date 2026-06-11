# config.py
import numpy as np

# 1. Cấu hình Mạng (Network Configuration)
NUM_APS = 6               # Số lượng trạm phát (B = 3)
NUM_STAS_PER_AP = 10       # Số lượng người dùng mỗi AP (K = 5)
BANDWIDTH = 20e6            # Băng thông kênh (W = 20 MHz)
FREQ_C = 5.0                # Tần số sóng mang (fc = 5 GHz)
P_MAX = 0.1                 # Công suất phát tối đa (P_max = 0.1 W)
AREA_SIZE = 100.0            # Kích thước vùng phục vụ 50m x 50m

# 2. Cấu hình Kênh truyền (Channel & Propagation)
L_W = 7.0                   # Suy hao do vật cản (L_w = 7 dB)
D_BP = 10.0                 # Khoảng cách điểm gãy (d_bp = 10 m)
SHADOW_STD = 1.5            # Độ lệch chuẩn Shadowing (1.5 dB)
NOISE_POWER_DBM = -100      # Tạp âm nhiệt (dBm) - Giả định
NOISE_POWER = 10 ** (NOISE_POWER_DBM / 10) * 1e-3  # Đổi sang Watt
TX_POWER_LEVELS = [0.01, 0.02, 0.05, 0.1]

# 3. Cấu hình cho Trí tuệ nhân tạo (AI Settings)
CCA_THRESHOLDS = [-82, -78, -74, -70, -66, -62, -58] # Tập ngưỡng CCA (dBm)
DEFAULT_CCA = -70           # Ngưỡng CCA mặc định
OBS_SIZE = 7

# --- CÔNG SUẤT VÀ NĂNG LƯỢNG ---
MAX_TX_POWER_DBM = 20  # Công suất tối đa của 1 AP (100mW)
MAX_TX_POWER_W = 10 ** (MAX_TX_POWER_DBM / 10) * 1e-3

# Tổng ngân sách công suất của toàn hệ thống (Ví dụ: 3 AP thì ngân sách tối đa là 300mW)
# Water-filling sẽ chia sẻ ngân sách này dựa trên độ sạch của kênh truyền
TOTAL_POWER_BUDGET_W = 0.3

# Khi train RL, không để thuật toán nền ghi đè công suất mà agent vừa chọn.
# Bật True nếu muốn chạy water-filling như một baseline/heuristic riêng.
USE_WATER_FILLING_BASELINE = False

# Giữ nguyên topology ban đầu trong quá trình train để đường học phản ánh tiến bộ
# của policy, thay vì bị nhiễu bởi việc reset ngẫu nhiên vị trí STA/shadowing.
FIXED_TRAIN_SCENARIO = True
TRAIN_SCENARIO_SEED = 2026
TRAIN_MOBILITY_ENABLED = False
