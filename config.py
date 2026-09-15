# config.py
import numpy as np

# 1. Cấu hình Mạng (Network Configuration)
NUM_APS = 6
NUM_STAS_PER_AP = 10          # default K, sẽ bị override bởi run_experiments.py
BANDWIDTH = 20e6
FREQ_C = 5.0
P_MAX = 0.1
AREA_SIZE = 100.0

# 2. Cấu hình Kênh truyền (Channel & Propagation)
L_W = 7.0
D_BP = 10.0
SHADOW_STD = 4.0
NOISE_POWER_DBM = -100
NOISE_POWER = 10 ** (NOISE_POWER_DBM / 10) * 1e-3
TX_POWER_LEVELS = [0.01, 0.02, 0.05, 0.1]
AP_CIRCUIT_POWER_W = 0.05

# AP_LOAD_PROFILE: số STA mặc định mỗi AP (không đồng đều, từ v8)
# Sẽ bị override bởi set_k() trong scripts/ khi sweep K
AP_LOAD_PROFILE = [14, 12, 10, 8, 9, 7]  # tổng 60 STA, heterogeneous load
AP_STA_RADIUS_MAX = [18, 24, 30, 36, 28, 34]
AP_SHADOW_BIAS_DB = [-2.0, -1.0, 0.5, 2.0, 3.5, 5.0]

# 3. Cấu hình AI
CCA_THRESHOLDS = [-82, -78, -74, -70, -66, -62, -58]
DEFAULT_CCA = -70
OBS_SIZE = 7
HYBRID_ACTION_SIZE = len(CCA_THRESHOLDS)
FULL_AI_ACTION_SIZE = len(CCA_THRESHOLDS) * len(TX_POWER_LEVELS)

# Legacy cooperation experiment. It is preserved for reproduction only and is
# not a standards-aligned Co-BF model. A Hybrid agent jointly selects its CCA
# threshold and one target: none / left BSS / right BSS.
COOPERATION_TARGET_OFFSETS = (None, -1, 1)
COOPERATIVE_OBS_SIZE = OBS_SIZE + 2
COOPERATIVE_HYBRID_ACTION_SIZE = len(CCA_THRESHOLDS) * len(COOPERATION_TARGET_OFFSETS)

# Historical fixed-suppression abstraction. Do not report it as true Co-BF:
# there is no antenna array, CSI, steering vector, or zero-forcing precoder.
COBF_INTERFERENCE_SUPPRESSION_DB = 10.0
COORDINATION_OVERHEAD_PER_LINK = 0.03
COORDINATION_MIN_HELPER_RATIO = 0.05
COORDINATION_MAX_HELPERS_PER_STA = 1

# Literature-aligned Co-SR environment (kept separate from the legacy
# CCA/Water-Filling environment).  One environment step is one MAPC TXOP.
COSR_TXOP_DURATION_S = 5e-3
COSR_FIXED_MAC_OVERHEAD_S = 0.2e-3
COSR_COORDINATION_OVERHEAD_S_PER_EXTRA_AP = 0.1e-3
COSR_PACKET_SIZE_BITS = 1500 * 8
COSR_MEAN_ARRIVALS_PER_STA = 0.45
COSR_INITIAL_MEAN_PACKETS_PER_STA = 2.0
COSR_MAX_QUEUE_PACKETS = 200
COSR_MAX_HOL_DELAY_TXOPS = 100
COSR_MAX_GROUP_APS = 2
COSR_COMPATIBILITY_TOLERANCE = 1e-9

# Approximate 802.11ax/11be 20 MHz, one-spatial-stream PHY table.  Keeping the
# table explicit is preferable to silently treating Shannon capacity as an
# achievable packet rate.  It can be replaced by a measured/standard table
# without changing the Co-SR scheduler.
COSR_MCS_SINR_THRESHOLDS_DB = [-5, -2, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
COSR_MCS_RATES_MBPS = [8.6, 17.2, 25.8, 34.4, 51.6, 68.8,
                       77.4, 86.0, 103.2, 114.7, 129.0, 143.4]

# Discrete powers used by the pairwise proportional-fair Co-SR allocator.
COSR_POWER_LEVELS_W = [0.01, 0.02, 0.05, 0.1]

# Reproducibility
GLOBAL_SEED = 2026

# Công suất
MAX_TX_POWER_DBM = 20
MAX_TX_POWER_W = 10 ** (MAX_TX_POWER_DBM / 10) * 1e-3
TOTAL_POWER_BUDGET_W = NUM_APS * 0.03
USE_WATER_FILLING_BASELINE = False

# Topology & Mobility
FIXED_TRAIN_SCENARIO = True
TRAIN_SCENARIO_SEED = 2026
TRAIN_MOBILITY_ENABLED = True

# Evaluation
EVAL_MODE = "fixed"

# Reward weights
REWARD_TOTAL_THROUGHPUT_REF_MBPS = 500.0
LOCAL_THROUGHPUT_REF_MBPS = 80.0
REWARD_GLOBAL_WEIGHT = 0.45
REWARD_LOCAL_WEIGHT = 0.10
REWARD_FAIRNESS_WEIGHT = 0.35
REWARD_ACTIVE_AP_WEIGHT = 0.10

# CCA stability controls
CCA_EWMA_ALPHA = 0.35
CCA_HYSTERESIS_DB = 2.0

# JFI reward smoothing (EMA) — addresses step-to-step JFI oscillation.
# Vấn đề: trước đây phần thưởng công bằng dùng jain_index TỨC THỜI của từng
# step, trong khi active-set/CCA đã có EWMA+hysteresis riêng. Vì mobility +
# việc active_ratio thay đổi khiến jain_index tức thời dao động mạnh giữa các
# step, agent nhận một tín hiệu thưởng công bằng rất nhiễu → khó học một chính
# sách ổn định, thể hiện ra ngoài là đường JFI huấn luyện "răng cưa" mạnh.
# Cách xử lý: dùng EMA của jain_index làm tín hiệu thưởng (không đổi info["jfi"]
# báo cáo — số liệu gốc dùng để vẽ Hình 1-4 và đánh giá vẫn là jain_index tức
# thời, không bị "làm đẹp" hộ).
# Clean baseline: avoid a hidden reward state that is absent from observations.
# Re-enable only as an explicit ablation or after adding the EMA to the state.
REWARD_JFI_SMOOTHING_ENABLED = False
REWARD_JFI_EMA_ALPHA = 0.3   # nhỏ hơn = mượt hơn, phản ứng chậm hơn với thay đổi thật

# RL Hyperparameters
LEARNING_RATE = 1e-4
LR_DECAY_STEP = 514   # scale theo TRAIN_EPISODES=1200, giữ tỷ lệ halving như bản 700ep (300/700)
LR_DECAY_GAMMA = 0.5
GAMMA = 0.95
BATCH_SIZE = 256
# VIB is optional, not part of the clean baseline.
VIB_BETA = 0.001
IB_BETA = 0.0  # backward-compatible default: deterministic encoder
TARGET_UPDATE_TAU = 0.005
REPLAY_BUFFER_CAPACITY = 100000

EPSILON_START = 1.0
EPSILON_MIN = 0.02
# Keep exploration identical across Hybrid and Full-AI. Different schedules
# confound an architecture comparison; 0.995 reaches epsilon floor near ep780.
EPSILON_DECAY = 0.995
EPSILON_DECAY_HYBRID = EPSILON_DECAY
EPSILON_DECAY_FULL_AI = EPSILON_DECAY

# Số bước tối thiểu trong replay buffer trước khi bắt đầu update Q-network
# (trước đây bắt đầu ngay khi đủ 1 batch = BATCH_SIZE ~ 2.5 episode, khiến
# encoder/Q-network học quá sớm trên dữ liệu ít & lệch → gây dip throughput/JFI
# ở khoảng ep 50-100 khi phần exploit dựa trên Q-values còn chưa hội tụ).
LEARNING_STARTS = 2000

# Curriculum cho phần thưởng "tiết kiệm AP": ramp tuyến tính từ 0 lên
# REWARD_ACTIVE_AP_WEIGHT trong N episode đầu, thay vì bật full ngay từ đầu.
# Mục tiêu: để agent học tốt throughput/fairness trước, rồi mới học đánh đổi
# tắt AP — tránh trường hợp ở K thấp (K=4,6) agent tắt AP quá sớm/quá mạnh
# trước khi hiểu rõ chi phí lên JFI, gây đảo ngược so với K cao.
REWARD_ACTIVE_AP_WARMUP_EPISODES = 150

# Same representation capacity for a controlled Hybrid-vs-Full comparison.
LATENT_SIZE = 16
LATENT_SIZE_HYBRID = LATENT_SIZE
LATENT_SIZE_FULL_AI = LATENT_SIZE

TRAIN_EPISODES = 1200   # tăng từ 700 — Full-AI hội tụ thật (dao động <2%) ở ep 477-570/700
                         # (68-81% ngân sách cũ), cần margin an toàn rộng hơn để chắc chắn hội tụ hẳn
TRAIN_STEPS_PER_EPISODE = 100
EVAL_INTERVAL = 50    # giãn ra từ 25 — giảm tổng overhead eval (trước đây eval tốn NHIỀU hơn cả train)
EVAL_EPISODES = 40    # tăng từ 30 — mỗi checkpoint eval ổn định hơn (ít nhiễu hơn)
PLOT_SMOOTHING_WINDOW = 25
PLOT_SKIP_INITIAL_EPISODES = 0

# 3 seeds để cân bằng giữa độ tin cậy thống kê và thời gian train
FINAL_SEEDS = [0, 1, 2]

assert len(AP_LOAD_PROFILE) == NUM_APS
assert len(AP_STA_RADIUS_MAX) == NUM_APS
assert len(AP_SHADOW_BIAS_DB) == NUM_APS
