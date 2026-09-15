"""
scripts/run_ablation_K10.py
============================
Ablation study — K = 10 | Full-AI action space | 3 seeds

Lần lượt bỏ TỪNG thành phần một khỏi hệ thống, giữ nguyên các phần còn lại,
để xem đóng góp riêng của mỗi thành phần vào kết quả:

    baseline  : clean baseline (Double DQN + QMIX, không VIB)
    no_ddqn   : bỏ Double DQN -> vanilla DQN (target vừa chọn vừa đánh giá action)
    no_qmix   : bỏ QMIX mixer -> VDN (Q_tot = tổng Q cục bộ)
    with_vib  : thêm VIB vào clean baseline

Chạy từ thư mục gốc project (mặc định chạy full 4 ablation x 3 seed):
    python scripts/run_ablation_K10.py

Chạy MỘT PHẦN để chia việc trên nhiều máy/tài khoản (mỗi máy 1 lệnh khác nhau,
chạy song song, không đụng file của nhau vì tên file luôn có seed/key):
    python scripts/run_ablation_K10.py --keys baseline,no_ddqn --seeds 0
    python scripts/run_ablation_K10.py --keys no_qmix,with_vib  --seeds 0
    python scripts/run_ablation_K10.py --keys baseline,no_ddqn,no_qmix,with_vib --seeds 1,2

Output lưu vào:  results/data/ (file .txt training curve + eval summary cho
mỗi ablation) và results/best_models/K10_ablation_<key>/ (checkpoint tốt nhất
theo TỪNG lần chạy — nếu chạy nhiều seed rời rạc cho cùng 1 key, best_models
sẽ chỉ giữ seed tốt nhất của LẦN CHẠY ĐÓ, không gộp seed từ lần chạy khác;
việc gộp mean/std qua các seed cần làm thủ công khi tổng hợp file .txt).
"""
import argparse
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts._base import run_ablation_k
import config as _cfg

DATA_DIR = "results/data"
K        = 10

ALL_ABLATION_CONFIGS = {
    "baseline": dict(key="baseline", label="Clean baseline (no VIB)", use_double_dqn=True,  use_qmix=True,  ib_beta=0.0),
    "no_ddqn":  dict(key="no_ddqn",  label="No Double DQN",         use_double_dqn=False, use_qmix=True,  ib_beta=0.0),
    "no_qmix":  dict(key="no_qmix",  label="VDN (no learned mixer)", use_double_dqn=True,  use_qmix=False, ib_beta=0.0),
    "with_vib": dict(key="with_vib", label="With VIB",              use_double_dqn=True,  use_qmix=True,  ib_beta=_cfg.VIB_BETA),
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--keys", type=str, default="baseline,no_ddqn,no_qmix,with_vib",
                     help="Danh sách ablation cách nhau dấu phẩy, vd: baseline,no_ddqn (mặc định: cả 4)")
parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in _cfg.FINAL_SEEDS),
                     help="Danh sách seed cách nhau dấu phẩy, vd: 0,1 (mặc định: config.FINAL_SEEDS)")
args = parser.parse_args()

SEEDS = [int(s) for s in args.seeds.split(",")]
selected_keys = [k.strip() for k in args.keys.split(",")]
for k in selected_keys:
    if k not in ALL_ABLATION_CONFIGS:
        raise ValueError(f"Key '{k}' không hợp lệ. Chọn trong: {list(ALL_ABLATION_CONFIGS.keys())}")
ABLATION_CONFIGS = [ALL_ABLATION_CONFIGS[k] for k in selected_keys]

print(f"\n{'='*60}")
print(f"  ABLATION STUDY — K={K} | Full-AI | keys={selected_keys} | seeds={SEEDS}")
print(f"{'='*60}")

run_ablation_k(
    K, SEEDS, ABLATION_CONFIGS, DATA_DIR,
    use_wf=False,                       # Full-AI action space (không water-filling)
    action_size=_cfg.FULL_AI_ACTION_SIZE,
    tag="ablation_clean",
)

print(f"\n✓ Xong! Data lưu tại: {DATA_DIR}/")
print(f"  Đã chạy: keys={selected_keys} | seeds={SEEDS} — K=10.")
