"""
scripts/run_ablation_K10_hybrid.py
====================================
Ablation study — K = 10 | Hybrid-AI + Water-Filling action space | 3 seeds

Bản song song với run_ablation_K10.py, nhưng trên nhánh Hybrid: agent chỉ học
chọn CCA threshold (action space nhỏ hơn), còn công suất phát được xử lý bởi
water-filling (không phải RL). So sánh 2 file cho biết đóng góp của
QMIX/Double DQN/VIB thay đổi thế nào khi RL không còn phải học toàn bộ hành
động (Full-AI) mà chỉ học một phần (Hybrid).

    baseline  : clean baseline (Double DQN + QMIX, không VIB)
    no_ddqn   : bỏ Double DQN -> vanilla DQN
    no_qmix   : bỏ QMIX mixer -> VDN
    with_vib  : thêm VIB vào clean baseline

Chạy từ thư mục gốc project (mặc định chạy full 4 ablation x 3 seed):
    python scripts/run_ablation_K10_hybrid.py

Chạy MỘT PHẦN để chia việc trên nhiều máy/tài khoản:
    python scripts/run_ablation_K10_hybrid.py --keys baseline,no_ddqn --seeds 0
    python scripts/run_ablation_K10_hybrid.py --keys no_qmix,with_vib  --seeds 0
    python scripts/run_ablation_K10_hybrid.py --keys baseline,no_ddqn,no_qmix,with_vib --seeds 1,2

Output lưu vào:  results/data/ (tên file có tiền tố "ablation_hybrid", tách
biệt hoàn toàn với bản Full-AI trong run_ablation_K10.py — tiền tố "ablation")
và results/best_models/K10_ablation_hybrid_<key>/. Chạy 2 script này theo thứ
tự nào cũng được, không ghi đè lẫn nhau.
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
print(f"  ABLATION STUDY — K={K} | Hybrid + Water-Filling | keys={selected_keys} | seeds={SEEDS}")
print(f"{'='*60}")

run_ablation_k(
    K, SEEDS, ABLATION_CONFIGS, DATA_DIR,
    use_wf=True,                          # Hybrid-AI: bật water-filling cho công suất
    action_size=_cfg.HYBRID_ACTION_SIZE,  # action space nhỏ hơn (chỉ CCA threshold)
    tag="ablation_clean_hybrid",
)

print(f"\n✓ Xong! Data lưu tại: {DATA_DIR}/")
print(f"  Đã chạy (Hybrid+WF): keys={selected_keys} | seeds={SEEDS} — K=10.")
print("  So kết quả với run_ablation_K10.py (Full-AI) để xem QMIX/DDQN/VIB đóng góp nhiều/ít hơn ở nhánh nào.")
