"""
scripts/run_v1_K10.py  —  Kaggle Version 1
==========================================
K = 10  |  Hybrid + Full-AI  |  3 seeds
~3-4 giờ trên Kaggle GPU

Chạy từ thư mục gốc project:
    python scripts/run_v1_K10.py

Output lưu vào:  results/data/
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts._base import run_k
import config as _cfg

SEEDS    = _cfg.FINAL_SEEDS   # [0, 1, 2]
DATA_DIR = "results/data"
K        = 10

MODEL_CONFIGS = [
    dict(model_key="hybrid",  use_wf=True,  action_size=_cfg.HYBRID_ACTION_SIZE),
    dict(model_key="full_ai", use_wf=False, action_size=_cfg.FULL_AI_ACTION_SIZE),
]

print(f"\n{'='*60}")
print(f"  VERSION 1 — K={K} | seeds={SEEDS}")
print(f"{'='*60}")

run_k(K, SEEDS, MODEL_CONFIGS, DATA_DIR, tag="v1")

print(f"\n✓ Xong! Data lưu tại: {DATA_DIR}/")
print("  Upload thư mục này lên Kaggle Dataset để dùng ở plot_final.py")
