"""Wi-Fi 8 user-centric multi-AP experiment at K=10.

This controlled comparison keeps DDQN + QMIX + VIB + water-filling fixed and
changes only multi-AP coordination:

    hybrid_vib : CCA-only Hybrid + WF (7 actions, no cooperation)
    coop_cobf  : CCA x {none, left, right} + WF (21 actions, user-centric Co-BF)

Run from the project root:
    python scripts/run_wifi8_cooperation_K10.py --keys hybrid_vib,coop_cobf --seeds 0,1,2
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from scripts._base import run_ablation_k


DATA_DIR = "results/data"
K = 10

ALL_CONFIGS = {
    "hybrid_vib": dict(
        key="hybrid_vib",
        label="Hybrid + WF + VIB (no cooperation)",
        use_double_dqn=True,
        use_qmix=True,
        ib_beta=config.VIB_BETA,
        use_wf=True,
        action_size=config.HYBRID_ACTION_SIZE,
        cooperation_enabled=False,
    ),
    "coop_cobf": dict(
        key="coop_cobf",
        label="Wi-Fi 8 user-centric Co-BF + WF + VIB",
        use_double_dqn=True,
        use_qmix=True,
        ib_beta=config.VIB_BETA,
        use_wf=True,
        action_size=config.COOPERATIVE_HYBRID_ACTION_SIZE,
        cooperation_enabled=True,
    ),
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--keys", default="hybrid_vib,coop_cobf",
    help="Comma-separated configs: hybrid_vib,coop_cobf",
)
parser.add_argument(
    "--seeds", default=",".join(str(seed) for seed in config.FINAL_SEEDS),
    help="Comma-separated seeds, default: 0,1,2",
)
args = parser.parse_args()

seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
keys = [value.strip() for value in args.keys.split(",") if value.strip()]
unknown = [key for key in keys if key not in ALL_CONFIGS]
if unknown:
    raise ValueError(f"Unknown keys {unknown}; choose from {list(ALL_CONFIGS)}")

print("=" * 72)
print(f"Wi-Fi 8 COOPERATION STUDY | K={K} | keys={keys} | seeds={seeds}")
print("=" * 72)

run_ablation_k(
    K,
    seeds,
    [ALL_CONFIGS[key] for key in keys],
    DATA_DIR,
    use_wf=True,
    action_size=config.HYBRID_ACTION_SIZE,
    tag="wifi8_cooperation",
)

print(f"Done. Data saved under {DATA_DIR}/ with tag wifi8_cooperation.")
