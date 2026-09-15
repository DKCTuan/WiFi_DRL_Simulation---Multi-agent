"""Command-line entry point for centralized masked DQN on Co-SR."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.cosr_dqn import train_cosr_dqn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--replay-warmup", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--arrival-rate", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenario-seed", type=int, default=2026,
        help="Fixed topology seed; independent of the DQN run seed.",
    )
    topology = parser.add_mutually_exclusive_group()
    topology.add_argument("--fixed-topology", dest="fixed_topology", action="store_true")
    topology.add_argument("--random-topology", dest="fixed_topology", action="store_false")
    parser.set_defaults(fixed_topology=True)
    parser.add_argument("--expert-steps", type=int, default=6000)
    parser.add_argument("--expert-pretrain-updates", type=int, default=500)
    parser.add_argument("--imitation-weight", type=float, default=0.30)
    parser.add_argument(
        "--imitation-floor", type=float, default=0.05,
        help="Minimum behavioral-cloning loss weight retained after decay (0 reproduces the prior run).",
    )
    parser.add_argument(
        "--lr-decay-after-episode", type=int, default=None,
        help="Halve or otherwise scale learning rate after this completed episode.",
    )
    parser.add_argument(
        "--lr-decay-gamma", type=float, default=0.50,
        help="Multiplier applied once at the learning-rate decay point.",
    )
    parser.add_argument("--output-dir", default="results_data/cosr_dqn")
    args = parser.parse_args()
    result = train_cosr_dqn(**vars(args))
    path = Path(args.output_dir) / f"cosr_dqn_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
