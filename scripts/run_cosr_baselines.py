"""Run Co-SR baselines across offered loads before training a neural policy.

Example:
    python scripts/run_cosr_baselines.py --seeds 0 1 2
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.cosr_baselines import BASELINE_SELECTORS, HierarchicalUCBPolicy
from env.cosr_env import CoSREnv


DEFAULT_EXPERIMENTS = (
    ("single_oldest", "full"),
    ("group_mnp", "full"),
    ("group_oldest", "full"),
    ("group_oldest", "pf_pair"),
    ("hierarchical_ucb", "pf_pair"),
)
EXPERIMENTS_BY_KEY = {
    "single_oldest_full": ("single_oldest", "full"),
    "group_mnp_full": ("group_mnp", "full"),
    "group_oldest_full": ("group_oldest", "full"),
    "group_oldest_pf": ("group_oldest", "pf_pair"),
    "hierarchical_ucb_pf": ("hierarchical_ucb", "pf_pair"),
}


def run_one(
    seed,
    policy_name,
    power_mode,
    episodes,
    steps,
    warmup_steps,
    num_stas,
    arrival_rate,
    topology_mode,
):
    rng = np.random.default_rng(seed)
    env = CoSREnv(
        num_stas_per_ap=num_stas,
        fixed_topology=(topology_mode == "fixed"),
        fixed_seed=2026 if topology_mode == "fixed" else None,
        mobility_enabled=False,
        power_mode=power_mode,
        max_steps=warmup_steps + steps,
        arrival_rate_per_sta=arrival_rate,
    )
    ucb = HierarchicalUCBPolicy(env) if policy_name == "hierarchical_ucb" else None
    totals = {
        "throughput": 0.0,
        "queue_packets": 0.0,
        "energy_efficiency": 0.0,
        "scheduled_group_size": 0.0,
        "invalid_actions": 0,
        "steps": 0,
    }
    window_values = []

    for episode in range(episodes):
        _, info = env.reset(seed=seed * 10000 + episode)
        mask = info["action_mask"]

        for _ in range(warmup_steps):
            action = (
                ucb.select(env, mask, rng)
                if ucb is not None
                else BASELINE_SELECTORS[policy_name](env, mask, rng)
            )
            _, _, terminated, truncated, info = env.step(action)
            if ucb is not None:
                ucb.update(env, info["executed_action"], info)
            mask = info["action_mask"]
            if terminated or truncated:
                raise RuntimeError("Warm-up exhausted the configured episode length")

        env.begin_measurement_window()
        for _ in range(steps):
            action = (
                ucb.select(env, mask, rng)
                if ucb is not None
                else BASELINE_SELECTORS[policy_name](env, mask, rng)
            )
            _, _, terminated, truncated, info = env.step(action)
            if ucb is not None:
                ucb.update(env, info["executed_action"], info)
            mask = info["action_mask"]
            totals["throughput"] += info["throughput"]
            totals["queue_packets"] += info["queue_packets"]
            totals["energy_efficiency"] += info["energy_efficiency"]
            totals["scheduled_group_size"] += info["scheduled_group_size"]
            totals["invalid_actions"] += int(info["invalid_action"])
            totals["steps"] += 1
            if terminated or truncated:
                break
        window_values.append(env.measurement_metrics())

    denom = max(totals["steps"], 1)
    return {
        "seed": seed,
        "policy": policy_name,
        "power_mode": power_mode,
        "arrival_rate_per_sta": float(arrival_rate),
        "topology_mode": topology_mode,
        "offered_load_mbps": float(np.mean([x["offered_load_mbps"] for x in window_values])),
        "throughput_mbps": totals["throughput"] / denom,
        "jfi": float(np.mean([x["jfi"] for x in window_values])),
        "mean_delay_txops": float(np.mean([x["mean_delay_txops"] for x in window_values])),
        "p95_delay_txops": float(np.mean([x["p95_delay_txops"] for x in window_values])),
        "p95_hol_txops": float(np.mean([x["p95_hol_txops"] for x in window_values])),
        "drop_rate": float(np.mean([x["drop_rate"] for x in window_values])),
        "mean_queue_packets": totals["queue_packets"] / denom,
        "queue_end_packets": float(np.mean([x["queue_end_packets"] for x in window_values])),
        "queue_growth_packets_per_txop": float(np.mean([
            x["queue_growth_packets_per_txop"] for x in window_values
        ])),
        "mean_group_size": totals["scheduled_group_size"] / denom,
        "energy_efficiency_mbps_per_w": totals["energy_efficiency"] / denom,
        "invalid_actions": totals["invalid_actions"],
    }


def aggregate(runs, experiments):
    summary = {}
    metric_keys = [
        "offered_load_mbps", "throughput_mbps", "jfi", "mean_delay_txops",
        "p95_delay_txops", "p95_hol_txops", "drop_rate",
        "mean_queue_packets", "queue_end_packets",
        "queue_growth_packets_per_txop", "mean_group_size",
        "energy_efficiency_mbps_per_w",
    ]
    for arrival_rate in sorted({run["arrival_rate_per_sta"] for run in runs}):
        load_key = f"arrival_rate_{arrival_rate:.2f}"
        summary[load_key] = {}
        for policy_name, power_mode in experiments:
            key = f"{policy_name}__{power_mode}"
            selected = [
                run for run in runs
                if run["arrival_rate_per_sta"] == arrival_rate
                and run["policy"] == policy_name
                and run["power_mode"] == power_mode
            ]
            summary[load_key][key] = {}
            for metric in metric_keys:
                values = np.asarray([run[metric] for run in selected], dtype=float)
                summary[load_key][key][metric + "_mean"] = float(values.mean())
                summary[load_key][key][metric + "_std"] = float(values.std(ddof=0))
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-stas", type=int, default=10)
    parser.add_argument(
        "--arrival-rates", nargs="+", type=float,
        default=[0.30, 0.35, 0.40, 0.45],
        help="Poisson packet arrivals per STA per TXOP.",
    )
    parser.add_argument(
        "--topology-mode", choices=("fixed", "random"), default="fixed",
        help="fixed controls scheduler comparisons; random tests topology generalization.",
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=tuple(EXPERIMENTS_BY_KEY),
        default=list(EXPERIMENTS_BY_KEY),
        help="Named baseline configurations to run.",
    )
    parser.add_argument("--output-dir", default="results_data/cosr_baselines")
    return parser.parse_args()


def main():
    args = parse_args()
    runs = []
    experiments = tuple(EXPERIMENTS_BY_KEY[key] for key in args.experiments)
    for arrival_rate in args.arrival_rates:
        for policy_name, power_mode in experiments:
            for seed in args.seeds:
                result = run_one(
                    seed, policy_name, power_mode,
                    args.episodes, args.steps, args.warmup_steps,
                    args.num_stas, arrival_rate, args.topology_mode,
                )
                runs.append(result)
                print(
                    f"load={result['offered_load_mbps']:.2f} Mbps | "
                    f"{policy_name:18s} {power_mode:7s} seed={seed} | "
                    f"thr={result['throughput_mbps']:.2f} Mbps | "
                    f"JFI={result['jfi']:.3f} | p95-delay={result['p95_delay_txops']:.2f} TXOP | "
                    f"queue-slope={result['queue_growth_packets_per_txop']:.2f} pkt/TXOP"
                )

    summary = aggregate(runs, experiments)
    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.output_dir, f"cosr_load_sweep_{stamp}.json")
    payload = {"config": vars(args), "runs": runs, "summary": summary}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved: {path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
