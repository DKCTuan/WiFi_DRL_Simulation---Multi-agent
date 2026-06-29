import csv
import os
import argparse

import config
from utils.plots import plot_learning_curve


def read_float(value):
    return float(value) if value not in ("", None) else None


def find_log_path(explicit_path=None, experiment=None):
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"Log file not found: {explicit_path}")
        return explicit_path

    if experiment:
        experiment_path = os.path.join("results", experiment, "training_logs.csv")
        if not os.path.exists(experiment_path):
            raise FileNotFoundError(f"Log file not found: {experiment_path}")
        return experiment_path

    candidates = [
        os.path.join("results", "no_water_filling", "training_logs.csv"),
        os.path.join("results", "water_filling", "training_logs.csv"),
        os.path.join("results", "training_logs.csv"),
    ]

    existing = [path for path in candidates if os.path.exists(path)]
    if not existing:
        raise FileNotFoundError("No training_logs.csv found under results/")
    return max(existing, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Plot MARL training curves from a CSV log.")
    parser.add_argument("--log", help="Path to training_logs.csv")
    parser.add_argument("--experiment", help="Experiment folder under results/, e.g. no_water_filling")
    args = parser.parse_args()

    log_path = find_log_path(args.log, args.experiment)
    output_dir = os.path.join(os.path.dirname(log_path), "plots")
    episodes = []
    throughput = []
    jfi = []
    eval_episodes = []
    eval_throughput = []
    eval_jfi = []
    eval_active_aps = []
    eval_energy_efficiency = []

    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode = int(row["Episode"])
            episodes.append(episode)
            throughput.append(float(row["Throughput_Mbps"]))
            jfi.append(float(row["JFI"]))

            eval_thr = read_float(row.get("Eval_Throughput_Mbps"))
            if eval_thr is not None:
                eval_episodes.append(episode)
                eval_throughput.append(eval_thr)
                eval_jfi.append(float(row["Eval_JFI"]))
                eval_active_aps.append(float(row["Eval_Active_APs"]))
                eval_ee = read_float(row.get("Eval_Energy_Efficiency_Mbps_per_W"))
                if eval_ee is not None:
                    eval_energy_efficiency.append(eval_ee)

    plot_learning_curve(
        throughput,
        jfi,
        save_dir=output_dir,
        eval_episodes=eval_episodes,
        eval_throughput=eval_throughput,
        eval_jfi=eval_jfi,
        eval_active_aps=eval_active_aps,
        eval_energy_efficiency=eval_energy_efficiency if len(eval_energy_efficiency) == len(eval_episodes) else None,
        num_agents=config.NUM_APS,
    )


if __name__ == "__main__":
    main()
