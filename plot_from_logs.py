import csv
import os

from utils.plots import plot_learning_curve


def read_float(value):
    return float(value) if value not in ("", None) else None


def main():
    log_path = os.path.join("results", "training_logs.csv")
    episodes = []
    throughput = []
    jfi = []
    eval_episodes = []
    eval_throughput = []
    eval_jfi = []
    eval_active_aps = []

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

    plot_learning_curve(
        throughput,
        jfi,
        save_dir=os.path.join("results", "plots"),
        eval_episodes=eval_episodes,
        eval_throughput=eval_throughput,
        eval_jfi=eval_jfi,
        eval_active_aps=eval_active_aps,
        num_agents=6,
    )


if __name__ == "__main__":
    main()
