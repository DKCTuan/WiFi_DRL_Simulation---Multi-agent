import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import config

from env.wifi_env import WiFiEnv
from agent.double_dqn import DoubleDQNAgent


def resolve_model_dir(experiment):
    experiment_dir = os.path.join("results", experiment, "models") if experiment else None
    if experiment == "water_filling":
        return experiment_dir

    candidates = [experiment_dir, os.path.join("results", "models")]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return experiment_dir


def run_visual_simulation(experiment="no_water_filling", action_size=None):
    print("=== ĐANG KHỞI CHẠY MÔ PHỎNG ĐỒ HỌA TRỰC QUAN 2D WI-FI MARL ===")

    # 1. Khởi tạo môi trường
    use_water_filling = experiment == "water_filling"
    action_size = action_size or (3 if use_water_filling else 5)
    env = WiFiEnv(use_water_filling=use_water_filling, action_size=action_size)
    num_agents = env.num_agents
    latent_size = 16
    model_dir = resolve_model_dir(experiment)

    # Kích hoạt chế độ kiểm thử (Không tò mò ngẫu nhiên nữa)
    agents = {
        agent_id: DoubleDQNAgent(state_size=config.OBS_SIZE, action_size=action_size, latent_size=latent_size)
        for agent_id in env.agent_ids
    }

    # Ép cấu hình chạy trên CPU để vẽ đồ thị động Matplotlib mượt mà nhất, không tốn tài nguyên GPU
    device = torch.device("cpu")

    for agent_id in env.agent_ids:
        agents[agent_id].epsilon = 0.0  # Tắt hoàn toàn thám hiểm để AI dùng 100% não khôn

        best_model_path = os.path.join(model_dir, f"best_ib_qmix_{agent_id}_model.pth")
        best_encoder_path = os.path.join(model_dir, f"best_ib_encoder_{agent_id}.pth")
        model_path = best_model_path if os.path.exists(best_model_path) else os.path.join(model_dir, f"ib_qmix_{agent_id}_model.pth")
        encoder_path = best_encoder_path if os.path.exists(best_encoder_path) else os.path.join(model_dir, f"ib_encoder_{agent_id}.pth")

        if os.path.exists(model_path) and os.path.exists(encoder_path):
            # Load trọng số an toàn lên CPU phục vụ render đồ họa
            agents[agent_id].q_network.load_state_dict(torch.load(model_path, map_location=device))
            agents[agent_id].encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            agents[agent_id].q_network.eval()
            agents[agent_id].encoder.eval()
            print(f" Loaded ĐỒNG BỘ bộ não khôn (Encoder + Q-Net) cho {agent_id}")

    # 2. Thiết lập cửa sổ giao diện đồ họa Matplotlib
    plt.ion() # Bật chế độ vẽ động (Interactive Mode)
    fig, ax = plt.subplots(figsize=(8, 7))

    states_dict, _ = env.reset(seed=42)

    # Vòng lặp quét 100 bước thời gian trong kịch bản di động
    for step in range(env.max_steps):
        actions_dict = {}
        for agent_id in env.agent_ids:
            # ✅ SỬA LỖI 1: Ép mảng trạng thái 3 chiều về float32 để khớp hoàn toàn với nơ-ron đầu vào
            state_input = np.array(states_dict[agent_id], dtype=np.float32)
            actions_dict[agent_id] = agents[agent_id].act(state_input)
            print(f"Step {step+1:02d} | {agent_id} | Action: {actions_dict[agent_id]} | CCA hiện tại: {env.aps[int(agent_id[-1])]['cca_threshold']} dBm")

        # Đẩy môi trường bước tiếp, lấy tọa độ STA mới dịch chuyển
        next_states_dict, _, _, _, info = env.step(actions_dict)
        states_dict = next_states_dict

        # 3. TIẾN HÀNH VẼ ĐỒ HỌA LÊN MÀN HÌNH
        ax.clear()
        ax.set_title(f"Mô phỏng Mạng lưới WiFi Multi-Agent (Step {step+1}/{env.max_steps})\nThông lượng tổng: {info['throughput']:.2f} Mbps | JFI: {info['jfi']:.3f}", fontsize=12, fontweight='bold')

        # ✅ SỬA LỖI 2: Đã xóa lệnh import config thừa thãi bên trong loop để tránh bug định danh
        margin = 40
        ax.set_xlim(-margin, config.AREA_SIZE + margin)
        ax.set_ylim(-margin, (config.AREA_SIZE * np.sqrt(3)/2) + margin)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', alpha=0.6)

        # Lấy danh sách các AP đang thỏa mãn ngưỡng CCA để hoạt động phát sóng
        active_aps = env._get_active_aps()
        active_ids = [ap['id'] for ap in active_aps]

        # Vẽ các trạm AP cố định
        for ap in env.aps:
            is_active = ap['id'] in active_ids
            color = 'green' if is_active else 'red'
            marker = '^' # Hình tam giác đại diện cho cột anten trạm phát

            # Vẽ vị trí AP
            ax.scatter(ap['x'], ap['y'], color=color, s=250, marker=marker, zorder=5, label="AP Hoạt động" if ap['id']==0 else "")
            ax.text(ap['x'] - 5, ap['y'] + 8, f"AP_{ap['id']}\nCCA: {ap['cca_threshold']}dBm", color='black', fontsize=9, fontweight='bold')

            # Vẽ vòng tròn mờ thể hiện ngưỡng CCA (Phình to thu nhỏ động khi AI vặn nút)
            radius_visual = abs(ap['cca_threshold']) - 20

            circle = plt.Circle((ap['x'], ap['y']), radius_visual, color=color, fill=True, alpha=0.04, linestyle='--', linewidth=1)
            ax.add_patch(circle)

        # Vẽ các thiết bị người dùng di động (STA)
        for sta in env.stas:
            # Vẽ chấm tròn đại diện cho điện thoại/laptop người dùng
            ax.scatter(sta['x'], sta['y'], color='blue', s=40, marker='o', zorder=4)

            # Vẽ nét đứt nối STA tới AP gốc của nó để chứng minh kết nối viễn thông
            associated_ap = env.aps[sta['ap_id']]
            if sta['ap_id'] in active_ids: # Nếu AP gốc hoạt động, đường nối màu xanh mượt
                ax.plot([associated_ap['x'], sta['x']], [associated_ap['y'], sta['y']], color='gray', linestyle=':', alpha=0.4)
            else: # Nếu AP bị đè nghẽn tắt sóng, đường nối chuyển sang đỏ (mất mạng)
                ax.plot([associated_ap['x'], sta['x']], [associated_ap['y'], sta['y']], color='red', linestyle=':', alpha=0.2)

        # Cập nhật và hiển thị cửa sổ giao diện
        plt.draw()
        plt.pause(0.1) # Tốc độ khung hình (0.1 giây đổi 1 cảnh phim)

    plt.ioff()
    print("=== MÔ PHỎNG HOÀN THÀNH CHUẨN CHỈNH! ===")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained WiFi MARL policy.")
    parser.add_argument("--experiment", default="no_water_filling", choices=["no_water_filling", "water_filling"])
    parser.add_argument("--action-size", type=int, help="Override action space size for the loaded model.")
    args = parser.parse_args()
    run_visual_simulation(experiment=args.experiment, action_size=args.action_size)
