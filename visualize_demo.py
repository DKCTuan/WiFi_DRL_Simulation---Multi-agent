# visualize_demo.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from env.wifi_env import WiFiEnv
from agent.double_dqn import DoubleDQNAgent

def run_visual_simulation():
    print("=== ĐANG KHỞI CHẠY MÔ PHỎNG ĐỒ HỌA TRỰC QUAN 2D WI-FI MARL ===")
    
    # 1. Khởi tạo môi trường
    env = WiFiEnv()
    num_agents = env.num_agents
    latent_size = 16
    
    # Kích hoạt chế độ kiểm thử (Không tò mò ngẫu nhiên nữa)
    agents = {agent_id: DoubleDQNAgent(state_size=1, action_size=3, latent_size=latent_size) for agent_id in env.agent_ids}
    for agent_id in env.agent_ids:
        agents[agent_id].epsilon = 0.0  # Tắt hoàn toàn thám hiểm để AI dùng 100% não khôn
        
        # Thử nạp trọng số đã huấn luyện (Nếu Tuấn chưa train xong thì nó sẽ chạy bằng não ngẫu nhiên ban đầu)
        model_path = f"results/models/ib_qmix_{agent_id}_model.pth"
        if os.path.exists(model_path):
            agents[agent_id].q_network.load_state_dict(torch.load(model_path))
            print(f" Loaded cấu hình não khôn cho {agent_id}")

    # 2. Thiết lập cửa sổ giao diện đồ họa Matplotlib
    plt.ion() # Bật chế độ vẽ động (Interactive Mode)
    fig, ax = plt.subplots(figsize=(8, 7))
    
    states_dict, _ = env.reset()
    
    # Vòng lặp quét 100 bước thời gian trong kịch bản di động
    for step in range(env.max_steps):
        actions_dict = {}
        for agent_id in env.agent_ids:
            actions_dict[agent_id] = agents[agent_id].act(states_dict[agent_id])
            
        # Đẩy môi trường bước tiếp, lấy tọa độ STA mới dịch chuyển
        next_states_dict, _, _, _, info = env.step(actions_dict)
        states_dict = next_states_dict
        
        # 3. TIẾN HÀNH VẼ ĐỒ HỌA LÊN MÀN HÌNH
        ax.clear()
        ax.set_title(f"Mô phỏng Mạng lưới WiFi Multi-Agent (Step {step+1}/100)\nThông lượng tổng: {info['throughput']:.2f} Mbps | JFI: {info['jfi']:.3f}", fontsize=12, fontweight='bold')
        ax.set_xlim(-50, 200)
        ax.set_ylim(-50, 200)
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
            # Ngưỡng CCA càng cao (-60dBm > -85dBm) thì bán kính nhạy cảm càng nhỏ
            radius_visual = 120 - abs(ap['cca_threshold']) 
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
    run_visual_simulation()