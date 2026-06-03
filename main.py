# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from env.wifi_env import WiFiEnv
from agent.double_dqn import DoubleDQNAgent
from agent.qmix_helper import QMixer

from utils.logger import ExperimentLogger
from utils.plots import plot_learning_curve

def train_marl():
    print("=== HUẤN LUYỆN MULTI-AGENT WIFI: QMIX + INFORMATION BOTTLENECK (RESEARCH VERSION) ===")
    
    env = WiFiEnv()
    num_agents = env.num_agents
    latent_size = 16 

    # 🌟 SỬA 1: Tự động phát hiện và ép cấu hình chạy trên Card rời RTX thông qua CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Hệ thống đang sử dụng: {device.type.upper()} để huấn luyện!")
    
    results_dir = "results"
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    logger = ExperimentLogger(save_dir=results_dir)
    
    # 1. Khởi tạo 3 Agent cục bộ tích hợp IB
    agents = {agent_id: DoubleDQNAgent(state_size=1, action_size=3, latent_size=latent_size) for agent_id in env.agent_ids}
    
    # 🌟 SỬA 2: Đẩy toàn bộ mạng nơ-ron cục bộ (Encoder, Q-Net, Target Q-Net) của từng AP lên Card rời
    for agent_id in env.agent_ids:
        agents[agent_id].encoder = agents[agent_id].encoder.to(device)
        agents[agent_id].q_network = agents[agent_id].q_network.to(device)
        agents[agent_id].target_network = agents[agent_id].target_network.to(device)
    
    global_state_dim = num_agents * 3 
    
    # Bổ sung song hành Mạng trộn chính và Mạng trộn mục tiêu (Target Mixer)
    q_mixer = QMixer(num_agents=num_agents, state_dim=global_state_dim)
    target_q_mixer = QMixer(num_agents=num_agents, state_dim=global_state_dim)
    
    # 🌟 SỬA 3: Đẩy bộ đôi mạng Mixer trung tâm lên Card rời và đồng bộ trọng số
    q_mixer = q_mixer.to(device)
    target_q_mixer = target_q_mixer.to(device)
    target_q_mixer.load_state_dict(q_mixer.state_dict()) 
    
    # Gom toàn bộ tham số vào Optimizer chung (Adam vẫn quản lý được khi các mô hình đã lên GPU)
    all_parameters = list(q_mixer.parameters())
    for agent_id in env.agent_ids:
        all_parameters += list(agents[agent_id].encoder.parameters())
        all_parameters += list(agents[agent_id].q_network.parameters())
        
    optimizer = optim.Adam(all_parameters, lr=0.001)
    gamma = 0.99
    batch_size = 64
    episodes = 1000
    beta_ib = 0.01 # Hệ số phạt nén thông tin rác của bộ lọc IB
    
    history_network_throughput = []
    history_jfi = []

    # 2. VÒNG LẶP HUẤN LUYỆN CHUẨN CTDE NÂNG CAO
    for e in range(episodes):
        states_dict, _ = env.reset() 
        total_network_throughput = 0
        total_jfi = 0
        total_marl_reward = 0
        
        # Mảng lưu trữ trạng thái hệ thống trước đó phục vụ tạo Global State thông minh
        import config
        last_throughputs = [0.0] * num_agents
        last_tx_powers = [config.P_MAX] * num_agents
        
        for step in range(100):
            actions_dict = {}
            for agent_id in env.agent_ids:
                actions_dict[agent_id] = agents[agent_id].act(states_dict[agent_id])
            
            # Thiết lập Vector Global State thông minh 9 chiều thời gian thực
            cca_array = [states_dict[aid][0] for aid in env.agent_ids]
            global_state = np.array(cca_array + last_throughputs + last_tx_powers, dtype=np.float32)
            
            # Thực thi kịch bản vật lý viễn thông
            next_states_dict, rewards_dict, terminated, truncated, info = env.step(actions_dict)
            
            # Cập nhật thông số từ phản hồi môi trường để làm Global State kế tiếp
            current_throughputs = info["ap_individual_throughputs"]
            current_tx_powers = [env.aps[i]["tx_power"] for i in range(num_agents)]
            next_cca_array = [next_states_dict[aid][0] for aid in env.agent_ids]
            
            next_global_state = np.array(next_cca_array + current_throughputs + current_tx_powers, dtype=np.float32)
            done = terminated or truncated
            
            # Đẩy bộ trải nghiệm có Global State giàu thông tin vào Replay Buffer
            for agent_id in env.agent_ids:
                agents[agent_id].memory.add(
                    states_dict[agent_id], actions_dict[agent_id], rewards_dict[agent_id],
                    next_states_dict[agent_id], done, global_state, next_global_state
                )
            
            # Cập nhật trạng thái đệm
            last_throughputs = current_throughputs
            last_tx_powers = current_tx_powers
            
            # --- TRÁI TIM CTDE: PHA HUẤN LUYỆN TẬP TRUNG TỐI ƯU ---
            if len(agents["ap_0"].memory) >= batch_size:
                batch_agent_qs = []
                batch_agent_next_qs = []
                global_states_tensor = None
                next_global_states_tensor = None
                dones_tensor = None
                
                # 🌟 SỬA 4: Khởi tạo biến lưu trữ phần thưởng tổng thể nằm sẵn trên Card rời
                team_rewards = torch.zeros(batch_size, 1, device=device)
                total_kl_loss = 0.0
                
                for idx, agent_id in enumerate(env.agent_ids):
                    s, a, r, s_next, d, g_s, g_s_next = agents[agent_id].memory.sample(batch_size)
                    
                    # 🌟 SỬA 5: Chuyển đổi toàn bộ mảng dữ liệu thô (Numpy/Tensor CPU) sang vùng nhớ của GPU (.to(device))
                    s = s.to(device)
                    a = a.to(device)
                    r = r.to(device)
                    s_next = s_next.to(device)
                    d = d.to(device)
                    g_s = g_s.to(device)
                    g_s_next = g_s_next.to(device)
                    
                    if idx == 0:
                        global_states_tensor = g_s
                        next_global_states_tensor = g_s_next
                        dones_tensor = d
                    team_rewards += r
                    
                    # Bộ lọc IB nén dữ liệu cục bộ (Chạy trực tiếp trên GPU)
                    mu, log_var = agents[agent_id].encoder(s)
                    z = agents[agent_id].encoder.reparameterize(mu, log_var)
                    
                    # Tổn hao KL Divergence
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
                    total_kl_loss += kl_loss
                    
                    # Giá trị Q hiện tại dự đoán bởi mạng nơ-ron cục bộ
                    q_values = agents[agent_id].q_network(z).gather(1, a)
                    batch_agent_qs.append(q_values)
                    
                    # Suy luận bước kế tiếp qua mạng Target cục bộ trên GPU
                    with torch.no_grad():
                        mu_next, log_var_next = agents[agent_id].encoder(s_next)
                        z_next = agents[agent_id].encoder.reparameterize(mu_next, log_var_next)
                        best_next_actions = agents[agent_id].q_network(z_next).argmax(1).unsqueeze(1)
                        next_q_values = agents[agent_id].target_network(z_next).gather(1, best_next_actions)
                        batch_agent_next_qs.append(next_q_values)
                
                chosen_qs = torch.cat(batch_agent_qs, dim=1)
                target_next_qs = torch.cat(batch_agent_next_qs, dim=1)
                
                # Mixer chính dự đoán Q_tot, Target Mixer tính Q_tot tiếp theo (Toàn bộ ma trận chạy trên GPU)
                q_tot_predicted = q_mixer(chosen_qs, global_states_tensor)
                with torch.no_grad():
                    q_tot_next = target_q_mixer(target_next_qs, next_global_states_tensor)
                    q_tot_target = team_rewards + (gamma * q_tot_next * (1 - dones_tensor))
                
                # Tổng hợp Loss chung
                td_loss = nn.MSELoss()(q_tot_predicted, q_tot_target.detach())
                marl_ib_loss = td_loss + (beta_ib * total_kl_loss)
                
                optimizer.zero_grad()
                marl_ib_loss.backward()
                optimizer.step()

            states_dict = next_states_dict
            total_network_throughput += info['throughput']
            total_jfi += info['jfi']
            total_marl_reward += sum(rewards_dict.values())
            
            if done:
                break
                
        # Cứ sau 5 ván, cập nhật trọng số cho cả Target Q-Net và Target Mixer!
        if (e + 1) % 5 == 0:
            target_q_mixer.load_state_dict(q_mixer.state_dict())
            for agent_id in env.agent_ids:
                agents[agent_id].target_network.load_state_dict(agents[agent_id].q_network.state_dict())

        for agent_id in env.agent_ids:
                    if agents[agent_id].epsilon > agents[agent_id].epsilon_min:
                        agents[agent_id].epsilon *= agents[agent_id].epsilon_decay
                
        avg_throughput = total_network_throughput / 100
        avg_jfi = total_jfi / 100
        avg_marl_reward = total_marl_reward / 100
        
        history_network_throughput.append(avg_throughput)
        history_jfi.append(avg_jfi)
        
        logger.log_episode(e + 1, avg_throughput, avg_jfi, avg_marl_reward)
        current_epsilon = agents["ap_0"].epsilon
        print(f"Ván {e+1:03d}/{episodes} | Tốc độ mạng: {avg_throughput:6.2f} Mbps | JFI: {avg_jfi:.3f} | Research-Reward: {avg_marl_reward:7.2f} | Epsilon: {current_epsilon:.2f}")
    
    print("\n=== HUẤN LUYỆN XONG! ĐANG ĐÓNG GÓI MÔ HÌNH VÀ GỌI UTILS... ===")
    
    # 🌟 SỬA 6: Khi lưu trữ mô hình .pth, ta chuyển bộ não về lại dạng CPU một chút để lưu file đồng bộ, tránh lỗi phân mảnh thiết bị khi load model sau này.
    for agent_id in env.agent_ids:
        torch.save(agents[agent_id].encoder.to("cpu").state_dict(), os.path.join(models_dir, f"ib_encoder_{agent_id}.pth"))
        torch.save(agents[agent_id].q_network.to("cpu").state_dict(), os.path.join(models_dir, f"ib_qmix_{agent_id}_model.pth"))
    torch.save(q_mixer.to("cpu").state_dict(), os.path.join(models_dir, "ib_qmix_mixer_model.pth"))
    print(f" 💾 Hệ thống đã lưu trữ gọn gàng cấu trúc mô hình tối giản!")
    
    plot_learning_curve(history_network_throughput, history_jfi, save_dir=os.path.join(results_dir, "plots"))
    
if __name__ == "__main__":
    train_marl()