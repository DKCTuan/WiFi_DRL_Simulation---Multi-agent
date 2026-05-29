# agent/double_dqn.py
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from agent.ib_helper import InformationBottleneckEncoder

class QNetwork(nn.Module):
    def __init__(self, latent_size=16, action_size=3):
        super(QNetwork, self).__init__()
        # Mạng Q nhận đầu vào là không gian ẩn latent_size đã được gọt tỉa nhiễu rác
        self.fc1 = nn.Linear(latent_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, latent_z):
        x = torch.relu(self.fc1(latent_z))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, global_state, next_global_state):
        self.memory.append((state, action, reward, next_state, done, global_state, next_global_state))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, global_states, next_global_states = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions).unsqueeze(1),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones).unsqueeze(1),
            torch.FloatTensor(np.array(global_states)),
            torch.FloatTensor(np.array(next_global_states))
        )

    def __len__(self):
        return len(self.memory)

class DoubleDQNAgent:
    def __init__(self, state_size=1, action_size=3, latent_size=16):
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        
        # Thêm bộ mã hóa Information Bottleneck cục bộ
        self.encoder = InformationBottleneckEncoder(input_dim=state_size, latent_dim=latent_size)
        
        self.q_network = QNetwork(latent_size, action_size)
        self.target_network = QNetwork(latent_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.memory = ReplayBuffer()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def act(self, state):
        # A. Pha khám phá ngẫu nhiên (Epsilon-Greedy)
        if np.random.rand() <= self.epsilon:
            import random
            return random.randrange(self.action_size)
            
        device = next(self.q_network.parameters()).device
        
        # Đẩy state_tensor lên đúng thiết bị GPU/CPU đó
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Bộ lọc IB nén dữ liệu (Mô hình và dữ liệu đã cùng trên GPU)
            mu, log_var = self.encoder(state_tensor)
            z = self.encoder.reparameterize(mu, log_var)
            
            # Dự đoán giá trị Q và lấy hành động có điểm cao nhất
            q_values = self.q_network(z)
            return torch.argmax(q_values, dim=1).item()