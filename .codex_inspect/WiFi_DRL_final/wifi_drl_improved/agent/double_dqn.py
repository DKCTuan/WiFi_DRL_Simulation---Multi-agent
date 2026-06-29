# agent/double_dqn.py
import torch
import torch.nn as nn
import numpy as np
import random
import copy
from collections import deque
from agent.ib_helper import InformationBottleneckEncoder
import config


class QNetwork(nn.Module):
    def __init__(self, latent_size=16, action_size=config.FULL_AI_ACTION_SIZE):
        super(QNetwork, self).__init__()
        # Mạng Q sâu hơn để biểu diễn policy với action space lớn
        self.fc1 = nn.Linear(latent_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

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
    def __init__(self, state_size=1, action_size=config.FULL_AI_ACTION_SIZE, latent_size=16,
                 epsilon_decay=config.EPSILON_DECAY):
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size

        # Bộ mã hóa Information Bottleneck cục bộ
        self.encoder = InformationBottleneckEncoder(input_dim=state_size, latent_dim=latent_size)
        self.target_encoder = copy.deepcopy(self.encoder)

        self.q_network = QNetwork(latent_size, action_size)
        self.target_network = QNetwork(latent_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.memory = ReplayBuffer()
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = epsilon_decay  # nhận từ ngoài để hybrid vs full-ai dùng khác nhau

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        device = next(self.q_network.parameters()).device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            mu, _ = self.encoder(state_tensor)
            q_values = self.q_network(mu)
            return torch.argmax(q_values, dim=1).item()
