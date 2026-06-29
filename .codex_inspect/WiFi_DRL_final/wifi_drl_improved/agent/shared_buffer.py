# agent/shared_buffer.py
import torch
import numpy as np
import random
from collections import deque

class SharedReplayBuffer:
    """
    Buffer dùng chung cho toàn bộ agents.
    Mỗi transition lưu trạng thái/hành động của TẤT CẢ agents cùng timestep.
    """
    def __init__(self, capacity=10000, num_agents=3):
        self.memory = deque(maxlen=capacity)
        self.num_agents = num_agents

    def add(self, states_all, actions_all, team_reward, next_states_all, done, global_state, next_global_state):
        """
        states_all: list[np.array] — state của từng agent, theo thứ tự agent_ids
        actions_all: list[int]
        team_reward: float — tổng reward của cả team
        """
        self.memory.append((states_all, actions_all, team_reward, next_states_all, done, global_state, next_global_state))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states_all, actions_all, team_rewards, next_states_all, dones, global_states, next_global_states = zip(*batch)
        
        # states_all: (batch_size, num_agents, state_dim)
        states_tensor = torch.FloatTensor(np.array(states_all))
        actions_tensor = torch.LongTensor(np.array(actions_all))
        rewards_tensor = torch.FloatTensor(np.array(team_rewards)).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(np.array(next_states_all))
        dones_tensor = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        global_states_tensor = torch.FloatTensor(np.array(global_states))
        next_global_states_tensor = torch.FloatTensor(np.array(next_global_states))
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor, global_states_tensor, next_global_states_tensor

    def __len__(self):
        return len(self.memory)