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

    def state_dict(self):
        """Compact tensor-only representation safe for weights-only loading."""
        if not self.memory:
            return {
                "capacity": int(self.memory.maxlen),
                "num_agents": int(self.num_agents),
                "empty": True,
            }
        columns = list(zip(*self.memory))
        return {
            "capacity": int(self.memory.maxlen),
            "num_agents": int(self.num_agents),
            "empty": False,
            "states": torch.from_numpy(np.asarray(columns[0], dtype=np.float32)),
            "actions": torch.from_numpy(np.asarray(columns[1], dtype=np.int64)),
            "rewards": torch.tensor(columns[2], dtype=torch.float32),
            "next_states": torch.from_numpy(np.asarray(columns[3], dtype=np.float32)),
            "dones": torch.tensor(columns[4], dtype=torch.bool),
            "global_states": torch.from_numpy(np.asarray(columns[5], dtype=np.float32)),
            "next_global_states": torch.from_numpy(np.asarray(columns[6], dtype=np.float32)),
        }

    def load_state_dict(self, state):
        if int(state["capacity"]) != self.memory.maxlen:
            raise ValueError("Replay buffer capacity differs from checkpoint")
        if int(state["num_agents"]) != self.num_agents:
            raise ValueError("Replay buffer agent count differs from checkpoint")
        self.memory.clear()
        if state.get("empty", False):
            return
        columns = [
            state[key].cpu().numpy()
            for key in (
                "states", "actions", "rewards", "next_states", "dones",
                "global_states", "next_global_states",
            )
        ]
        for row in zip(*columns):
            states, actions, reward, next_states, done, global_state, next_global = row
            self.memory.append((
                [np.array(value, dtype=np.float32, copy=True) for value in states],
                np.asarray(actions, dtype=np.int64).tolist(),
                float(reward),
                [np.array(value, dtype=np.float32, copy=True) for value in next_states],
                bool(done),
                np.array(global_state, dtype=np.float32, copy=True),
                np.array(next_global, dtype=np.float32, copy=True),
            ))
