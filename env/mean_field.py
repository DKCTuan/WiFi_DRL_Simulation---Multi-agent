# env/mean_field.py
"""
Mean-field cooperation module for AP coordination over wired backhaul (Đóng góp 3).
Mỗi AP tổng hợp thông điệp trung bình z̄_N(b) từ các AP lân cận để hỗ trợ ra quyết định CCA.
"""
import numpy as np
import torch


def get_circular_neighbor_indices(agent_idx, num_agents):
    """
    Trả về danh sách chỉ số AP lân cận trên vòng tròn.
    Ví dụ: AP 0 kề AP 5 (trái) và AP 1 (phải).
    """
    left = (agent_idx - 1) % num_agents
    right = (agent_idx + 1) % num_agents
    return [left, right]


def compute_mean_field_z(all_latents, num_agents):
    """
    all_latents: tensor shape (batch, num_agents, latent_size) hoặc list of arrays
    Trả về tensor cùng shape chứa z̄_N(b) cho từng AP.
    """
    if isinstance(all_latents, torch.Tensor):
        if all_latents.dim() == 2:
            # (num_agents, latent_size)
            mf_list = []
            for i in range(num_agents):
                nbrs = get_circular_neighbor_indices(i, num_agents)
                mf_list.append(all_latents[nbrs].mean(dim=0))
            return torch.stack(mf_list, dim=0)
        elif all_latents.dim() == 3:
            # (batch, num_agents, latent_size)
            mf_list = []
            for i in range(num_agents):
                nbrs = get_circular_neighbor_indices(i, num_agents)
                mf_list.append(all_latents[:, nbrs, :].mean(dim=1))
            return torch.stack(mf_list, dim=1)
    else:
        # list of numpy arrays
        mf_list = []
        for i in range(num_agents):
            nbrs = get_circular_neighbor_indices(i, num_agents)
            mf_list.append(np.mean([all_latents[j] for j in nbrs], axis=0))
        return mf_list
