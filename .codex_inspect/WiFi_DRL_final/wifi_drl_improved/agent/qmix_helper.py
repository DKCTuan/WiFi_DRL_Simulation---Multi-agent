# agent/qmix_helper.py
import torch
import torch.nn as nn

class QMixer(nn.Module):
    def __init__(self, num_agents=3, state_dim=3):
        super(QMixer, self).__init__()
        self.num_agents = num_agents
        self.embed_dim = 32
        
        # Lớp ẩn 1: Nhận state hệ thống để tạo ra trọng số cho lớp trộn 1
        self.hyper_w1 = nn.Linear(state_dim, num_agents * self.embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, self.embed_dim)
        
        # Lớp ẩn 2: Tạo trọng số để đưa từ lớp ẩn ra điểm Q_tot cuối cùng
        self.hyper_w2 = nn.Linear(state_dim, self.embed_dim * 1)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, global_state):
        """
        agent_qs: Tensor kích thước (batch_size, num_agents) chứa điểm Q cục bộ
        global_state: Tensor kích thước (batch_size, state_dim) chứa toàn bộ trạng thái hệ thống
        """
        batch_size = agent_qs.size(0)
        global_state = global_state.view(-1, global_state.size(-1))
        agent_qs = agent_qs.view(-1, 1, self.num_agents)
        
        # 🌟 BẪY TOÁN HỌC: Ép trọng số đơn điệu bằng torch.abs() theo đúng Mục V-B của bài báo
        w1 = torch.abs(self.hyper_w1(global_state))
        w1 = w1.view(-1, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(global_state).view(-1, 1, self.embed_dim)
        
        # Lớp ẩn trộn phi tuyến 1
        hidden = torch.relu(torch.bmm(agent_qs, w1) + b1)
        
        # Tạo trọng số lớp 2 và ép dương
        w2 = torch.abs(self.hyper_w2(global_state))
        w2 = w2.view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(-1, 1, 1)
        
        # Tính toán ra tổng điểm tối ưu toàn mạng Q_tot
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size, -1)