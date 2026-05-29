# agent/ib_helper.py
import torch
import torch.nn as nn

class InformationBottleneckEncoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=16):
        super(InformationBottleneckEncoder, self).__init__()
        # Lớp nén thông tin đầu vào thành các tham số phân phối
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_log_var = nn.Linear(32, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std