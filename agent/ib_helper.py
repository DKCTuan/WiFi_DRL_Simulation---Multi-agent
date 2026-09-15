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
        # Clamp để tránh std = exp(0.5*log_var) bùng nổ theo hàm mũ khi log_var
        # bị đẩy lên giá trị dương lớn trong lúc train (nhất là khi IB_BETA rất
        # nhỏ như 0.001, áp lực kéo log_var về 0 gần như không đáng kể so với
        # TD-loss). Không clamp có thể khiến z = mu + eps*std cực kỳ nhiễu ở một
        # vài seed cụ thể, làm training "gãy" ngẫu nhiên (variance cao bất
        # thường, không phản ánh hiệu ứng thật của VIB). Đây là thực hành chuẩn
        # trong các cài đặt VAE/VIB, không đổi công thức KL hay bản chất bottleneck.
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std