"""LSTM 变分自编码器（主线补充决策 5）。

编码器 LSTM 的最后隐状态 → μ、logσ²（latent 维）→ 重参数化得到 z → 重复 W 次 → 解码器 LSTM → Linear。
损失 = 重构 + β·KL（β 来自 config，默认 1.0），两项都按"每个窗口"计：
    重构 = 0.5 × 窗口内全部元素的平方误差之和（单位方差高斯的负对数似然，去掉常数）
    KL   = KL(q(z|x) ‖ N(0, I))，对 latent 各维求和
再对一批窗口取平均。两项用同一尺度（都是求和），β = 1 时就是负 ELBO。
打分时不采样（z = μ），分数 = 窗口 MSE（重构误差），口径与 LSTM-AE 相同；同样输出最后一秒的逐传感器平方误差。
验证损失（早停依据）同样用 z = μ 计算，不受采样噪声影响。
"""

import torch
from torch import nn

from src.models.torch_base import TorchWindowModel


class LSTMVariationalAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, latent: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)
        self.decoder = nn.LSTM(latent, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.encoder(x)
        return self.to_mu(h[-1]), self.to_logvar(h[-1])

    def decode(self, z: torch.Tensor, window: int) -> torch.Tensor:
        decoded, _ = self.decoder(z.unsqueeze(1).expand(-1, window, -1))
        return self.out(decoded)

    def forward(self, x: torch.Tensor, sample: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return self.decode(z, x.shape[1]), mu, logvar


class LSTMVAEModel(TorchWindowModel):
    name = "lstm_vae"

    def build_net(self, n_features: int, window: int) -> nn.Module:
        p = self.params
        return LSTMVariationalAutoencoder(n_features, p["hidden"], p["layers"], p["latent"])

    def batch_loss(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict]:
        recon, mu, logvar = self.net(x, sample=training)
        mse = ((recon - x) ** 2).mean()
        recon_term = 0.5 * mse * x.shape[1] * x.shape[2]
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=1)).mean()
        return recon_term + self.params["beta"] * kl, {"mse": mse.item(), "kl": kl.item()}

    def batch_scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon, _, _ = self.net(x, sample=False)
        err = (recon - x) ** 2
        return err.mean(dim=(1, 2)), err[:, -1, :]
