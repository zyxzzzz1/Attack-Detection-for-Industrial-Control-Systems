"""LSTM autoencoder (main model, main-line decision 5; main-model configuration fixed at W=60, hidden=64, constraint 9).

Encoder LSTM takes the last hidden state of the last layer -> repeated W times -> decoder LSTM -> Linear back to the feature dimension.
Score = window MSE (mean over W×D elements); per_sensor_err = squared error of each sensor at the window's last second.
"""

import torch
from torch import nn

from src.models.torch_base import TorchWindowModel


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        repeated = h[-1].unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, W, hidden)
        decoded, _ = self.decoder(repeated)
        return self.out(decoded)


class LSTMAEModel(TorchWindowModel):
    name = "lstm_ae"

    def build_net(self, n_features: int, window: int) -> nn.Module:
        return LSTMAutoencoder(n_features, self.params["hidden"], self.params["layers"])

    def batch_loss(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict]:
        return ((self.net(x) - x) ** 2).mean(), {}

    def batch_scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        err = (self.net(x) - x) ** 2
        return err.mean(dim=(1, 2)), err[:, -1, :]
