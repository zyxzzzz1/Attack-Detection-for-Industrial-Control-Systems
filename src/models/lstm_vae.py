"""LSTM variational autoencoder (main-line decision 5).

Last hidden state of the encoder LSTM -> μ, logσ² (latent dim) -> reparameterize to z -> repeat W times -> decoder LSTM -> Linear.
Loss = reconstruction + β·KL (β from config, default 1.0), both counted "per window":
    reconstruction = 0.5 × sum of squared errors over all elements of the window (negative log-likelihood of a unit-variance Gaussian, constants dropped)
    KL             = KL(q(z|x) ‖ N(0, I)), summed over latent dimensions
then averaged over the batch of windows. Both terms are on the same scale (both sums), so with β = 1 this is the negative ELBO.
Scoring does not sample (z = μ); score = window MSE (reconstruction error), the same definition as LSTM-AE; it likewise outputs the per-sensor squared error of the last second.
The validation loss (used for early stopping) also uses z = μ, so it is free of sampling noise.
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
