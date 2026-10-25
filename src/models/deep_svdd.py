"""Deep SVDD (one-class, main-line decision 5).

LSTM encoder (all biases removed, no normalization layers with affine parameters) -> last hidden state -> bias-free Linear -> representation vector.
Center c = mean output of the untrained network over all training windows; components with absolute value below eps are set to ±eps (sign kept, 0 becomes +eps), then fixed.
Removing biases and keeping c away from 0 rules out the trivial solution of mapping every input to the same point.
Loss = mean squared distance to c; score = squared distance to c. No per-sensor errors, no attribution.
Collapse check: the threshold stage records the variance and coefficient of variation of the val_thr scores and warns when they are close to 0.
"""

import torch
from torch import nn

from src.data.windowing import WindowDataset
from src.models.torch_base import TorchWindowModel


class SVDDEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, rep_dim: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, bias=False)
        self.proj = nn.Linear(hidden, rep_dim, bias=False)
        self.register_buffer("center", torch.zeros(rep_dim))  # Saved with the model, not updated during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        return self.proj(h[-1])


class DeepSVDDModel(TorchWindowModel):
    name = "deep_svdd"

    def build_net(self, n_features: int, window: int) -> nn.Module:
        p = self.params
        return SVDDEncoder(n_features, p["hidden"], p["layers"], p["rep_dim"])

    def on_fit_start(self, train_ds: WindowDataset) -> None:
        """Determine the center c from the initial network's forward output, then keep it fixed."""
        eps = self.params["eps"]
        self.net.eval()
        total, n = torch.zeros_like(self.net.center, dtype=torch.float64), 0
        with torch.no_grad():
            for _, x in train_ds.iter_batches(self.train_cfg["eval_batch_size"]):
                out = self.net(x.to(self.device))
                total += out.double().sum(dim=0)
                n += len(x)
        c = (total / n).float()
        raw = c.abs()  # |c| before replacement: recorded in the manifest to judge whether the center is little more than signs
        small = c.abs() < eps
        c[small] = torch.where(c[small] < 0, -eps, eps)
        self.net.center.copy_(c)
        self.extra["deep_svdd_center"] = {
            "rep_dim": len(c), "eps": eps, "n_replaced": int(small.sum()),
            "abs_c_before_min": raw.min().item(), "abs_c_before_median": raw.median().item(),
            "abs_c_before_max": raw.max().item(), "n_init_windows": n}
        self.log.info("Deep SVDD 中心：%d 维，其中 %d 维因接近 0 被设为 ±%g；|c| 最小 %.4g，最大 %.4g",
                      len(c), int(small.sum()), eps, c.abs().min().item(), c.abs().max().item())

    def _dist(self, x: torch.Tensor) -> torch.Tensor:
        return ((self.net(x) - self.net.center) ** 2).sum(dim=1)

    def batch_loss(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict]:
        return self._dist(x).mean(), {}

    def batch_scores(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self._dist(x), None
