"""Deep SVDD（单类，主线补充决策 5）。

LSTM 编码器（去掉所有 bias，不用带 affine 的归一化层）→ 最后隐状态 → 无 bias 的 Linear → 表示向量。
中心 c = 未训练的网络在全部训练窗口上输出的均值；绝对值小于 eps 的分量设为 ±eps（保留符号，0 取 +eps），之后固定。
去掉 bias 并让 c 远离 0，是为了排除"所有输入都映射到同一个点"的平凡解。
损失 = 到 c 的平方距离的均值；分数 = 到 c 的平方距离。不输出逐传感器误差，不参与归因。
坍塌检查：threshold 阶段记录 val_thr 分数的方差与变异系数，接近 0 时给出警告。
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
        self.register_buffer("center", torch.zeros(rep_dim))  # 随模型一起保存，训练中不更新

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        return self.proj(h[-1])


class DeepSVDDModel(TorchWindowModel):
    name = "deep_svdd"

    def build_net(self, n_features: int, window: int) -> nn.Module:
        p = self.params
        return SVDDEncoder(n_features, p["hidden"], p["layers"], p["rep_dim"])

    def on_fit_start(self, train_ds: WindowDataset) -> None:
        """用初始网络的前向输出确定中心 c，之后固定。"""
        eps = self.params["eps"]
        self.net.eval()
        total, n = torch.zeros_like(self.net.center, dtype=torch.float64), 0
        with torch.no_grad():
            for _, x in train_ds.iter_batches(self.train_cfg["eval_batch_size"]):
                out = self.net(x.to(self.device))
                total += out.double().sum(dim=0)
                n += len(x)
        c = (total / n).float()
        raw = c.abs()  # 替换之前的 |c|：记进 manifest，用来判断中心是否几乎只剩符号
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
