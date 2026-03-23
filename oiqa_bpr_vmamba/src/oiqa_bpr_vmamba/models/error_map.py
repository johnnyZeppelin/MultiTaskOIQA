from __future__ import annotations

import math
import torch
from torch import nn


class NormalizedLogErrorMap(nn.Module):
    def __init__(self, epsilon: float = 0.1) -> None:
        super().__init__()
        alpha = epsilon / (255.0 ** 2)
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        self._log_alpha = math.log(alpha)

    def forward(self, pseudo_ref: torch.Tensor, distorted: torch.Tensor) -> torch.Tensor:
        diff2 = (pseudo_ref - distorted).pow(2)
        e = torch.log(self.alpha + diff2) / self._log_alpha
        return e.clamp(0.0, 1.0)
