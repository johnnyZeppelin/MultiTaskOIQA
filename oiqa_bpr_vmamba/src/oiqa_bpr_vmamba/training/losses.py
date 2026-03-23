from __future__ import annotations

from typing import Any
import torch
from torch import nn


class MultiTaskLoss(nn.Module):
    def __init__(self, mos_weight: float = 1.0, distortion_weight: float = 0.1, compression_weight: float = 0.1) -> None:
        super().__init__()
        self.mos_weight = mos_weight
        self.distortion_weight = distortion_weight
        self.compression_weight = compression_weight
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        lq = self.mse(outputs['quality'], batch['mos'])
        total = self.mos_weight * lq
        logs = {'loss_quality': float(lq.detach().cpu())}

        if 'distortion_logits' in outputs:
            ld = self.ce(outputs['distortion_logits'], batch['distortion_level'])
            total = total + self.distortion_weight * ld
            logs['loss_distortion'] = float(ld.detach().cpu())
        else:
            logs['loss_distortion'] = 0.0

        if 'compression_logits' in outputs:
            lc = self.ce(outputs['compression_logits'], batch['compression_type'])
            total = total + self.compression_weight * lc
            logs['loss_compression'] = float(lc.detach().cpu())
        else:
            logs['loss_compression'] = 0.0

        logs['loss_total'] = float(total.detach().cpu())
        return total, logs
