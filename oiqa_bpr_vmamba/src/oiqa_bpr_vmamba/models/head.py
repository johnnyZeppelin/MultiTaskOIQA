from __future__ import annotations

import torch
from torch import nn


class MultiTaskHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        shared_dim: int,
        aux_hidden_dim: int,
        num_distortion_levels: int,
        num_compression_classes: int,
        use_auxiliary_tasks: bool = True,
    ) -> None:
        super().__init__()
        self.use_auxiliary_tasks = use_auxiliary_tasks
        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.quality_head = nn.Linear(shared_dim, 1)
        if use_auxiliary_tasks:
            self.distortion_head = nn.Sequential(
                nn.Linear(shared_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(1024, aux_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(aux_hidden_dim, num_distortion_levels),
            )
            self.compression_head = nn.Sequential(
                nn.Linear(shared_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(1024, aux_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(aux_hidden_dim, num_compression_classes),
            )
        else:
            self.distortion_head = None
            self.compression_head = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(x)
        out = {'quality': self.quality_head(shared).squeeze(-1)}
        if self.use_auxiliary_tasks:
            out['distortion_logits'] = self.distortion_head(shared)
            out['compression_logits'] = self.compression_head(shared)
        return out
