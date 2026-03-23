from __future__ import annotations

import torch
from torch import nn


class BSMSFAUnit1(nn.Module):
    def __init__(self, local_dim: int, global_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(local_dim + global_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([local_feat, global_feat], dim=-1)
        return self.fc(x)


class BSMSFAUnit2(nn.Module):
    def __init__(self, prev_dim: int, local_dim: int, global_dim: int, out_dim: int) -> None:
        super().__init__()
        self.prev_proj = nn.Linear(prev_dim, prev_dim)
        self.fc = nn.Sequential(
            nn.Linear(prev_dim + local_dim + global_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, prev_feat: torch.Tensor, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        prev_proj = self.prev_proj(prev_feat)
        updated = self.fc(torch.cat([prev_feat, local_feat, global_feat], dim=-1))
        return torch.cat([prev_proj, updated], dim=-1)


class BSMSFA(nn.Module):
    def __init__(self, local_dims: list[int], global_dims: list[int], fused_dim: int) -> None:
        super().__init__()
        assert len(local_dims) == 3 and len(global_dims) == 3, 'BS-MSFA expects 3 scales.'
        self.unit1 = BSMSFAUnit1(local_dims[0], global_dims[0], fused_dim)
        self.unit2 = BSMSFAUnit2(fused_dim, local_dims[1], global_dims[1], fused_dim)
        self.unit3 = BSMSFAUnit2(fused_dim * 2, local_dims[2], global_dims[2], fused_dim)
        self.out_dim = fused_dim * 3

    def forward(self, local_feats: list[torch.Tensor], global_feats: list[torch.Tensor]) -> torch.Tensor:
        f1 = self.unit1(local_feats[0], global_feats[0])
        f2 = self.unit2(f1, local_feats[1], global_feats[1])
        f3 = self.unit3(f2, local_feats[2], global_feats[2])
        return f3


class SimpleConcatFusion(nn.Module):
    def __init__(self, local_dims: list[int], global_dims: list[int], fused_dim: int) -> None:
        super().__init__()
        in_dim = sum(local_dims) + sum(global_dims)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, fused_dim * 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.out_dim = fused_dim * 3

    def forward(self, local_feats: list[torch.Tensor], global_feats: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(local_feats + global_feats, dim=-1)
        return self.proj(x)
