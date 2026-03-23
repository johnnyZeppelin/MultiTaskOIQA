from __future__ import annotations

from typing import Any
import torch
from torch import nn
import torch.nn.functional as F

from oiqa_bpr_vmamba.models.error_map import NormalizedLogErrorMap
from oiqa_bpr_vmamba.models.backbones import LocalResNetBackbone, GlobalBackboneFactory
from oiqa_bpr_vmamba.models.bs_msfa import BSMSFA, SimpleConcatFusion
from oiqa_bpr_vmamba.models.head import MultiTaskHead


class OIQABPRVMamba(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.num_viewports = int(cfg['num_viewports'])
        self.use_local = bool(cfg.get('use_local', True))
        self.use_global = bool(cfg.get('use_global', True))
        self.use_bs_msfa = bool(cfg.get('use_bs_msfa', True))
        self.use_auxiliary_tasks = bool(cfg.get('use_auxiliary_tasks', True))
        self.error_map = NormalizedLogErrorMap()

        if self.use_local:
            self.restoration_local = LocalResNetBackbone(cfg.get('local_backbone_name', 'resnet50'), pretrained=cfg.get('pretrained', True))
            self.degradation_local = LocalResNetBackbone(cfg.get('local_backbone_name', 'resnet50'), pretrained=cfg.get('pretrained', True))
            local_dims = self.restoration_local.spec.channels
        else:
            self.restoration_local = None
            self.degradation_local = None
            local_dims = [0, 0, 0]

        if self.use_global:
            self.global_backbone = GlobalBackboneFactory.build(
                cfg.get('global_backbone_type', 'timm_hierarchical'),
                cfg.get('global_backbone_name', 'vmamba_tiny'),
                pretrained=cfg.get('pretrained', True),
                fallback_name=cfg.get('global_backbone_fallback'),
            )
            global_dims = self.global_backbone.spec.channels
        else:
            self.global_backbone = None
            global_dims = [0, 0, 0]

        self.local_dims = local_dims
        self.global_dims = global_dims

        fused_dim = int(cfg.get('fused_dim', 256))
        if self.use_bs_msfa:
            self.restoration_fusion = BSMSFA(local_dims, global_dims, fused_dim)
            self.degradation_fusion = BSMSFA(local_dims, global_dims, fused_dim)
        else:
            self.restoration_fusion = SimpleConcatFusion(local_dims, global_dims, fused_dim)
            self.degradation_fusion = SimpleConcatFusion(local_dims, global_dims, fused_dim)

        fused_total_dim = (self.restoration_fusion.out_dim + self.degradation_fusion.out_dim) * self.num_viewports
        self.pre_head = nn.Sequential(
            nn.Linear(fused_total_dim, int(cfg.get('shared_dim', 1024))),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.head = MultiTaskHead(
            in_dim=int(cfg.get('shared_dim', 1024)),
            shared_dim=int(cfg.get('shared_dim', 1024)),
            aux_hidden_dim=int(cfg.get('aux_hidden_dim', 64)),
            num_distortion_levels=int(cfg.get('num_distortion_levels', 5)),
            num_compression_classes=len(cfg.get('compression_classes', ['JPEG', 'AVC', 'HEVC', 'ref'])),
            use_auxiliary_tasks=self.use_auxiliary_tasks,
        )

    @staticmethod
    def _gap_feats(feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]

    def _encode_local_branch(self, distorted_vps: torch.Tensor, pseudo_vps: torch.Tensor, backbone: nn.Module, global_feats: list[torch.Tensor]) -> torch.Tensor:
        B, V, C, H, W = distorted_vps.shape
        error_maps = self.error_map(
            pseudo_vps.reshape(B * V, C, H, W),
            distorted_vps.reshape(B * V, C, H, W),
        )
        local_feats = backbone(error_maps)
        local_feats = [feat.reshape(B, V, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in local_feats]
        per_view_vectors = []
        for v in range(V):
            local_vecs = self._gap_feats([feat[:, v] for feat in local_feats])
            fused = self.restoration_fusion(local_vecs, global_feats) if backbone is self.restoration_local else self.degradation_fusion(local_vecs, global_feats)
            per_view_vectors.append(fused)
        return torch.cat(per_view_vectors, dim=-1)

    def _make_zero_feats(self, batch_size: int, device: torch.device, dims: list[int]) -> list[torch.Tensor]:
        return [torch.zeros(batch_size, d, device=device) for d in dims]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        distorted_global = batch['distorted_global']
        B = distorted_global.shape[0]
        device = distorted_global.device

        if self.use_global:
            global_feats_map = self.global_backbone(distorted_global)
            global_feats = self._gap_feats(global_feats_map)
        else:
            global_feats = self._make_zero_feats(B, device, self.global_dims)

        if self.use_local:
            restored_branch = self._encode_local_branch(batch['distorted_viewports'], batch['restored_viewports'], self.restoration_local, global_feats)
            degraded_branch = self._encode_local_branch(batch['distorted_viewports'], batch['degraded_viewports'], self.degradation_local, global_feats)
        else:
            zero_local = self._make_zero_feats(B, device, self.local_dims)
            per_view_restored, per_view_degraded = [], []
            for _ in range(self.num_viewports):
                per_view_restored.append(self.restoration_fusion(zero_local, global_feats))
                per_view_degraded.append(self.degradation_fusion(zero_local, global_feats))
            restored_branch = torch.cat(per_view_restored, dim=-1)
            degraded_branch = torch.cat(per_view_degraded, dim=-1)

        fused = torch.cat([restored_branch, degraded_branch], dim=-1)
        fused = self.pre_head(fused)
        return self.head(fused)
