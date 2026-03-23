from __future__ import annotations

import torch

from oiqa_bpr_vmamba.models.network import OIQABPRVMamba


def test_forward_smoke() -> None:
    cfg = {
        'num_viewports': 20,
        'num_distortion_levels': 5,
        'compression_classes': ['JPEG', 'AVC', 'HEVC', 'ref'],
        'local_backbone_name': 'resnet50',
        'global_backbone_type': 'timm_hierarchical',
        'global_backbone_name': 'swin_tiny_patch4_window7_224',
        'global_backbone_fallback': 'swin_tiny_patch4_window7_224',
        'pretrained': False,
        'fused_dim': 32,
        'shared_dim': 64,
        'aux_hidden_dim': 16,
        'use_local': True,
        'use_global': True,
        'use_bs_msfa': True,
        'use_auxiliary_tasks': True,
    }
    model = OIQABPRVMamba(cfg)
    batch = {
        'distorted_global': torch.rand(2, 3, 224, 224),
        'distorted_viewports': torch.rand(2, 20, 3, 64, 64),
        'restored_viewports': torch.rand(2, 20, 3, 64, 64),
        'degraded_viewports': torch.rand(2, 20, 3, 64, 64),
    }
    out = model(batch)
    assert out['quality'].shape == (2,)
    assert out['distortion_logits'].shape[0] == 2
    assert out['compression_logits'].shape[0] == 2
