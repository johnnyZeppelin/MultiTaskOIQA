from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pandas as pd
import torch

from oiqa_bpr_vmamba.cli.infer_single_image import (
    build_single_batch,
    deterministic_mock_quality,
    lookup_mos_from_csv,
    resolve_checkpoint_path,
    run_transparent_mock_inference,
)
from oiqa_bpr_vmamba.cli.generate_mock_checkpoint import build_mock_state_dict


def _write_img(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.random.rand(size[1], size[0], 3) * 255).astype('uint8')
    Image.fromarray(arr).save(path)


def test_resolve_checkpoint_path_alias(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / 'run'
    ckpt_dir.mkdir()
    (ckpt_dir / 'best.pt').write_bytes(b'0')
    assert resolve_checkpoint_path('best', str(ckpt_dir)) == ckpt_dir / 'best.pt'


def test_build_single_batch_from_roots(tmp_path: Path) -> None:
    image = tmp_path / '326.png'
    restored_image = tmp_path / '326_r.png'
    _write_img(image, (64, 32))
    _write_img(restored_image, (64, 32))
    vp_root = tmp_path / 'vp'
    rvp_root = tmp_path / 'rvp'
    for i in range(1, 3):
        _write_img(vp_root / f'326_fov{i}.png', (16, 16))
        _write_img(rvp_root / f'326_fov{i}_r.png', (16, 16))
    args = SimpleNamespace(
        image=str(image),
        restored_image=str(restored_image),
        num_viewports=2,
        viewports=None,
        restored_viewports=None,
        viewport_root=str(vp_root),
        restored_viewport_root=str(rvp_root),
        viewport_pattern='{stem}_fov{idx}.png',
        restored_viewport_pattern='{stem}_fov{idx}_r.png',
        degradation_seed=1,
    )
    cfg = {
        'model': {'image_size': [32, 64], 'viewport_size': [16, 16], 'num_viewports': 2},
        'degradation': {'seed': 1},
    }
    batch = build_single_batch(args, cfg)
    assert tuple(batch['distorted_global'].shape) == (1, 3, 32, 64)
    assert tuple(batch['distorted_viewports'].shape) == (1, 2, 3, 16, 16)
    assert torch.is_tensor(batch['degraded_viewports'])


def test_build_mock_state_dict_hits_target() -> None:
    state_dict, total = build_mock_state_dict(target_params=1000, dtype=torch.float16, seed=1, tensor_prefix='mock_param', chunk_size=128)
    assert total == 1000
    assert sum(v.numel() for v in state_dict.values()) == 1000


def test_lookup_mos_from_csv_and_deterministic_mock(tmp_path: Path) -> None:
    csv_path = tmp_path / 'mos.csv'
    pd.DataFrame([
        {'fu': 'data/CVIQ/001.png', 'mos': 50.0},
        {'fu': 'data/CVIQ/002.png', 'mos': 70.0},
    ]).to_csv(csv_path, index=False)
    mos, info = lookup_mos_from_csv(tmp_path / '001.png', csv_path)
    assert mos == 50.0
    assert info['match_type'] == 'basename'
    q1, d1 = deterministic_mock_quality(50.0, '001', 'mock.pt', max_relative_error=0.02)
    q2, d2 = deterministic_mock_quality(50.0, '001', 'mock.pt', max_relative_error=0.02)
    assert q1 == q2
    assert d1 == d2
    assert abs(d1) <= 0.02


def test_run_transparent_mock_inference(tmp_path: Path) -> None:
    image = tmp_path / '326.png'
    restored_image = tmp_path / '326_r.png'
    _write_img(image, (64, 32))
    _write_img(restored_image, (64, 32))
    vp_root = tmp_path / 'vp'
    rvp_root = tmp_path / 'rvp'
    for i in range(1, 3):
        _write_img(vp_root / f'326_fov{i}.png', (16, 16))
        _write_img(rvp_root / f'326_fov{i}_r.png', (16, 16))
    csv_path = tmp_path / 'mos.csv'
    pd.DataFrame([{'fu': 'data/OI/326.png', 'mos': 88.0}]).to_csv(csv_path, index=False)
    ckpt_path = tmp_path / 'mock.pt'
    torch.save({'model': {'mock_param_0000': torch.randn(16)}, 'transparent_mock': True, 'mock_checkpoint': True}, ckpt_path)
    args = SimpleNamespace(
        image=str(image),
        restored_image=str(restored_image),
        num_viewports=2,
        viewports=None,
        restored_viewports=None,
        viewport_root=str(vp_root),
        restored_viewport_root=str(rvp_root),
        viewport_pattern='{stem}_fov{idx}.png',
        restored_viewport_pattern='{stem}_fov{idx}_r.png',
        mock_mos_csv=str(csv_path),
        mock_global_column='fu',
        mock_mos_column='mos',
        mock_max_relative_error=0.02,
    )
    result = run_transparent_mock_inference(args, ckpt_path)
    assert result['transparent_mock'] is True
    assert result['matched_mos'] == 88.0
    assert result['num_viewports_resolved'] == 2
    assert abs(result['relative_delta']) <= 0.02
