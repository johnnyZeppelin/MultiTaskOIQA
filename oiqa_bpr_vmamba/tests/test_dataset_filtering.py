from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from oiqa_bpr_vmamba.data.cviq_dataset import CVIQDataset


def _make_img(path: Path, size: tuple[int, int] = (16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color=(128, 128, 128)).save(path)


def test_dataset_can_filter_compression_types(tmp_path: Path) -> None:
    rows = []
    for idx, ctype in enumerate(['JPEG', 'AVC'], start=1):
        distorted_global = tmp_path / f'{idx:03d}.png'
        restored_global = tmp_path / 'restored' / f'{idx:03d}_r.png'
        _make_img(distorted_global)
        _make_img(restored_global)
        row = {
            'image_id': f'{idx:03d}',
            'distorted_global_path': str(distorted_global),
            'restored_global_path': str(restored_global),
            'mos': float(idx),
            'compression_type': ctype,
            'distortion_level': 0,
        }
        for v in range(1, 21):
            vp = tmp_path / ctype / f'{idx:03d}_fov{v}.png'
            rvp = tmp_path / 'restored' / ctype / f'{idx:03d}_fov{v}_r.png'
            dvp = tmp_path / 'degraded' / ctype / f'{idx:03d}_fov{v}_d.png'
            _make_img(vp)
            _make_img(rvp)
            _make_img(dvp)
            row[f'viewport_{v:02d}'] = str(vp)
            row[f'restored_viewport_{v:02d}'] = str(rvp)
            row[f'degraded_viewport_{v:02d}'] = str(dvp)
        rows.append(row)

    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)

    ds = CVIQDataset(
        manifest_csv=manifest,
        split_csv=None,
        image_size=(16, 16),
        viewport_size=(16, 16),
        num_viewports=20,
        allowed_compression_types=['JPEG'],
    )
    assert len(ds) == 1
    sample = ds[0]
    assert sample['compression_type'].item() == 0
