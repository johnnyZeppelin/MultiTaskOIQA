from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from oiqa_bpr_vmamba.data.degradation import DegradationConfig, synthesize_random_degradation


COMPRESSION_TO_ID = {'JPEG': 0, 'AVC': 1, 'HEVC': 2, 'ref': 3}


@dataclass
class CVIQSample:
    image_id: str
    distorted_global_path: str
    restored_global_path: str
    mos: float
    compression_type: str
    distortion_level: int
    viewport_paths: list[str]
    restored_viewport_paths: list[str]
    degraded_viewport_paths: list[str]


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def resize_image(img: Image.Image, size_hw: tuple[int, int]) -> Image.Image:
    h, w = size_hw
    return img.resize((w, h), resample=Image.BICUBIC)


class CVIQDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        split_csv: str | Path | None,
        image_size: tuple[int, int],
        viewport_size: tuple[int, int],
        num_viewports: int = 20,
        online_degradation_cfg: dict[str, Any] | None = None,
        use_precomputed_degraded: bool = True,
        allowed_compression_types: Iterable[str] | None = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_csv)
        if split_csv is not None:
            split_df = pd.read_csv(split_csv)
            keep_ids = set(split_df['image_id'].astype(str).tolist())
            self.df = self.df[self.df['image_id'].astype(str).isin(keep_ids)].reset_index(drop=True)

        if allowed_compression_types is not None:
            allowed = {str(x) for x in allowed_compression_types}
            self.df = self.df[self.df['compression_type'].astype(str).isin(allowed)].reset_index(drop=True)

        self.image_size = image_size
        self.viewport_size = viewport_size
        self.num_viewports = num_viewports
        self.use_precomputed_degraded = use_precomputed_degraded
        self.deg_cfg = DegradationConfig.from_dict(online_degradation_cfg or {})

        self.viewport_cols = [f'viewport_{i:02d}' for i in range(1, num_viewports + 1)]
        self.restored_cols = [f'restored_viewport_{i:02d}' for i in range(1, num_viewports + 1)]
        self.degraded_cols = [f'degraded_viewport_{i:02d}' for i in range(1, num_viewports + 1)]

    def __len__(self) -> int:
        return len(self.df)

    def _load_rgb(self, path: str | Path, resize_hw: tuple[int, int]) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        img = resize_image(img, resize_hw)
        return pil_to_tensor(img)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        distorted_global = self._load_rgb(row['distorted_global_path'], self.image_size)
        restored_global = self._load_rgb(row['restored_global_path'], self.image_size)

        distorted_vps, restored_vps, degraded_vps = [], [], []
        for i in range(self.num_viewports):
            vp = self._load_rgb(row[self.viewport_cols[i]], self.viewport_size)
            rvp = self._load_rgb(row[self.restored_cols[i]], self.viewport_size)
            distorted_vps.append(vp)
            restored_vps.append(rvp)

            degraded_path = Path(row[self.degraded_cols[i]])
            if self.use_precomputed_degraded and degraded_path.exists():
                dvp = self._load_rgb(degraded_path, self.viewport_size)
            else:
                img = Image.open(row[self.viewport_cols[i]]).convert('RGB')
                img = resize_image(img, self.viewport_size)
                seed = hash((str(row['image_id']), i)) % (2**31 - 1)
                dvp = pil_to_tensor(synthesize_random_degradation(img, self.deg_cfg, seed=seed))
            degraded_vps.append(dvp)

        compression_name = str(row['compression_type'])
        if compression_name not in COMPRESSION_TO_ID:
            raise KeyError(f'Unknown compression_type={compression_name}. Known values: {sorted(COMPRESSION_TO_ID)}')

        return {
            'image_id': str(row['image_id']),
            'distorted_global': distorted_global,
            'restored_global': restored_global,
            'distorted_viewports': torch.stack(distorted_vps, dim=0),
            'restored_viewports': torch.stack(restored_vps, dim=0),
            'degraded_viewports': torch.stack(degraded_vps, dim=0),
            'mos': torch.tensor(float(row['mos']), dtype=torch.float32),
            'compression_type': torch.tensor(COMPRESSION_TO_ID[compression_name], dtype=torch.long),
            'distortion_level': torch.tensor(int(row['distortion_level']), dtype=torch.long),
        }
