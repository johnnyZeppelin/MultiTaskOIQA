from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

from oiqa_bpr_vmamba.data.degradation import DegradationConfig, synthesize_random_degradation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--output-root', type=str, required=True)
    parser.add_argument('--num-viewports', type=int, default=20)
    parser.add_argument('--seed', type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DegradationConfig(seed=args.seed)
    manifest = pd.read_csv(args.manifest)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    vp_cols = [f'viewport_{i:02d}' for i in range(1, args.num_viewports + 1)]
    deg_cols = [f'degraded_viewport_{i:02d}' for i in range(1, args.num_viewports + 1)]

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc='synthesize degraded viewports'):
        for vp_col, deg_col in zip(vp_cols, deg_cols):
            src = Path(row[vp_col])
            dst = Path(row[deg_col])
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue
            img = Image.open(src).convert('RGB')
            seed = hash((str(row['image_id']), vp_col, args.seed)) % (2**31 - 1)
            out = synthesize_random_degradation(img, cfg, seed=seed)
            out.save(dst)
    print(f'Degraded viewports written under {out_root}')


if __name__ == '__main__':
    main()
