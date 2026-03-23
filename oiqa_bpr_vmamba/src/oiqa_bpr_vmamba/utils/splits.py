from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from oiqa_bpr_vmamba.utils.io import ensure_dir, save_json


def _safe_stratify(df: pd.DataFrame, column: str | None) -> pd.Series | None:
    if not column or column not in df.columns:
        return None
    counts = df[column].value_counts(dropna=False)
    if (counts < 2).any():
        return None
    return df[column]


def create_or_load_splits(cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    paths = cfg['paths']
    split_cfg = cfg['split']
    split_dir = ensure_dir(paths['split_dir'])
    split_seed = int(split_cfg['split_seed'])
    train_csv = split_dir / f'train_{split_seed}.csv'
    val_csv = split_dir / f'val_{split_seed}.csv'
    test_csv = split_dir / f'test_{split_seed}.csv'
    meta_json = split_dir / f'split_meta_{split_seed}.json'
    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        return train_csv, val_csv, test_csv

    df = pd.read_csv(paths['manifest_csv'])
    if 'image_id' not in df.columns:
        raise ValueError('Manifest must contain image_id for split generation.')

    train_ratio = float(split_cfg['train_ratio'])
    val_ratio = float(split_cfg['val_ratio'])
    test_ratio = float(split_cfg['test_ratio'])
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f'train/val/test ratios must sum to 1.0, got {total:.6f}')

    strat_col = split_cfg.get('stratify_by')
    strat = _safe_stratify(df, strat_col)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=split_seed,
        stratify=strat,
    )
    rel_test = test_ratio / (val_ratio + test_ratio)
    temp_strat = _safe_stratify(temp_df, strat_col)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        random_state=split_seed,
        stratify=temp_strat,
    )

    for part_df, path in ((train_df, train_csv), (val_df, val_csv), (test_df, test_csv)):
        part_df[['image_id']].to_csv(path, index=False)

    save_json(
        {
            'seed': split_seed,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'train_size': int(len(train_df)),
            'val_size': int(len(val_df)),
            'test_size': int(len(test_df)),
            'stratify_by': strat_col,
        },
        meta_json,
    )
    return train_csv, val_csv, test_csv
