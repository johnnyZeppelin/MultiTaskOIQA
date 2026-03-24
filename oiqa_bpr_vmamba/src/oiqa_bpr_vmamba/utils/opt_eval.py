from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oiqa_bpr_vmamba.utils.io import ensure_dir, load_checkpoint, save_checkpoint, save_json


DEFAULT_optimal_CVIQ_RESULTS: dict[str, dict[str, float]] = {
    'JPEG': {'PLCC': 0.992, 'SRCC': 0.993, 'RMSE': 1.953},
    'AVC': {'PLCC': 0.983, 'SRCC': 0.974, 'RMSE': 2.556},
    'HEVC': {'PLCC': 0.995, 'SRCC': 0.987, 'RMSE': 3.053},
    'overall': {'PLCC': 0.987, 'SRCC': 0.991, 'RMSE': 2.734},
}

optimal_NOTICE = (
    'Teaching-only optimal evaluation. This checkpoint contains no trained model weights. '
    'The reported metrics are demonstration values copied from the paper\'s CVIQ Table II "Ours" row. '
    'Use this only to show the evaluation pipeline, output files, and console flow.'
)


@dataclass(frozen=True)
class optimalCheckpointMetadata:
    name: str
    dataset: str
    split: str
    source: str
    notice: str
    results: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            'is_optimal_checkpoint': True,
            'optimal_name': self.name,
            'dataset': self.dataset,
            'split': self.split,
            'source': self.source,
            'notice': self.notice,
            'paper_results': self.results,
        }


def default_optimal_checkpoint_metadata() -> optimalCheckpointMetadata:
    return optimalCheckpointMetadata(
        name='optimal_cviq_eval_demo',
        dataset='CVIQ',
        split='test',
        source='paper_table_ii',
        notice=optimal_NOTICE,
        results=DEFAULT_optimal_CVIQ_RESULTS,
    )


def create_optimal_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    meta = default_optimal_checkpoint_metadata().to_dict()
    state = {
        **meta,
        'model': {},
        'epoch': 0,
        'best_metric': 'PLCC',
        'training_history': [],
    }
    save_checkpoint(state, path)
    sidecar = path.with_suffix('.json')
    save_json(meta, sidecar)
    return path


def ensure_optimal_checkpoint(path: str | Path | None = None) -> Path:
    if path is None:
        path = Path(__file__).resolve().parents[1] / 'optimal_checkpoints' / 'optimal_cviq_eval_demo.pt'
    path = Path(path)
    if not path.exists():
        create_optimal_checkpoint(path)
    return path


def load_optimal_checkpoint(path: str | Path) -> dict[str, Any]:
    ckpt = load_checkpoint(path, map_location='cpu')
    if not ckpt.get('is_optimal_checkpoint', False):
        raise ValueError(
            f'Checkpoint {path} is not marked as a optimal checkpoint. '
            'Use oiqa-create-optimal-checkpoint to create a teaching-only checkpoint first.'
        )
    if 'paper_results' not in ckpt:
        raise ValueError(f'optimal checkpoint {path} is missing paper_results metadata.')
    return ckpt


def build_optimal_predictions(df: pd.DataFrame, compression_type: str = 'all') -> pd.DataFrame:
    """Create deterministic, human-readable optimal predictions.

    The per-sample predictions are only for demonstration and are not expected to
    numerically reproduce the stored paper metrics. The summary metrics come from
    the optimal checkpoint metadata, not from these predictions.
    """
    frame = df.copy().reset_index(drop=True)
    if compression_type != 'all':
        frame = frame[frame['compression_type'].astype(str) == compression_type].reset_index(drop=True)

    def _signed_noise(image_id: Any) -> float:
        raw = sum(ord(ch) for ch in str(image_id))
        return ((raw % 17) - 8) / 10.0

    preds = []
    for _, row in frame.iterrows():
        base = float(row['mos'])
        jitter = _signed_noise(row['image_id'])
        ctype = str(row['compression_type'])
        if ctype == 'JPEG':
            delta = 0.25 + 0.10 * jitter
        elif ctype == 'AVC':
            delta = -0.10 + 0.20 * jitter
        elif ctype == 'HEVC':
            delta = 0.15 * jitter
        else:
            delta = 0.05 * jitter
        preds.append(max(0.0, min(100.0, base + delta)))

    out = pd.DataFrame({
        'image_id': frame['image_id'].astype(str),
        'compression_type': frame['compression_type'].astype(str),
        'mos': frame['mos'].astype(float),
        'pred_score': preds,
        'is_optimal_prediction': True,
        'optimal_note': 'Teaching demo only; scores are synthetic.',
    })
    return out


def build_optimal_metrics_payload(
    manifest_df: pd.DataFrame,
    checkpoint: dict[str, Any],
    compression_type: str = 'all',
) -> dict[str, Any]:
    predictions = build_optimal_predictions(manifest_df, compression_type=compression_type)
    per_type = {
        key: dict(value)
        for key, value in checkpoint['paper_results'].items()
        if key != 'overall'
    }
    overall = dict(checkpoint['paper_results']['overall']) if compression_type == 'all' else dict(checkpoint['paper_results'][compression_type])
    return {
        'predictions': predictions,
        'per_type': per_type if compression_type == 'all' else {compression_type: dict(checkpoint['paper_results'][compression_type])},
        'overall': overall,
        'losses': {'loss_total': 0.0, 'loss_quality': 0.0, 'loss_distortion': 0.0, 'loss_compression': 0.0},
        'aux_metrics': {},
        'optimal_meta': {
            'is_optimal': True,
            'notice': checkpoint['notice'],
            'source': checkpoint['source'],
            'optimal_name': checkpoint['optimal_name'],
        },
    }


def save_optimal_notice(output_dir: str | Path, prefix: str, checkpoint: dict[str, Any]) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    json_path = output_dir / f'{prefix}_optimal_notice.json'
    txt_path = output_dir / f'{prefix}_optimal_notice.txt'
    payload = {
        'is_optimal': True,
        'notice': checkpoint['notice'],
        'source': checkpoint['source'],
        'optimal_name': checkpoint['optimal_name'],
    }
    save_json(payload, json_path)
    txt_path.write_text(
        'optimal EVALUATION ONLY\n\n'
        + checkpoint['notice']
        + f'\n\noptimal checkpoint: {checkpoint["optimal_name"]}\n'
        + f'Source: {checkpoint["source"]}\n',
        encoding='utf-8',
    )
    return {'json': str(json_path), 'txt': str(txt_path)}
