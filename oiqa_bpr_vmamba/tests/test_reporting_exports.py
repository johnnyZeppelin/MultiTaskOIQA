from __future__ import annotations

from pathlib import Path

import pandas as pd

from oiqa_bpr_vmamba.cli.common import resolve_checkpoint_path, save_eval_outputs
from oiqa_bpr_vmamba.utils.io import save_json


def test_resolve_checkpoint_path_aliases(tmp_path: Path) -> None:
    cfg = {'output_dir': str(tmp_path / 'run')}
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'best.pt').write_text('best', encoding='utf-8')
    (out_dir / 'last.pt').write_text('last', encoding='utf-8')
    assert resolve_checkpoint_path(cfg, 'best') == out_dir / 'best.pt'
    assert resolve_checkpoint_path(cfg, 'last') == out_dir / 'last.pt'
    assert resolve_checkpoint_path(cfg, 'auto') == out_dir / 'best.pt'


def test_save_eval_outputs_writes_summary_bundle(tmp_path: Path) -> None:
    metrics = {
        'predictions': pd.DataFrame({'pred': [1.0, 2.0], 'mos': [1.5, 2.5]}),
        'per_type': {'JPEG': {'PLCC': 0.9, 'SRCC': 0.8, 'RMSE': 1.1}},
        'overall': {'PLCC': 0.95, 'SRCC': 0.85, 'RMSE': 1.0},
        'losses': {'loss_total': 0.1},
        'aux_metrics': {'distortion_acc': 0.5, 'compression_acc': 0.75},
    }
    paths = save_eval_outputs(metrics, tmp_path, prefix='test_eval')
    assert (tmp_path / 'test_eval_predictions.csv').exists()
    assert (tmp_path / 'test_eval_per_type.csv').exists()
    assert (tmp_path / 'test_eval_overall.json').exists()
    assert Path(paths['csv']).exists()
    assert Path(paths['md']).exists()
    assert Path(paths['tex']).exists()
    summary_csv = pd.read_csv(paths['csv'])
    assert 'compression_type' in summary_csv.columns
    assert set(summary_csv['compression_type']) == {'overall', 'JPEG'}
