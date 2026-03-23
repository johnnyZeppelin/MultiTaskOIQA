from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import yaml

from oiqa_bpr_vmamba.cli.common import existing_run_is_complete, run_logged_subprocess
from oiqa_bpr_vmamba.utils.config import load_yaml_config
from oiqa_bpr_vmamba.utils.io import ensure_dir, load_json, save_json


ABLATIONS = {
    'baseline': {},
    'no_local': {'model': {'use_local': False}},
    'no_global': {'model': {'use_global': False}},
    'no_bs_msfa': {'model': {'use_bs_msfa': False}},
    'no_auxiliary': {'model': {'use_auxiliary_tasks': False}},
    'vit_backbone': {'model': {'global_backbone_type': 'vit', 'global_backbone_name': 'vit_base_patch16_224'}},
}


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--work-dir', type=str, default='runs/ablations')
    parser.add_argument('--ablations', type=str, default='all', help='Comma-separated names or all')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--resume', action='store_true', help='Pass --resume auto to each child run.')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--include-baseline', action='store_true')
    return parser.parse_args()


def _selected_ablations(spec: str, include_baseline: bool) -> dict[str, dict]:
    if spec == 'all':
        names = list(ABLATIONS.keys()) if include_baseline else [k for k in ABLATIONS if k != 'baseline']
        return {name: ABLATIONS[name] for name in names}
    names = [x.strip() for x in spec.split(',') if x.strip()]
    if include_baseline and 'baseline' not in names:
        names = ['baseline', *names]
    unknown = [x for x in names if x not in ABLATIONS]
    if unknown:
        raise ValueError(f'Unknown ablations: {unknown}. Known: {sorted(ABLATIONS)}')
    return {name: ABLATIONS[name] for name in names}


def _load_test_row(output_dir: Path) -> dict:
    row = {}
    metrics_path = output_dir / 'test_overall.json'
    per_type_path = output_dir / 'test_per_type.csv'
    run_summary_path = output_dir / 'run_summary.json'
    if metrics_path.exists():
        row.update(load_json(metrics_path))
    if run_summary_path.exists():
        summary = load_json(run_summary_path)
        row['best_epoch'] = summary.get('best_epoch', -1)
        row['best_val_metric_name'] = summary.get('best_val_metric_name')
        row['best_val_metric_value'] = summary.get('best_val_metric_value')
    if per_type_path.exists():
        per_type_df = pd.read_csv(per_type_path, index_col=0)
        for comp_name, comp_row in per_type_df.iterrows():
            for metric_name in ['PLCC', 'SRCC', 'RMSE']:
                if metric_name in comp_row:
                    row[f'{comp_name}_{metric_name}'] = comp_row[metric_name]
    return row


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml_config(args.config)
    work_dir = ensure_dir(args.work_dir)
    records = []

    selected = _selected_ablations(args.ablations, include_baseline=args.include_baseline)
    for name, patch in selected.items():
        cfg = deep_update(base_cfg, patch)
        cfg['experiment_name'] = f"{base_cfg.get('experiment_name', 'exp')}_{name}"
        cfg['output_dir'] = str(work_dir / name)
        if args.epochs is not None:
            cfg['training']['epochs'] = args.epochs
        if args.batch_size is not None:
            cfg['training']['batch_size'] = args.batch_size
        tmp_cfg = work_dir / f'{name}.yaml'
        with tmp_cfg.open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        row = {'ablation': name, 'output_dir': cfg['output_dir']}
        output_dir = Path(cfg['output_dir'])
        should_skip = args.skip_existing and existing_run_is_complete(output_dir)
        if should_skip:
            row['returncode'] = 0
            row['skipped_existing'] = True
        else:
            cmd = [sys.executable, '-m', 'oiqa_bpr_vmamba.cli.train_cviq', '--config', str(tmp_cfg)]
            if args.device is not None:
                cmd += ['--device', args.device]
            if args.resume:
                cmd += ['--resume', 'auto']
            print('Running:', ' '.join(cmd))
            result = run_logged_subprocess(cmd, output_dir / 'train.log')
            row['returncode'] = int(result.returncode)
            row['skipped_existing'] = False
            row['command'] = ' '.join(cmd)

        row.update(_load_test_row(output_dir))
        records.append(row)

    if records:
        df = pd.DataFrame(records)
        baseline_row = None
        if 'baseline' in df['ablation'].values:
            baseline_row = df[df['ablation'] == 'baseline'].iloc[0].to_dict()
            for metric_name in ['PLCC', 'SRCC', 'RMSE']:
                if metric_name in df.columns and metric_name in baseline_row and pd.notna(baseline_row[metric_name]):
                    delta_col = f'delta_{metric_name}_vs_baseline'
                    if metric_name == 'RMSE':
                        df[delta_col] = baseline_row[metric_name] - df[metric_name]
                    else:
                        df[delta_col] = df[metric_name] - baseline_row[metric_name]

        df.to_csv(work_dir / 'ablation_summary.csv', index=False)
        best_row = None
        valid_df = df[df['returncode'] == 0].copy()
        if 'PLCC' in valid_df.columns and not valid_df.empty:
            best_row = valid_df.sort_values('PLCC', ascending=False).iloc[0].to_dict()
        save_json(
            {
                'best_by_plcc': best_row,
                'baseline': baseline_row,
                'num_runs': len(records),
                'ablations_run': list(df['ablation']),
            },
            work_dir / 'ablation_meta.json',
        )
        print(df)


if __name__ == '__main__':
    main()
