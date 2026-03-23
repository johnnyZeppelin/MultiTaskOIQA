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


DEFAULT_PROTOCOLS = [
    {'name': '50_50', 'train_ratio': 0.5, 'val_ratio': 0.0, 'test_ratio': 0.5},
    {'name': '60_40', 'train_ratio': 0.6, 'val_ratio': 0.0, 'test_ratio': 0.4},
    {'name': '70_30', 'train_ratio': 0.7, 'val_ratio': 0.0, 'test_ratio': 0.3},
    {'name': '80_20', 'train_ratio': 0.8, 'val_ratio': 0.0, 'test_ratio': 0.2},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--work-dir', type=str, default='runs/split_protocols')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed-start', type=int, default=3407)
    parser.add_argument('--protocols', type=str, default='all', help='Comma-separated protocol names or all')
    parser.add_argument('--val-from-train-ratio', type=float, default=0.1, help='When protocol val_ratio is 0, carve this fraction from the train set for validation.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--skip-existing', action='store_true')
    return parser.parse_args()


def _selected_protocols(spec: str) -> list[dict]:
    if spec == 'all':
        return DEFAULT_PROTOCOLS
    wanted = {x.strip() for x in spec.split(',') if x.strip()}
    out = [proto for proto in DEFAULT_PROTOCOLS if proto['name'] in wanted]
    missing = sorted(wanted - {proto['name'] for proto in out})
    if missing:
        raise ValueError(f'Unknown protocols: {missing}. Known: {[p["name"] for p in DEFAULT_PROTOCOLS]}')
    return out


def _materialize_ratios(protocol: dict, val_from_train_ratio: float) -> tuple[float, float, float]:
    if protocol['val_ratio'] > 0:
        return protocol['train_ratio'], protocol['val_ratio'], protocol['test_ratio']
    train_ratio = protocol['train_ratio'] * (1.0 - val_from_train_ratio)
    val_ratio = protocol['train_ratio'] * val_from_train_ratio
    test_ratio = protocol['test_ratio']
    return train_ratio, val_ratio, test_ratio


def _load_repeat_row(output_dir: Path) -> dict:
    row = {}
    metrics_path = output_dir / 'test_overall.json'
    summary_path = output_dir / 'run_summary.json'
    if metrics_path.exists():
        row.update(load_json(metrics_path))
    if summary_path.exists():
        summary = load_json(summary_path)
        row['best_epoch'] = summary.get('best_epoch', -1)
        row['best_val_metric_name'] = summary.get('best_val_metric_name')
        row['best_val_metric_value'] = summary.get('best_val_metric_value')
    return row


def _aggregate_per_type(proto_dir: Path, per_repeat_rows: list[dict]) -> None:
    frames = []
    for row in per_repeat_rows:
        repeat = int(row['repeat'])
        per_type_path = proto_dir / f'repeat_{repeat}' / 'test_per_type.csv'
        if not per_type_path.exists():
            continue
        df = pd.read_csv(per_type_path, index_col=0).reset_index().rename(columns={'index': 'compression_type'})
        df['repeat'] = repeat
        frames.append(df)
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(proto_dir / 'per_type_per_repeat.csv', index=False)
    metric_cols = [c for c in ['PLCC', 'SRCC', 'RMSE'] if c in merged.columns]
    if metric_cols:
        summary = merged.groupby('compression_type')[metric_cols].agg(['mean', 'std'])
        summary.to_csv(proto_dir / 'per_type_average.csv')


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml_config(args.config)
    work_dir = ensure_dir(args.work_dir)
    all_rows = []

    for protocol in _selected_protocols(args.protocols):
        proto_dir = ensure_dir(work_dir / protocol['name'])
        per_repeat = []
        train_ratio, val_ratio, test_ratio = _materialize_ratios(protocol, args.val_from_train_ratio)
        for repeat_idx in range(args.repeats):
            seed = int(args.seed_start) + repeat_idx
            cfg = copy.deepcopy(base_cfg)
            cfg['seed'] = seed
            cfg['split']['train_ratio'] = train_ratio
            cfg['split']['val_ratio'] = val_ratio
            cfg['split']['test_ratio'] = test_ratio
            cfg['split']['split_seed'] = seed
            cfg['output_dir'] = str(proto_dir / f'repeat_{repeat_idx + 1}')
            cfg['experiment_name'] = f"{base_cfg.get('experiment_name', 'exp')}_{protocol['name']}_repeat{repeat_idx + 1}"
            if args.epochs is not None:
                cfg['training']['epochs'] = args.epochs
            if args.batch_size is not None:
                cfg['training']['batch_size'] = args.batch_size
            tmp_cfg = proto_dir / f'{protocol["name"]}_repeat_{repeat_idx + 1}.yaml'
            with tmp_cfg.open('w', encoding='utf-8') as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            output_dir = Path(cfg['output_dir'])
            row = {
                'protocol': protocol['name'],
                'repeat': repeat_idx + 1,
                'seed': seed,
                'effective_train_ratio': train_ratio,
                'effective_val_ratio': val_ratio,
                'effective_test_ratio': test_ratio,
            }
            should_skip = args.skip_existing and existing_run_is_complete(output_dir)
            if should_skip:
                row['returncode'] = 0
                row['skipped_existing'] = True
            else:
                cmd = [sys.executable, '-m', 'oiqa_bpr_vmamba.cli.train_cviq', '--config', str(tmp_cfg)]
                if args.device is not None:
                    cmd += ['--device', args.device]
                print('Running:', ' '.join(cmd))
                result = run_logged_subprocess(cmd, output_dir / 'train.log')
                row['returncode'] = int(result.returncode)
                row['skipped_existing'] = False
                row['command'] = ' '.join(cmd)

            row.update(_load_repeat_row(output_dir))
            if row['returncode'] == 0 and 'PLCC' in row:
                per_repeat.append(row.copy())
            all_rows.append(row)

        if per_repeat:
            df = pd.DataFrame(per_repeat)
            df.to_csv(proto_dir / 'per_repeat.csv', index=False)
            metric_cols = [c for c in ['PLCC', 'SRCC', 'RMSE'] if c in df.columns]
            avg = {'protocol': protocol['name'], 'num_repeats': len(df)}
            if metric_cols:
                avg.update({f'{metric}_mean': float(df[metric].mean()) for metric in metric_cols})
                avg.update({f'{metric}_std': float(df[metric].std(ddof=0)) for metric in metric_cols})
            save_json(avg, proto_dir / 'average.json')
            _aggregate_per_type(proto_dir, per_repeat)

    if all_rows:
        all_df = pd.DataFrame(all_rows)
        all_df.to_csv(work_dir / 'all_protocols_per_repeat.csv', index=False)
        valid = all_df[(all_df['returncode'] == 0) & all_df['PLCC'].notna()].copy() if 'PLCC' in all_df.columns else pd.DataFrame()
        if not valid.empty and {'PLCC', 'SRCC', 'RMSE'}.issubset(valid.columns):
            summary = valid.groupby('protocol')[['PLCC', 'SRCC', 'RMSE']].agg(['mean', 'std'])
            summary.to_csv(work_dir / 'all_protocols_average.csv')
            save_json(
                {
                    'protocols': list(valid['protocol'].unique()),
                    'repeats': args.repeats,
                    'seed_start': args.seed_start,
                },
                work_dir / 'split_protocols_meta.json',
            )
            print(summary)


if __name__ == '__main__':
    main()
