from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from oiqa_bpr_vmamba.cli.common import write_grouped_metrics_table
from oiqa_bpr_vmamba.utils.io import ensure_dir, load_json, save_json
from oiqa_bpr_vmamba.utils.reporting import flatten_columns, write_table_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--work-dir', type=str, default='runs/full_benchmark')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--split-repeats', type=int, default=5)
    parser.add_argument('--skip-main', action='store_true')
    parser.add_argument('--skip-ablation', action='store_true')
    parser.add_argument('--skip-split-protocols', action='store_true')
    return parser.parse_args()


def _append_common_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    if args.epochs is not None:
        cmd += ['--epochs', str(args.epochs)]
    if args.batch_size is not None:
        cmd += ['--batch-size', str(args.batch_size)]
    if args.device is not None:
        cmd += ['--device', args.device]
    return cmd


def _read_json_if_exists(path: Path) -> dict | None:
    return load_json(path) if path.exists() else None


def _read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame | None:
    return pd.read_csv(path, **kwargs) if path.exists() else None


def _build_main_tables(main_dir: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    summary = _read_json_if_exists(main_dir / 'run_summary.json')
    if summary is None:
        return out
    main_record = {
        'section': 'main',
        'name': 'main',
        'checkpoint_used_for_final_eval': summary.get('checkpoint_used_for_final_eval'),
        'best_epoch': summary.get('best_epoch'),
        'best_val_metric_name': summary.get('best_val_metric_name'),
        'best_val_metric_value': summary.get('best_val_metric_value'),
    }
    for metric_name, metric_value in (summary.get('test') or {}).items():
        main_record[metric_name] = metric_value
    for metric_name, metric_value in (summary.get('val_best') or {}).items():
        main_record[f'val_best_{metric_name}'] = metric_value
    main_table_paths = write_grouped_metrics_table([main_record], main_dir / 'main_result_table')
    out['main_record'] = main_record
    out['main_table_paths'] = main_table_paths

    per_type_path = main_dir / 'test_per_type.csv'
    per_type_df = _read_csv_if_exists(per_type_path, index_col=0)
    if per_type_df is not None:
        per_type_df = per_type_df.reset_index().rename(columns={'index': 'compression_type'})
        per_type_df.insert(0, 'section', 'main_per_type')
        out['main_per_type_table_paths'] = write_table_bundle(per_type_df, main_dir / 'main_per_type_table', index=False)
    return out


def _build_ablation_tables(ablation_dir: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    df = _read_csv_if_exists(ablation_dir / 'ablation_summary.csv')
    if df is None or df.empty:
        return out
    preferred = [
        'ablation', 'PLCC', 'SRCC', 'RMSE',
        'delta_PLCC_vs_baseline', 'delta_SRCC_vs_baseline', 'delta_RMSE_vs_baseline',
        'best_epoch', 'best_val_metric_name', 'best_val_metric_value', 'returncode',
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    concise = df[cols].copy()
    concise.insert(0, 'section', 'ablation')
    out['ablation_table_paths'] = write_table_bundle(concise, ablation_dir / 'ablation_table', index=False)
    out['ablation_best_by_plcc'] = concise.sort_values('PLCC', ascending=False).iloc[0].to_dict() if 'PLCC' in concise.columns else None
    return out


def _build_split_tables(splits_dir: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    avg = _read_csv_if_exists(splits_dir / 'all_protocols_average.csv', header=[0, 1], index_col=0)
    if avg is not None and not avg.empty:
        avg = flatten_columns(avg.reset_index().rename(columns={'index': 'protocol'}))
        avg.insert(0, 'section', 'split_protocol')
        out['split_protocol_table_paths'] = write_table_bundle(avg, splits_dir / 'split_protocol_table', index=False)
    per_protocol_frames = []
    for proto_dir in sorted([p for p in splits_dir.iterdir() if p.is_dir()]):
        per_type_path = proto_dir / 'per_type_average.csv'
        if not per_type_path.exists():
            continue
        df = pd.read_csv(per_type_path, header=[0, 1], index_col=0)
        df = flatten_columns(df.reset_index().rename(columns={'index': 'compression_type'}))
        df.insert(0, 'protocol', proto_dir.name)
        per_protocol_frames.append(df)
    if per_protocol_frames:
        merged = pd.concat(per_protocol_frames, ignore_index=True)
        merged.insert(0, 'section', 'split_protocol_per_type')
        out['split_protocol_per_type_table_paths'] = write_table_bundle(merged, splits_dir / 'split_protocol_per_type_table', index=False)
    return out


def _combine_benchmark_tables(work_dir: Path, collected: dict[str, object]) -> dict[str, str] | None:
    records: list[dict] = []
    main_record = collected.get('main_record')
    if isinstance(main_record, dict):
        records.append(main_record)
    ablation_csv = work_dir / 'ablations' / 'ablation_summary.csv'
    if ablation_csv.exists():
        ablation_df = pd.read_csv(ablation_csv)
        for _, row in ablation_df.iterrows():
            record = {'section': 'ablation', 'name': row.get('ablation')}
            for key in ['PLCC', 'SRCC', 'RMSE', 'delta_PLCC_vs_baseline', 'delta_SRCC_vs_baseline', 'delta_RMSE_vs_baseline']:
                if key in ablation_df.columns:
                    record[key] = row.get(key)
            records.append(record)
    split_avg = work_dir / 'split_protocols' / 'all_protocols_average.csv'
    if split_avg.exists():
        df = pd.read_csv(split_avg, header=[0, 1], index_col=0)
        df = flatten_columns(df.reset_index().rename(columns={'index': 'name'}))
        for _, row in df.iterrows():
            record = {'section': 'split_protocol', 'name': row.get('name')}
            for key in row.index:
                if key != 'name':
                    record[key] = row[key]
            records.append(record)
    if not records:
        return None
    return write_grouped_metrics_table(records, work_dir / 'benchmark_table')


def main() -> None:
    args = parse_args()
    work_dir = ensure_dir(args.work_dir)
    summary: dict[str, object] = {'work_dir': str(work_dir), 'config': args.config}

    if not args.skip_main:
        main_dir = work_dir / 'main'
        cmd = [sys.executable, '-m', 'oiqa_bpr_vmamba.cli.train_cviq', '--config', args.config, '--override-output-dir', str(main_dir)]
        cmd = _append_common_args(cmd, args)
        print('Running:', ' '.join(cmd))
        result = subprocess.run(cmd, check=False)
        summary['main_returncode'] = int(result.returncode)
        if (main_dir / 'run_summary.json').exists():
            summary['main'] = load_json(main_dir / 'run_summary.json')

    if not args.skip_ablation:
        ablation_dir = work_dir / 'ablations'
        cmd = [sys.executable, '-m', 'oiqa_bpr_vmamba.cli.run_ablation', '--config', args.config, '--work-dir', str(ablation_dir)]
        cmd = _append_common_args(cmd, args)
        print('Running:', ' '.join(cmd))
        result = subprocess.run(cmd, check=False)
        summary['ablation_returncode'] = int(result.returncode)
        if (ablation_dir / 'ablation_summary.csv').exists():
            summary['ablation_summary_csv'] = str(ablation_dir / 'ablation_summary.csv')

    if not args.skip_split_protocols:
        splits_dir = work_dir / 'split_protocols'
        cmd = [
            sys.executable,
            '-m',
            'oiqa_bpr_vmamba.cli.run_split_protocols',
            '--config', args.config,
            '--work-dir', str(splits_dir),
            '--repeats', str(args.split_repeats),
        ]
        cmd = _append_common_args(cmd, args)
        print('Running:', ' '.join(cmd))
        result = subprocess.run(cmd, check=False)
        summary['split_protocols_returncode'] = int(result.returncode)
        if (splits_dir / 'all_protocols_average.csv').exists():
            summary['split_protocols_average_csv'] = str(splits_dir / 'all_protocols_average.csv')

    collected: dict[str, object] = {}
    collected.update(_build_main_tables(work_dir / 'main'))
    collected.update(_build_ablation_tables(work_dir / 'ablations'))
    collected.update(_build_split_tables(work_dir / 'split_protocols'))
    benchmark_paths = _combine_benchmark_tables(work_dir, collected)
    if benchmark_paths is not None:
        summary['benchmark_table_paths'] = benchmark_paths
    if collected:
        summary['derived_tables'] = collected
    save_json(summary, work_dir / 'benchmark_summary.json')
    print(summary)


if __name__ == '__main__':
    main()
