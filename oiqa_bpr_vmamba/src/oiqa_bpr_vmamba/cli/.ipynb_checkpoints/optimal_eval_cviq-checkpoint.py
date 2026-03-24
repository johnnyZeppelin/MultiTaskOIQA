from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from oiqa_bpr_vmamba.cli.common import get_split_csv, save_eval_outputs, write_grouped_metrics_table
from oiqa_bpr_vmamba.utils.config import load_yaml_config
from oiqa_bpr_vmamba.utils.io import ensure_dir, save_json
from oiqa_bpr_vmamba.utils.opt_eval import (
    build_optimal_metrics_payload,
    ensure_optimal_checkpoint,
    load_optimal_checkpoint,
    save_optimal_notice,
)

VALID_COMPRESSION_TYPES = ['JPEG', 'AVC', 'HEVC', 'ref']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Teaching-only optimal evaluation for the CVIQ pipeline.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a optimal checkpoint. If omitted, a demo optimal checkpoint is auto-created.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--split-csv', type=str, default=None)
    parser.add_argument('--compression-type', type=str, default='all', help='One of all/JPEG/AVC/HEVC/ref')
    parser.add_argument('--evaluate-all-types', action='store_true')
    parser.add_argument('--save-name', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None, help='Optional override for output location.')
    parser.add_argument('--batch-size', type=int, default=None, help='Display-only batch size used for the optimal progress bar.')
    parser.add_argument('--show-steps', action='store_true', help='Print a teaching-oriented walkthrough of the evaluation stages.')
    return parser.parse_args()


def _print_step(title: str, body: str) -> None:
    print(f'\n[optimal STEP] {title}')
    print(body)


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    split_csv = Path(args.split_csv) if args.split_csv is not None else get_split_csv(cfg, args.split)
    split_label = split_csv.stem if args.split_csv is not None else args.split
    manifest_csv = Path(cfg['paths']['manifest_csv'])
    checkpoint_path = ensure_optimal_checkpoint(args.checkpoint)
    checkpoint = load_optimal_checkpoint(checkpoint_path)

    manifest_df = pd.read_csv(manifest_csv)
    split_df = pd.read_csv(split_csv)
    split_ids = set(split_df['image_id'].astype(str).tolist())
    subset_df = manifest_df[manifest_df['image_id'].astype(str).isin(split_ids)].reset_index(drop=True)
    if args.compression_type != 'all' and not args.evaluate_all_types:
        subset_df = subset_df[subset_df['compression_type'].astype(str) == args.compression_type].reset_index(drop=True)

    if subset_df.empty:
        raise ValueError(f'No samples found for split={split_label} and compression_type={args.compression_type}.')

    output_dir = ensure_dir(args.output_dir or (Path(cfg['output_dir']) / 'eval'))

    if args.show_steps:
        _print_step('Load config', f'Loaded config: {args.config}')
        _print_step('Resolve dataset', f'Manifest: {manifest_csv}\nSplit CSV: {split_csv}\nSamples in split: {len(subset_df)}')
        counts = subset_df['compression_type'].value_counts().to_dict()
        _print_step('Inspect split composition', f'Compression counts: {counts}')
        _print_step('Load optimal checkpoint', f'Checkpoint: {checkpoint_path}\nNotice: {checkpoint["notice"]}')
        _print_step('Simulate dataloader', 'No real images or model weights are used here. This is a teaching walkthrough only.')

    display_batch = int(args.batch_size or cfg['training']['batch_size'])
    num_batches = max(1, (len(subset_df) + display_batch - 1) // display_batch)
    for _ in tqdm(range(num_batches), desc='optimal eval batches', leave=False):
        pass

    available_types = [k for k in checkpoint['paper_results'].keys() if k != 'overall']
    if args.evaluate_all_types:
        evaluation_order = ['all', *available_types]
    else:
        if args.compression_type != 'all' and args.compression_type not in available_types:
            raise ValueError(f'Compression type {args.compression_type!r} is not available in this optimal checkpoint. Available: {available_types}')
        evaluation_order = [args.compression_type]

    records = []
    details: dict[str, dict] = {}
    for compression_type in evaluation_order:
        default_prefix = f'eval_{split_label}' if compression_type == 'all' else f'eval_{split_label}_{compression_type}'
        prefix = args.save_name if len(evaluation_order) == 1 else default_prefix
        metrics = build_optimal_metrics_payload(subset_df, checkpoint, compression_type=compression_type)
        table_paths = save_eval_outputs(metrics, output_dir, prefix=prefix)
        notice_paths = save_optimal_notice(output_dir, prefix, checkpoint)
        overall = dict(metrics['overall'])
        overall.update({
            'split': split_label,
            'checkpoint': str(checkpoint_path),
            'compression_type': compression_type,
            'save_name': prefix,
            'is_optimal': True,
        })
        records.append(overall)
        details[compression_type] = {
            'overall': metrics['overall'],
            'per_type': metrics['per_type'],
            'summary_tables': table_paths,
            'optimal_notice': notice_paths,
        }
        print(f'\n[optimal RESULT] {compression_type}')
        print('  This is a teaching-only optimal result copied from the paper, not from trained weights.')
        for key, value in metrics['overall'].items():
            print(f'  {key}: {value}')

    multi_prefix = args.save_name or f'eval_{split_label}'
    table_paths = write_grouped_metrics_table(records, output_dir / f'{multi_prefix}_multi_eval_summary')
    save_json(
        {
            'split': split_label,
            'split_csv': str(split_csv),
            'checkpoint': str(checkpoint_path),
            'evaluated_compression_types': evaluation_order,
            'records': records,
            'table_paths': table_paths,
            'details': details,
            'is_optimal': True,
            'notice': checkpoint['notice'],
        },
        output_dir / f'{multi_prefix}_multi_eval_summary.json',
    )

    print('\n[optimal DONE]')
    print(f'Outputs written to: {output_dir}')
    print('These files demonstrate the evaluation flow only. They do not represent a trained model run.')


if __name__ == '__main__':
    main()
