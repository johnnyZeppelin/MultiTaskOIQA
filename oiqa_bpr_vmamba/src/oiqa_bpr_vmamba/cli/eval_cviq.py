from __future__ import annotations

import argparse
from pathlib import Path

from oiqa_bpr_vmamba.cli.common import (
    build_eval_loader,
    build_model_criterion_optimizer_scheduler,
    get_split_csv,
    resolve_checkpoint_path,
    resolve_device,
    save_eval_outputs,
    write_grouped_metrics_table,
)
from oiqa_bpr_vmamba.training.trainer import Trainer
from oiqa_bpr_vmamba.utils.config import load_yaml_config
from oiqa_bpr_vmamba.utils.io import ensure_dir, load_checkpoint, save_json

VALID_COMPRESSION_TYPES = ['JPEG', 'AVC', 'HEVC', 'ref']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='best / last / auto / explicit path')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--split-csv', type=str, default=None, help='Optional explicit split CSV. Overrides --split.')
    parser.add_argument('--compression-type', type=str, default='all', help='One of all/JPEG/AVC/HEVC/ref')
    parser.add_argument('--evaluate-all-types', action='store_true', help='Evaluate overall plus each compression type separately.')
    parser.add_argument('--save-name', type=str, default=None, help='Prefix for output files. Defaults to eval_<split> or eval_<split>_<type>.')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def _evaluate_single(
    cfg: dict,
    trainer: Trainer,
    split_csv: Path,
    split_name: str,
    compression_type: str,
    batch_size: int | None,
    num_workers: int | None,
    output_dir: Path,
    save_name: str,
) -> dict:
    compression_types = None if compression_type == 'all' else [compression_type]
    loader = build_eval_loader(
        cfg,
        split_csv=split_csv,
        compression_types=compression_types,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    metrics = trainer.evaluate(loader, split_name=split_name)
    table_paths = save_eval_outputs(metrics, output_dir, prefix=save_name)
    return {
        'compression_type': compression_type,
        'overall': metrics['overall'],
        'per_type': metrics['per_type'],
        'aux_metrics': metrics.get('aux_metrics', {}),
        'summary_tables': table_paths,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    split_csv = Path(args.split_csv) if args.split_csv is not None else get_split_csv(cfg, args.split)
    device = resolve_device(args.device)
    model, criterion, _, _ = build_model_criterion_optimizer_scheduler(cfg, device, trainable=False)
    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint)
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)

    output_dir = ensure_dir(Path(checkpoint_path).resolve().parent)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=None,
        scheduler=None,
        device=device,
        output_dir=output_dir,
        amp=False,
        clip_grad_norm=None,
        fit_nonlinear_mapping=bool(cfg['evaluation']['fit_nonlinear_mapping']),
    )

    if args.evaluate_all_types:
        evaluation_order = ['all', *VALID_COMPRESSION_TYPES]
    else:
        evaluation_order = [args.compression_type]

    records = []
    details: dict[str, dict] = {}
    for compression_type in evaluation_order:
        default_prefix = f'eval_{args.split}' if compression_type == 'all' else f'eval_{args.split}_{compression_type}'
        prefix = args.save_name if len(evaluation_order) == 1 else default_prefix
        result = _evaluate_single(
            cfg=cfg,
            trainer=trainer,
            split_csv=split_csv,
            split_name=args.split,
            compression_type=compression_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=output_dir,
            save_name=prefix,
        )
        overall = dict(result['overall'])
        overall.update(result.get('aux_metrics', {}))
        overall['split'] = args.split
        overall['checkpoint'] = str(checkpoint_path)
        overall['compression_type'] = compression_type
        overall['save_name'] = prefix
        records.append(overall)
        details[compression_type] = result
        print(f'[{compression_type}] overall:', result['overall'])
        if result['per_type']:
            print(f'[{compression_type}] per type:', result['per_type'])

    multi_prefix = args.save_name or f'eval_{args.split}'
    table_paths = write_grouped_metrics_table(records, output_dir / f'{multi_prefix}_multi_eval_summary')
    save_json(
        {
            'split': args.split,
            'split_csv': str(split_csv),
            'checkpoint': str(checkpoint_path),
            'evaluated_compression_types': evaluation_order,
            'records': records,
            'table_paths': table_paths,
        },
        output_dir / f'{multi_prefix}_multi_eval_summary.json',
    )


if __name__ == '__main__':
    main()
