from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from oiqa_bpr_vmamba.cli.common import (
    build_dataloaders,
    build_model_criterion_optimizer_scheduler,
    resolve_device,
    save_eval_outputs,
    save_resolved_config,
    save_run_summary,
)
from oiqa_bpr_vmamba.training.trainer import Trainer
from oiqa_bpr_vmamba.utils.config import load_yaml_config
from oiqa_bpr_vmamba.utils.io import ensure_dir, load_checkpoint
from oiqa_bpr_vmamba.utils.seed import seed_everything
from oiqa_bpr_vmamba.utils.splits import create_or_load_splits


VALID_BEST_METRICS = {'PLCC', 'SRCC', 'RMSE'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--override-output-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint, or "auto" for output_dir/last.pt if it exists.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--accumulation-steps', type=int, default=None)
    parser.add_argument('--best-metric', type=str, default=None, choices=sorted(VALID_BEST_METRICS))
    parser.add_argument('--early-stopping-patience', type=int, default=None)
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-eval-batches', type=int, default=None)
    parser.add_argument('--skip-test', action='store_true')
    return parser.parse_args()


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.override_output_dir is not None:
        cfg['output_dir'] = args.override_output_dir
    if args.epochs is not None:
        cfg['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['training']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        cfg['training']['num_workers'] = args.num_workers
    if args.accumulation_steps is not None:
        cfg['training']['accumulation_steps'] = args.accumulation_steps
    if args.best_metric is not None:
        cfg['training']['best_metric'] = args.best_metric
    if args.early_stopping_patience is not None:
        cfg['training']['early_stopping_patience'] = args.early_stopping_patience
    if args.max_train_batches is not None:
        cfg['training']['max_train_batches'] = args.max_train_batches
    if args.max_eval_batches is not None:
        cfg['training']['max_eval_batches'] = args.max_eval_batches
    return cfg


def _resolve_resume_path(output_dir: Path, resume_arg: str | None) -> Path | None:
    if resume_arg is None:
        return None
    if resume_arg == 'auto':
        candidate = output_dir / 'last.pt'
        return candidate if candidate.exists() else None
    path = Path(resume_arg)
    return path if path.exists() else None


def _best_metric_cfg(cfg: dict) -> tuple[str, bool]:
    metric_name = str(cfg['training'].get('best_metric', 'PLCC')).upper()
    if metric_name not in VALID_BEST_METRICS:
        raise ValueError(f'Unsupported best_metric={metric_name}. Expected one of {sorted(VALID_BEST_METRICS)}')
    maximize = metric_name != 'RMSE'
    return metric_name, maximize


def main() -> None:
    args = parse_args()
    cfg = _apply_cli_overrides(load_yaml_config(args.config), args)
    seed_everything(int(cfg['seed']))
    output_dir = ensure_dir(cfg['output_dir'])
    save_resolved_config(cfg, output_dir)

    train_csv, val_csv, test_csv = create_or_load_splits(cfg)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, train_csv, val_csv, test_csv)

    device = resolve_device(args.device)
    model, criterion, optimizer, scheduler = build_model_criterion_optimizer_scheduler(cfg, device, trainable=True)

    best_metric_name, maximize_best_metric = _best_metric_cfg(cfg)
    start_epoch = 1
    history: list[dict] = []
    resume_path = _resolve_resume_path(output_dir, args.resume)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        amp=bool(cfg['training']['amp']),
        clip_grad_norm=float(cfg['training']['clip_grad_norm']) if cfg['training'].get('clip_grad_norm') is not None else None,
        fit_nonlinear_mapping=bool(cfg['evaluation']['fit_nonlinear_mapping']),
        accumulation_steps=int(cfg['training'].get('accumulation_steps', 1)),
        max_train_batches=cfg['training'].get('max_train_batches'),
        max_eval_batches=cfg['training'].get('max_eval_batches'),
        best_metric_name=best_metric_name,
        maximize_best_metric=maximize_best_metric,
        early_stopping_patience=cfg['training'].get('early_stopping_patience'),
    )

    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        if optimizer is not None and ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler is not None and ckpt.get('scheduler') is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        trainer.best_val_plcc = float(ckpt.get('best_val_plcc', float('-inf')))
        trainer.best_val_metric = float(ckpt.get('best_val_metric', float('-inf' if maximize_best_metric else 'inf')))
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        history_path = output_dir / 'history.csv'
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict(orient='records')
        print(f'Resumed from {resume_path} at epoch {start_epoch}.')

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(cfg['training']['epochs']),
        save_every=int(cfg['training']['save_every']),
        start_epoch=start_epoch,
        history=history,
    )

    best_path = output_dir / 'best.pt'
    if not best_path.exists():
        best_path = output_dir / 'last.pt'
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best['model'])

    val_metrics = trainer.evaluate(val_loader, split_name='val_best')
    save_eval_outputs(val_metrics, output_dir, prefix='val_best')

    summary = {
        'checkpoint_used_for_final_eval': str(best_path),
        'best_epoch': int(best.get('epoch', -1)),
        'best_val_plcc': float(best.get('best_val_plcc', float('-inf'))),
        'best_val_metric_name': best_metric_name,
        'best_val_metric_value': float(best.get('best_val_metric', float('-inf' if maximize_best_metric else 'inf'))),
        'num_history_rows': len(history),
        'train_split_csv': str(train_csv),
        'val_split_csv': str(val_csv),
        'test_split_csv': str(test_csv),
        'val_best': val_metrics['overall'],
        'val_best_per_type': val_metrics['per_type'],
        'accumulation_steps': int(cfg['training'].get('accumulation_steps', 1)),
        'max_train_batches': cfg['training'].get('max_train_batches'),
        'max_eval_batches': cfg['training'].get('max_eval_batches'),
    }

    if not args.skip_test:
        test_metrics = trainer.evaluate(test_loader, split_name='test')
        save_eval_outputs(test_metrics, output_dir, prefix='test')
        summary['test'] = test_metrics['overall']
        summary['test_per_type'] = test_metrics['per_type']
        print('Test overall:', test_metrics['overall'])
        print('Test per type:', test_metrics['per_type'])

    save_run_summary(summary, output_dir)
    print('Final run summary:', summary)


if __name__ == '__main__':
    main()
