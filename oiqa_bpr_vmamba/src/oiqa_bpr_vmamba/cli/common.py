from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence
import subprocess

import pandas as pd
import torch
from torch.utils.data import DataLoader

from oiqa_bpr_vmamba.data.cviq_dataset import CVIQDataset
from oiqa_bpr_vmamba.models.network import OIQABPRVMamba
from oiqa_bpr_vmamba.training.losses import MultiTaskLoss
from oiqa_bpr_vmamba.utils.io import ensure_dir, load_json, save_json, save_yaml
from oiqa_bpr_vmamba.utils.reporting import evaluation_summary_dataframe, flatten_columns, write_table_bundle
from oiqa_bpr_vmamba.utils.splits import create_or_load_splits


SPLIT_TO_INDEX = {'train': 0, 'val': 1, 'test': 2}


def resolve_device(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def resolve_checkpoint_path(cfg: dict[str, Any], checkpoint_arg: str) -> Path:
    checkpoint_arg = str(checkpoint_arg)
    output_dir = Path(cfg['output_dir'])
    if checkpoint_arg == 'best':
        return output_dir / 'best.pt'
    if checkpoint_arg == 'last':
        return output_dir / 'last.pt'
    if checkpoint_arg == 'auto':
        best_path = output_dir / 'best.pt'
        if best_path.exists():
            return best_path
        return output_dir / 'last.pt'
    return Path(checkpoint_arg)

def save_resolved_config(cfg: dict[str, Any], output_dir: str | Path, filename: str = 'resolved_config.yaml') -> Path:
    output_dir = ensure_dir(output_dir)
    out_path = output_dir / filename
    save_yaml(cfg, out_path)
    return out_path


def _dataset_kwargs_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(
        manifest_csv=cfg['paths']['manifest_csv'],
        image_size=tuple(cfg['model']['image_size']),
        viewport_size=tuple(cfg['model']['viewport_size']),
        num_viewports=int(cfg['model']['num_viewports']),
        online_degradation_cfg=cfg.get('degradation', {}),
        use_precomputed_degraded=bool(cfg.get('data', {}).get('use_precomputed_degraded', True)),
    )


def build_dataset(
    cfg: dict[str, Any],
    split_csv: str | Path | None,
    compression_types: Iterable[str] | None = None,
) -> CVIQDataset:
    kwargs = _dataset_kwargs_from_cfg(cfg)
    return CVIQDataset(split_csv=split_csv, allowed_compression_types=compression_types, **kwargs)


def build_dataloaders(
    cfg: dict[str, Any],
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = build_dataset(cfg, train_csv)
    val_ds = build_dataset(cfg, val_csv)
    test_ds = build_dataset(cfg, test_csv)

    loader_kwargs = dict(
        batch_size=int(cfg['training']['batch_size']),
        num_workers=int(cfg['training']['num_workers']),
        pin_memory=True,
        persistent_workers=bool(int(cfg['training']['num_workers']) > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def build_eval_loader(
    cfg: dict[str, Any],
    split_csv: str | Path | None,
    compression_types: Iterable[str] | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    dataset = build_dataset(cfg, split_csv=split_csv, compression_types=compression_types)
    workers = int(cfg['training']['num_workers'] if num_workers is None else num_workers)
    return DataLoader(
        dataset,
        batch_size=int(batch_size or cfg['training']['batch_size']),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=bool(workers > 0),
    )


def build_model_criterion_optimizer_scheduler(
    cfg: dict[str, Any],
    device: torch.device,
    trainable: bool = True,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer | None, torch.optim.lr_scheduler._LRScheduler | None]:
    model = OIQABPRVMamba(cfg['model'])
    model.to(device)

    criterion = MultiTaskLoss(
        mos_weight=float(cfg['loss']['mos_weight']),
        distortion_weight=float(cfg['loss']['distortion_weight']),
        compression_weight=float(cfg['loss']['compression_weight']),
    )
    criterion.to(device)

    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    if trainable:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg['training']['lr']),
            momentum=float(cfg['training']['momentum']),
            weight_decay=float(cfg['training']['weight_decay']),
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(cfg['training']['scheduler_gamma']),
        )
    return model, criterion, optimizer, scheduler


def get_split_csv(cfg: dict[str, Any], split_name: str) -> Path:
    split_paths = create_or_load_splits(cfg)
    return Path(split_paths[SPLIT_TO_INDEX[split_name]])


def save_eval_outputs(
    metrics: dict[str, Any],
    output_dir: str | Path,
    prefix: str,
) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    metrics['predictions'].to_csv(output_dir / f'{prefix}_predictions.csv', index=False)
    per_type_df = pd.DataFrame(metrics['per_type']).T
    per_type_df.to_csv(output_dir / f'{prefix}_per_type.csv')
    save_json(metrics['overall'], output_dir / f'{prefix}_overall.json')
    save_json(metrics['losses'], output_dir / f'{prefix}_losses.json')
    save_json(metrics.get('aux_metrics', {}), output_dir / f'{prefix}_aux_metrics.json')
    summary_df = evaluation_summary_dataframe(metrics)
    table_paths = write_table_bundle(summary_df, output_dir / f'{prefix}_summary', index=False)
    return table_paths




def write_grouped_metrics_table(records: list[dict[str, Any]], output_prefix: str | Path) -> dict[str, str]:
    if not records:
        df = pd.DataFrame(columns=['section', 'name'])
    else:
        df = pd.DataFrame(records)
        df = flatten_columns(df)
    return write_table_bundle(df, output_prefix, index=False)

def save_run_summary(summary: dict[str, Any], output_dir: str | Path, filename: str = 'run_summary.json') -> None:
    save_json(summary, ensure_dir(output_dir) / filename)


def run_logged_subprocess(cmd: Sequence[str], log_path: str | Path) -> subprocess.CompletedProcess[str]:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(list(cmd), check=False, text=True, capture_output=True)
    with log_path.open('w', encoding='utf-8') as f:
        f.write('$ ' + ' '.join(cmd) + '\n\n')
        if result.stdout:
            f.write('[stdout]\n')
            f.write(result.stdout)
            if not result.stdout.endswith('\n'):
                f.write('\n')
        if result.stderr:
            f.write('\n[stderr]\n')
            f.write(result.stderr)
            if not result.stderr.endswith('\n'):
                f.write('\n')
        f.write(f'\n[returncode]\n{result.returncode}\n')
    return result


def existing_run_is_complete(output_dir: str | Path, require_test_metrics: bool = True) -> bool:
    output_dir = Path(output_dir)
    summary_path = output_dir / 'run_summary.json'
    if not summary_path.exists():
        return False
    try:
        summary = load_json(summary_path)
    except Exception:
        return False
    if require_test_metrics:
        return (output_dir / 'test_overall.json').exists() or 'test' in summary
    return True
