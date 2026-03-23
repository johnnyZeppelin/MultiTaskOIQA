from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from oiqa_bpr_vmamba.training.metrics import compute_metrics
from oiqa_bpr_vmamba.utils.io import ensure_dir, save_checkpoint, save_json


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        device: torch.device,
        output_dir: str | Path,
        amp: bool = True,
        clip_grad_norm: float | None = None,
        fit_nonlinear_mapping: bool = True,
        accumulation_steps: int = 1,
        max_train_batches: int | None = None,
        max_eval_batches: int | None = None,
        best_metric_name: str = 'PLCC',
        maximize_best_metric: bool = True,
        early_stopping_patience: int | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = ensure_dir(output_dir)
        self.amp = amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.clip_grad_norm = clip_grad_norm
        self.fit_nonlinear_mapping = fit_nonlinear_mapping
        self.accumulation_steps = max(1, int(accumulation_steps))
        self.max_train_batches = max_train_batches
        self.max_eval_batches = max_eval_batches
        self.best_metric_name = best_metric_name
        self.maximize_best_metric = maximize_best_metric
        self.early_stopping_patience = early_stopping_patience
        self.best_val_metric = float('-inf') if maximize_best_metric else float('inf')
        self.best_val_plcc = float('-inf')
        self._epochs_without_improvement = 0

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if torch.is_tensor(v) else v
        return out

    def _should_limit_batches(self, step_idx: int, limit: int | None) -> bool:
        return limit is not None and step_idx >= limit

    def _is_improved(self, current: float) -> bool:
        if self.maximize_best_metric:
            return current > self.best_val_metric
        return current < self.best_val_metric

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        if self.optimizer is None:
            raise RuntimeError('Optimizer is required for training.')
        self.model.train()
        running = {'loss_total': 0.0, 'loss_quality': 0.0, 'loss_distortion': 0.0, 'loss_compression': 0.0}
        self.optimizer.zero_grad(set_to_none=True)
        num_batches = 0
        effective_num_batches = min(len(loader), self.max_train_batches) if self.max_train_batches is not None else len(loader)
        pbar = tqdm(loader, desc=f'train {epoch}', leave=False)
        for step_idx, batch in enumerate(pbar):
            if self._should_limit_batches(step_idx, self.max_train_batches):
                break
            batch = self._move_batch(batch)
            with torch.cuda.amp.autocast(enabled=self.amp):
                outputs = self.model(batch)
                loss, logs = self.criterion(outputs, batch)
                scaled_loss = loss / self.accumulation_steps
            self.scaler.scale(scaled_loss).backward()

            should_step = ((step_idx + 1) % self.accumulation_steps == 0) or (step_idx + 1 == effective_num_batches)
            if should_step:
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            for key in running:
                running[key] += logs[key]
            num_batches += 1
            lr = float(self.optimizer.param_groups[0]['lr'])
            pbar.set_postfix(loss=f"{logs['loss_total']:.4f}", lr=f'{lr:.2e}', acc=f'{self.accumulation_steps}')

        if self.scheduler is not None:
            self.scheduler.step()
        denom = max(1, num_batches)
        logs_out = {k: v / denom for k, v in running.items()}
        logs_out['lr'] = float(self.optimizer.param_groups[0]['lr'])
        logs_out['num_batches'] = float(num_batches)
        return logs_out

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str) -> dict[str, Any]:
        self.model.eval()
        all_rows = []
        running = {'loss_total': 0.0, 'loss_quality': 0.0, 'loss_distortion': 0.0, 'loss_compression': 0.0}
        correct_distortion = 0
        correct_compression = 0
        num_samples = 0
        num_batches = 0

        for step_idx, batch in enumerate(tqdm(loader, desc=f'eval {split_name}', leave=False)):
            if self._should_limit_batches(step_idx, self.max_eval_batches):
                break
            batch = self._move_batch(batch)
            outputs = self.model(batch)
            loss, logs = self.criterion(outputs, batch)
            _ = loss
            for key in running:
                running[key] += logs[key]
            pred = outputs['quality'].detach().cpu().numpy().tolist()
            mos = batch['mos'].detach().cpu().numpy().tolist()
            ctype = batch['compression_type'].detach().cpu().numpy().tolist()
            ids = batch['image_id']

            if 'distortion_logits' in outputs:
                distortion_pred = outputs['distortion_logits'].argmax(dim=1)
                correct_distortion += int((distortion_pred == batch['distortion_level']).sum().item())
            if 'compression_logits' in outputs:
                compression_pred = outputs['compression_logits'].argmax(dim=1)
                correct_compression += int((compression_pred == batch['compression_type']).sum().item())

            for image_id, p, m, c in zip(ids, pred, mos, ctype):
                all_rows.append({'image_id': image_id, 'pred': p, 'mos': m, 'compression_type_id': int(c)})
            num_samples += len(ids)
            num_batches += 1

        df = pd.DataFrame(all_rows)
        if len(df) > 1:
            overall = compute_metrics(df['pred'], df['mos'], fit_nonlinear_mapping=self.fit_nonlinear_mapping)
        else:
            overall = {'PLCC': 0.0, 'SRCC': 0.0, 'RMSE': 0.0}
        per_type = {}
        type_map = {0: 'JPEG', 1: 'AVC', 2: 'HEVC', 3: 'ref'}
        for type_id, name in type_map.items():
            sub = df[df['compression_type_id'] == type_id]
            if len(sub) > 1:
                per_type[name] = compute_metrics(sub['pred'], sub['mos'], fit_nonlinear_mapping=self.fit_nonlinear_mapping)

        denom = max(1, num_batches)
        losses = {k: v / denom for k, v in running.items()}
        aux_metrics = {
            'distortion_acc': float(correct_distortion / max(1, num_samples)),
            'compression_acc': float(correct_compression / max(1, num_samples)),
            'num_samples': int(num_samples),
            'num_batches': int(num_batches),
        }
        overall.update(aux_metrics)
        return {
            'overall': overall,
            'per_type': per_type,
            'losses': losses,
            'predictions': df,
            'aux_metrics': aux_metrics,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_every: int = 10,
        start_epoch: int = 1,
        history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        history = [] if history is None else history
        best_state_path = self.output_dir / 'best.pt'
        for epoch in range(start_epoch, epochs + 1):
            train_logs = self.train_one_epoch(train_loader, epoch)
            val_logs = self.evaluate(val_loader, split_name='val')
            metric_value = float(val_logs['overall'].get(self.best_metric_name, float('-inf' if self.maximize_best_metric else 'inf')))
            history_row = {
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_logs.items()},
                **{f'val_{k}': v for k, v in val_logs['overall'].items()},
                **{f'val_loss_{k}': v for k, v in val_logs['losses'].items()},
            }
            history.append(history_row)
            pd.DataFrame(history).to_csv(self.output_dir / 'history.csv', index=False)
            save_json(history_row, self.output_dir / f'epoch_{epoch:03d}_summary.json')
            print(
                f'Epoch {epoch}: '
                f"train_loss={train_logs['loss_total']:.4f}, "
                f"val_PLCC={val_logs['overall']['PLCC']:.4f}, "
                f"val_SRCC={val_logs['overall']['SRCC']:.4f}, "
                f"val_RMSE={val_logs['overall']['RMSE']:.4f}, "
                f'{self.best_metric_name}={metric_value:.4f}'
            )

            state = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                'best_val_plcc': self.best_val_plcc,
                'best_val_metric': self.best_val_metric,
                'best_metric_name': self.best_metric_name,
                'history_len': len(history),
            }

            if self._is_improved(metric_value):
                self.best_val_metric = metric_value
                self.best_val_plcc = float(val_logs['overall']['PLCC'])
                self._epochs_without_improvement = 0
                state['best_val_plcc'] = self.best_val_plcc
                state['best_val_metric'] = self.best_val_metric
                save_checkpoint(state, best_state_path)
                save_json(
                    {
                        'epoch': epoch,
                        'monitor_metric': self.best_metric_name,
                        'monitor_value': self.best_val_metric,
                        'val': val_logs['overall'],
                        'losses': val_logs['losses'],
                        'aux_metrics': val_logs['aux_metrics'],
                    },
                    self.output_dir / 'best_metrics.json',
                )
                val_logs['predictions'].to_csv(self.output_dir / 'best_val_predictions.csv', index=False)
            else:
                self._epochs_without_improvement += 1

            save_checkpoint(state, self.output_dir / 'last.pt')
            save_json(
                {
                    'epoch': epoch,
                    'monitor_metric': self.best_metric_name,
                    'monitor_value': metric_value,
                    'val': val_logs['overall'],
                    'losses': val_logs['losses'],
                    'aux_metrics': val_logs['aux_metrics'],
                },
                self.output_dir / 'last_metrics.json',
            )
            if epoch % save_every == 0:
                save_checkpoint(state, self.output_dir / f'epoch_{epoch:03d}.pt')

            if self.early_stopping_patience is not None and self._epochs_without_improvement >= self.early_stopping_patience:
                print(
                    f'Early stopping at epoch {epoch} after '
                    f'{self._epochs_without_improvement} epochs without improvement in {self.best_metric_name}.'
                )
                break
        return history
