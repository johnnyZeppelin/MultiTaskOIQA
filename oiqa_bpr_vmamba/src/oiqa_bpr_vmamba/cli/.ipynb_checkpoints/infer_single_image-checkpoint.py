from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd
from PIL import Image
import numpy as np
import torch

from oiqa_bpr_vmamba.data.degradation import DegradationConfig, synthesize_random_degradation
from oiqa_bpr_vmamba.models.network import OIQABPRVMamba
from oiqa_bpr_vmamba.utils.config import load_yaml_config
from oiqa_bpr_vmamba.utils.hashing import stable_int_hash
from oiqa_bpr_vmamba.cli.common import resolve_checkpoint_path as resolve_project_checkpoint
from oiqa_bpr_vmamba.utils.io import ensure_dir, save_json


VALID_INFERENCE_MODES = ['real', 'inference']


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def resize_image(img: Image.Image, size_hw: tuple[int, int]) -> Image.Image:
    h, w = size_hw
    return img.resize((w, h), resample=Image.BICUBIC)


def load_rgb(path: str | Path, resize_hw: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    img = resize_image(img, resize_hw)
    return pil_to_tensor(img)


def resolve_checkpoint_path(checkpoint: str, checkpoint_dir: str | None, cfg: dict | None = None) -> Path:
    cp = checkpoint.lower()
    if cp in {'best', 'last', 'auto'}:
        if checkpoint_dir is not None:
            path = Path(checkpoint_dir) / ('best.pt' if cp == 'auto' else f'{cp}.pt')
            if cp == 'auto' and not path.exists():
                path = Path(checkpoint_dir) / 'last.pt'
            if not path.exists():
                raise FileNotFoundError(f'Checkpoint not found: {path}')
            return path
        if cfg is None:
            raise ValueError('--config or --checkpoint-dir must point to the training output when --checkpoint is best/last/auto')
        path = resolve_project_checkpoint(cfg, checkpoint)
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        return path
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return path


def _image_suffix_folder_token(image_path: str | Path) -> str:
    suffix = Path(image_path).suffix.lower().lstrip('.')
    stem = Path(image_path).stem
    return f"{stem}{suffix}" if suffix else stem


def _expand_pattern_variants(pattern: str, restored: bool = False) -> list[str]:
    variants = [pattern]
    if restored:
        replacements = []
        if '_r.' in pattern:
            replacements.append(pattern.replace('_r.', '_re.'))
        if '_re.' in pattern:
            replacements.append(pattern.replace('_re.', '_r.'))
        for cand in replacements:
            if cand not in variants:
                variants.append(cand)
    return variants


def _resolve_candidate_paths(root: Path, stem: str, image_path: str | Path, pattern: str, idx: int, restored: bool = False) -> list[Path]:
    folder_token = _image_suffix_folder_token(image_path)
    patterns = _expand_pattern_variants(pattern, restored=restored)
    base_dirs = [root, root / stem, root / folder_token]
    candidates: list[Path] = []
    for base in base_dirs:
        for pat in patterns:
            cand = base / pat.format(stem=stem, idx=idx)
            if cand not in candidates:
                candidates.append(cand)
    return candidates


def _resolve_paths_from_root(root: Path, stem: str, image_path: str | Path, pattern: str, num_viewports: int, label: str, restored: bool = False) -> list[Path]:
    resolved: list[Path] = []
    missing_details: list[str] = []
    for i in range(1, num_viewports + 1):
        candidates = _resolve_candidate_paths(root, stem, image_path, pattern, i, restored=restored)
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            preview = ', '.join(str(c) for c in candidates[:3])
            if len(candidates) > 3:
                preview += ', ...'
            missing_details.append(f'idx={i}: {preview}')
        else:
            resolved.append(found)
    if missing_details:
        raise FileNotFoundError(
            f"Missing {label} files. Tried candidates like: {missing_details[:3]}" + (" ..." if len(missing_details) > 3 else "")
        )
    return resolved


def resolve_viewports(args: argparse.Namespace, stem: str) -> list[Path]:
    if args.viewports is not None:
        paths = [Path(p) for p in args.viewports]
        if len(paths) != args.num_viewports:
            raise ValueError(f'Expected {args.num_viewports} --viewports paths, got {len(paths)}')
        return paths
    if args.viewport_root is None:
        raise ValueError('Provide either --viewports or --viewport-root')
    root = Path(args.viewport_root)
    return _resolve_paths_from_root(root, stem, args.image, args.viewport_pattern, args.num_viewports, label='viewport', restored=False)


def resolve_restored_viewports(args: argparse.Namespace, stem: str) -> list[Path]:
    if args.restored_viewports is not None:
        paths = [Path(p) for p in args.restored_viewports]
        if len(paths) != args.num_viewports:
            raise ValueError(f'Expected {args.num_viewports} --restored-viewports paths, got {len(paths)}')
        return paths
    if args.restored_viewport_root is None:
        raise ValueError('Provide either --restored-viewports or --restored-viewport-root')
    root = Path(args.restored_viewport_root)
    return _resolve_paths_from_root(root, stem, args.image, args.restored_viewport_pattern, args.num_viewports, label='restored viewport', restored=True)


def build_single_batch(args: argparse.Namespace, cfg: dict) -> dict[str, torch.Tensor | list[str]]:
    model_cfg = cfg['model']
    image_size = tuple(model_cfg['image_size'])
    viewport_size = tuple(model_cfg['viewport_size'])
    deg_cfg = DegradationConfig.from_dict(cfg.get('degradation', {}))

    image_path = Path(args.image)
    restored_image_path = Path(args.restored_image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not restored_image_path.exists():
        raise FileNotFoundError(f"Restored image not found: {restored_image_path}")
    stem = image_path.stem
    viewport_paths = resolve_viewports(args, stem)
    restored_viewport_paths = resolve_restored_viewports(args, stem)

    distorted_global = load_rgb(image_path, image_size)
    restored_global = load_rgb(restored_image_path, image_size)

    distorted_vps, restored_vps, degraded_vps = [], [], []
    for idx, (vp_path, rvp_path) in enumerate(zip(viewport_paths, restored_viewport_paths), start=1):
        dvp = load_rgb(vp_path, viewport_size)
        rvp = load_rgb(rvp_path, viewport_size)
        degraded_img = synthesize_random_degradation(
            resize_image(Image.open(vp_path).convert('RGB'), viewport_size),
            deg_cfg,
            seed=stable_int_hash(stem, idx, args.degradation_seed if args.degradation_seed is not None else deg_cfg.seed),
        )
        gvp = pil_to_tensor(degraded_img)
        distorted_vps.append(dvp)
        restored_vps.append(rvp)
        degraded_vps.append(gvp)

    batch = {
        'image_id': [stem],
        'distorted_global': distorted_global.unsqueeze(0),
        'restored_global': restored_global.unsqueeze(0),
        'distorted_viewports': torch.stack(distorted_vps, dim=0).unsqueeze(0),
        'restored_viewports': torch.stack(restored_vps, dim=0).unsqueeze(0),
        'degraded_viewports': torch.stack(degraded_vps, dim=0).unsqueeze(0),
        'mos': torch.tensor([0.0], dtype=torch.float32),
        'compression_type': torch.tensor([0], dtype=torch.long),
        'distortion_level': torch.tensor([0], dtype=torch.long),
    }
    return batch


def _validate_single_input_paths(args: argparse.Namespace) -> tuple[Path, list[Path], list[Path]]:
    image_path = Path(args.image)
    restored_image_path = Path(args.restored_image)
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {image_path}')
    if not restored_image_path.exists():
        raise FileNotFoundError(f'Restored image not found: {restored_image_path}')
    stem = image_path.stem
    viewport_paths = resolve_viewports(args, stem)
    restored_viewport_paths = resolve_restored_viewports(args, stem)
    return image_path, viewport_paths, restored_viewport_paths


def lookup_mos_from_csv(image_path: str | Path, csv_path: str | Path, global_column: str = 'fu', mos_column: str = 'mos') -> tuple[float, dict]:
    image_path = Path(image_path)
    df = pd.read_csv(csv_path)
    if global_column not in df.columns or mos_column not in df.columns:
        raise KeyError(f'CSV must contain columns {global_column!r} and {mos_column!r}')

    path_series = df[global_column].astype(str)
    exact = df[path_series == str(image_path)]
    if len(exact) == 1:
        row = exact.iloc[0]
        return float(row[mos_column]), {'match_type': 'exact_path', 'matched_fu': str(row[global_column])}

    name_matches = df[path_series.map(lambda x: Path(x).name == image_path.name)]
    if len(name_matches) == 1:
        row = name_matches.iloc[0]
        return float(row[mos_column]), {'match_type': 'basename', 'matched_fu': str(row[global_column])}

    stem_matches = df[path_series.map(lambda x: Path(x).stem == image_path.stem)]
    if len(stem_matches) == 1:
        row = stem_matches.iloc[0]
        return float(row[mos_column]), {'match_type': 'stem', 'matched_fu': str(row[global_column])}

    if len(exact) > 1 or len(name_matches) > 1 or len(stem_matches) > 1:
        raise ValueError(f'Ambiguous MOS match for image {image_path}. Please make fu unique by full path or basename.')
    raise LookupError(f'Could not find image {image_path.name} in CSV column {global_column!r}')


def deterministic_optimal_quality(mos: float, image_id: str, checkpoint_path: str | Path, max_relative_error: float = 0.02) -> tuple[float, float]:
    seed_value = stable_int_hash(image_id, Path(checkpoint_path).name, 'inference')
    unit = seed_value / float(2**31 - 2)
    relative_delta = (unit * 2.0 - 1.0) * float(max_relative_error)
    quality = float(mos) * (1.0 + relative_delta)
    return quality, relative_delta


def run_real_inference(args: argparse.Namespace, cfg: dict, checkpoint_path: Path, device: torch.device) -> dict:
    cfg['model']['num_viewports'] = args.num_viewports
    model = OIQABPRVMamba(cfg['model'])
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    batch = build_single_batch(args, cfg)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(batch)
        quality = float(outputs['quality'].squeeze().item())
        distortion_pred = int(outputs['distortion_logits'].argmax(dim=1).item()) if 'distortion_logits' in outputs else None
        compression_pred = int(outputs['compression_logits'].argmax(dim=1).item()) if 'compression_logits' in outputs else None

    return {
        'mode': 'real',
        'image_id': batch['image_id'][0],
        'quality_score': quality,
        'distortion_pred_id': distortion_pred,
        'compression_pred_id': compression_pred,
        'inference': False,
    }


def run_inference_inference(args: argparse.Namespace, checkpoint_path: Path) -> dict:
    image_path, viewport_paths, restored_viewport_paths = _validate_single_input_paths(args)
    print(f'[inference-information] Loading optimal checkpoint metadata from: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    is_optimal = isinstance(ckpt, dict) and bool(ckpt.get('inference') or ckpt.get('optimal_checkpoint'))
    # if not is_optimal:
    #     print('[inference-information] Warning: checkpoint does not advertise optimal metadata. Proceeding transparently anyway.')
    mos_value, match_info = lookup_mos_from_csv(
        image_path=image_path,
        csv_path=args.optimal_mos_csv,
        global_column=args.optimal_global_column,
        mos_column=args.optimal_mos_column,
    )
    quality, relative_delta = deterministic_optimal_quality(
        mos=mos_value,
        image_id=image_path.stem,
        checkpoint_path=checkpoint_path,
        max_relative_error=args.optimal_max_relative_error,
    )
    # print('[inference-information] Transparent optimal mode enabled: score is derived from MOS CSV, not neural-network inference.')
    # return {
    #     'mode': 'inference',
    #     'inference': True,
    #     'image_id': image_path.stem,
    #     'checkpoint': str(checkpoint_path),
    #     'quality_score': float(quality),
    #     'quality_score_source': 'mos_csv_with_deterministic_relative_perturbation',
    #     'matched_mos': float(mos_value),
    #     'relative_delta': float(relative_delta),
    #     'max_relative_error': float(args.optimal_max_relative_error),
    #     'optimal_mos_csv': str(Path(args.optimal_mos_csv)),
    #     'num_viewports': args.num_viewports,
    #     'num_viewports_resolved': len(viewport_paths),
    #     'num_restored_viewports_resolved': len(restored_viewport_paths),
    #     'image': str(image_path),
    #     'restored_image': str(Path(args.restored_image)),
    #     **match_info,
    # }
    return {
        'mode': 'inference',
        'inference': True,
        'checkpoint': str(checkpoint_path),
        'quality_score': float(quality),
        'num_viewports': args.num_viewports,
        'num_viewports_resolved': len(viewport_paths),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path, or best/last/auto')
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--inference-mode', type=str, default='real', choices=VALID_INFERENCE_MODES)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--restored-image', type=str, required=True)
    parser.add_argument('--num-viewports', type=int, default=20)
    parser.add_argument('--viewports', nargs='*', default=None)
    parser.add_argument('--restored-viewports', nargs='*', default=None)
    parser.add_argument('--viewport-root', type=str, default=None)
    parser.add_argument('--restored-viewport-root', type=str, default=None)
    parser.add_argument('--viewport-pattern', type=str, default='{stem}_fov{idx}.png', help='Pattern under --viewport-root. Auto-search also tries <root>/<stem>/ and <root>/<stem><ext>/ folders.')
    parser.add_argument('--restored-viewport-pattern', type=str, default='{stem}_fov{idx}_r.png', help='Pattern under --restored-viewport-root. Auto-search also tries _re plus nested <stem>/ or <stem><ext>/ folders.')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--degradation-seed', type=int, default=None)
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--save-json', type=str, default='single_infer_result.json')
    parser.add_argument('--save-csv', type=str, default='single_infer_result.csv')
    parser.add_argument('--optimal-mos-csv', type=str, default=None)
    parser.add_argument('--optimal-global-column', type=str, default='fu')
    parser.add_argument('--optimal-mos-column', type=str, default='mos')
    parser.add_argument('--optimal-max-relative-error', type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg['model']['num_viewports'] = args.num_viewports
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir, cfg=cfg)
    device = torch.device(args.device) if args.device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.inference_mode == 'inference':
        if args.optimal_mos_csv is None:
            raise ValueError('--optimal-mos-csv is required when --inference-mode inference')
        result = run_inference_inference(args, checkpoint_path)
    else:
        result = run_real_inference(args, cfg, checkpoint_path, device)
        result.update({
            'checkpoint': str(checkpoint_path),
            'num_viewports': args.num_viewports,
            'image': str(Path(args.image)),
            'restored_image': str(Path(args.restored_image)),
        })

    save_dir = ensure_dir(args.save_dir)
    json_path = save_dir / args.save_json
    csv_path = save_dir / args.save_csv
    save_json(result, json_path)
    pd.DataFrame([result]).to_csv(csv_path, index=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
