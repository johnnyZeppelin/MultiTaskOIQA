from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


KNOWN_COMPRESSIONS = ['JPEG', 'AVC', 'HEVC', 'ref']
VIEWPORT_ANCHORS = ('view_ports', 'viewports')


def _normalize_path(path_str: str, path_prefix: Path | None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    if path_prefix is None:
        return p
    return path_prefix / p


def _relative_after_anchor(path: Path, anchors: tuple[str, ...]) -> Path:
    parts = list(path.parts)
    for anchor in anchors:
        if anchor in parts:
            idx = parts.index(anchor)
            return Path(*parts[idx + 1 :])
    return Path(path.name)


def _infer_compression_type(path: Path) -> str:
    parts = set(path.parts)
    for name in KNOWN_COMPRESSIONS:
        if name in parts:
            return name
    raise ValueError(
        f'Could not infer compression type from path: {path}. '
        'Please either keep JPEG/AVC/HEVC/ref in the path or pass --compression-column.'
    )


def _resolve_root(root_arg: str | None, path_prefix: Path | None, fallback_root: Path) -> Path:
    if root_arg is None:
        return fallback_root
    p = Path(root_arg)
    if p.is_absolute():
        return p
    if path_prefix is None:
        return p
    return path_prefix / p


def _dataset_home_from_global(global_path: Path) -> Path:
    return global_path.parent


def _make_restored_global(path: Path, root: Path) -> Path:
    return root / f'{path.stem}_r{path.suffix}'


def _make_restored_viewport(path: Path, root: Path) -> Path:
    rel = _relative_after_anchor(path, VIEWPORT_ANCHORS)
    return (root / rel.parent / f'{rel.stem}_r{rel.suffix}').resolve()


def _make_degraded_viewport(path: Path, root: Path) -> Path:
    rel = _relative_after_anchor(path, VIEWPORT_ANCHORS)
    return (root / rel.parent / f'{rel.stem}_d{rel.suffix}').resolve()


def _infer_compression(row: pd.Series, viewport_paths: list[Path], compression_column: str | None) -> str:
    if compression_column and compression_column in row.index and pd.notna(row[compression_column]):
        value = str(row[compression_column]).strip()
        if value:
            return value
    for vp in viewport_paths:
        try:
            return _infer_compression_type(vp)
        except ValueError:
            continue
    raise ValueError(
        'Could not infer compression_type from viewport paths for this row. '
        'Either include JPEG/AVC/HEVC/ref in the relative path or provide --compression-column.'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num-viewports', type=int, default=20)
    parser.add_argument('--mos-column', type=str, default='mos')
    parser.add_argument('--global-column', type=str, default='fu')
    parser.add_argument('--compression-column', type=str, default=None)
    parser.add_argument('--distortion-level-column', type=str, default=None)
    parser.add_argument('--num-distortion-levels', type=int, default=5)

    parser.add_argument(
        '--path-prefix',
        type=str,
        default=None,
        help='Optional filesystem prefix placed before relative CSV paths, e.g. F:/ws/dataset for CSV entries like data/CVIQ/....',
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default=None,
        help='Backward-compatible alias of --path-prefix.',
    )
    parser.add_argument(
        '--global-restored-root',
        type=str,
        default=None,
        help='Override root for restored ERP images. Relative values are resolved under --path-prefix.',
    )
    parser.add_argument(
        '--viewport-restored-root',
        type=str,
        default=None,
        help='Override root for restored viewports. Relative values are resolved under --path-prefix.',
    )
    parser.add_argument(
        '--degraded-root',
        type=str,
        default=None,
        help='Override root for degraded viewports. Relative values are resolved under --path-prefix.',
    )
    return parser.parse_args()


def _build_sample(
    row: pd.Series,
    viewport_cols: list[str],
    args: argparse.Namespace,
    path_prefix: Path | None,
) -> dict[str, Any]:
    global_path = _normalize_path(str(row[args.global_column]), path_prefix)
    viewport_paths = [_normalize_path(str(row[c]), path_prefix) for c in viewport_cols]

    dataset_home = _dataset_home_from_global(global_path)
    global_restored_root = _resolve_root(
        args.global_restored_root,
        path_prefix,
        dataset_home / 'restored' / f'{dataset_home.name}_restored',
    )
    viewport_restored_root = _resolve_root(
        args.viewport_restored_root,
        path_prefix,
        dataset_home / 'restored' / 'view_ports_restored',
    )
    degraded_root = _resolve_root(
        args.degraded_root,
        path_prefix,
        dataset_home / 'degraded' / 'view_ports_degraded',
    )

    compression_type = _infer_compression(row, viewport_paths, args.compression_column)
    sample: dict[str, Any] = {
        'image_id': str(global_path.stem).replace('_r', ''),
        'distorted_global_path': str(global_path),
        'restored_global_path': str(_make_restored_global(global_path, global_restored_root)),
        'mos': float(row[args.mos_column]),
        'compression_type': compression_type,
    }
    for i, vp in enumerate(viewport_paths, start=1):
        sample[f'viewport_{i:02d}'] = str(vp)
        sample[f'restored_viewport_{i:02d}'] = str(_make_restored_viewport(vp, viewport_restored_root))
        sample[f'degraded_viewport_{i:02d}'] = str(_make_degraded_viewport(vp, degraded_root))
    if args.distortion_level_column and args.distortion_level_column in row.index and pd.notna(row[args.distortion_level_column]):
        sample['distortion_level'] = int(row[args.distortion_level_column])
    return sample


def main() -> None:
    args = parse_args()
    path_prefix = Path(args.path_prefix or args.dataset_root) if (args.path_prefix or args.dataset_root) else None
    df = pd.read_csv(args.csv)
    viewport_cols = [f'f{i:02d}' for i in range(1, args.num_viewports + 1)]
    missing = [c for c in [args.global_column, args.mos_column, *viewport_cols] if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required CSV columns: {missing}')

    rows = [_build_sample(row, viewport_cols, args, path_prefix) for _, row in df.iterrows()]
    out_df = pd.DataFrame(rows)

    if 'distortion_level' not in out_df.columns:
        bins = min(args.num_distortion_levels, out_df['mos'].nunique())
        out_df['distortion_level'] = pd.qcut(out_df['mos'], q=bins, labels=False, duplicates='drop').astype(int)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f'Saved manifest to {out_path}')


if __name__ == '__main__':
    main()
