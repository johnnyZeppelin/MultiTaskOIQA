from __future__ import annotations

from pathlib import Path

from oiqa_bpr_vmamba.cli.build_cviq_manifest import (
    _make_degraded_viewport,
    _make_restored_viewport,
    _normalize_path,
)


def test_path_prefix_resolution() -> None:
    prefix = Path('/tmp/prefix')
    rel = 'data/CVIQ/view_ports/326/326_fov15.png'
    assert _normalize_path(rel, prefix) == prefix / rel


def test_viewport_path_mirroring() -> None:
    src = Path('/tmp/prefix/data/CVIQ/view_ports/326/326_fov15.png')
    restored_root = Path('/tmp/prefix/data/CVIQ/restored/view_ports_restored')
    degraded_root = Path('/tmp/prefix/data/CVIQ/degraded/view_ports_degraded')
    assert _make_restored_viewport(src, restored_root) == restored_root / '326' / '326_fov15_r.png'
    assert _make_degraded_viewport(src, degraded_root) == degraded_root / '326' / '326_fov15_d.png'


def test_viewport_path_mirroring_with_compression_folder() -> None:
    src = Path('/tmp/prefix/data/CVIQ/view_ports/JPEG/326_fov15.png')
    restored_root = Path('/tmp/prefix/data/CVIQ/restored/view_ports_restored')
    assert _make_restored_viewport(src, restored_root) == restored_root / 'JPEG' / '326_fov15_r.png'
