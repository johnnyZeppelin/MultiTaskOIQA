from __future__ import annotations

from pathlib import Path

from oiqa_bpr_vmamba.cli.common import existing_run_is_complete, run_logged_subprocess
from oiqa_bpr_vmamba.utils.io import save_json


def test_run_logged_subprocess_writes_log(tmp_path: Path) -> None:
    log_path = tmp_path / 'child.log'
    result = run_logged_subprocess(['python', '-c', 'print("hello")'], log_path)
    assert result.returncode == 0
    text = log_path.read_text(encoding='utf-8')
    assert '[stdout]' in text
    assert 'hello' in text


def test_existing_run_is_complete_checks_summary_and_test_file(tmp_path: Path) -> None:
    out_dir = tmp_path / 'run'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({'best_epoch': 1}, out_dir / 'run_summary.json')
    assert not existing_run_is_complete(out_dir)
    save_json({'PLCC': 0.9, 'SRCC': 0.8, 'RMSE': 1.2}, out_dir / 'test_overall.json')
    assert existing_run_is_complete(out_dir)
