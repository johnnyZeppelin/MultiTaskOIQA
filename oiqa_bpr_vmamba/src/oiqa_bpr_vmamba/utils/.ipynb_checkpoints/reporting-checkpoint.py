from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _escape_latex(value: Any) -> str:
    text = '' if value is None else str(value)
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    frame = df.copy().where(pd.notna(df), '')
    headers = [str(c) for c in frame.columns]
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for _, row in frame.iterrows():
        lines.append('| ' + ' | '.join(str(v) for v in row.tolist()) + ' |')
    return '\n'.join(lines) + '\n'


def dataframe_to_latex(df: pd.DataFrame) -> str:
    frame = df.copy().where(pd.notna(df), '')
    cols = len(frame.columns)
    align = 'l' + 'c' * max(0, cols - 1)
    header = ' & '.join(_escape_latex(c) for c in frame.columns) + ' \\\n'
    body_lines = [
        ' & '.join(_escape_latex(v) for v in row.tolist()) + ' \\\n'
        for _, row in frame.iterrows()
    ]
    return (
        '\\begin{tabular}{' + align + '}\n'
        '\\hline\n'
        + header +
        '\\hline\n'
        + ''.join(body_lines) +
        '\\hline\n'
        '\\end{tabular}\n'
    )


def write_table_bundle(df: pd.DataFrame, output_prefix: str | Path, index: bool = False) -> dict[str, str]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix('.csv')
    md_path = prefix.with_suffix('.md')
    tex_path = prefix.with_suffix('.tex')
    df.to_csv(csv_path, index=index)
    frame = df.reset_index(drop=True) if index else df.copy()
    md_path.write_text(dataframe_to_markdown(frame), encoding='utf-8')
    tex_path.write_text(dataframe_to_latex(frame), encoding='utf-8')
    return {'csv': str(csv_path), 'md': str(md_path), 'tex': str(tex_path)}


def evaluation_summary_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    overall = dict(metrics.get('overall', {}))
    overall.update(metrics.get('aux_metrics', {}))
    overall['section'] = 'overall'
    overall['compression_type'] = 'overall'
    rows.append(overall)
    per_type = metrics.get('per_type', {}) or {}
    for comp_name, values in per_type.items():
        row = {'section': 'per_type', 'compression_type': comp_name}
        row.update(values)
        rows.append(row)
    preferred_cols = ['section', 'compression_type', 'PLCC', 'SRCC', 'RMSE', 'distortion_acc', 'compression_acc']
    df = pd.DataFrame(rows)
    other_cols = [c for c in df.columns if c not in preferred_cols]
    return df[[c for c in preferred_cols if c in df.columns] + other_cols]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        '__'.join(str(x) for x in col if str(x) != '').strip('_') if isinstance(col, tuple) else str(col)
        for col in out.columns
    ]
    return out
