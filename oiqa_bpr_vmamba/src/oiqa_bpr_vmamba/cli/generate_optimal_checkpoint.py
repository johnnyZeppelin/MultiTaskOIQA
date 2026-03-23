from __future__ import annotations

import argparse
from pathlib import Path

import torch

from oiqa_bpr_vmamba.utils.io import ensure_dir, save_checkpoint, save_json

DTYPE_MAP = {
    'float16': torch.float16,
    'float32': torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a transparent mock checkpoint with approximately N parameters.')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--target-params', type=int, default=80_000_000)
    parser.add_argument('--dtype', type=str, default='float16', choices=sorted(DTYPE_MAP))
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--tensor-prefix', type=str, default='mock_param')
    parser.add_argument('--chunk-size', type=int, default=5_000_000, help='Maximum parameters per tensor chunk.')
    parser.add_argument('--save-metadata-json', action='store_true')
    return parser.parse_args()


def build_mock_state_dict(target_params: int, dtype: torch.dtype, seed: int, tensor_prefix: str, chunk_size: int) -> tuple[dict[str, torch.Tensor], int]:
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    state_dict: dict[str, torch.Tensor] = {}
    remaining = int(target_params)
    total = 0
    idx = 0
    while remaining > 0:
        this_chunk = min(remaining, int(chunk_size))
        tensor = torch.randn(this_chunk, generator=gen, dtype=dtype)
        state_dict[f'{tensor_prefix}_{idx:04d}'] = tensor
        total += int(tensor.numel())
        remaining -= this_chunk
        idx += 1
    return state_dict, total


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    dtype = DTYPE_MAP[args.dtype]
    state_dict, total_params = build_mock_state_dict(
        target_params=args.target_params,
        dtype=dtype,
        seed=args.seed,
        tensor_prefix=args.tensor_prefix,
        chunk_size=args.chunk_size,
    )
    checkpoint = {
        'model': state_dict,
        'mock_checkpoint': True,
        'transparent_mock': True,
        'target_num_params': int(args.target_params),
        'actual_num_params': int(total_params),
        'dtype': str(dtype).replace('torch.', ''),
        'seed': int(args.seed),
        'tensor_prefix': args.tensor_prefix,
        'notes': 'Random transparent mock checkpoint. Not trained. Intended for benchmarking pipeline overhead only.',
    }
    save_checkpoint(checkpoint, output_path)
    meta = {
        'checkpoint_path': str(output_path),
        'target_num_params': int(args.target_params),
        'actual_num_params': int(total_params),
        'dtype': checkpoint['dtype'],
        'seed': int(args.seed),
        'num_tensors': len(state_dict),
    }
    if args.save_metadata_json:
        save_json(meta, output_path.with_suffix('.json'))
    print(meta)


if __name__ == '__main__':
    main()
