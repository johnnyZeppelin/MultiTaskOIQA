from __future__ import annotations

import argparse

from oiqa_bpr_vmamba.utils.opt_eval import create_optimal_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create a teaching-only optimal checkpoint for CVIQ evaluation demos.'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save the teaching-only optimal checkpoint (.pt).',
    )
    # parser.add_argument('--name', type=str, default=None, help='Optional logical name stored inside the checkpoint metadata.')
    parser.add_argument('--dataset', type=str, default='CVIQ')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--source', type=str, default='paper_table_ii')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = create_optimal_checkpoint(
        args.output,
        # name=args.name,
        dataset=args.dataset,
        split=args.split,
        source=args.source,
    )
    print(f'Saved teaching-only optimal checkpoint to: {path}')
    print(f'Saved metadata sidecar to: {path.with_suffix(".json")}')
    print('This checkpoint contains no trained weights and is for teaching/demo use only.')


if __name__ == '__main__':
    main()
