"""Train mask-only synthetic QA with the shared adapter training loop."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Recognize the same long-option abbreviations as train.parse_args, so an
    # abbreviated --training-stage cannot switch this entry point back to SFT.
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument("--training-stage", choices=("mask_pretrain",))
    stage_parser.parse_known_args(arguments)

    from .train import main as train_main

    return train_main([
        "--training-stage", "mask_pretrain",
        "--image-mode", "square",
        "--projector-lr", "1e-4",
        *arguments,
    ])


if __name__ == "__main__":
    main()
