#!/usr/bin/env python3
"""
Convert ann-benchmarks SIFT1M HDF5 into Phase 1 .npy layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def convert_sift1m(hdf5_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as handle:
        base = np.asarray(handle["train"], dtype=np.float32)
        query = np.asarray(handle["test"], dtype=np.float32)
        groundtruth = np.asarray(handle["neighbors"], dtype=np.int32)

    np.save(output_dir / "base.npy", base)
    np.save(output_dir / "query.npy", query)
    np.save(output_dir / "groundtruth.npy", groundtruth)

    print("Saved SIFT1M arrays:")
    print(f"- base:        {base.shape} {base.dtype}")
    print(f"- query:       {query.shape} {query.dtype}")
    print(f"- groundtruth: {groundtruth.shape} {groundtruth.dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SIFT1M HDF5 to data/sift1m/*.npy format."
    )
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        default=Path("sift-128-euclidean.hdf5"),
        help="Path to ann-benchmarks SIFT HDF5 file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sift1m"),
        help="Output directory for converted .npy files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_sift1m(args.hdf5_path, args.output_dir)
