#!/usr/bin/env python3
"""
Build Phase 1 MS MARCO base/query embeddings and passage IDs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def collect_passages(n_passages: int, seed: int, streaming: bool) -> tuple[list[str], np.ndarray]:
    # Avoid torch shared-memory path in restricted environments.
    os.environ.setdefault("USE_TORCH", "0")
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    ds = load_dataset("ms_marco", "v1.1", split="train", streaming=streaming)

    texts: list[str] = []
    ids: list[int] = []

    if streaming:
        for row_idx, row in enumerate(ds):
            for passage in row["passages"]["passage_text"]:
                passage = (passage or "").strip()
                if passage:
                    texts.append(passage)
                    ids.append(row_idx)
                if len(texts) >= n_passages:
                    break
            if len(texts) >= n_passages:
                break
    else:
        all_indices = rng.choice(len(ds), size=min(n_passages, len(ds)), replace=False)
        for idx in all_indices:
            row = ds[int(idx)]
            passage_list = row["passages"]["passage_text"]
            for passage in passage_list:
                passage = (passage or "").strip()
                if passage:
                    texts.append(passage)
                    ids.append(int(idx))
                if len(texts) >= n_passages:
                    break
            if len(texts) >= n_passages:
                break

    return texts, np.asarray(ids, dtype=np.int64)


def collect_queries(n_queries: int, seed: int) -> list[str]:
    os.environ.setdefault("USE_TORCH", "0")
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    ds = load_dataset("ms_marco", "v1.1", split="validation")
    queries = [(q or "").strip() for q in ds["query"]]
    queries = [q for q in queries if q]
    choose = min(n_queries, len(queries))
    idx = rng.choice(len(queries), size=choose, replace=False)
    return [queries[int(i)] for i in idx]


def encode_texts(model_name: str, texts: list[str], batch_size: int) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return l2_normalize(embeddings).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MS MARCO embeddings for Phase 1.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/msmarco"))
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--n-passages", type=int, default=1_000_000)
    parser.add_argument("--n-queries", type=int, default=6_980)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode when reading the train split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting passages...")
    passages, passage_ids = collect_passages(
        n_passages=args.n_passages,
        seed=args.seed,
        streaming=not args.no_streaming,
    )
    print(f"Collected {len(passages)} passages")

    print("Collecting queries...")
    queries = collect_queries(n_queries=args.n_queries, seed=args.seed)
    print(f"Collected {len(queries)} queries")

    print("Encoding passages...")
    base = encode_texts(args.model_name, passages, args.batch_size)
    print("Encoding queries...")
    query = encode_texts(args.model_name, queries, args.batch_size)

    np.save(output_dir / "base.npy", base)
    np.save(output_dir / "query.npy", query)
    np.save(output_dir / "passage_ids.npy", passage_ids[: len(base)])

    print("Saved MS MARCO artifacts:")
    print(f"- base:        {base.shape} {base.dtype}")
    print(f"- query:       {query.shape} {query.dtype}")
    print(f"- passage_ids: {passage_ids[: len(base)].shape} {passage_ids.dtype}")


if __name__ == "__main__":
    main()
