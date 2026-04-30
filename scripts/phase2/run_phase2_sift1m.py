#!/usr/bin/env python3
"""
Run Phase 2 tasks for SIFT1M end-to-end.

Outputs:
- per-method JSONs in results/phase2
- consolidated CSV: results/phase2/sift1m_phase2_results.csv
- Pareto-like plot:  results/phase2/sift1m_phase2_recall_qps.png
- short report:      results/phase2/sift1m_phase2_report.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 SIFT1M baselines.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/sift1m"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/phase2"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--hnsw-M", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=100)
    parser.add_argument("--faiss-nbits", type=int, default=256)
    return parser.parse_args()


def run_cmd(cmd: list[str], required: bool = True) -> bool:
    print(f"\n$ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        if required:
            raise
        return False


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # 1) HNSW baseline (required)
    run_cmd(
        [
            py,
            "scripts/phase2/reproduce_hnsw_sift1m.py",
            "--data-dir",
            str(args.data_dir),
            "--k",
            str(args.k),
            "--warmup",
            str(args.warmup),
            "--max-queries",
            str(args.max_queries),
            "--M",
            str(args.hnsw_M),
            "--ef-construction",
            str(args.hnsw_ef_construction),
            "--ef-search",
            str(args.hnsw_ef_search),
            "--output-json",
            str(args.results_dir / "hnsw_sift1m.json"),
        ]
    )

    # 2) FALCONN baseline (optional) then fallback if unavailable
    falconn_ok = run_cmd(
        [
            py,
            "scripts/phase2/reproduce_falconn_sift1m.py",
            "--data-dir",
            str(args.data_dir),
            "--k",
            str(args.k),
            "--warmup",
            str(args.warmup),
            "--max-queries",
            str(args.max_queries),
            "--output-json",
            str(args.results_dir / "falconn_sift1m.json"),
        ],
        required=False,
    )
    if not falconn_ok:
        print("\nFALCONN run unavailable on this machine; using FAISS fallback.")
        run_cmd(
            [
                py,
                "scripts/phase2/reproduce_faiss_lsh_sift1m.py",
                "--data-dir",
                str(args.data_dir),
                "--k",
                str(args.k),
                "--warmup",
                str(args.warmup),
                "--max-queries",
                str(args.max_queries),
                "--nbits",
                str(args.faiss_nbits),
                "--output-json",
                str(args.results_dir / "faiss_lsh_sift1m.json"),
            ],
            required=True,
        )

    # 3) Consolidate JSON outputs
    rows: list[dict[str, float | str | int]] = []
    for path in sorted(args.results_dir.glob("*_sift1m.json")):
        obj = load_json(path)
        metrics = obj["metrics"]
        rows.append(
            {
                "method": obj["method"],
                "dataset": obj.get("dataset", "sift1m"),
                "recall": metrics["recall"],
                "qps": metrics["qps"],
                "p50_us": metrics["p50_us"],
                "p95_us": metrics["p95_us"],
                "p99_us": metrics["p99_us"],
                "mean_us": metrics["mean_us"],
                "n_queries": metrics["n_queries"],
                "json_path": str(path),
            }
        )

    df = pd.DataFrame(rows).sort_values("qps", ascending=False)
    csv_path = args.results_dir / "sift1m_phase2_results.csv"
    df.to_csv(csv_path, index=False)

    # 4) Plot recall vs QPS
    fig_path = args.results_dir / "sift1m_phase2_recall_qps.png"
    plt.figure(figsize=(8, 5))
    for _, row in df.iterrows():
        plt.scatter(row["recall"], row["qps"], s=80, label=row["method"])
        plt.annotate(row["method"], (row["recall"], row["qps"]), xytext=(6, 4), textcoords="offset points")
    plt.yscale("log")
    plt.xlabel("Recall@10")
    plt.ylabel("QPS (log scale)")
    plt.title("Phase 2 SIFT1M Baselines")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # 5) Short markdown report
    report_path = args.results_dir / "sift1m_phase2_report.md"
    top_recall = df.sort_values("recall", ascending=False).iloc[0]
    top_qps = df.sort_values("qps", ascending=False).iloc[0]
    report = [
        "# SIFT1M Phase 2 Report",
        "",
        "## Outputs",
        f"- Results CSV: `{csv_path}`",
        f"- Plot: `{fig_path}`",
        "",
        "## Headline",
        f"- Best recall method: `{top_recall['method']}` ({top_recall['recall']:.4f})",
        f"- Best QPS method: `{top_qps['method']}` ({top_qps['qps']:.2f})",
        "",
        "## Table",
        df.to_markdown(index=False),
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nPhase 2 SIFT1M run complete.")
    print(f"- CSV:    {csv_path}")
    print(f"- Plot:   {fig_path}")
    print(f"- Report: {report_path}")


if __name__ == "__main__":
    main()
