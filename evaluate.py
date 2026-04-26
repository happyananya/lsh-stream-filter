import numpy as np
from typing import Sequence


# ------------------------------------------------------------------
# Core metrics
# ------------------------------------------------------------------

def compute_metrics(filter_admits: Sequence[bool], oracle_admits: Sequence[bool]) -> dict:
    """
    Compare filter decisions against oracle ground truth.

    filter_admits : per-vector admission decisions from LSHAdmissionFilter
    oracle_admits : per-vector admission decisions from BruteForceOracle
                    (True = truly novel, False = truly redundant)

    Returns dict with keys:
        precision        — of vectors the filter stored, fraction truly novel
        recall           — of truly novel vectors, fraction the filter kept
        compression_ratio — fraction of the full stream stored by the filter
        tp, fp, fn, tn   — raw counts
    """
    assert len(filter_admits) == len(oracle_admits), "length mismatch"

    fa = np.asarray(filter_admits, dtype=bool)
    oa = np.asarray(oracle_admits, dtype=bool)

    tp = int(np.sum(fa & oa))    # filter admits, oracle admits  (correct keep)
    fp = int(np.sum(fa & ~oa))   # filter admits, oracle rejects (false admit)
    fn = int(np.sum(~fa & oa))   # filter rejects, oracle admits (false reject)
    tn = int(np.sum(~fa & ~oa))  # filter rejects, oracle rejects (correct reject)

    precision         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall            = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    compression_ratio = float(np.mean(fa))

    return dict(
        precision=precision,
        recall=recall,
        compression_ratio=compression_ratio,
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def print_metrics(metrics: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}precision={metrics['precision']:.3f}  "
        f"recall={metrics['recall']:.3f}  "
        f"compression={metrics['compression_ratio']:.3f}  "
        f"(TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']})"
    )


# ------------------------------------------------------------------
# Baselines
# ------------------------------------------------------------------

class FIFOBaseline:
    """
    Fixed-size buffer; evicts the oldest entry when full.
    insert() always returns True (every vector is "stored" momentarily),
    but admitted tracks whether the vector is in the buffer at end of stream.
    For streaming evaluation we treat every insert as admitted=True and
    measure memory as buffer size / stream length.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list[np.ndarray] = []
        self.admit_log: list[bool] = []

    def insert(self, vec: np.ndarray) -> bool:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(vec)
        self.admit_log.append(True)
        return True

    def reset(self) -> None:
        self.buffer = []
        self.admit_log = []

    def __len__(self) -> int:
        return len(self.buffer)


class RandomEvictionBaseline:
    """
    Fixed-size buffer; evicts a uniformly random entry when full.
    """

    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.buffer: list[np.ndarray] = []
        self.admit_log: list[bool] = []

    def insert(self, vec: np.ndarray) -> bool:
        if len(self.buffer) >= self.capacity:
            evict_idx = int(self.rng.integers(0, len(self.buffer)))
            self.buffer.pop(evict_idx)
        self.buffer.append(vec)
        self.admit_log.append(True)
        return True

    def reset(self) -> None:
        self.buffer = []
        self.admit_log = []

    def __len__(self) -> int:
        return len(self.buffer)


# ------------------------------------------------------------------
# Smoke tests
# ------------------------------------------------------------------

if __name__ == "__main__":
    # --- test 1: perfect filter (matches oracle exactly) ---
    oracle  = [True, False, True, False, True]
    perfect = [True, False, True, False, True]
    m = compute_metrics(perfect, oracle)
    assert m["precision"] == 1.0, f"expected 1.0, got {m['precision']}"
    assert m["recall"]    == 1.0, f"expected 1.0, got {m['recall']}"
    assert abs(m["compression_ratio"] - 3/5) < 1e-9

    # --- test 2: filter admits everything ---
    admits_all = [True] * 5
    m = compute_metrics(admits_all, oracle)
    # TP=3 (novel admitted), FP=2 (redundant admitted)
    assert m["tp"] == 3 and m["fp"] == 2
    assert abs(m["precision"] - 3/5) < 1e-9
    assert m["recall"] == 1.0
    assert m["compression_ratio"] == 1.0

    # --- test 3: filter rejects everything ---
    rejects_all = [False] * 5
    m = compute_metrics(rejects_all, oracle)
    assert m["precision"] == 0.0  # no admits => 0 by convention
    assert m["recall"]    == 0.0
    assert m["fn"] == 3

    # --- test 4: one false reject, one false admit ---
    # oracle: T F T F T
    # filter: F F T T T  (missed first novel, admitted second redundant)
    filter_4 = [False, False, True, True, True]
    m = compute_metrics(filter_4, oracle)
    # TP: positions 2,4 → 2; FP: position 3 → 1; FN: position 0 → 1
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1
    assert abs(m["precision"] - 2/3) < 1e-9
    assert abs(m["recall"]    - 2/3) < 1e-9

    # --- test 5: FIFO baseline evicts correctly ---
    fifo = FIFOBaseline(capacity=3)
    vecs = [np.array([float(i)]) for i in range(5)]
    for v in vecs:
        fifo.insert(v)
    assert len(fifo.buffer) == 3
    assert fifo.buffer[0][0] == 2.0, "oldest 0,1 should have been evicted"

    # --- test 6: RandomEviction baseline stays at capacity ---
    rand_ev = RandomEvictionBaseline(capacity=3, seed=0)
    for v in vecs:
        rand_ev.insert(v)
    assert len(rand_ev.buffer) == 3

    print("All evaluate.py smoke tests passed.")
    print_metrics(compute_metrics(perfect, oracle), label="perfect filter")
    print_metrics(compute_metrics(admits_all, oracle), label="admits-all filter")
