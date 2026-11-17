import time
import statistics as st
from contextlib import contextmanager
from typing import Dict, List, Any


class LatencyTracker:
    """
    Precision latency tracker for the voice turn pipeline.
    Tracks individual stage durations (STT, LLM, TTS) and total server processing time.
    """

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self._start_time = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.timings[name] = round(elapsed_ms, 2)

    def set_stage(self, name: str, value_ms: float):
        self.timings[name] = round(value_ms, 2)

    def finish(self) -> Dict[str, float]:
        elapsed_total = (time.perf_counter() - self._start_time) * 1000.0
        self.timings["server_total"] = round(elapsed_total, 2)
        return self.timings


def summarize(values: List[float]) -> Dict[str, Any]:
    """
    Computes standard summary statistics for latency or error rate distributions.
    Returns: {n, avg, min, max, p95}
    """
    if not values:
        return {"n": 0, "avg": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    avg = round(float(st.mean(sorted_vals)), 2)
    min_v = round(float(sorted_vals[0]), 2)
    max_v = round(float(sorted_vals[-1]), 2)
    p95_index = min(n - 1, int(0.95 * (n - 1)))
    p95_v = round(float(sorted_vals[p95_index]), 2)

    return {
        "n": n,
        "avg": avg,
        "min": min_v,
        "max": max_v,
        "p95": p95_v,
    }
