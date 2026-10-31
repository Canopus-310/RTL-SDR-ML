"""
Per-stage latency measurement for the scan pipeline.

A StageTimer accumulates time.perf_counter() durations per named stage:

    timer = StageTimer()            # also marks the end-to-end start
    with timer.stage("detection"):
        ...

Stages that run once per detected signal (extraction .. threat) are entered
once per signal and add up, so each stage holds the total for the scan.
"""

import time
from contextlib import contextmanager

import numpy as np

# (key, table label, compact label) in pipeline order
STAGES = [
    ("capture", "Capture (incl. settle)", "cap"),
    ("detection", "Detection", "det"),
    ("bursts", "Burst detection + FHSS", "bst"),
    ("extraction", "Bandwidth + extraction", "bw+x"),
    ("features", "Feature extraction", "feat"),
    ("ml", "ML classification", "ml"),
    ("fingerprint", "Fingerprinting", "fp"),
    ("threat", "Threat assessment", "thr"),
    ("report", "Report writing", "rep"),
    ("plots", "Plot saving (--plot)", "plt"),
]
PER_SIGNAL_STAGES = ("extraction", "features", "ml", "fingerprint", "threat")
# Shown (table, JSON) only when they took time, so runs without them look as before
OPTIONAL_STAGES = ("plots",)
PROCESSING_STAGES = [key for key, _, _ in STAGES if key not in ("capture",) + OPTIONAL_STAGES]


def _fmt_ms(ms):
    return f"{ms:.1f}" if ms < 10 else f"{ms:.0f}"


class StageTimer:
    """Accumulates wall-clock milliseconds per pipeline stage."""

    def __init__(self):
        self.start = time.perf_counter()
        self.ms = {key: 0.0 for key, _, _ in STAGES}
        self.signals = 0

    @contextmanager
    def stage(self, key):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.ms[key] += 1000 * (time.perf_counter() - t0)

    def add(self, key, ms):
        """Record a duration measured elsewhere (e.g. in the stream reader thread)."""
        self.ms[key] += ms

    def elapsed_ms(self):
        """Milliseconds since the timer was created."""
        return 1000 * (time.perf_counter() - self.start)

    def processing_ms(self):
        """Stages after capture (detection .. report writing)."""
        return sum(self.ms[key] for key in PROCESSING_STAGES)

    def as_dict(self, end_to_end_ms=None):
        result = {
            "stages_ms": {key: round(ms, 2) for key, ms in self.ms.items()
                          if key not in OPTIONAL_STAGES or ms > 0},
            "signals": self.signals,
            "per_signal_stages": list(PER_SIGNAL_STAGES),
            "processing_ms": round(self.processing_ms(), 2),
        }
        if end_to_end_ms is not None:
            result["end_to_end_ms"] = round(end_to_end_ms, 2)
        return result

    def format_table(self, end_to_end_ms):
        """Stage / ms / % of end-to-end, plus the processing and end-to-end totals."""
        lines = [f"  {'Stage':<30}{'Time (ms)':>11}{'% of total':>12}",
                 "  " + "-" * 53]
        for key, label, _ in STAGES:
            if key in OPTIONAL_STAGES and self.ms[key] == 0:
                continue
            if key in PER_SIGNAL_STAGES:
                label += f" (x{self.signals})"
            lines.append(f"  {label:<30}{self.ms[key]:>11.1f}"
                         f"{100 * self.ms[key] / end_to_end_ms:>11.1f}%")
        other = end_to_end_ms - sum(self.ms.values())
        lines.append(f"  {'Other (printing, overhead)':<30}{other:>11.1f}"
                     f"{100 * other / end_to_end_ms:>11.1f}%")
        lines.append("  " + "-" * 53)
        lines.append(f"  Signals: {self.signals} (stages marked xN are summed over all signals)")
        lines.append(f"  Processing (detection .. report): {self.processing_ms():.1f} ms")
        lines.append(f"  End-to-end (capture start .. results printed): {end_to_end_ms:.1f} ms")
        return "\n".join(lines)

    def compact(self, keys=None):
        """One-line breakdown, e.g. 'det 152 bw+x 310 feat 640 ml 25 fp 0.1 thr 0.3'."""
        keys = keys or [key for key, _, _ in STAGES]
        short = {key: abbrev for key, _, abbrev in STAGES}
        return " ".join(f"{short[key]} {_fmt_ms(self.ms[key])}" for key in keys)


def stage_stats(samples):
    """
    Mean / max / p95 for each key across a list of {key: ms} dicts.

    Returns {key: {"mean_ms", "max_ms", "p95_ms"}}, keys in first-seen order.
    """
    keys = list(dict.fromkeys(key for sample in samples for key in sample))
    stats = {}
    for key in keys:
        values = np.array([sample[key] for sample in samples if key in sample])
        stats[key] = {"mean_ms": round(float(values.mean()), 2),
                      "max_ms": round(float(values.max()), 2),
                      "p95_ms": round(float(np.percentile(values, 95)), 2)}
    return stats
