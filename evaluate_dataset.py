"""
Evaluate the LoRa dechirp rule and the ML classifier on labelled recordings.

Expects baseband WAV clips (SDR++ format, or live_scan.py --save-wav) in:
    <root>/lora_hop/   clips containing the LoRa transmitter
    <root>/noise/      clips without it
The center frequency is read from SDR++-style names (baseband_<freq>Hz_...),
otherwise --freq is used.

Each clip runs through the normal pipeline (analyze_iq). A clip counts as LoRa
for the rule if any detection fires it (dechirp score >= LORA_THRESHOLD_DB),
and for the ML if any detection is labelled LORA. The clip's score is the
highest dechirp score among its detections (none if nothing was detected).
A threshold is then suggested from the real scores: the one with the best
accuracy, placed in the middle of the widest gap that achieves it.

Usage:
    python evaluate_dataset.py [--root rf_dataset] [--freq 866000000]
"""

import argparse
import io
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np

from lora_dsp import LORA_THRESHOLD_DB
from rf_scanner import load_recording, analyze_iq, load_components
from threat_engine import ThreatEngine

LABELS = {"lora_hop": True, "noise": False}


def clip_center_freq(path, default_hz):
    match = re.search(r"_(\d{6,})Hz", os.path.basename(path))
    return float(match.group(1)) if match else default_hz


def evaluate_clip(path, default_hz, clf, fingerprinter):
    """Run the pipeline on one clip and summarize its detections."""
    iq, sample_rate = load_recording(path)
    center = clip_center_freq(path, default_hz)
    detections, _ = analyze_iq(iq, sample_rate, center, clf, fingerprinter, ThreatEngine(),
                               verbose=False)
    best = max(detections, key=lambda d: d["dechirp_score_db"], default=None)
    return {
        "clip": os.path.basename(path),
        "center_mhz": center / 1e6,
        "detections": len(detections),
        "score_db": None if best is None else best["dechirp_score_db"],
        "duty": None if best is None else best["duty_cycle"],
        "fhss": any(d.get("fhss") for d in detections),
        "rule": any(d["lora_rule"] for d in detections),
        "ml": any(d["ml_class"] == "LORA" for d in detections),
        "ml_labels": sorted({d["ml_class"] for d in detections}),
    }


def suggest_threshold(results):
    """Threshold on the clip score with the best accuracy, centered in the widest such gap."""
    scores = sorted({r["score_db"] for r in results if r["score_db"] is not None})
    if not scores:
        return None, None
    candidates = [scores[0] - 1.0] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + 1.0]

    def accuracy(th):
        return np.mean([((r["score_db"] is not None and r["score_db"] >= th) == r["label"])
                        for r in results])

    best_acc = max(accuracy(th) for th in candidates)
    best = [th for th in candidates if accuracy(th) == best_acc]
    gaps = {th: min((abs(th - s) for s in scores), default=0) for th in best}
    return max(best, key=lambda th: gaps[th]), best_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate the LoRa rule and ML on labelled clips.")
    parser.add_argument("--root", default="rf_dataset")
    parser.add_argument("--freq", type=float, default=866000000,
                        help="center frequency for clips whose name doesn't include it")
    args = parser.parse_args()

    clips = [(os.path.join(args.root, folder, name), label)
             for folder, label in LABELS.items()
             if os.path.isdir(os.path.join(args.root, folder))
             for name in sorted(os.listdir(os.path.join(args.root, folder)))
             if name.lower().endswith(".wav")]
    if not clips:
        print(f"No WAV clips found in {args.root}\\lora_hop or {args.root}\\noise - nothing to evaluate.")
        return 1

    with redirect_stdout(io.StringIO()):
        clf, fingerprinter, _ = load_components()
    clf.n_jobs = 1

    results = []
    print(f"{'Clip':<44} {'Label':<6} {'Det':>3} {'Dechirp':>8} {'Duty':>6} {'FHSS':>5} "
          f"{'Rule':>5} {'ML':>5}  ML labels")
    for path, label in clips:
        r = evaluate_clip(path, args.freq, clf, fingerprinter)
        r["label"] = label
        results.append(r)
        score = "-" if r["score_db"] is None else f"{r['score_db']:.1f}"
        duty = "-" if r["duty"] is None else f"{r['duty']:.0%}"
        print(f"{r['clip'][:44]:<44} {'LoRa' if label else 'noise':<6} {r['detections']:>3} "
              f"{score:>8} {duty:>6} {'yes' if r['fhss'] else 'no':>5} "
              f"{'yes' if r['rule'] else 'no':>5} {'yes' if r['ml'] else 'no':>5}  "
              f"{', '.join(r['ml_labels']) or '-'}")

    n = len(results)
    rule_ok = sum(r["rule"] == r["label"] for r in results)
    ml_ok = sum(r["ml"] == r["label"] for r in results)
    print(f"\nClips: {n} ({sum(r['label'] for r in results)} LoRa, "
          f"{sum(not r['label'] for r in results)} noise)")
    print(f"Rule (dechirp >= {LORA_THRESHOLD_DB} dB) correct: {rule_ok}/{n}")
    print(f"ML (any detection labelled LORA) correct:   {ml_ok}/{n}")
    threshold, acc = suggest_threshold(results)
    if threshold is not None:
        print(f"Suggested threshold from this data: {threshold:.1f} dB "
              f"(clip accuracy {acc:.0%}; current {LORA_THRESHOLD_DB} dB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
