"""
Report figures for the test scenarios: one combined two-panel figure each
(spectrum on top, waterfall below), run through the deployed pipeline.

    python spectrum_report.py
    python spectrum_report.py --scenario fm=fm_test.wav@91100000 --scenario beacon_off=my_capture.wav@866000000

Default scenarios use the recordings in the repo; replace them with your own
captures (e.g. from live_scan.py --save-wav) via --scenario name=file.wav@freq_hz.
Writes results/report/figures/fig1N_scenario_<name>.png/.pdf (numbered from 10,
after the model-comparison figures) plus the per-capture spectrum/waterfall pair
in results/spectra/. Run it before compare_models.py so the report includes them.
"""

import argparse
import io
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

from rf_scanner import analyze_iq, load_components, load_recording
from spectrum_plots import save_capture_plots
from threat_engine import ThreatEngine

DEFAULT_SCENARIOS = [
    ("fm", "fm_test.wav", 91100000, "FM broadcast (real recording)"),
    ("fixed_beacon", "lora_test.wav", 866000000, "Fixed LoRa beacon (real recording)"),
    ("hopping_beacon", "synthetic_fhss.wav", 866000000, "Hopping LoRa beacon (synthetic)"),
    ("beacon_off", "synthetic_noise.wav", 866000000, "Beacon off, noise only (synthetic)"),
]
FIG_DIR = os.path.join("results", "report", "figures")


def parse_scenario(text):
    name, rest = text.split("=", 1)
    path, freq = rest.rsplit("@", 1)
    return name, path, float(freq), f"{name.replace('_', ' ')} ({os.path.basename(path)})"


def main():
    parser = argparse.ArgumentParser(description="Combined spectrum/waterfall figures for each scenario.")
    parser.add_argument("--scenario", action="append", type=parse_scenario,
                        help="name=file.wav@center_freq_hz (repeatable; replaces the defaults)")
    args = parser.parse_args()
    scenarios = args.scenario or DEFAULT_SCENARIOS

    with redirect_stdout(io.StringIO()):
        clf, fingerprinter, _ = load_components()
    saved = []
    for k, (name, path, freq, label) in enumerate(scenarios):
        if not os.path.exists(path):
            print(f"[!] {name}: {path} not found - skipped")
            continue
        iq, sr = load_recording(path)
        details = {}
        detections, _ = analyze_iq(iq, sr, freq, clf, fingerprinter, ThreatEngine(), verbose=False,
                                   details=details)
        verdicts = ", ".join(sorted({d["verdict"] for d in detections})) or "no detections"
        when = datetime.now()
        paths = save_capture_plots(details, detections, freq, None, when,
                                   combined_path=os.path.join(FIG_DIR, f"fig{10 + k}_scenario_{name}"),
                                   combined_title=f"{label}  |  {freq / 1e6:.3f} MHz  |  {verdicts}")
        saved += paths
        print(f"[OK] {name}: {len(detections)} detection(s), {verdicts}")
    print("\nSaved files:")
    for p in saved:
        print("  " + os.path.normpath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
