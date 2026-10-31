# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Offline RF signal-intelligence pipeline for RTL-SDR recordings: detect signals in a baseband IQ WAV capture, classify them with a Random Forest, match them against known drone RF profiles, score drone threat, and write reports. Targets the sub-GHz ISM bands (433/868/915 MHz). It's a flat set of standalone Python scripts with no package structure, test suite, linter or build step.

## Commands

```bash
pip install -r requirements.txt            # numpy, scipy, scikit-learn, matplotlib, joblib, pyrtlsdr (a .venv exists locally)

python generate_test_signals.py            # writes synthetic_{drone_telemetry,fm_broadcast,interference}.wav (2.4 MS/s, 5 s)
python rf_scanner.py <file.wav> <center_freq_hz>   # full pipeline, e.g. synthetic_drone_telemetry.wav 433900000
python train_rf_classifier.py              # regenerates rf_classifier_model.joblib + rf_classifier_config.json
python live_scan.py [--freq HZ] [--gain DB] [--duration S] [--loop | --stream] [--save-wav]   # capture from the dongle via pyrtlsdr (SDR++ must be closed)

# Standalone demos / single-stage checks (closest thing to tests)
python threat_engine.py                    # runs 4 canned assessments through ThreatEngine + DroneFingerprinter
python drone_fingerprints.py               # prints the profile DB and sample matches
# Each of these takes <file.wav> <center_freq_hz> and shows matplotlib plots
python capture_spectrum.py <file.wav> <center_freq_hz>
python signal_detector.py <file.wav> <center_freq_hz>
python feature_extractor.py <file.wav> <center_freq_hz>   # rule-based classifier
python ml_classifier.py <file.wav> <center_freq_hz>
```

Run scripts from the repo root: `rf_scanner.py` and `ml_classifier.py` load `rf_classifier_model.joblib` / `rf_classifier_config.json` by relative path, and reports (`rf_threat_report_<timestamp>.json/.txt`) are written to the CWD. `*.wav` and report files are gitignored, except under `Examples/`.

## Architecture

`rf_scanner.py` holds the pipeline. `analyze_iq()` covers steps 2–5 and returns `(detections, noise_floor)`; `load_components()` and `report_scan()` handle setup and step 6. Both `rf_scanner.main()` (WAV input) and `live_scan.py` (pyrtlsdr input) call these, so analysis changes go there. Live IQ is normalized as `(x − 127.5)/127.5`, matching pyrtlsdr's `read_samples`; it must not pass through `load_recording`'s `/32768`. `analyze_iq(..., verbose=False, burst_lookup=f)` is what `live_scan.py --stream` uses: it runs quietly and gets burstiness from `SignalTracker`'s presence across blocks instead of the ML class. Stream mode also sets `clf.n_jobs = 1`; the model is saved with `n_jobs=-1`, which adds about 1 s of worker-pool overhead per block, and the outputs are identical. For each recording:

1. `load_recording`: WAV is 2-channel int16 (I, Q), normalized by 32768. This is SDR++'s baseband format and also what `generate_test_signals.save_iq` writes.
2. `detect_signals`: two-sided Welch PSD (nperseg 4096, fftshifted). Peaks must clear median noise floor + 10 dB with prominence 5. Frequencies are offsets from `center_freq` (the CLI arg), so a wrong center frequency breaks band and fingerprint matching downstream.
3. Per peak: `estimate_bandwidth` (−20 dB walk-out), then `extract_signal_iq` (FFT bandpass on the first 0.5 s), then `extract_ml_features`, which produces the 10-feature vector passed to the joblib model.
4. `DroneFingerprinter.get_best_match` (`drone_fingerprints.py`) scores the frequency, bandwidth and burst pattern against `DRONE_PROFILES`. Only profiles with `detectable_by_rtlsdr: True` are matched; the DJI 2.4/5.8 GHz entries are reference-only.
5. `ThreatEngine.assess` (`threat_engine.py`) combines band membership (`DRONE_BANDS`), ML class/confidence, fingerprint score, bandwidth, burstiness and power into a 0–100 score. That score maps to `SEVERITY_LEVELS` (NONE/LOW/MEDIUM/HIGH/CRITICAL), with per-factor reasons and recommendations.
6. `ScanReport` (`scan_report.py`) collects the detection dicts built in `rf_scanner.main` and saves the JSON + text reports.

### Invariants to keep in sync

- **Feature extraction is duplicated** in `train_rf_classifier.extract_features`, `rf_scanner.extract_ml_features` and `ml_classifier.extract_features`. All three must produce the same 10 features in the same order as `FEATURE_NAMES` / `rf_classifier_config.json`. If one changes, update the others and retrain. The training copy uses the module constant `SAMPLE_RATE` (2.4 MS/s); the inference copies take `sample_rate` from the WAV.
- **Class names**: the model's classes are `FM_BROADCAST, DRONE_TELEMETRY, DRONE_VIDEO, INTERFERENCE, NOISE` (`CLASSES` in the training script). `NARROWBAND_SIGNAL` comes only from the rule-based classifier in `feature_extractor.py`, but `threat_engine.SUSPECT_ML_CLASSES` and the burst check in `rf_scanner` also reference it. Keep the class lists in the training script, `threat_engine.DRONE_ML_CLASSES`/`SUSPECT_ML_CLASSES` and `rf_scanner` consistent.
- `rf_classifier_model.joblib` is committed. Changing the training data or features means retraining and committing the new model and config together.

### Known limitations (from README)

- Training data is fully synthetic (`train_rf_classifier.generate_signal`). Real captures get only ~50–65% confidence, a deliberate domain-gap demonstration.
- Strong signals produce sidelobe peaks that show up as separate detections.
- Crossfire, ELRS and FrSky R9 overlap in frequency, so several fingerprint matches can be reported for one signal.
- The tool is receive-only and analyzes signal characteristics only. It never decodes payload contents, and new features should stay within that scope (see the README's legal disclaimer).
