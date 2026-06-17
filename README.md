# RTL-SDR Drone RF Threat Detection System

A software-defined radio (SDR) pipeline that captures RF signals, detects and classifies them using machine learning, matches them against a database of known drone communication profiles, and produces threat assessment reports. Built around a low-cost RTL-SDR dongle.

This project monitors the sub-GHz ISM bands (433/868/915 MHz) where drone telemetry and control links operate, identifying potential unauthorized drone activity through RF signal intelligence.

## Overview

The system runs a full signal-intelligence pipeline:

1. **Capture** — ingest IQ samples from an RTL-SDR recording
2. **Detect** — automatically find signals above the noise floor
3. **Classify** — identify signal type using a trained Random Forest model
4. **Fingerprint** — match against known drone RF profiles (SiK, Crossfire, ExpressLRS, FrSky)
5. **Assess** — score each signal for drone-related threat using multiple risk factors
6. **Report** — generate timestamped JSON and text security assessment reports

## Hardware

This project was built and tested with the following hardware:

![Hardware setup](Examples/hardware.jpeg)

| Component | Detail |
|-----------|--------|
| **SDR dongle** | NooElec NESDR SMArt v5 (RTL2832U chipset + R820T2 tuner) |
| **Frequency range** | ~24 MHz – 1.7 GHz |
| **Antenna** | Telescoping antenna (adjustable length for different bands) |
| **Antenna base** | Magnetic mount base with coaxial cable (SMA) |
| **Connection** | USB 2.0 cable (dongle to host PC) |
| **Host OS** | Windows |
| **Capture software** | SDR++ (for IQ recording) |

The telescoping antenna mounts on the magnetic base, which connects to the dongle via the coaxial cable. The dongle connects to the PC over USB. Recordings are captured in SDR++ as baseband IQ WAV files and analyzed offline by this pipeline.

> **Note on frequency coverage:** The RTL-SDR tuner range covers the 433/868/915 MHz drone telemetry bands. It does **not** reach 2.4 GHz / 5.8 GHz, where DJI OcuSync and similar video links operate — those profiles are included in the database for reference but flagged as not RTL-SDR detectable.

## Screenshots

### Real-world signal capture
FM broadcast spectrum captured from the RTL-SDR and processed in Python — two stations clearly visible at ~103.5 and ~104.5 MHz:

![Spectrum capture](Examples/spectrum.png)

### Automatic signal detection
The detector finds signals above the noise floor (red triangles); the spectrogram below shows their persistence over time:

![Signal detection](Examples/detection.png)

### ML classification
The trained Random Forest model classifies each detected signal — here correctly identifying both as FM broadcast:

![ML classification](Examples/classification.png)

## Installation

```bash
# Clone the repo
git clone https://github.com/Megh089/rtl-drone-rf.git
cd rtl-drone-rf

# Install dependencies
pip install -r requirements.txt
```

For live capture from hardware, you'll also need a working RTL-SDR driver setup (e.g. Zadig + WinUSB on Windows). Recordings can also be made with SDR++ and analyzed offline — no special drivers needed for the analysis pipeline itself.

## Usage

### Full pipeline (recommended)

Run the complete detect → classify → fingerprint → assess → report pipeline on a recording:

```bash
python rf_scanner.py <recording.wav> <center_freq_hz>

# Example: analyze a 433 MHz capture
python rf_scanner.py recording.wav 433900000
```

This produces `rf_threat_report_<timestamp>.json` and `.txt` with full findings.

### Individual tools

```bash
# Plot the spectrum of a recording
python capture_spectrum.py recording.wav 433900000

# Auto-detect signals with a waterfall view
python signal_detector.py recording.wav 433900000

# Rule-based feature extraction and classification
python feature_extractor.py recording.wav 433900000

# ML-powered classification
python ml_classifier.py recording.wav 433900000
```

### Training the ML model

The repo includes a pre-trained model, but you can regenerate it:

```bash
python train_rf_classifier.py
```

This generates synthetic training signals across five classes (FM broadcast, drone telemetry, drone video, interference, noise), extracts spectral features, and trains a Random Forest classifier.

### Generating test signals

To create synthetic recordings for testing without hardware:

```bash
python generate_test_signals.py
```

### Viewing the drone profile database

```bash
python drone_fingerprints.py
```

## Components

| File | Purpose |
|------|---------|
| `rf_scanner.py` | Main orchestrator — runs the full pipeline |
| `capture_spectrum.py` | Basic IQ capture and spectrum plot |
| `signal_detector.py` | Auto-detection with spectrogram |
| `feature_extractor.py` | Rule-based signal classification |
| `ml_classifier.py` | ML-powered classification |
| `train_rf_classifier.py` | Random Forest model training |
| `generate_test_signals.py` | Synthetic signal generator |
| `drone_fingerprints.py` | Drone RF profile database + matcher |
| `threat_engine.py` | Multi-factor threat scoring |
| `scan_report.py` | Report generation |

## Drone Profile Database

The fingerprint database includes RF profiles for common drone communication systems, sourced from published specifications and documentation:

- **SiK Telemetry Radio** (433/915 MHz) — MAVLink telemetry, GFSK, FHSS
- **TBS Crossfire** (868/915 MHz) — RC control, LoRa modulation
- **ExpressLRS** (868/915 MHz) — open-source RC control, LoRa
- **FrSky R9** (868/915 MHz) — RC control, LoRa
- **DJI OcuSync** (2.4/5.8 GHz) — reference only, outside RTL-SDR range

## How It Works

The threat assessment scores each detected signal across multiple factors: frequency band membership (is it in a known drone band?), ML classification result, fingerprint match strength, signal bandwidth, burst pattern, and signal strength (proximity). These combine into a 0–100 threat score mapped to severity levels (NONE / LOW / MEDIUM / HIGH / CRITICAL), with transparent per-factor reasoning so every score can be explained.

## Limitations and Honest Notes

- **Synthetic training data:** The ML model is trained on synthetic signals. It achieves high accuracy on synthetic test data but shows reduced confidence (~50–65%) on real-world captures — a demonstration of the domain gap between synthetic and real RF data. Real-world deployment would benefit from retraining on captured samples.
- **Sidelobe detection:** The detector can pick up spectral sidelobes of strong signals as separate detections.
- **LoRa systems overlap:** Crossfire, ExpressLRS, and FrSky R9 share frequency bands and can't be distinguished by frequency alone — the fingerprint matcher reports all plausible matches.
- **Receive-only:** This is a passive monitoring tool. The RTL-SDR cannot transmit.

## Legal Disclaimer

This tool is intended for **educational and security research purposes** and for monitoring **your own equipment** (e.g. your own drone's telemetry link).

Radio regulations vary by jurisdiction. In many countries, passively receiving radio signals is permitted, but **intercepting, decoding, recording, or divulging the contents of private communications you are not party to may be illegal.** In Canada, digital scanner receivers may be subject to licensing under the Radiocommunication Act (RSS-135), and it is an offence to divulge or use intercepted private radiocommunications.

This tool analyzes RF signal *characteristics* (frequency, bandwidth, modulation) for classification — it does not decode message contents. **Users are solely responsible for ensuring their use complies with all applicable laws and regulations in their jurisdiction.** Do not use this tool to intercept private communications or transmit on any frequency without proper authorization.

The author provides this software as-is, with no warranty, and accepts no liability for misuse.

## License

MIT License — see LICENSE file.