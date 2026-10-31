"""
RF Drone Threat Scanner — Main Entry Point

Chains the full pipeline: detect -> classify -> fingerprint -> threat assess -> report.

Usage:
    python rf_scanner.py <recording.wav> <center_freq_hz>

Examples:
    python rf_scanner.py recording.wav 433900000
    python rf_scanner.py synthetic_drone_telemetry.wav 433900000
"""

import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch, find_peaks
import joblib
import json

from drone_fingerprints import DroneFingerprinter
from threat_engine import ThreatEngine
from scan_report import ScanReport


def load_recording(path):
    """Load IQ recording from WAV file."""
    sample_rate, data = wavfile.read(path)
    if data.ndim == 2:
        iq = data[:, 0] + 1j * data[:, 1]
    else:
        iq = data.astype(np.float64)
    iq = iq / 32768.0
    return iq, sample_rate


def detect_signals(iq, sample_rate, center_freq):
    """Detect signals in the spectrum. Returns freqs, psd, peaks, properties."""
    freqs, psd = welch(iq, fs=sample_rate, nperseg=4096, return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    freqs_hz = freqs + center_freq
    freqs_mhz = freqs_hz / 1e6
    psd_db = 10 * np.log10(psd + 1e-20)

    noise_floor = np.median(psd_db)
    threshold = noise_floor + 10
    peaks, props = find_peaks(psd_db, height=threshold, distance=50, prominence=5)

    return freqs, freqs_hz, freqs_mhz, psd_db, noise_floor, threshold, peaks, props


def extract_signal_iq(iq, sample_rate, freqs, peak_idx, bw_hz):
    """Extract a signal's IQ using FFT bandpass filter."""
    chunk = iq[:int(0.5 * sample_rate)]
    fft_iq = np.fft.fft(chunk)
    fft_freqs = np.fft.fftfreq(len(fft_iq), 1 / sample_rate)
    sig_center = freqs[peak_idx]
    half_bw = max(bw_hz, 50000) / 2
    mask = np.abs(fft_freqs - sig_center) < half_bw
    return np.fft.ifft(fft_iq * mask)


def estimate_bandwidth(freqs_hz, psd_db, peak_idx, noise_floor):
    """Estimate signal bandwidth."""
    peak_power = psd_db[peak_idx]
    bw_threshold = max(peak_power - 20, noise_floor + 3)
    left = peak_idx
    while left > 0 and psd_db[left] > bw_threshold:
        left -= 1
    right = peak_idx
    while right < len(psd_db) - 1 and psd_db[right] > bw_threshold:
        right += 1
    return freqs_hz[right] - freqs_hz[left], left, right


def extract_ml_features(iq, sample_rate):
    """Extract the 10 features used by the ML model."""
    freqs, psd = welch(iq, fs=sample_rate, nperseg=min(1024, len(iq)),
                       return_onesided=False)
    psd_db = 10 * np.log10(np.fft.fftshift(psd) + 1e-20)
    freqs = np.fft.fftshift(freqs)
    psd_linear = np.fft.fftshift(psd)

    noise_floor = np.median(psd_db)
    peak_power = np.max(psd_db)
    snr_est = peak_power - noise_floor

    above = psd_db > (noise_floor + 6)
    if np.any(above):
        indices = np.where(above)[0]
        bw_hz = (indices[-1] - indices[0]) * (freqs[1] - freqs[0])
    else:
        bw_hz = 0

    geo_mean = np.exp(np.mean(np.log(psd_linear + 1e-20)))
    arith_mean = np.mean(psd_linear)
    spectral_flatness = geo_mean / (arith_mean + 1e-20)

    amplitude = np.abs(iq)
    amp_mean = np.mean(amplitude)
    amp_std = np.std(amplitude) / (amp_mean + 1e-10)
    amp_kurtosis = np.mean((amplitude - amp_mean) ** 4) / (np.std(amplitude) ** 4 + 1e-20)

    phase = np.unwrap(np.angle(iq))
    inst_freq = np.diff(phase) * sample_rate / (2 * np.pi)
    freq_std = np.std(inst_freq)
    freq_range = np.percentile(inst_freq, 95) - np.percentile(inst_freq, 5)

    threshold = np.median(amplitude) + np.std(amplitude)
    duty_cycle = np.mean(amplitude > threshold)

    peaks, _ = find_peaks(psd_db, height=noise_floor + 8, distance=20)
    num_peaks = len(peaks)

    return [bw_hz, peak_power, snr_est, spectral_flatness,
            amp_std, amp_kurtosis, freq_std, freq_range,
            duty_cycle, num_peaks]


def analyze_iq(iq, sample_rate, center_freq, clf, fingerprinter, threat_engine,
               verbose=True, burst_lookup=None):
    """
    Detect, classify, fingerprint and threat-assess every signal in an IQ capture.

    Returns (detections, noise_floor): a list of detection dicts (one per
    detected peak, in the format ScanReport.add_detection expects) and the
    estimated noise floor in dB.

    verbose=False suppresses the console output. burst_lookup, if given, is
    called as burst_lookup(freq_mhz) -> bool to decide whether a signal is
    bursty; otherwise burstiness is inferred from the ML class.
    """
    log = print if verbose else (lambda *args, **kwargs: None)

    # --- Detect signals ---
    log(f"\n[*] Scanning spectrum around {center_freq/1e6:.3f} MHz...")
    (freqs, freqs_hz, freqs_mhz, psd_db,
     noise_floor, threshold, peaks, props) = detect_signals(iq, sample_rate, center_freq)

    log(f"    Noise floor: {noise_floor:.1f} dB")
    log(f"    Threshold:   {threshold:.1f} dB")
    log(f"    Signals detected: {len(peaks)}")

    detections = []

    # --- Process each detection ---
    if len(peaks) == 0:
        log("\n[*] No signals above threshold — band appears clear.")
    else:
        log(f"\n[*] Analyzing {len(peaks)} detected signal(s)...\n")

    for i, peak in enumerate(peaks):
        freq_mhz = freqs_mhz[peak]
        power = psd_db[peak]
        prominence = props["prominences"][i]
        bw_hz, left, right = estimate_bandwidth(freqs_hz, psd_db, peak, noise_floor)
        bw_khz = bw_hz / 1000

        # ML classification
        sig_iq = extract_signal_iq(iq, sample_rate, freqs, peak, bw_hz)
        features = extract_ml_features(sig_iq, sample_rate)
        ml_class = clf.predict([features])[0]
        ml_proba = clf.predict_proba([features])[0]
        ml_confidence = float(np.max(ml_proba))

        # Fingerprint matching
        if burst_lookup is not None:
            is_bursty = burst_lookup(freq_mhz)
        else:
            is_bursty = ml_class in ["DRONE_TELEMETRY", "NARROWBAND_SIGNAL"]
        fp_match = fingerprinter.get_best_match(
            freq_mhz=freq_mhz, bandwidth_khz=bw_khz,
            is_bursty=is_bursty)

        # Threat assessment
        assessment = threat_engine.assess({
            "freq_mhz": freq_mhz,
            "bandwidth_khz": bw_khz,
            "power_db": power,
            "ml_class": ml_class,
            "ml_confidence": ml_confidence,
            "fingerprint_match": fp_match,
            "is_bursty": is_bursty,
        })

        # Print live result
        sev = assessment["severity"]
        score = assessment["threat_score"]
        log(f"  Signal {i+1}: {freq_mhz:.3f} MHz  |  {bw_khz:.0f} kHz  |  "
              f"ML: {ml_class} ({ml_confidence:.0%})  |  "
              f"Threat: {score}/100 ({sev})")

        if fp_match:
            profile, fp_score, _ = fp_match
            log(f"           Fingerprint: {profile['name']} ({fp_score}/100)")

        # Add to report
        detection_record = {
            "freq_mhz": round(freq_mhz, 3),
            "power_db": round(power, 1),
            "bandwidth_khz": round(bw_khz, 1),
            "prominence_db": round(float(prominence), 1),
            "ml_class": ml_class,
            "ml_confidence": round(ml_confidence, 2),
            "fingerprint_name": fp_match[0]["name"] if fp_match else None,
            "fingerprint_score": fp_match[1] if fp_match else None,
            "threat_score": score,
            "threat_severity": sev,
            "threat_factors": assessment["factors"],
            "recommendations": assessment["recommendations"],
        }
        detections.append(detection_record)

    return detections, noise_floor


def load_components():
    """Load the ML model, fingerprint DB and threat engine. Returns None if the model is missing."""
    print("\n[*] Loading components...")
    try:
        clf = joblib.load("rf_classifier_model.joblib")
        with open("rf_classifier_config.json") as f:
            config = json.load(f)
        print(f"    ML model loaded ({config['accuracy']:.0%} accuracy)")
    except FileNotFoundError:
        print("    [!] ML model not found — run train_rf_classifier.py first")
        return None

    fingerprinter = DroneFingerprinter()
    print(f"    Fingerprint DB loaded ({len(fingerprinter.detectable)} drone profiles)")

    threat_engine = ThreatEngine()
    print(f"    Threat engine ready")
    return clf, fingerprinter, threat_engine


def report_scan(source, center_freq, sample_rate, num_samples,
                detections, noise_floor, threat_engine):
    """Print the threat summary and save JSON + text reports. Returns the filenames."""
    duration = num_samples / sample_rate

    # --- Build report ---
    report = ScanReport(source, center_freq / 1e6, sample_rate)
    report.set_environment(noise_floor, len(detections), duration, num_samples)
    for detection_record in detections:
        report.add_detection(detection_record)

    # --- Print threat summary ---
    threat_engine.summary()

    # --- Save reports ---
    json_name, txt_name = report.save()
    print(f"\n[OK] Reports saved:")
    print(f"     {json_name}")
    print(f"     {txt_name}")
    return json_name, txt_name


def main():
    # --- Parse arguments ---
    if len(sys.argv) < 3:
        print("Usage: python rf_scanner.py <recording.wav> <center_freq_hz>")
        print("Example: python rf_scanner.py synthetic_drone_telemetry.wav 433900000")
        return

    recording_path = sys.argv[1]
    center_freq = float(sys.argv[2])

    # --- Load components ---
    print("=" * 60)
    print("RTL-SDR DRONE RF THREAT SCANNER")
    print("=" * 60)

    components = load_components()
    if components is None:
        return
    clf, fingerprinter, threat_engine = components

    # --- Load and analyze recording ---
    print(f"\n[*] Loading recording: {recording_path}")
    try:
        iq, sample_rate = load_recording(recording_path)
    except Exception as e:
        print(f"    [!] Failed to load recording: {e}")
        return

    duration = len(iq) / sample_rate
    print(f"    {len(iq):,} samples, {duration:.1f}s, {sample_rate/1e6:.1f} MS/s")

    detections, noise_floor = analyze_iq(iq, sample_rate, center_freq,
                                         clf, fingerprinter, threat_engine)
    report_scan(recording_path, center_freq, sample_rate, len(iq),
                detections, noise_floor, threat_engine)

    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()