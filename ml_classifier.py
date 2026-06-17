"""
ML-powered RF signal classifier.
Loads a trained model and classifies signals in IQ recordings.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch, find_peaks
import joblib
import json
import sys
from datetime import datetime


def extract_features(iq, sample_rate):
    """Extract the same 10 features used in training."""
    freqs, psd = welch(iq, fs=sample_rate, nperseg=min(1024, len(iq)), return_onesided=False)
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


def estimate_bandwidth(freqs_hz, psd_db, peak_idx, noise_floor):
    peak_power = psd_db[peak_idx]
    bw_threshold = max(peak_power - 20, noise_floor + 3)
    left = peak_idx
    while left > 0 and psd_db[left] > bw_threshold:
        left -= 1
    right = peak_idx
    while right < len(psd_db) - 1 and psd_db[right] > bw_threshold:
        right += 1
    return freqs_hz[right] - freqs_hz[left], left, right


def main():
    if len(sys.argv) < 2:
        print("Usage: python ml_classifier.py <recording.wav> [center_freq_hz]")
        print("Example: python ml_classifier.py synthetic_drone_telemetry.wav 433900000")
        return

    recording = sys.argv[1]
    center_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 100e6

    # Load model
    clf = joblib.load("rf_classifier_model.joblib")
    with open("rf_classifier_config.json") as f:
        config = json.load(f)

    # Load recording
    sample_rate, data = wavfile.read(recording)
    if data.ndim == 2:
        iq = data[:, 0] + 1j * data[:, 1]
    else:
        iq = data.astype(np.float64)
    iq = iq / 32768.0

    print("=" * 60)
    print("RF SIGNAL CLASSIFIER (ML-POWERED)")
    print("=" * 60)
    print(f"Recording:  {recording}")
    print(f"Samples:    {len(iq)}")
    print(f"Duration:   {len(iq)/sample_rate:.1f}s")
    print(f"Center:     {center_freq/1e6:.3f} MHz")
    print(f"Model:      {config['accuracy']:.0%} accuracy on {len(config['classes'])} classes")

    # Compute spectrum for detection
    freqs, psd = welch(iq, fs=sample_rate, nperseg=4096, return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    freqs_hz = freqs + center_freq
    freqs_mhz = freqs_hz / 1e6
    psd_db = 10 * np.log10(psd + 1e-20)

    noise_floor = np.median(psd_db)
    threshold = noise_floor + 10
    peaks, props = find_peaks(psd_db, height=threshold, distance=50, prominence=5)

    print(f"\nNoise floor: {noise_floor:.1f} dB")
    print(f"Threshold:   {threshold:.1f} dB")
    print(f"Signals detected: {len(peaks)}")
    print("=" * 60)

    results = []
    for i, peak in enumerate(peaks):
        freq_mhz = freqs_mhz[peak]
        power = psd_db[peak]
        prominence = props["prominences"][i]
        bw_hz, left, right = estimate_bandwidth(freqs_hz, psd_db, peak, noise_floor)

       # Extract signal IQ using FFT bandpass filter (no decimation artifacts)
        fft_iq = np.fft.fft(iq[:int(0.5 * sample_rate)])  # use 0.5s chunk
        fft_freqs = np.fft.fftfreq(len(fft_iq), 1 / sample_rate)
        sig_center_offset = freqs[peak]
        half_bw = max(bw_hz, 50000) / 2
        mask = np.abs(fft_freqs - sig_center_offset) < half_bw
        fft_filtered = fft_iq * mask
        sig_chunk = np.fft.ifft(fft_filtered)

        # ML classification
        features = extract_features(sig_chunk, sample_rate)
        ml_class = clf.predict([features])[0]
        ml_proba = clf.predict_proba([features])[0]
        confidence = np.max(ml_proba)

        # Threat assessment
        if ml_class in ["DRONE_TELEMETRY", "DRONE_VIDEO"]:
            threat = "ALERT - potential drone signal"
        elif ml_class == "INTERFERENCE":
            threat = "WARNING - unknown interference"
        else:
            threat = "NORMAL"

        result = {
            "frequency_mhz": freq_mhz,
            "power_db": power,
            "prominence_db": prominence,
            "bandwidth_khz": bw_hz / 1000,
            "classification": ml_class,
            "confidence": confidence,
            "threat": threat,
        }
        results.append(result)

        print(f"\n--- Signal {i+1} ---")
        print(f"  Frequency:      {freq_mhz:.3f} MHz")
        print(f"  Power:          {power:.1f} dB")
        print(f"  Bandwidth:      {bw_hz/1000:.1f} kHz")
        print(f"  ML Class:       {ml_class}")
        print(f"  Confidence:     {confidence:.1%}")
        print(f"  Threat Level:   {threat}")

    # Save report
    report = {
        "metadata": {
            "tool": "RTL-SDR Drone RF Signal Classifier",
            "recording": recording,
            "center_freq_mhz": center_freq / 1e6,
            "model_accuracy": config["accuracy"],
            "scan_date": datetime.now().isoformat(),
        },
        "summary": {
            "signals_detected": len(results),
            "drone_signals": sum(1 for r in results if "DRONE" in r["classification"]),
            "alerts": sum(1 for r in results if r["threat"] != "NORMAL"),
        },
        "detections": results,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"rf_scan_{stamp}.json"
    with open(report_name, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freqs_mhz, psd_db, linewidth=0.5, alpha=0.8)
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')

    colors = {
        "FM_BROADCAST": "green", "DRONE_TELEMETRY": "red",
        "DRONE_VIDEO": "orange", "INTERFERENCE": "purple", "NOISE": "gray"
    }
    for r, peak in zip(results, peaks):
        c = colors.get(r["classification"], "blue")
        ax.plot(freqs_mhz[peak], psd_db[peak], 'v', color=c, markersize=12)
        ax.annotate(f"{r['classification']}\n{r['confidence']:.0%}",
                    (freqs_mhz[peak], psd_db[peak] + 1),
                    ha='center', fontsize=7, color=c, fontweight='bold')

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("ML-Powered RF Signal Classification")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\n[OK] Report saved: {report_name}")
    print("=" * 60)
    print(f"SCAN COMPLETE: {len(results)} signals, {report['summary']['alerts']} alerts")
    print("=" * 60)


if __name__ == "__main__":
    main()