"""
Train an ML classifier to distinguish RF signal types.
Generates varied synthetic training data, extracts features, trains Random Forest.
"""
import numpy as np
from scipy.signal import welch, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

from lora_dsp import lora_packet, dechirp_score
from rf_scanner import extract_signal_iq

SAMPLE_RATE = 2400000
# Snippet length varies per sample: the pipeline classifies anything from a
# 15 ms hop (burst window) to 0.5 s (steady signals), so length must not be a cue.
MIN_DURATION, MAX_DURATION = 0.015, 0.5
NUM_SAMPLES_PER_CLASS = 300

np.random.seed(42)
rng = np.random.default_rng(42)


def generate_lora(t):
    """1-3 LoRa packets (SF 7-9, BW 125/250/500 kHz) at a random offset; returns (signal, center, in-band SNR, bw)."""
    sf = int(rng.integers(7, 10))
    bw = float(rng.choice([125e3, 250e3, 500e3]))
    offset = rng.uniform(-300e3, 300e3)
    signal = np.zeros(len(t), dtype=complex)
    n_packets = int(rng.integers(1, 4))
    slot = len(t) // n_packets
    for i in range(n_packets):
        airtime = rng.uniform(0.6, 1.0) * slot / SAMPLE_RATE
        packet = lora_packet(rng, sf, bw, airtime, SAMPLE_RATE)
        start = i * slot + int(rng.integers(0, slot - len(packet) + 1))
        signal[start:start + len(packet)] = packet
    signal *= np.exp(2j * np.pi * offset * t)
    return signal, offset, rng.uniform(0, 30), bw


def generate_signal(sig_type):
    """
    Generate one synthetic signal snippet with random variation.

    Returns (iq, center_hz, bw_hz): the snippet (random length and amplitude),
    the signal's center offset (the dechirp feature is computed around it) and
    its occupied bandwidth (the pipeline band-passes each signal before features).
    """
    duration = np.random.uniform(MIN_DURATION, MAX_DURATION)
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    inband_snr = None
    
    if sig_type == "FM_BROADCAST":
        # Wide FM with realistic variation
        num_tones = np.random.randint(1, 6)
        audio = np.zeros(len(t))
        for _ in range(num_tones):
            audio += np.random.uniform(0.3, 1.0) * np.sin(
                2 * np.pi * np.random.uniform(200, 15000) * t)
        # Add some noise to the audio (realistic)
        audio += np.random.normal(0, 0.1, len(t))
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        deviation = np.random.uniform(40000, 80000)
        phase = 2 * np.pi * deviation * np.cumsum(audio) / SAMPLE_RATE
        # Random center offset (real stations aren't perfectly centered)
        offset = np.random.uniform(-200000, 200000)
        signal = np.exp(1j * (phase + 2 * np.pi * offset * t))
        # Vary amplitude (fading)
        fade = 1.0 + 0.3 * np.sin(2 * np.pi * np.random.uniform(0.5, 5) * t)
        signal = signal * fade
        snr = np.random.uniform(8, 35)  # wider SNR range
        center = offset
        bw = 2 * (deviation + 15000)
        
    elif sig_type == "DRONE_TELEMETRY":
        # Narrow FSK: two tones, bursty, ~10-30kHz bandwidth
        freq1 = np.random.uniform(30000, 60000)
        freq2 = freq1 + np.random.uniform(3000, 8000)
        baud = np.random.choice([9600, 19200, 57600])
        samples_per_bit = max(1, SAMPLE_RATE // baud)
        num_bits = len(t) // samples_per_bit
        bits = np.random.randint(0, 2, num_bits)
        signal = np.zeros(len(t), dtype=complex)
        for i, bit in enumerate(bits):
            s, e = i * samples_per_bit, min((i+1) * samples_per_bit, len(t))
            f = freq1 if bit == 0 else freq2
            signal[s:e] = 0.5 * np.exp(1j * 2 * np.pi * f * t[s:e])
        # Bursty: random duty cycle
        duty = np.random.uniform(0.3, 0.8)
        burst = int(duty * len(t))
        gap = len(t) - burst
        mask = np.concatenate([np.ones(burst), np.zeros(gap)])
        np.random.shuffle(mask)  # randomize burst position
        signal = signal * mask
        snr = np.random.uniform(10, 25)
        center = (freq1 + freq2) / 2
        bw = (freq2 - freq1) + 2 * baud
        
    elif sig_type == "DRONE_VIDEO":
        # Wider analog signal, continuous, moderate bandwidth
        freq = np.random.uniform(40000, 80000)
        mod_freq = np.random.uniform(15000, 30000)
        modulator = np.sin(2 * np.pi * mod_freq * t)
        deviation = np.random.uniform(100000, 200000)
        phase = 2 * np.pi * (freq * t + deviation * np.cumsum(modulator) / SAMPLE_RATE)
        signal = np.exp(1j * phase)
        snr = np.random.uniform(12, 28)
        center = freq
        bw = 2 * (deviation + mod_freq)
        
    elif sig_type == "INTERFERENCE":
        # CW or narrowband noise, very narrow
        freq = np.random.uniform(20000, 100000)
        signal = 0.3 * np.exp(1j * 2 * np.pi * freq * t)
        # Add slight random drift
        drift = np.random.uniform(-100, 100)
        signal *= np.exp(1j * 2 * np.pi * drift * t * t)
        snr = np.random.uniform(8, 20)
        center = freq
        bw = 3000

    elif sig_type == "LORA":
        signal, center, inband_snr, lora_bw = generate_lora(t)
        bw = lora_bw
        snr = 1  # noise set from inband_snr below
        
    else:  # NOISE
        signal = np.zeros(len(t), dtype=complex)
        center = np.random.uniform(-200000, 200000)
        bw = np.random.uniform(50000, 200000)
        # Sometimes add very faint random signals barely above noise 
        if np.random.random() > 0.5:
            freq = np.random.uniform(10000, 200000)
            signal = 0.02 * np.exp(1j * 2 * np.pi * freq * t)       
            center = freq
            bw = 3000
        snr = np.random.uniform(0, 5)
    
    # Add noise
    noise = (np.random.normal(0, 1, len(t)) + 1j * np.random.normal(0, 1, len(t))) * 0.01
    if snr > 0:
        sig_power = np.mean(np.abs(signal) ** 2) + 1e-20
        noise_power = sig_power / (10 ** (snr / 10))
        noise = noise * np.sqrt(noise_power) / 0.01
    if inband_snr is not None:
        # LoRa: noise set for an in-band SNR (signal vs noise within its bandwidth)
        sigma = np.sqrt(1.0 / 10 ** (inband_snr / 10) / lora_bw * SAMPLE_RATE / 2)
        noise = sigma * (np.random.normal(0, 1, len(t)) + 1j * np.random.normal(0, 1, len(t)))
    # Random gain (+-20 dB) so absolute power isn't a class cue
    gain = 10 ** (np.random.uniform(-20, 20) / 20)
    return gain * (signal + noise), center, bw


def extract_features(iq):
    """Extract RF features from an IQ snippet."""
    # Power spectrum
    freqs, psd = welch(iq, fs=SAMPLE_RATE, nperseg=min(1024, len(iq)), return_onesided=False)
    psd_db = 10 * np.log10(np.fft.fftshift(psd) + 1e-20)
    freqs = np.fft.fftshift(freqs)
    
    # Basic spectral stats
    noise_floor = np.median(psd_db)
    peak_power = np.max(psd_db)
    snr_est = peak_power - noise_floor
    
    # Bandwidth: width where power > noise + 6dB
    above = psd_db > (noise_floor + 6)
    if np.any(above):
        indices = np.where(above)[0]
        bw_hz = (indices[-1] - indices[0]) * (freqs[1] - freqs[0])
    else:
        bw_hz = 0
    
    # Spectral flatness (low = tonal, high = noisy)
    psd_linear = np.fft.fftshift(psd)
    geo_mean = np.exp(np.mean(np.log(psd_linear + 1e-20)))
    arith_mean = np.mean(psd_linear)
    spectral_flatness = geo_mean / (arith_mean + 1e-20)
    
    # Amplitude statistics
    amplitude = np.abs(iq)
    amp_mean = np.mean(amplitude)
    amp_std = np.std(amplitude) / (amp_mean + 1e-10)
    amp_kurtosis = np.mean((amplitude - amp_mean) ** 4) / (np.std(amplitude) ** 4 + 1e-20)
    
    # Instantaneous frequency stats
    phase = np.unwrap(np.angle(iq))
    inst_freq = np.diff(phase) * SAMPLE_RATE / (2 * np.pi)
    freq_std = np.std(inst_freq)
    freq_range = np.percentile(inst_freq, 95) - np.percentile(inst_freq, 5)
    
    # Duty cycle estimate (fraction of time signal is "on")
    threshold = np.median(amplitude) + np.std(amplitude)
    duty_cycle = np.mean(amplitude > threshold)
    
    # Number of spectral peaks
    peaks, _ = find_peaks(psd_db, height=noise_floor + 8, distance=20)
    num_peaks = len(peaks)
    
    return [
        bw_hz, peak_power, snr_est, spectral_flatness,
        amp_std, amp_kurtosis, freq_std, freq_range,
        duty_cycle, num_peaks
    ]


FEATURE_NAMES = [
    "bandwidth_hz", "peak_power_db", "snr_db", "spectral_flatness",
    "amp_variation", "amp_kurtosis", "freq_std_hz", "freq_range_hz",
    "duty_cycle", "num_spectral_peaks", "dechirp_score_db"
]

CLASSES = ["FM_BROADCAST", "DRONE_TELEMETRY", "DRONE_VIDEO", "INTERFERENCE", "NOISE", "LORA"]


def snippet_features(iq, center_hz, bw_hz):
    """
    The 10 spectral/amplitude features plus the dechirp score, computed the way
    rf_scanner.analyze_iq does: spectral features on the signal band-passed
    around center_hz (same extract_signal_iq filter), dechirp on the raw IQ.
    """
    band, _ = extract_signal_iq(iq, SAMPLE_RATE, center_hz, bw_hz)
    return extract_features(band) + [dechirp_score(iq, SAMPLE_RATE, center_hz)["score_db"]]


def main():
    print("="*60)
    print("RF SIGNAL CLASSIFIER - TRAINING")
    print("="*60)
    
    # Generate training data
    print(f"\nGenerating {NUM_SAMPLES_PER_CLASS} samples per class...")
    X, y = [], []
    for cls in CLASSES:
        print(f"  Generating {cls}...")
        for _ in range(NUM_SAMPLES_PER_CLASS):
            sig, center, bw = generate_signal(cls)
            # The pipeline's center and bandwidth estimates are off a little; mimic that
            features = snippet_features(sig, center + np.random.normal(0, 2000),
                                        bw * np.random.uniform(0.9, 1.3))
            X.append(features)
            y.append(cls)
    
    X = np.array(X)
    y = np.array(y)
    print(f"Total samples: {len(X)}")
    print(f"Features per sample: {len(FEATURE_NAMES)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"\nAccuracy: {accuracy:.1%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\nFeature Importance:")
    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                        key=lambda x: x[1], reverse=True)
    for name, imp in importances:
        bar = "#" * int(imp * 50)
        print(f"  {name:22s} {imp:.3f} {bar}")
    
    # Save
    joblib.dump(clf, "rf_classifier_model.joblib")
    with open("rf_classifier_config.json", "w") as f:
        json.dump({
            "features": FEATURE_NAMES,
            "classes": CLASSES,
            "accuracy": float(accuracy),
            "sample_rate": SAMPLE_RATE,
        }, f, indent=2)
    
    print(f"\nModel saved: rf_classifier_model.joblib")
    print(f"Config saved: rf_classifier_config.json")
    print("="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()