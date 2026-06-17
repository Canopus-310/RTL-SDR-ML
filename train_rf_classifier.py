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

SAMPLE_RATE = 2400000
DURATION = 0.5  # half-second snippets for training
NUM_SAMPLES_PER_CLASS = 200

np.random.seed(42)


def generate_signal(sig_type):
    """Generate one synthetic signal snippet with random variation."""
    t = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
    
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
        
    elif sig_type == "DRONE_VIDEO":
        # Wider analog signal, continuous, moderate bandwidth
        freq = np.random.uniform(40000, 80000)
        mod_freq = np.random.uniform(15000, 30000)
        modulator = np.sin(2 * np.pi * mod_freq * t)
        deviation = np.random.uniform(100000, 200000)
        phase = 2 * np.pi * (freq * t + deviation * np.cumsum(modulator) / SAMPLE_RATE)
        signal = np.exp(1j * phase)
        snr = np.random.uniform(12, 28)
        
    elif sig_type == "INTERFERENCE":
        # CW or narrowband noise, very narrow
        freq = np.random.uniform(20000, 100000)
        signal = 0.3 * np.exp(1j * 2 * np.pi * freq * t)
        # Add slight random drift
        drift = np.random.uniform(-100, 100)
        signal *= np.exp(1j * 2 * np.pi * drift * t * t)
        snr = np.random.uniform(8, 20)
        
    else:  # NOISE
        signal = np.zeros(len(t), dtype=complex)
        # Sometimes add very faint random signals barely above noise 
        if np.random.random() > 0.5:
            freq = np.random.uniform(10000, 200000)
            signal = 0.02 * np.exp(1j * 2 * np.pi * freq * t)       
        snr = np.random.uniform(0, 5)
    
    # Add noise
    noise = (np.random.normal(0, 1, len(t)) + 1j * np.random.normal(0, 1, len(t))) * 0.01
    if snr > 0:
        sig_power = np.mean(np.abs(signal) ** 2) + 1e-20
        noise_power = sig_power / (10 ** (snr / 10))
        noise = noise * np.sqrt(noise_power) / 0.01
    return signal + noise


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
    "duty_cycle", "num_spectral_peaks"
]

CLASSES = ["FM_BROADCAST", "DRONE_TELEMETRY", "DRONE_VIDEO", "INTERFERENCE", "NOISE"]


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
            sig = generate_signal(cls)
            features = extract_features(sig)
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