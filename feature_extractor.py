import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io import wavfile
from scipy.signal import welch, find_peaks

if len(sys.argv) < 2:
    print("Usage: python feature_extractor.py <recording.wav> [center_freq_hz]")
    print("Example: python feature_extractor.py synthetic_drone_telemetry.wav 433900000")
    sys.exit(1)

recording = sys.argv[1]
center_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 100e6

sample_rate, data = wavfile.read(recording)

iq = data[:, 0] + 1j * data[:, 1]
iq = iq / 32768.0
print(f"Loaded {len(iq)} IQ samples at {sample_rate} Hz")

# --- Compute spectrum ---
freqs, psd = welch(iq, fs=sample_rate, nperseg=4096, return_onesided=False)
freqs = np.fft.fftshift(freqs)
psd = np.fft.fftshift(psd)
freqs_hz = freqs + center_freq
freqs_mhz = freqs_hz / 1e6
psd_db = 10 * np.log10(psd + 1e-20)

# --- Detect signals ---
noise_floor = np.median(psd_db)
threshold = noise_floor + 10
peaks, props = find_peaks(psd_db, height=threshold, distance=50, prominence=5)

print(f"\nNoise floor: {noise_floor:.1f} dB")
print(f"Signals detected: {len(peaks)}")
print("=" * 70)


def estimate_bandwidth(freqs_hz, psd_db, peak_idx, noise_floor):
    """Walk outward from peak until power drops to within 3dB of noise."""
    peak_power = psd_db[peak_idx]
    bw_threshold = max(peak_power - 20, noise_floor + 3)

    # walk left
    left = peak_idx
    while left > 0 and psd_db[left] > bw_threshold:
        left -= 1

    # walk right
    right = peak_idx
    while right < len(psd_db) - 1 and psd_db[right] > bw_threshold:
        right += 1

    bw_hz = freqs_hz[right] - freqs_hz[left]
    return bw_hz, left, right


def estimate_modulation(iq_signal, sample_rate):
    """Simple modulation indicators based on signal statistics."""
    # Instantaneous frequency via phase derivative
    phase = np.unwrap(np.angle(iq_signal))
    inst_freq = np.diff(phase) * sample_rate / (2 * np.pi)

    # Amplitude variation
    amplitude = np.abs(iq_signal)
    amp_std = np.std(amplitude) / (np.mean(amplitude) + 1e-10)

    # Frequency deviation (key FM indicator)
    freq_std = np.std(inst_freq)
    freq_range = np.percentile(inst_freq, 95) - np.percentile(inst_freq, 5)

    return {
        "amp_variation": amp_std,
        "freq_std_hz": freq_std,
        "freq_range_hz": freq_range,
    }


def classify_signal(bandwidth_hz, mod_features, freq_mhz):
    """Rule-based classification from extracted features."""
    reasons = []

    # Check frequency band
    if 88 <= freq_mhz <= 108:
        reasons.append("frequency in FM broadcast band (88-108 MHz)")
    elif 430 <= freq_mhz <= 440:
        reasons.append("frequency in 433 MHz ISM band (common drone telemetry)")
    elif 900 <= freq_mhz <= 930:
        reasons.append("frequency in 915 MHz ISM band (common drone telemetry)")

    # Check bandwidth
    if bandwidth_hz > 100000:
        reasons.append(f"wide bandwidth ({bandwidth_hz/1000:.0f} kHz) consistent with FM broadcast")
        sig_type = "FM_BROADCAST"
    elif bandwidth_hz > 20000:
        reasons.append(f"medium bandwidth ({bandwidth_hz/1000:.0f} kHz) consistent with voice/data")
        sig_type = "NARROWBAND_SIGNAL"
    else:
        reasons.append(f"narrow bandwidth ({bandwidth_hz/1000:.0f} kHz) consistent with telemetry/control")
        sig_type = "TELEMETRY_CANDIDATE"

    # Check modulation
    if mod_features["freq_range_hz"] > 50000:
        reasons.append("high frequency deviation suggests FM modulation")
    elif mod_features["amp_variation"] > 0.3:
        reasons.append("high amplitude variation suggests AM or digital modulation")

    return sig_type, reasons


# --- Analyze each detected signal ---
for i, peak in enumerate(peaks):
    freq_mhz = freqs_mhz[peak]
    power = psd_db[peak]
    prominence = props["prominences"][i]

    # Bandwidth
    bw_hz, left, right = estimate_bandwidth(freqs_hz, psd_db, peak, noise_floor)

    # Extract the signal's IQ for modulation analysis
    bin_width = freqs_hz[1] - freqs_hz[0]
    sig_center_offset = freqs[peak]
    half_bw = bw_hz / 2
    # Filter: keep only this signal's bandwidth
    t = np.arange(len(iq)) / sample_rate
    sig_iq = iq * np.exp(-1j * 2 * np.pi * sig_center_offset * t)
    # Simple low-pass by decimation
    dec_factor = max(1, int(sample_rate / bw_hz / 2))
    sig_iq_filtered = sig_iq[::dec_factor][:10000]  # take a manageable chunk

    # Modulation features
    mod = estimate_modulation(sig_iq_filtered, sample_rate / dec_factor)

    # Classify
    sig_type, reasons = classify_signal(bw_hz, mod, freq_mhz)

    print(f"\n--- Signal {i+1} ---")
    print(f"  Frequency:    {freq_mhz:.3f} MHz")
    print(f"  Power:        {power:.1f} dB")
    print(f"  Prominence:   {prominence:.1f} dB above surroundings")
    print(f"  Bandwidth:    {bw_hz/1000:.1f} kHz")
    print(f"  Amp variation: {mod['amp_variation']:.3f}")
    print(f"  Freq deviation: {mod['freq_range_hz']/1000:.1f} kHz")
    print(f"  Classification: {sig_type}")
    print(f"  Reasoning:")
    for r in reasons:
        print(f"    - {r}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)