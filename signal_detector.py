import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io import wavfile
from scipy.signal import welch, find_peaks

if len(sys.argv) < 2:
    print("Usage: python signal_detector.py <recording.wav> [center_freq_hz]")
    print("Example: python signal_detector.py recording.wav 103959385")
    sys.exit(1)

recording = sys.argv[1]
center_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 100e6

sample_rate, data = wavfile.read(recording)

iq = data[:, 0] + 1j * data[:, 1]
iq = iq / 32768.0
print(f"Loaded {len(iq)} IQ samples at {sample_rate} Hz")

# --- Compute power spectrum ---
freqs, psd = welch(iq, fs=sample_rate, nperseg=4096, return_onesided=False)
freqs = np.fft.fftshift(freqs)
psd = np.fft.fftshift(psd)
freqs_mhz = freqs / 1e6 + center_freq / 1e6
psd_db = 10 * np.log10(psd + 1e-20)

# --- Detect signals (peaks above noise floor) ---
noise_floor = np.median(psd_db)
threshold = noise_floor + 10   # signals must be 10 dB above noise
peaks, properties = find_peaks(psd_db, height=threshold, distance=50, prominence=5)

print(f"\nNoise floor: {noise_floor:.1f} dB")
print(f"Detection threshold: {threshold:.1f} dB")
print(f"Signals detected: {len(peaks)}")
print("-" * 50)

for i, peak in enumerate(peaks):
    freq = freqs_mhz[peak]
    power = psd_db[peak]
    prominence = properties.get("prominences", [0])[i] if "prominences" in properties else 0
    print(f"  Signal {i+1}: {freq:.3f} MHz  |  power: {power:.1f} dB  |  prominence: {prominence:.1f} dB")

# --- Plot with detections marked ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Full spectrum with detections
ax1.plot(freqs_mhz, psd_db, linewidth=0.5)
ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold:.0f} dB)')
ax1.axhline(y=noise_floor, color='orange', linestyle=':', alpha=0.7, label=f'Noise floor ({noise_floor:.0f} dB)')
ax1.plot(freqs_mhz[peaks], psd_db[peaks], 'rv', markersize=10, label='Detected signals')
ax1.set_xlabel("Frequency (MHz)")
ax1.set_ylabel("Power (dB)")
ax1.set_title("Spectrum with Auto-Detected Signals")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Waterfall-style spectrogram
chunk_size = 4096
num_chunks = min(len(iq) // chunk_size, 500)
spectrogram = np.zeros((num_chunks, chunk_size))
for i in range(num_chunks):
    chunk = iq[i * chunk_size:(i + 1) * chunk_size]
    spec = np.fft.fftshift(np.abs(np.fft.fft(chunk)) ** 2)
    spectrogram[i] = 10 * np.log10(spec + 1e-20)

extent = [freqs_mhz[0], freqs_mhz[-1], num_chunks * chunk_size / sample_rate, 0]
ax2.imshow(spectrogram, aspect='auto', extent=extent, cmap='viridis')
ax2.set_xlabel("Frequency (MHz)")
ax2.set_ylabel("Time (seconds)")
ax2.set_title("Spectrogram (Waterfall)")

plt.tight_layout()
plt.show()