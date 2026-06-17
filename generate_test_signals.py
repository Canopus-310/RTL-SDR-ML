"""
Generate synthetic IQ recordings that mimic different signal types.
Used to test and demonstrate the classifier on drone-like signals.
"""
import numpy as np
from scipy.io import wavfile

sample_rate = 2400000
duration = 5  # seconds
t = np.arange(int(sample_rate * duration)) / sample_rate


def add_noise(signal, snr_db=15):
    noise = np.random.normal(0, 1, len(signal)) + 1j * np.random.normal(0, 1, len(signal))
    sig_power = np.mean(np.abs(signal) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    return signal + noise * np.sqrt(noise_power)


def save_iq(filename, iq, sample_rate):
    iq_scaled = np.column_stack([
        np.clip(iq.real * 32767, -32768, 32767).astype(np.int16),
        np.clip(iq.imag * 32767, -32768, 32767).astype(np.int16)
    ])
    wavfile.write(filename, sample_rate, iq_scaled)
    print(f"[OK] Saved {filename} ({len(iq)} samples, {duration}s)")


# --- 1. Simulated drone telemetry (FSK, narrow, bursty) ---
print("\nGenerating drone telemetry signal (433 MHz style)...")
drone_signal = np.zeros(len(t), dtype=complex)
# FSK: switches between two frequencies (common in SiK radios)
freq1, freq2 = 50000, 55000  # 50/55 kHz offset from center
bit_rate = 19200  # typical SiK radio baud rate
bits = np.random.randint(0, 2, int(duration * bit_rate))
samples_per_bit = sample_rate // bit_rate
for i, bit in enumerate(bits):
    start = i * samples_per_bit
    end = min(start + samples_per_bit, len(t))
    if end > len(t):
        break
    freq = freq1 if bit == 0 else freq2
    drone_signal[start:end] = 0.5 * np.exp(1j * 2 * np.pi * freq * t[start:end])

# Make it bursty: transmit for 200ms, silent for 300ms
burst_len = int(0.2 * sample_rate)
gap_len = int(0.3 * sample_rate)
cycle = burst_len + gap_len
mask = np.zeros(len(t))
pos = 0
while pos < len(t):
    mask[pos:pos + burst_len] = 1.0
    pos += cycle
drone_signal = drone_signal * mask
drone_signal = add_noise(drone_signal, snr_db=20)
save_iq("synthetic_drone_telemetry.wav", drone_signal, sample_rate)

# --- 2. Simulated FM broadcast (wide, continuous) ---
print("Generating FM broadcast signal...")
audio = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 3000 * t)
audio = audio / np.max(np.abs(audio))
fm_deviation = 75000  # standard FM broadcast deviation
phase = 2 * np.pi * fm_deviation * np.cumsum(audio) / sample_rate
fm_signal = np.exp(1j * phase)
fm_signal = add_noise(fm_signal, snr_db=25)
save_iq("synthetic_fm_broadcast.wav", fm_signal, sample_rate)

# --- 3. Simulated unknown/interference (continuous wave + noise) ---
print("Generating unknown CW interference signal...")
cw_signal = 0.3 * np.exp(1j * 2 * np.pi * 80000 * t)
cw_signal = add_noise(cw_signal, snr_db=12)
save_iq("synthetic_interference.wav", cw_signal, sample_rate)

print("\nDone! Now run feature_extractor.py on each file.")