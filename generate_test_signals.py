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
    print(f"[OK] Saved {filename} ({len(iq)} samples, {len(iq) / sample_rate:g}s)")


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


# --- 4. Simulated LoRa (868 MHz band style), receiver tuned to 866.0 MHz ---
# lora_dsp.lora_packet builds each packet from LoRa chirps: preamble upchirps,
# 2 sync downchirps, then cyclically shifted payload upchirps (exact symbol timing).
from lora_dsp import lora_packet

rng = np.random.default_rng(866)


def place_packet(sig, packet, start_s, offset_hz):
    """Write a baseband packet into sig at start_s, shifted to offset_hz from the tuned center."""
    i0 = int(start_s * sample_rate)
    seg = packet[:len(sig) - i0]
    ts = np.arange(i0, i0 + len(seg)) / sample_rate
    sig[i0:i0 + len(seg)] = seg * np.exp(2j * np.pi * offset_hz * ts)


def add_dongle_noise(sig, inband_snr_db, amplitude, bw):
    """Noise for the given in-band SNR during a packet, plus a small DC offset like a real dongle."""
    noise_power = (amplitude ** 2 * sample_rate / bw) / 10 ** (inband_snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(sig)) + 1j * rng.standard_normal(len(sig)))
    return sig + noise + 0.01 + 0.008j


# 4a. Fixed-frequency beacon: SF9 / 125 kHz at 866.0 MHz, ~170 ms packets about once a second
print("Generating fixed-frequency LoRa beacon (SF9/125 kHz at 866.0 MHz)...")
lora_fixed = np.zeros(len(t), dtype=complex)
start = 0.3
while start + 0.17 < duration:
    place_packet(lora_fixed, 0.3 * lora_packet(rng, 9, 125e3, 0.17, sample_rate), start, 0.0)
    start += 1.0 + rng.uniform(-0.05, 0.05)
save_iq("synthetic_lora_fixed.wav", add_dongle_noise(lora_fixed, 20, 0.3, 125e3), sample_rate)

# 4b. Frequency-hopping link: 4 channels at 865.25 / 865.75 / 866.25 / 866.75 MHz,
# SF7 / 500 kHz, ~15 ms packets, ~10 hops/s, random channel, never the same twice in a row
print("Generating frequency-hopping LoRa signal (4 ch, SF7/500 kHz, 15 ms hops)...")
fhss_channels = [-750e3, -250e3, 250e3, 750e3]  # offsets from 866.0 MHz
hop_rate = 10           # packets (hops) per second
packet_s = 0.015
fhss_signal = np.zeros(len(t), dtype=complex)
channel = None
start = 0.02
while start + packet_s < duration:
    channel = rng.choice([c for c in range(len(fhss_channels)) if c != channel])
    place_packet(fhss_signal, 0.3 * lora_packet(rng, 7, 500e3, packet_s, sample_rate),
                 start, fhss_channels[channel])
    start += 1 / hop_rate + rng.uniform(-0.01, 0.01)
save_iq("synthetic_fhss.wav", add_dongle_noise(fhss_signal, 15, 0.3, 500e3), sample_rate)

# 4c. Noise only (same noise level and DC offset as the hopping file)
print("Generating noise-only capture...")
save_iq("synthetic_noise.wav", add_dongle_noise(np.zeros(len(t), dtype=complex), 15, 0.3, 500e3),
        sample_rate)

# 4d. LoRaWAN-like IoT device (smart meter / sensor): SF9 / 125 kHz, one ~185 ms
# uplink every few seconds on one of the 3 default India (IN865) channels
# 865.0625 / 865.4025 / 865.985 MHz. 10 s long, so it holds a few packets.
print("Generating LoRaWAN-like IoT signal (SF9/125 kHz, 3 fixed channels, 1 packet / ~3 s)...")
iot_duration = 10
lorawan_channels = [865.0625e6 - 866e6, 865.4025e6 - 866e6, 865.985e6 - 866e6]
lorawan = np.zeros(int(iot_duration * sample_rate), dtype=complex)
start = 0.5
while start + 0.185 < iot_duration:
    place_packet(lorawan, 0.3 * lora_packet(rng, 9, 125e3, 0.185, sample_rate),
                 start, lorawan_channels[rng.integers(len(lorawan_channels))])
    start += rng.uniform(2.5, 3.5)
save_iq("synthetic_lorawan.wav", add_dongle_noise(lorawan, 20, 0.3, 125e3), sample_rate)

print("\nDone! Now run feature_extractor.py on each file.")
print("Analyze synthetic_lora_fixed / synthetic_fhss / synthetic_noise / synthetic_lorawan at "
      "866000000 Hz; the others at 433900000 Hz.")
