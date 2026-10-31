"""
Shared DSP for the model comparison: the pipeline's own feature extraction and
detection (no copies), plus synthetic generators with an explicit in-band SNR.

    features(iq, sample_rate, center_hz, bw_hz)
        the 11 ML features exactly as the pipeline computes them for a known
        signal slice: rf_scanner.extract_signal_iq band-pass, then
        rf_scanner.extract_ml_features, plus the dechirp score on the raw slice
    detect_regions(iq, sample_rate, center_freq)       = rf_scanner.detect_regions
    region_features(iq, sample_rate, center_freq, target, regions)
                                                       = rf_scanner.target_features
    make_signal(cls, rng)        clean signal (unit power while on) + center/bandwidth
    noise_for_snr(n, snr_db, bw_hz, rng)   white noise giving that in-band SNR
    snippet(cls, snr_db, rng)    signal + noise, ready for features()
    capture(cls, snr_db, rng)    a 0.5 s capture the signal spans (bursty classes keep their gaps),
                                 for end-to-end tests through detect_regions
    train_row / test_row / e2e_row   one dataset row each for compare_models.py
                                 (kept here so parallel workers only import this
                                 light module, not the ML libraries)

SNR_DEFINITION (in-band): SNR = P_signal / (N0 * B), where P_signal is the
signal's mean power while it is transmitting, N0 the noise power spectral
density and B the signal's occupied bandwidth. Noise is white over the full
2.4 MHz capture, so out-of-band noise does not count against the signal.
"""

import numpy as np

from lora_dsp import lora_packet, dechirp_score
from rf_scanner import (detect_regions, extract_ml_features, extract_signal_iq,
                        target_features as region_features)

SAMPLE_RATE = 2400000
CLASSES = ["FM_BROADCAST", "DRONE_TELEMETRY", "DRONE_VIDEO", "INTERFERENCE", "NOISE", "LORA"]
FEATURE_NAMES = [
    "bandwidth_hz", "peak_power_db", "snr_db", "spectral_flatness",
    "amp_variation", "amp_kurtosis", "freq_std_hz", "freq_range_hz",
    "duty_cycle", "num_spectral_peaks", "dechirp_score_db",
]
SNR_DEFINITION = ("In-band SNR = P_signal / (N0 x B): signal power while transmitting divided "
                  "by the noise power inside the signal's occupied bandwidth B (noise is white "
                  "over the full 2.4 MHz capture).")
MIN_DURATION, MAX_DURATION = 0.015, 0.5   # snippet lengths, as in train_rf_classifier.py
CAPTURE_S = 0.5                           # end-to-end capture length (one --stream block)

__all__ = ["features", "detect_regions", "region_features", "make_signal", "noise_for_snr",
           "snippet", "capture", "CLASSES", "FEATURE_NAMES", "SNR_DEFINITION", "SAMPLE_RATE"]


def features(iq, sample_rate, center_hz, bw_hz):
    """The pipeline's 11 features for a slice whose signal is at center_hz with bandwidth bw_hz."""
    band, _ = extract_signal_iq(iq, sample_rate, center_hz, bw_hz)
    return extract_ml_features(band, sample_rate) + [dechirp_score(iq, sample_rate, center_hz)["score_db"]]


def _bursts_mask(rng, n, duty):
    """1-3 contiguous on-periods covering about duty of n samples."""
    mask = np.zeros(n)
    k = int(rng.integers(1, 4))
    slot = n // k
    for i in range(k):
        on = int(duty * slot)
        start = i * slot + int(rng.integers(0, slot - on + 1))
        mask[start:start + on] = 1
    return mask


def make_signal(cls, rng, duration_s=None, sample_rate=SAMPLE_RATE):
    """
    One clean signal of class cls, unit power while on, random offset/phase/parameters.

    Returns (signal, center_hz, bw_hz). Classes mirror train_rf_classifier.py:
    FM broadcast, 2-FSK telemetry (SiK-like, bursty), analog video FM, CW
    interference, LoRa (SF 7-9, BW 125/250/500 kHz, 1-3 packets) and NOISE
    (no signal; its center/bandwidth are random and only used for features).
    """
    if duration_s is None:
        duration_s = rng.uniform(MIN_DURATION, MAX_DURATION)
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    offset = rng.uniform(-300e3, 300e3)

    if cls == "FM_BROADCAST":
        audio = sum(rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * rng.uniform(200, 15000) * t)
                    for _ in range(int(rng.integers(1, 6))))
        audio = audio / (np.max(np.abs(audio)) + 1e-12)
        dev = rng.uniform(40e3, 80e3)
        sig = np.exp(2j * np.pi * (offset * t + dev * np.cumsum(audio) / sample_rate))
        bw = 2 * (dev + 15e3)
    elif cls == "DRONE_TELEMETRY":
        shift = rng.uniform(3e3, 8e3)
        baud = int(rng.choice([9600, 19200, 57600]))
        spb = max(1, sample_rate // baud)
        bits = rng.integers(0, 2, n // spb + 1)
        f = offset + np.where(np.repeat(bits, spb)[:n] == 0, -shift / 2, shift / 2)
        sig = np.exp(2j * np.pi * np.cumsum(f) / sample_rate) * _bursts_mask(rng, n, rng.uniform(0.3, 0.8))
        bw = shift + 2 * baud
    elif cls == "DRONE_VIDEO":
        mod = rng.uniform(15e3, 30e3)
        dev = rng.uniform(100e3, 200e3)
        sig = np.exp(2j * np.pi * (offset * t + dev * np.cumsum(np.sin(2 * np.pi * mod * t)) / sample_rate))
        bw = 2 * (dev + mod)
    elif cls == "INTERFERENCE":
        drift = rng.uniform(-100, 100)
        sig = np.exp(2j * np.pi * (offset * t + drift * t * t))
        bw = 3e3
    elif cls == "LORA":
        sf = int(rng.integers(7, 10))
        lbw = float(rng.choice([125e3, 250e3, 500e3]))
        sig = np.zeros(n, dtype=complex)
        k = int(rng.integers(1, 4))
        slot = n // k
        for i in range(k):
            packet = lora_packet(rng, sf, lbw, rng.uniform(0.6, 1.0) * slot / sample_rate, sample_rate)
            start = i * slot + int(rng.integers(0, slot - len(packet) + 1))
            sig[start:start + len(packet)] = packet
        sig *= np.exp(2j * np.pi * offset * t)
        bw = lbw
    elif cls == "NOISE":
        sig = np.zeros(n, dtype=complex)
        bw = rng.uniform(50e3, 200e3)
    else:
        raise ValueError(cls)
    return sig * np.exp(1j * rng.uniform(0, 2 * np.pi)), offset, bw


def noise_for_snr(n, snr_db, bw_hz, rng, sample_rate=SAMPLE_RATE):
    """White complex noise over the full band such that a unit-power signal in bw_hz has snr_db in-band."""
    n0 = 1.0 / (10 ** (snr_db / 10) * bw_hz)
    sigma = np.sqrt(n0 * sample_rate / 2)
    return sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def snippet(cls, snr_db, rng):
    """(iq, center_hz, bw_hz): signal + noise at snr_db in-band, random amplitude (+-20 dB)."""
    sig, center, bw = make_signal(cls, rng)
    gain = 10 ** (rng.uniform(-20, 20) / 20)
    return gain * (sig + noise_for_snr(len(sig), snr_db, bw, rng)), center, bw


def capture(cls, snr_db, rng, capture_s=CAPTURE_S, sample_rate=SAMPLE_RATE):
    """
    (iq, center_hz, bw_hz): a capture_s capture as the receiver would see it. The
    signal spans the whole capture: continuous classes (FM, video, CW) are on
    throughout, bursty classes (telemetry, LoRa) keep their own on/off pattern.
    """
    sig, center, bw = make_signal(cls, rng, capture_s)
    iq = sig + noise_for_snr(len(sig), snr_db, bw, rng)
    return 10 ** (rng.uniform(-20, 20) / 20) * iq, center, bw


def train_row(seed, ci, i):
    """Features of one training snippet (in-band SNR ~ U(0, 30) dB, slightly-off center/bandwidth)."""
    rng = np.random.default_rng([seed, 0, ci, i])
    iq, center, bw = snippet(CLASSES[ci], rng.uniform(0, 30), rng)
    # the pipeline's center/bandwidth estimates are a little off; mimic that
    return features(iq, SAMPLE_RATE, center + rng.normal(0, 2000), bw * rng.uniform(0.9, 1.3))


def test_row(seed, snr, ci, i):
    """Features of one test snippet at a fixed in-band SNR, from the true signal slice."""
    rng = np.random.default_rng([seed, 1, snr + 100, ci, i])
    iq, center, bw = snippet(CLASSES[ci], snr, rng)
    return features(iq, SAMPLE_RATE, center, bw)


def e2e_row(seed, snr, ci, i, center_freq=866e6):
    """
    One end-to-end capture through detect_regions. Signal classes: whether a target
    lies on the signal (within max(bw/2, 25 kHz)) and that target's features.
    NOISE: whether anything was detected and every target's features.
    """
    rng = np.random.default_rng([seed, 2, snr + 100, ci, i])
    cls = CLASSES[ci]
    iq, center, bw = capture(cls, snr, rng)
    regions = detect_regions(iq, SAMPLE_RATE, center_freq)
    targets = regions["targets"]
    if cls == "NOISE":
        return {"detected": bool(targets),
                "features": [region_features(iq, SAMPLE_RATE, center_freq, t, regions)[4] for t in targets]}
    near = [t for t in targets if abs(t["center_hz"] - center) <= max(bw / 2, 25e3)]
    if not near:
        return {"detected": False, "features": None}
    t = min(near, key=lambda t: abs(t["center_hz"] - center))
    return {"detected": True, "features": region_features(iq, SAMPLE_RATE, center_freq, t, regions)[4]}
