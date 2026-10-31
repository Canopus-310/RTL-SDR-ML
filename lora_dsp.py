"""
LoRa signal generation and the dechirp score.

    lora_packet()     baseband LoRa packet: preamble upchirps, 2 sync downchirps,
                      then cyclically shifted payload upchirps
    dechirp_score()   how LoRa-like a slice of IQ is: dechirp with each candidate
                      (SF, BW) reference upchirp, FFT, best peak-to-average ratio

Run directly to print dechirp scores for synthetic LoRa / FM / FSK / noise at
20, 10, 5 and 0 dB SNR (the table LORA_THRESHOLD_DB was chosen from):

    python lora_dsp.py
"""

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 2400000
CANDIDATE_SFS = (7, 8, 9, 10)
CANDIDATE_BWS = (125e3, 250e3, 500e3)
MAX_WINDOWS = 64          # dechirp windows per candidate, spread evenly over the slice
WINDOW_PERCENTILE = 50    # score = median over windows (higher percentiles let FM sweeps through)
EULER_GAMMA = 0.5772
SCORE_FLOOR_DB = -10.0    # returned when the slice is too short for any candidate
# Rule-based "LoRa: yes" threshold, chosen from the table printed by main():
# the highest non-LoRa score is 2.9 dB (FM at 20 dB SNR) and the lowest LoRa
# score is 10.3 dB (SF7/500k at 0 dB SNR); 6 dB leaves ~3-4 dB margin each way.
LORA_THRESHOLD_DB = 6.0


def lora_packet(rng, sf, bw, airtime_s, sample_rate=SAMPLE_RATE, preamble=12):
    """
    One baseband LoRa packet of airtime_s seconds, centered at 0 Hz.

    Symbols last exactly 2**sf / bw seconds (timing is computed per sample, so
    non-integer samples per symbol don't drift). An upchirp with value v starts
    at -bw/2 + v/2**sf * bw and sweeps up, wrapping at +bw/2; the preamble is
    value 0, followed by 2 downchirps and random payload values.
    """
    ts = 2 ** sf / bw
    t = np.arange(int(round(airtime_s * sample_rate))) / sample_rate
    k = (t // ts).astype(int)
    tau = t - k * ts
    n_sym = k[-1] + 1
    values = np.zeros(n_sym)
    values[preamble + 2:] = rng.integers(0, 2 ** sf, max(0, n_sym - preamble - 2))
    down = np.zeros(n_sym, dtype=bool)
    down[preamble:preamble + 2] = True
    f_up = -bw / 2 + (values[k] / 2 ** sf * bw + bw * tau / ts) % bw
    f = np.where(down[k], bw / 2 - bw * tau / ts, f_up)
    return np.exp(2j * np.pi * np.cumsum(f) / sample_rate)


def dechirp_score(iq, sample_rate, center_hz=0.0):
    """
    Dechirp score in dB for a slice of IQ whose signal is centered at center_hz.

    For each candidate bandwidth the slice is shifted to 0 Hz and resampled to
    BW (one sample per chip, as a LoRa receiver does). For each SF it is cut
    into 2**SF-sample windows (up to MAX_WINDOWS, spread evenly over the slice),
    multiplied by the conjugate reference upchirp and FFT'd: a LoRa symbol
    collapses to a single tone, anything else stays spread. The window's
    peak-to-average power ratio is compared with what noise gives for that FFT
    size (ln N + 0.577), so noise scores ~0 dB for every candidate. The score is
    the WINDOW_PERCENTILE-th percentile over windows (so gaps between packets
    don't hide them), best over candidates.

    Returns {"score_db", "sf", "bw_hz"} (SCORE_FLOOR_DB if the slice is too short).
    """
    n = np.arange(len(iq))
    x = iq * np.exp(-2j * np.pi * center_hz * n / sample_rate)
    best = {"score_db": SCORE_FLOOR_DB, "sf": None, "bw_hz": None}
    for bw in CANDIDATE_BWS:
        ratio = Fraction(int(bw), int(sample_rate)).limit_denominator(1000)
        y = resample_poly(x, ratio.numerator, ratio.denominator)
        for sf in CANDIDATE_SFS:
            N = 2 ** sf
            available = len(y) // N
            if available == 0:
                continue
            picks = np.unique(np.linspace(0, available - 1, min(available, MAX_WINDOWS)).astype(int))
            windows = y[:available * N].reshape(available, N)[picks]
            k = np.arange(N)
            ref = np.exp(1j * np.pi * (k ** 2 / N - k))
            power = np.abs(np.fft.fft(windows * np.conj(ref), axis=1)) ** 2
            par = power.max(axis=1) / (power.mean(axis=1) + 1e-30)
            score = (10 * np.log10(np.percentile(par, WINDOW_PERCENTILE))
                     - 10 * np.log10(np.log(N) + EULER_GAMMA))
            if score > best["score_db"]:
                best = {"score_db": float(score), "sf": sf, "bw_hz": bw}
    return best


def _with_noise(rng, signal, snr_db, bandwidth_hz, sample_rate=SAMPLE_RATE):
    """Add white noise so the in-band SNR (signal power vs noise in bandwidth_hz) is snr_db."""
    noise_psd = np.mean(np.abs(signal) ** 2) / 10 ** (snr_db / 10) / bandwidth_hz
    sigma = np.sqrt(noise_psd * sample_rate / 2)
    return signal + sigma * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))


def _test_signals(rng, duration_s=0.05, sample_rate=SAMPLE_RATE):
    """(name, iq, occupied bandwidth) for the comparison table, all centered at 0 Hz."""
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    audio = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 3000 * t)
    fm = np.exp(2j * np.pi * 75e3 * np.cumsum(audio / np.max(np.abs(audio))) / sample_rate)
    spb = sample_rate // 19200
    bits = rng.integers(0, 2, len(t) // spb + 1)
    fsk_f = np.where(np.repeat(bits, spb)[:len(t)] == 0, -2.5e3, 2.5e3)
    fsk = np.exp(2j * np.pi * np.cumsum(fsk_f) / sample_rate)
    return [
        ("LoRa SF7/500k (15 ms)", lora_packet(rng, 7, 500e3, 0.015), 500e3),
        ("LoRa SF9/125k (50 ms)", lora_packet(rng, 9, 125e3, duration_s), 125e3),
        ("FM, 75 kHz dev (50 ms)", fm, 180e3),
        ("2-FSK 19.2k (50 ms)", fsk, 25e3),
        ("Noise only (50 ms)", np.zeros(len(t), dtype=complex), None),
    ]


def main():
    rng = np.random.default_rng(1)
    snrs = (20, 10, 5, 0)
    trials = 10
    print("Dechirp score (dB over the noise expectation), mean [min..max] of "
          f"{trials} trials; SNR is in-band; candidates SF {CANDIDATE_SFS}, "
          f"BW {tuple(int(b/1e3) for b in CANDIDATE_BWS)} kHz")
    print(f"{'Signal':<24}" + "".join(f"{f'{s} dB SNR':>20}" for s in snrs))
    for name, _, _ in _test_signals(rng):
        cells = []
        for snr in snrs:
            scores = []
            for _ in range(trials):
                sig = {n: (s, b) for n, s, b in _test_signals(rng)}[name]
                iq = (_with_noise(rng, sig[0], snr, sig[1]) if sig[1] else
                      _with_noise(rng, np.ones(len(sig[0])), 0, SAMPLE_RATE) - 1)
                scores.append(dechirp_score(iq, SAMPLE_RATE)["score_db"])
            cells.append(f"{np.mean(scores):6.1f} [{min(scores):4.1f}..{max(scores):4.1f}]")
        print(f"{name:<24}" + "".join(f"{c:>20}" for c in cells))
    print(f"\nRule: LoRa if score >= {LORA_THRESHOLD_DB} dB")


if __name__ == "__main__":
    main()
