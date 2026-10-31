"""
Short-time burst detection and FHSS grouping.

rf_scanner.detect_signals averages power over the whole capture (Welch), so a
short burst is diluted by its duty cycle: a 15 ms hop loses ~21 dB in a 2 s
scan. This module looks at the capture in ~4 ms slices instead:

    short_time_spectrum()  slice PSDs, per-bin noise, max-hold and threshold
    detect_bursts()        connected time-frequency regions above the threshold
    refine_burst_timing()  burst start/duration at single-FFT (~0.2 ms) resolution
    group_channels()       bursts grouped by center frequency, with duty cycle
    find_fhss_emitters()   channels that hop as one transmitter

Frequencies are offsets from the tuned center in Hz; times are seconds from
the start of the IQ array.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import get_window
from scipy.stats import gamma

NFFT = 512                   # 4.7 kHz bins at 2.4 MS/s
SLICE_SECONDS = 0.004        # each slice averages ~19 FFTs, so its noise is stable
CHUNK_SLICES = 128           # slices per FFT batch (bounds memory on long captures)
BURST_PFA = 1e-6             # false-alarm probability per bin per slice, from noise statistics
MIN_THRESHOLD_DB = 6.0       # floor for real dongles: spurs, 8-bit quantization, impulses
MIN_BURST_BINS = 3           # a burst must span >= 3 adjacent bins (~14 kHz)
PERSISTENT_FRACTION = 0.5    # narrow lines above threshold this often (DC spike, spurs) are removed
LINE_MAX_BINS = 5            # "narrow": a strong tone spreads over ~3-5 bins with the Hann window
GAP_BINS = LINE_MAX_BINS + 1  # gaps up to this many bins inside a burst are bridged
ARTIFACT_DB = 20.0           # weaker bursts only present during a burst this much stronger are its spurs
EDGE_MIN_SLICES = 2          # bursts cut by the capture start/end need this many slices to be measured
BURST_BW_DROP_DB = 6.0       # burst bandwidth: width within this many dB of the burst's peak
REFINE_MAX_S = 0.5           # bursts longer than this keep slice timing (continuous signals)
REFINE_MIN_SNR_DB = 10.0     # weaker bursts keep slice timing (per-FFT power too noisy)
CHANNEL_TOLERANCE_HZ = 25e3  # bursts this close (or within 1/4 bandwidth) share a channel
PRESENCE_MARGIN_HZ = 100e3   # presence is measured over the channel's extent widened by this
PRESENCE_DB = 1.0            # channel counts as "on" when that band's total power is this far over noise
CONTINUOUS_DUTY = 0.5        # channels on at least this fraction of slices are continuous
MAX_CENTER_SPREAD = 0.15     # a transmitter's bursts keep their center within this x bandwidth (std)
FHSS_MIN_CHANNELS = 3        # fewer channels can't be told apart from independent transmitters
FHSS_BW_TOLERANCE = 0.3      # hop channels have bandwidths within +-30% of each other
FHSS_MIN_ALTERNATION = 0.6   # fraction of consecutive bursts that change channel
FHSS_MAX_OVERLAP = 0.2       # fraction of bursts overlapping another channel's burst in time


def short_time_spectrum(iq, sample_rate):
    """
    PSD of consecutive ~4 ms slices, plus the noise reference and threshold.

    Each slice averages k non-overlapping Hann-windowed FFTs with the same
    density scaling as scipy's welch, so slice dB values compare directly with
    the Welch PSD in rf_scanner. The per-bin noise is the bin's median over
    time (bursts are on less than half the time), capped at the band-wide
    median so continuous signals stay visible above it. The threshold is set
    for BURST_PFA from the gamma distribution of a k-FFT average of noise.
    """
    k = max(1, int(round(SLICE_SECONDS * sample_rate / NFFT)))
    n_slices = len(iq) // (k * NFFT)
    window = get_window("hann", NFFT)
    scale = 1.0 / (sample_rate * np.sum(window ** 2))

    power = np.empty((n_slices, NFFT))
    for first in range(0, n_slices, CHUNK_SLICES):
        last = min(first + CHUNK_SLICES, n_slices)
        segs = iq[first * k * NFFT:last * k * NFFT].reshape(-1, NFFT) * window
        spec = np.abs(np.fft.fft(segs, axis=1)) ** 2 * scale
        power[first:last] = np.fft.fftshift(spec.reshape(last - first, k, NFFT).mean(axis=1), axes=1)

    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / sample_rate))
    bin_median = np.median(power, axis=0) if n_slices else np.ones(NFFT)
    noise = np.minimum(bin_median, np.median(bin_median))

    noise_median = gamma.median(k, scale=1 / k)
    stat_db = 10 * np.log10(gamma.isf(BURST_PFA, k, scale=1 / k) / noise_median)

    return {
        "freqs": freqs,
        "bin_hz": sample_rate / NFFT,
        "slice_s": k * NFFT / sample_rate,
        "power": power,
        "noise": noise,
        "maxhold_db": 10 * np.log10(power.max(axis=0) + 1e-20) if n_slices else None,
        "threshold_db": max(stat_db, MIN_THRESHOLD_DB),
        "statistical_threshold_db": stat_db,
    }


def detect_bursts(spectrum):
    """
    Find bursts: connected regions of (slice, bin) cells above the threshold.

    Narrow lines that are on most of the time (the DC spike, spurs) are removed
    first and recorded in spectrum["line_bins"], then gaps of up to GAP_BINS
    inside a slice are bridged so a signal straddling a removed line stays one
    burst. Each burst reports its start time, duration, center frequency,
    bandwidth (-BURST_BW_DROP_DB width of its average spectrum), frequency
    extent, peak PSD (dB) and peak SNR over noise. Weak bursts that only occur
    while a burst ARTIFACT_DB stronger is on (keying splatter, spurs from a
    strong nearby transmitter) are dropped and counted in spectrum["artifacts"].
    So are slivers shorter than EDGE_MIN_SLICES cut off by the start or end of
    the capture: a few ms of a chirp covers only part of its band, so their
    center and bandwidth are wrong (the full burst is in the neighbouring block).
    """
    power, noise, freqs = spectrum["power"], spectrum["noise"], spectrum["freqs"]
    spectrum["line_bins"] = np.zeros(len(freqs), dtype=bool)
    spectrum["artifacts"] = 0
    if len(power) == 0:
        return []
    snr_db = 10 * np.log10(power / noise)
    mask = snr_db > spectrum["threshold_db"]

    persistent = mask.mean(axis=0) > PERSISTENT_FRACTION
    lines, n_lines = ndimage.label(persistent)
    for run in ndimage.find_objects(lines):
        if run[0].stop - run[0].start <= LINE_MAX_BINS:
            mask[:, run[0]] = False
            spectrum["line_bins"][run[0]] = True

    mask |= ndimage.binary_closing(mask, structure=np.ones((1, GAP_BINS + 1), dtype=bool))

    labels, _ = ndimage.label(mask)
    bin_hz, slice_s = spectrum["bin_hz"], spectrum["slice_s"]
    bursts = []
    for label, (t_sl, f_sl) in enumerate(ndimage.find_objects(labels), start=1):
        region = labels[t_sl, f_sl] == label
        if region.sum(axis=1).max() < MIN_BURST_BINS:
            continue
        cut = t_sl.start == 0 or t_sl.stop == len(power)
        if cut and t_sl.stop - t_sl.start < EDGE_MIN_SLICES:
            spectrum["artifacts"] += 1
            continue
        avg_db = 10 * np.log10(power[t_sl, f_sl].mean(axis=0))
        top = int(np.argmax(avg_db))
        lo = hi = top
        while lo > 0 and avg_db[lo - 1] >= avg_db[top] - BURST_BW_DROP_DB:
            lo -= 1
        while hi < len(avg_db) - 1 and avg_db[hi + 1] >= avg_db[top] - BURST_BW_DROP_DB:
            hi += 1
        bursts.append({
            "start_s": t_sl.start * slice_s,
            "duration_s": (t_sl.stop - t_sl.start) * slice_s,
            "center_hz": (freqs[f_sl.start + lo] + freqs[f_sl.start + hi]) / 2,
            "bandwidth_hz": (hi - lo + 1) * bin_hz,
            "f_lo_hz": freqs[f_sl.start] - bin_hz / 2,
            "f_hi_hz": freqs[f_sl.stop - 1] + bin_hz / 2,
            "peak_db": float(10 * np.log10(power[t_sl, f_sl][region].max())),
            "snr_db": float(snr_db[t_sl, f_sl][region].max()),
        })

    def is_artifact(b):
        return any(s["peak_db"] - b["peak_db"] >= ARTIFACT_DB
                   and s["start_s"] - slice_s <= b["start_s"]
                   and b["start_s"] + b["duration_s"] <= s["start_s"] + s["duration_s"] + slice_s
                   for s in bursts)

    kept = [b for b in bursts if not is_artifact(b)]
    spectrum["artifacts"] += len(bursts) - len(kept)
    return kept


def refine_burst_timing(bursts, iq, sample_rate, spectrum):
    """
    Re-measure each short, strong burst's start and duration from single FFTs.

    Slice timing over-reads by up to a slice at each end, because a partly
    filled slice still crosses the threshold. Here the burst's band power is
    taken from each NFFT-sample FFT (~0.2 ms) around the burst, and the burst
    runs from the first to the last FFT within BURST_BW_DROP_DB of its plateau
    (the median of its strongest half). Long or weak bursts keep slice timing.
    """
    window = get_window("hann", NFFT)
    freqs, slice_s = spectrum["freqs"], spectrum["slice_s"]
    for b in bursts:
        core = (np.abs(freqs - b["center_hz"]) <= b["bandwidth_hz"] / 2) & ~spectrum["line_bins"]
        if (b["duration_s"] > REFINE_MAX_S or b["snr_db"] < REFINE_MIN_SNR_DB
                or core.sum() < MIN_BURST_BINS):
            continue
        first = max(0, int((b["start_s"] - slice_s) * sample_rate) // NFFT * NFFT)
        last = min(len(iq), int((b["start_s"] + b["duration_s"] + slice_s) * sample_rate))
        n_fft = (last - first) // NFFT
        segs = iq[first:first + n_fft * NFFT].reshape(n_fft, NFFT) * window
        band = (np.abs(np.fft.fftshift(np.fft.fft(segs, axis=1), axes=1)) ** 2)[:, core].sum(axis=1)
        plateau = np.median(np.sort(band)[len(band) // 2:])
        on = np.flatnonzero(band >= plateau / 10 ** (BURST_BW_DROP_DB / 10))
        if len(on):
            b["start_s"] = (first + on[0] * NFFT) / sample_rate
            b["duration_s"] = (on[-1] - on[0] + 1) * NFFT / sample_rate
    return bursts


def group_channels(bursts, spectrum):
    """
    Group bursts into channels by center frequency.

    A channel's duty cycle is the fraction of slices in which the total power
    over its frequency extent, widened by PRESENCE_MARGIN_HZ and excluding
    always-on lines like the DC spike, is PRESENCE_DB over the noise total.
    Summing many bins makes this far steadier than any single bin, so a
    continuous signal whose energy wanders (an FM station's modulation, which
    only lights part of its band at a time) still reads as continuous.

    A channel is "stable" when its bursts' centers have a standard deviation
    of at most MAX_CENTER_SPREAD x its bandwidth. A real transmitter's channel
    stays put; energy wandering inside a weak FM station's band does not.
    """
    channels = []
    for b in sorted(bursts, key=lambda b: b["center_hz"]):
        ch = channels[-1] if channels else None
        if ch is not None and abs(b["center_hz"] - ch["center_hz"]) <= max(
                CHANNEL_TOLERANCE_HZ, 0.25 * max(b["bandwidth_hz"], ch["bandwidth_hz"])):
            ch["bursts"].append(b)
            ch["center_hz"] = float(np.median([x["center_hz"] for x in ch["bursts"]]))
            ch["bandwidth_hz"] = float(np.median([x["bandwidth_hz"] for x in ch["bursts"]]))
        else:
            channels.append({"bursts": [b], "center_hz": b["center_hz"],
                             "bandwidth_hz": b["bandwidth_hz"]})

    freqs, line_bins = spectrum["freqs"], spectrum["line_bins"]
    for ch in channels:
        ch["bursts"].sort(key=lambda b: b["start_s"])
        ch["f_lo_hz"] = min(b["f_lo_hz"] for b in ch["bursts"])
        ch["f_hi_hz"] = max(b["f_hi_hz"] for b in ch["bursts"])
        band = ((freqs >= ch["f_lo_hz"] - PRESENCE_MARGIN_HZ)
                & (freqs <= ch["f_hi_hz"] + PRESENCE_MARGIN_HZ) & ~line_bins)
        excess_db = 10 * np.log10(spectrum["power"][:, band].sum(axis=1) / spectrum["noise"][band].sum())
        ch["duty"] = float((excess_db > PRESENCE_DB).mean())
        ch["continuous"] = ch["duty"] >= CONTINUOUS_DUTY
        centers = [b["center_hz"] for b in ch["bursts"]]
        ch["stable"] = bool(np.std(centers) <= MAX_CENTER_SPREAD * ch["bandwidth_hz"])
        ch["peak_db"] = max(b["peak_db"] for b in ch["bursts"])
        ch["snr_db"] = max(b["snr_db"] for b in ch["bursts"])
        ch["strongest"] = max(ch["bursts"], key=lambda b: b["peak_db"])
    return channels


def find_fhss_emitters(channels):
    """
    Group intermittent channels that behave like one frequency-hopping transmitter.

    Only intermittent, stable channels are considered. Channels qualify
    together when there are at least FHSS_MIN_CHANNELS of them,
    their bandwidths are within FHSS_BW_TOLERANCE, consecutive bursts mostly
    change channel (one transmitter can only hop), and bursts on different
    channels rarely overlap in time (one transmitter is on one channel at a
    time). Each emitter reports its channels, hop rate, dwell and hop interval.
    """
    candidates = [ch for ch in channels if not ch["continuous"] and ch["stable"]]
    used = set()
    emitters = []
    for seed in sorted(candidates, key=lambda ch: -len(ch["bursts"])):
        if id(seed) in used:
            continue
        group = [ch for ch in candidates if id(ch) not in used and
                 abs(ch["bandwidth_hz"] / seed["bandwidth_hz"] - 1) <= FHSS_BW_TOLERANCE]
        if len(group) < FHSS_MIN_CHANNELS:
            continue

        hops = sorted(((b, i) for i, ch in enumerate(group) for b in ch["bursts"]),
                      key=lambda hop: hop[0]["start_s"])
        if len(hops) < FHSS_MIN_CHANNELS + 1:
            continue
        alternation = np.mean([a[1] != b[1] for a, b in zip(hops, hops[1:])])
        overlapping = sum(
            any(j != i and b["start_s"] < o["start_s"] + o["duration_s"]
                and o["start_s"] < b["start_s"] + b["duration_s"] for o, j in hops)
            for b, i in hops)
        if alternation < FHSS_MIN_ALTERNATION or overlapping / len(hops) > FHSS_MAX_OVERLAP:
            continue

        starts = np.array([b["start_s"] for b, _ in hops])
        span = starts[-1] - starts[0]
        group.sort(key=lambda ch: ch["center_hz"])
        emitters.append({
            "channels": group,
            "bursts": [b for b, _ in hops],
            "center_hz": float(np.mean([ch["center_hz"] for ch in group])),
            "bandwidth_hz": float(np.median([ch["bandwidth_hz"] for ch in group])),
            "peak_db": max(ch["peak_db"] for ch in group),
            "snr_db": max(ch["snr_db"] for ch in group),
            "strongest": max((ch["strongest"] for ch in group), key=lambda b: b["peak_db"]),
            "hop_rate_hz": (len(hops) - 1) / span if span > 0 else 0.0,
            "dwell_s": float(np.median([b["duration_s"] for b, _ in hops])),
            "hop_interval_s": float(np.median(np.diff(starts))),
            "alternation": float(alternation),
        })
        used.update(id(ch) for ch in group)
    return emitters
