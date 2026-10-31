"""
RF Drone Threat Scanner — Main Entry Point

Chains the full pipeline: detect -> classify -> fingerprint -> threat assess -> report.

Usage:
    python rf_scanner.py <recording.wav> <center_freq_hz>

Examples:
    python rf_scanner.py recording.wav 433900000
    python rf_scanner.py synthetic_drone_telemetry.wav 433900000
"""

import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch, find_peaks
import joblib
import json

from drone_fingerprints import DroneFingerprinter
from threat_engine import ThreatEngine
from scan_report import ScanReport
from stage_timer import StageTimer
from burst_detector import (short_time_spectrum, detect_bursts, refine_burst_timing,
                            group_channels, find_fhss_emitters, CONTINUOUS_DUTY)
from lora_dsp import dechirp_score, LORA_THRESHOLD_DB

# Burst detection uses a 6 dB threshold; a burst channel with no Welch peak and no
# FHSS grouping to back it up is only reported if it reaches this SNR.
BURST_ONLY_MIN_SNR_DB = 12.0


def load_recording(path):
    """Load IQ recording from WAV file."""
    sample_rate, data = wavfile.read(path)
    if data.ndim == 2:
        iq = data[:, 0] + 1j * data[:, 1]
    else:
        iq = data.astype(np.float64)
    iq = iq / 32768.0
    return iq, sample_rate


def detect_signals(iq, sample_rate, center_freq):
    """Detect signals in the spectrum. Returns freqs, psd, peaks, properties."""
    freqs, psd = welch(iq, fs=sample_rate, nperseg=4096, return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    freqs_hz = freqs + center_freq
    freqs_mhz = freqs_hz / 1e6
    psd_db = 10 * np.log10(psd + 1e-20)

    noise_floor = np.median(psd_db)
    threshold = noise_floor + 10
    peaks, props = find_peaks(psd_db, height=threshold, distance=50, prominence=5)

    return freqs, freqs_hz, freqs_mhz, psd_db, noise_floor, threshold, peaks, props


def extract_signal_iq(iq, sample_rate, center_hz, bw_hz, window_s=None):
    """
    Extract a signal's IQ using an FFT bandpass filter around center_hz (offset from tune).

    window_s is the (start, stop) time in seconds to take the IQ from, capped at
    0.5 s; by default the first 0.5 s of the capture. Returns (iq, (start_s, stop_s)).
    """
    max_len = int(0.5 * sample_rate)
    if window_s is None:
        start, stop = 0, max_len
    else:
        start = int(window_s[0] * sample_rate)
        stop = min(int(window_s[1] * sample_rate), start + max_len)
    chunk = iq[start:stop]
    fft_iq = np.fft.fft(chunk)
    fft_freqs = np.fft.fftfreq(len(fft_iq), 1 / sample_rate)
    half_bw = max(bw_hz, 50000) / 2
    mask = np.abs(fft_freqs - center_hz) < half_bw
    return np.fft.ifft(fft_iq * mask), (start / sample_rate, (start + len(chunk)) / sample_rate)


def estimate_bandwidth(freqs_hz, psd_db, peak_idx, noise_floor):
    """Estimate signal bandwidth."""
    peak_power = psd_db[peak_idx]
    bw_threshold = max(peak_power - 20, noise_floor + 3)
    left = peak_idx
    while left > 0 and psd_db[left] > bw_threshold:
        left -= 1
    right = peak_idx
    while right < len(psd_db) - 1 and psd_db[right] > bw_threshold:
        right += 1
    return freqs_hz[right] - freqs_hz[left], left, right


def extract_ml_features(iq, sample_rate):
    """Extract the 10 features used by the ML model."""
    freqs, psd = welch(iq, fs=sample_rate, nperseg=min(1024, len(iq)),
                       return_onesided=False)
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


def plan_targets(freqs, psd_db, peaks, channels, emitters, spectrum):
    """
    Decide what gets analyzed: one target per reported signal, sorted by frequency.

      "welch"        a Welch peak outside any intermittent channel: analyzed as
                     before, with IQ from the first 0.5 s
      "welch+burst"  Welch peaks inside the same intermittent channel (e.g. one
                     LoRa beacon split in two by the DC notch): merged into one
                     target at the strongest peak, IQ from the channel's strongest burst
      "burst"        an intermittent channel the Welch average missed, if its
                     strongest burst is BURST_ONLY_MIN_SNR_DB over noise (a lone
                     weak burst is the likeliest false alarm, so it needs more margin)
      "fhss"         a frequency-hopping emitter; Welch peaks on its channels are
                     absorbed into it

    Only intermittent channels with a stable center count; continuous ones are
    left to the Welch detector. A "welch" target is marked steady when it sits
    on a continuous channel or an always-on narrow line (on 100% of the time).
    """
    intermittent = [ch for ch in channels if not ch["continuous"] and ch["stable"]]
    hopping = {id(ch) for e in emitters for ch in e["channels"]}
    line_freqs = spectrum["freqs"][spectrum["line_bins"]]

    def channel_at(f):
        inside = [ch for ch in intermittent if ch["f_lo_hz"] <= f <= ch["f_hi_hz"]]
        return min(inside, key=lambda ch: abs(ch["center_hz"] - f), default=None)

    def steady_at(f):
        return (any(ch["continuous"] and ch["f_lo_hz"] <= f <= ch["f_hi_hz"] for ch in channels)
                or bool(np.any(np.abs(line_freqs - f) <= spectrum["bin_hz"])))

    targets, merged = [], {}
    for i, peak in enumerate(peaks):
        ch = channel_at(freqs[peak])
        if ch is None:
            targets.append({"kind": "welch", "peak": peak, "index": i, "center_hz": freqs[peak],
                            "steady": steady_at(freqs[peak])})
        elif id(ch) not in hopping:
            best = merged.get(id(ch))
            if best is None or psd_db[peak] > psd_db[best["peak"]]:
                merged[id(ch)] = {"kind": "welch+burst", "peak": peak, "index": i,
                                  "center_hz": freqs[peak], "channel": ch}
    targets += merged.values()
    targets += [{"kind": "burst", "channel": ch, "center_hz": ch["center_hz"]}
                for ch in intermittent if id(ch) not in merged and id(ch) not in hopping
                and ch["snr_db"] >= BURST_ONLY_MIN_SNR_DB]
    targets += [{"kind": "fhss", "emitter": e, "center_hz": e["center_hz"]} for e in emitters]
    return sorted(targets, key=lambda t: t["center_hz"])


def measured_duty(bursts, capture_s):
    """Fraction of the capture the bursts were on (refined burst durations), capped at 1."""
    return min(1.0, sum(b["duration_s"] for b in bursts) / capture_s)


def describe_target(target, center_freq, freqs_hz, psd_db, props, noise_floor, capture_s):
    """
    Frequency, power, bandwidth, IQ window and measured duty cycle for one target.

    Welch-based targets keep the Welch peak's power, prominence and -20 dB
    bandwidth. Burst-based targets use the burst PSD (comparable to Welch's dB
    scale), the burst SNR as prominence and the burst's -6 dB bandwidth. Any
    target backed by bursts takes its IQ from the strongest burst.

    duty_cycle is the measured on-time over the capture: the target's bursts
    (all hops for an FHSS emitter), 1.0 for a steady Welch signal, or None when
    the burst detector saw nothing to measure. is_bursty means duty < CONTINUOUS_DUTY.
    """
    info = _target_signal(target, center_freq, freqs_hz, psd_db, props, noise_floor)
    kind = target["kind"]
    if kind == "welch":
        duty, bursts = (1.0 if target["steady"] else None), []
    else:
        bursts = target["emitter" if kind == "fhss" else "channel"]["bursts"]
        duty = measured_duty(bursts, capture_s)
    info["duty_cycle"] = duty
    info["is_bursty"] = duty is not None and duty < CONTINUOUS_DUTY
    # Packets (bursts) per second over the capture; None for a steady signal
    info["bursts"] = len(bursts)
    info["packet_rate_hz"] = len(bursts) / capture_s if bursts else None
    return info


def _target_signal(target, center_freq, freqs_hz, psd_db, props, noise_floor):
    """Frequency, power, prominence, bandwidth, IQ center/window and FHSS info of a target."""
    kind = target["kind"]
    if kind in ("welch", "welch+burst"):
        peak = target["peak"]
        bw_hz, _, _ = estimate_bandwidth(freqs_hz, psd_db, peak, noise_floor)
        info = {"freq_mhz": (freqs_hz[peak]) / 1e6, "power_db": psd_db[peak],
                "prominence_db": float(props["prominences"][target["index"]]),
                "bw_hz": bw_hz, "iq_center_hz": target["center_hz"], "iq_bw_hz": bw_hz,
                "window_s": None, "fhss": None}
        if kind == "welch+burst":
            # Report the channel center: which merged Welch peak is strongest varies between captures
            ch = target["channel"]
            info["freq_mhz"] = (ch["center_hz"] + center_freq) / 1e6
            info["iq_center_hz"] = ch["center_hz"]
            b = ch["strongest"]
            info["window_s"] = (b["start_s"], b["start_s"] + b["duration_s"])
        return info

    source = target["channel"] if kind == "burst" else target["emitter"]
    b = source["strongest"]
    info = {"freq_mhz": (source["center_hz"] + center_freq) / 1e6, "power_db": source["peak_db"],
            "prominence_db": source["snr_db"], "bw_hz": source["bandwidth_hz"],
            "iq_center_hz": b["center_hz"], "iq_bw_hz": b["bandwidth_hz"],
            "window_s": (b["start_s"], b["start_s"] + b["duration_s"]), "fhss": None}
    if kind == "fhss":
        e = target["emitter"]
        info["fhss"] = {
            "channels_mhz": [round(float(ch["center_hz"] + center_freq) / 1e6, 3) for ch in e["channels"]],
            "hops": len(e["bursts"]),
            "hop_rate_hz": round(e["hop_rate_hz"], 2),
            "dwell_ms": round(1000 * e["dwell_s"], 1),
            "hop_interval_ms": round(1000 * e["hop_interval_s"], 1),
            "channel_bandwidth_khz": round(e["bandwidth_hz"] / 1e3, 1),
        }
    return info


def detect_regions(iq, sample_rate, center_freq, timer=None):
    """
    Run both detectors and plan one target per signal (no classification).

    Welch average (detect_signals) for steady signals, then the short-time burst
    detector, FHSS grouping and plan_targets. Returns a dict with the Welch
    products (freqs, freqs_hz, freqs_mhz, psd_db, noise_floor, threshold, peaks,
    props), the burst products (spectrum, bursts, channels, emitters) and the
    targets. analyze_iq and the model comparison's end-to-end test both use it.
    """
    if timer is None:
        timer = StageTimer()
    with timer.stage("detection"):
        (freqs, freqs_hz, freqs_mhz, psd_db,
         noise_floor, threshold, peaks, props) = detect_signals(iq, sample_rate, center_freq)
    with timer.stage("bursts"):
        spectrum = short_time_spectrum(iq, sample_rate)
        bursts = refine_burst_timing(detect_bursts(spectrum), iq, sample_rate, spectrum)
        channels = group_channels(bursts, spectrum)
        emitters = find_fhss_emitters(channels)
        targets = plan_targets(freqs, psd_db, peaks, channels, emitters, spectrum)
    return {"freqs": freqs, "freqs_hz": freqs_hz, "freqs_mhz": freqs_mhz, "psd_db": psd_db,
            "noise_floor": noise_floor, "threshold": threshold, "peaks": peaks, "props": props,
            "spectrum": spectrum, "bursts": bursts, "channels": channels, "emitters": emitters,
            "targets": targets}


def target_features(iq, sample_rate, center_freq, target, regions, timer=None):
    """
    Describe one target and compute its 11 ML features.

    Returns (info, sig_iq, iq_window, lora, features): describe_target's info,
    the band-passed IQ and its time window, the dechirp result on the raw IQ
    slice, and the feature vector (10 spectral/amplitude features + dechirp score).
    """
    if timer is None:
        timer = StageTimer()
    # Bandwidth + signal extraction (IQ from the burst's own time slice if it has one)
    with timer.stage("extraction"):
        info = describe_target(target, center_freq, regions["freqs_hz"], regions["psd_db"],
                               regions["props"], regions["noise_floor"], len(iq) / sample_rate)
        sig_iq, iq_window = extract_signal_iq(iq, sample_rate, info["iq_center_hz"],
                                              info["iq_bw_hz"], info["window_s"])
    # Features: the 10 spectral/amplitude features plus the dechirp score,
    # computed on this target's own raw IQ slice around its center
    with timer.stage("features"):
        raw = iq[int(iq_window[0] * sample_rate):int(iq_window[1] * sample_rate)]
        lora = dechirp_score(raw, sample_rate, info["iq_center_hz"])
        features = extract_ml_features(sig_iq, sample_rate) + [lora["score_db"]]
    return info, sig_iq, iq_window, lora, features


def analyze_iq(iq, sample_rate, center_freq, clf, fingerprinter, threat_engine,
               verbose=True, timer=None, details=None):
    """
    Detect, classify, fingerprint and threat-assess every signal in an IQ capture.

    Two detectors run side by side: the Welch average (steady signals) and the
    short-time burst detector (bursts and frequency hopping, see
    burst_detector.py); plan_targets combines them into one target per signal.

    Returns (detections, noise_floor): a list of detection dicts (one per
    detected signal, in the format ScanReport.add_detection expects) and the
    estimated noise floor in dB.

    Burstiness is measured (duty cycle from the burst detector's on/off times)
    and LoRa modulation is checked with the dechirp score on each target's own
    IQ slice; both feed fingerprint matching and the threat score.

    verbose=False suppresses the console output. timer, if given, is a
    StageTimer that receives the time spent in each stage. details, if given,
    is a dict that receives detect_regions' products (for plotting).
    """
    log = print if verbose else (lambda *args, **kwargs: None)
    if timer is None:
        timer = StageTimer()

    # --- Detect signals ---
    log(f"\n[*] Scanning spectrum around {center_freq/1e6:.3f} MHz...")
    regions = detect_regions(iq, sample_rate, center_freq, timer)
    if details is not None:
        details.update(regions)
    noise_floor, threshold, peaks = regions["noise_floor"], regions["threshold"], regions["peaks"]
    spectrum, bursts, channels = regions["spectrum"], regions["bursts"], regions["channels"]
    emitters, targets = regions["emitters"], regions["targets"]

    log(f"    Noise floor: {noise_floor:.1f} dB")
    log(f"    Threshold:   {threshold:.1f} dB")
    log(f"    Signals detected (Welch average): {len(peaks)}")

    # --- Short-time burst detection, FHSS grouping, merge with Welch (in detect_regions) ---
    timer.signals += len(targets)

    n_intermittent = sum(1 for ch in channels if not ch["continuous"] and ch["stable"])
    log(f"    Burst scan: {len(spectrum['power'])} slices of {spectrum['slice_s']*1000:.1f} ms, "
        f"threshold {spectrum['threshold_db']:.1f} dB over noise")
    log(f"    Bursts: {len(bursts)} in {n_intermittent} intermittent channel(s)"
        + (f" ({spectrum['artifacts']} spur/splatter bursts ignored)" if spectrum["artifacts"] else ""))
    for e in emitters:
        log(f"    FHSS emitter: {len(e['channels'])} channels, {len(e['bursts'])} hops, "
            f"{e['hop_rate_hz']:.1f} hops/s, dwell {e['dwell_s']*1000:.1f} ms")

    detections = []

    # --- Process each detection ---
    if len(targets) == 0:
        log("\n[*] No signals above threshold — band appears clear.")
    else:
        log(f"\n[*] Analyzing {len(targets)} detected signal(s)...\n")

    for i, target in enumerate(targets):
        info, sig_iq, iq_window, lora, features = target_features(
            iq, sample_rate, center_freq, target, regions, timer)
        freq_mhz = info["freq_mhz"]
        power = info["power_db"]
        prominence = info["prominence_db"]
        bw_hz = info["bw_hz"]
        bw_khz = bw_hz / 1000
        fhss = info["fhss"]
        duty = info["duty_cycle"]
        is_bursty = info["is_bursty"]

        is_lora = lora["score_db"] >= LORA_THRESHOLD_DB
        with timer.stage("ml"):
            ml_class = clf.predict([features])[0]
            ml_proba = clf.predict_proba([features])[0]
            ml_confidence = float(np.max(ml_proba))

        # Fingerprint matching with the measured burstiness/duty cycle
        with timer.stage("fingerprint"):
            fp_match = fingerprinter.get_best_match(
                freq_mhz=freq_mhz, bandwidth_khz=bw_khz,
                is_bursty=is_bursty, duty_cycle=duty,
                hopping=fhss is not None, lora=is_lora)
        lora_info = ({"score_db": round(lora["score_db"], 1), "sf": lora["sf"],
                      "bw_khz": lora["bw_hz"] / 1e3} if is_lora else None)

        # Threat assessment
        with timer.stage("threat"):
            assessment = threat_engine.assess({
                "freq_mhz": freq_mhz,
                "bandwidth_khz": bw_khz,
                "power_db": power,
                "ml_class": ml_class,
                "ml_confidence": ml_confidence,
                "fingerprint_match": fp_match,
                "is_bursty": is_bursty,
                "duty_cycle": duty,
                "fhss": fhss,
                "lora": lora_info,
                "dechirp_score_db": lora["score_db"],
                "packet_rate_hz": info["packet_rate_hz"],
            })

        # Print live result
        sev = assessment["severity"]
        score = assessment["threat_score"]
        kind = {"welch": "", "welch+burst": "  |  burst", "burst": "  |  burst only",
                "fhss": "  |  FHSS"}[target["kind"]]
        lora_text = (f"LoRa: yes ({lora['score_db']:.1f} dB, SF{lora['sf']}/{lora['bw_hz']/1e3:.0f}k)"
                     if is_lora else f"LoRa: no ({lora['score_db']:.1f} dB)")
        duty_text = "duty ?" if duty is None else f"duty {duty:.0%}{' bursty' if is_bursty else ''}"
        log(f"  Signal {i+1}: {freq_mhz:.3f} MHz  |  {bw_khz:.0f} kHz  |  "
              f"ML: {ml_class} ({ml_confidence:.0%})  |  {lora_text}  |  {duty_text}  |  "
              f"Threat: {score}/100 ({sev}){kind}")

        if fp_match:
            profile, fp_score, _ = fp_match
            log(f"           Fingerprint: {profile['name']} ({fp_score}/100)")
        if fhss:
            log(f"           FHSS: {', '.join(f'{c:.3f}' for c in fhss['channels_mhz'])} MHz  |  "
                f"{fhss['hop_rate_hz']:.1f} hops/s  |  dwell {fhss['dwell_ms']:.1f} ms")
        log(f"           Verdict: {assessment['verdict']} — {'; '.join(assessment['verdict_reasons'])}")

        # Add to report
        detection_record = {
            "verdict": assessment["verdict"],
            "verdict_reasons": assessment["verdict_reasons"],
            "freq_mhz": round(freq_mhz, 3),
            "power_db": round(power, 1),
            "bandwidth_khz": round(bw_khz, 1),
            "prominence_db": round(float(prominence), 1),
            "ml_class": ml_class,
            "ml_confidence": round(ml_confidence, 2),
            "fingerprint_name": fp_match[0]["name"] if fp_match else None,
            "fingerprint_id": fp_match[0]["id"] if fp_match else None,
            "fingerprint_score": fp_match[1] if fp_match else None,
            "threat_score": score,
            "threat_severity": sev,
            "threat_factors": assessment["factors"],
            "recommendations": assessment["recommendations"],
            "detected_by": target["kind"],
            "iq_window_s": [round(iq_window[0], 4), round(iq_window[1], 4)],
            "duty_cycle": None if duty is None else round(duty, 3),
            "is_bursty": is_bursty,
            "bursts": info["bursts"],
            "packet_rate_hz": None if info["packet_rate_hz"] is None else round(info["packet_rate_hz"], 3),
            "dechirp_score_db": round(lora["score_db"], 1),
            "lora_rule": bool(is_lora),
            "lora_sf": lora["sf"] if is_lora else None,
            "lora_bw_khz": lora["bw_hz"] / 1e3 if is_lora else None,
        }
        if fhss:
            detection_record["fhss"] = fhss
        detections.append(detection_record)

    return detections, noise_floor


def load_components(model="rf"):
    """
    Load the ML model, fingerprint DB and threat engine. Returns None if the model is missing.

    model "rf" is the default deployed Random Forest (rf_classifier_model.joblib);
    "xgb", "lgbm", "cat" and "ensemble" load models/<name>.joblib written by
    compare_models.py. Every model returns class names, so nothing else changes.
    """
    print("\n[*] Loading components...")
    if model == "rf":
        try:
            clf = joblib.load("rf_classifier_model.joblib")
            with open("rf_classifier_config.json") as f:
                config = json.load(f)
            print(f"    ML model loaded ({config['accuracy']:.0%} accuracy)")
        except FileNotFoundError:
            print("    [!] ML model not found — run train_rf_classifier.py first")
            return None
    else:
        from models import load_model
        try:
            clf, meta = load_model(model)
        except FileNotFoundError:
            print(f"    [!] Model models/{model}.joblib not found — run compare_models.py first")
            return None
        print(f"    ML model loaded ({model}: CV macro-F1 {meta['cv_macro_f1_mean']:.3f} "
              f"+- {meta['cv_macro_f1_std']:.3f})")

    fingerprinter = DroneFingerprinter()
    print(f"    Fingerprint DB loaded ({len(fingerprinter.detectable)} drone profiles)")

    threat_engine = ThreatEngine()
    print(f"    Threat engine ready")
    return clf, fingerprinter, threat_engine


def report_scan(source, center_freq, sample_rate, num_samples,
                detections, noise_floor, threat_engine, timer=None):
    """
    Print the threat summary and save JSON + text reports. Returns the filenames.

    timer, if given, is a StageTimer whose "report" stage receives the time
    spent building and saving the reports.
    """
    if timer is None:
        timer = StageTimer()
    duration = num_samples / sample_rate

    # --- Build report ---
    with timer.stage("report"):
        report = ScanReport(source, center_freq / 1e6, sample_rate)
        report.set_environment(noise_floor, len(detections), duration, num_samples)
        for detection_record in detections:
            report.add_detection(detection_record)

    # --- Print threat summary ---
    threat_engine.summary()

    # --- Save reports ---
    with timer.stage("report"):
        json_name, txt_name = report.save()
    print(f"\n[OK] Reports saved:")
    print(f"     {json_name}")
    print(f"     {txt_name}")
    return json_name, txt_name


def main():
    # --- Parse arguments ---
    if len(sys.argv) < 3:
        print("Usage: python rf_scanner.py <recording.wav> <center_freq_hz>")
        print("Example: python rf_scanner.py synthetic_drone_telemetry.wav 433900000")
        return

    recording_path = sys.argv[1]
    center_freq = float(sys.argv[2])

    # --- Load components ---
    print("=" * 60)
    print("RTL-SDR DRONE RF THREAT SCANNER")
    print("=" * 60)

    components = load_components()
    if components is None:
        return
    clf, fingerprinter, threat_engine = components

    # --- Load and analyze recording ---
    print(f"\n[*] Loading recording: {recording_path}")
    try:
        iq, sample_rate = load_recording(recording_path)
    except Exception as e:
        print(f"    [!] Failed to load recording: {e}")
        return

    duration = len(iq) / sample_rate
    print(f"    {len(iq):,} samples, {duration:.1f}s, {sample_rate/1e6:.1f} MS/s")

    detections, noise_floor = analyze_iq(iq, sample_rate, center_freq,
                                         clf, fingerprinter, threat_engine)
    report_scan(recording_path, center_freq, sample_rate, len(iq),
                detections, noise_floor, threat_engine)

    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()