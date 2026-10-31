"""
RF Drone Threat Scanner — Live RTL-SDR Capture

Replaces the SDR++ record step: tunes the RTL-SDR directly with pyrtlsdr,
captures baseband IQ, and runs the same pipeline as rf_scanner.py
(detect -> classify -> fingerprint -> threat assess -> report).

Usage:
    python live_scan.py [--freq HZ] [--gain DB] [--duration S] [--loop | --stream] [--save-wav]

Examples:
    python live_scan.py                          # 5 s at 865.8 MHz, gain ~15 dB
    python live_scan.py --freq 433900000 --save-wav
    python live_scan.py --loop                   # rescan until Ctrl+C
    python live_scan.py --stream                 # continuous: live signal table, summary on Ctrl+C

SDR++ must be stopped (or closed) first — only one program can open the dongle.
"""

import argparse
import ctypes
import json
import os
import queue
import sys
import threading
from datetime import datetime

import numpy as np
from scipy.io import wavfile

from rf_scanner import analyze_iq, load_components, report_scan
from stage_timer import StageTimer, STAGES, stage_stats
from burst_detector import CONTINUOUS_DUTY
from threat_engine import (ThreatEngine, SEVERITY_LEVELS, VERDICTS, decide_verdict,
                           severity_for_score)

SAMPLE_RATE = 2400000     # rate the ML model was trained at and SDR++ recorded at
SETTLE_SECONDS = 0.2      # discarded after tuning: PLL settling + stale USB buffer data
CHUNK_BYTES = 2 * 262144  # 256k IQ samples per USB read
USB_PACKET = 512          # librtlsdr sync reads must be a multiple of this

# --stream settings
BLOCK_SECONDS = 0.5       # IQ per analyzed block
QUEUE_BLOCKS = 2          # blocks buffered before the oldest is dropped
TRACK_TOLERANCE_KHZ = 20  # a detection joins the nearest track within this distance
FHSS_CHANNEL_TOLERANCE_KHZ = 50  # a frequency within this of a hop channel belongs to that emitter
TABLE_ROWS = 20           # live table shows the top tracks by max threat
# Per-block stages in stream mode (no per-block report), and their summary labels
STREAM_STAGES = ["detection", "bursts", "extraction", "features", "ml", "fingerprint", "threat"]
STREAM_STAT_LABELS = {**{key: label for key, label, _ in STAGES},
                      "capture": "Capture (block read)",
                      "processing": "Processing (analyze_iq)",
                      "latency": "Latency (captured -> printed)"}

# libusb error codes returned by rtlsdr_open
LIBUSB_ERROR_ACCESS = -3
LIBUSB_ERROR_NOT_FOUND = -5
LIBUSB_ERROR_BUSY = -6
LIBUSB_ERROR_NOT_SUPPORTED = -12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture IQ from an RTL-SDR and run the drone RF threat scan.")
    parser.add_argument("--freq", type=float, default=865800000,
                        help="center frequency in Hz (default: 865800000; tune ~0.2 MHz "
                             "below the target so it stays clear of the DC spike)")
    parser.add_argument("--gain", type=float, default=15,
                        help="tuner gain in dB, snapped to the nearest step the tuner "
                             "supports (default: 15)")
    parser.add_argument("--duration", type=float, default=2,
                        help="seconds of IQ to capture per scan (default: 5)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--loop", action="store_true",
                      help="keep scanning until Ctrl+C")
    mode.add_argument("--stream", action="store_true",
                      help=f"continuous mode: analyze {BLOCK_SECONDS:g} s blocks as they arrive, "
                           "track signals in a live table, write a summary report on Ctrl+C "
                           "(--duration is ignored)")
    parser.add_argument("--save-wav", action="store_true",
                        help="also save each capture as an SDR++-style Int16 stereo WAV")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be greater than 0")
    if args.stream and args.save_wav:
        parser.error("--save-wav is not supported with --stream")
    return args


def import_rtlsdr():
    """Import pyrtlsdr. Returns (RtlSdr, librtlsdr), or None with an explanation printed."""
    try:
        from rtlsdr import RtlSdr
        from rtlsdr.librtlsdr import librtlsdr
    except ImportError as e:
        if getattr(e, "name", None) == "rtlsdr":
            print("    [!] pyrtlsdr is not installed - run: pip install pyrtlsdr pyrtlsdrlib")
        else:
            # pyrtlsdr raises ImportError at import time when it can't load the DLL
            print("    [!] pyrtlsdr could not load librtlsdr (rtlsdr.dll).")
            print("        Install the bundled DLLs with:  pip install pyrtlsdrlib")
            print("        See 'Live capture with pyrtlsdr' in README.md.")
        return None
    return RtlSdr, librtlsdr


def open_sdr(RtlSdr, librtlsdr):
    """Open the first RTL-SDR. Returns the device, or None with an explanation printed."""
    if librtlsdr.rtlsdr_get_device_count() == 0:
        print("    [!] No RTL-SDR dongle found.")
        print("        Check it is plugged in, then re-run. If it is plugged in, the WinUSB")
        print("        driver may be missing - see the Zadig step in README.md.")
        return None

    try:
        return RtlSdr()
    except IOError as e:
        code = getattr(e, "errno", None)
        print(f"    [!] RTL-SDR found but could not be opened: {e}")
        if code in (LIBUSB_ERROR_ACCESS, LIBUSB_ERROR_BUSY):
            print("        Another program is using the dongle. Press Stop in SDR++ or close it,")
            print("        and any other SDR software, then re-run.")
        elif code in (LIBUSB_ERROR_NOT_SUPPORTED, LIBUSB_ERROR_NOT_FOUND):
            print("        The dongle is not using the WinUSB driver. Install it with Zadig")
            print("        (see README.md), then unplug and replug the dongle.")
        else:
            print("        Most likely SDR++ or another SDR program is holding it - close it")
            print("        and re-run. Otherwise unplug and replug the dongle.")
        return None


def read_iq(sdr, num_samples):
    """
    Read num_samples IQ samples in CHUNK_BYTES pieces.

    Returns complex128 normalized like pyrtlsdr's read_samples(): (x - 127.5) / 127.5.
    """
    num_bytes = 2 * num_samples
    raw = np.empty(num_bytes, dtype=np.uint8)
    pos = 0
    while pos < num_bytes:
        n = min(CHUNK_BYTES, num_bytes - pos)
        n_read = -(-n // USB_PACKET) * USB_PACKET
        # read_bytes reuses its buffer, so copy out before the next read
        raw[pos:pos + n] = np.ctypeslib.as_array(sdr.read_bytes(n_read))[:n]
        pos += n

    iq = raw.astype(np.float64).view(np.complex128)
    iq /= 127.5
    iq -= (1 + 1j)
    return iq


def save_wav(iq, sample_rate, center_freq, when):
    """Save IQ the way SDR++'s baseband recorder does: stereo Int16, I left, Q right."""
    name = f"baseband_{int(center_freq)}Hz_{when:%H-%M-%S_%d-%m-%Y}.wav"
    stereo = np.empty((len(iq), 2), dtype=np.int16)
    stereo[:, 0] = np.clip(iq.real * 32767, -32768, 32767)
    stereo[:, 1] = np.clip(iq.imag * 32767, -32768, 32767)
    wavfile.write(name, sample_rate, stereo)
    return name


def add_timings_to_report(json_name, timings):
    """Add a "timings" section to a saved scan report (done afterwards so report writing is counted)."""
    with open(json_name) as f:
        report = json.load(f)
    report["timings"] = timings
    with open(json_name, "w") as f:
        json.dump(report, f, indent=2, default=str)


def run_scan(sdr, center_freq, args, clf, fingerprinter):
    """Capture one block of IQ, analyze it, save the reports and print the stage timings."""
    sample_rate = int(sdr.sample_rate)
    num_samples = int(args.duration * sample_rate)

    print(f"\n[*] Capturing {args.duration:g}s at {center_freq/1e6:.3f} MHz...")
    timer = StageTimer()  # end-to-end clock starts at capture start
    with timer.stage("capture"):
        read_iq(sdr, int(SETTLE_SECONDS * sample_rate))  # discarded
        started = datetime.now()
        iq = read_iq(sdr, num_samples)
    print(f"    {len(iq):,} samples, {len(iq)/sample_rate:.1f}s, {sample_rate/1e6:.1f} MS/s")

    source = f"live RTL-SDR capture ({started:%Y-%m-%d %H:%M:%S})"
    if args.save_wav:
        with timer.stage("report"):
            source = save_wav(iq, sample_rate, center_freq, started)
        print(f"    IQ saved: {source}")

    # Fresh engine per scan so the summary only covers this capture
    threat_engine = ThreatEngine()
    detections, noise_floor = analyze_iq(iq, sample_rate, center_freq,
                                         clf, fingerprinter, threat_engine, timer=timer)
    json_name, _ = report_scan(source, center_freq, sample_rate, len(iq),
                               detections, noise_floor, threat_engine, timer=timer)
    end_to_end_ms = timer.elapsed_ms()

    add_timings_to_report(json_name, timer.as_dict(end_to_end_ms))
    print("\n[*] Stage timings")
    print(timer.format_table(end_to_end_ms))


class BlockReader(threading.Thread):
    """Reads fixed-size IQ blocks into a small queue, dropping the oldest when it's full."""

    def __init__(self, sdr, block_samples, settle_samples):
        super().__init__(daemon=True)
        self.sdr = sdr
        self.block_samples = block_samples
        self.settle_samples = settle_samples
        self.blocks = queue.Queue(maxsize=QUEUE_BLOCKS)
        self.stop_event = threading.Event()
        self.blocks_read = 0
        self.blocks_dropped = 0
        self.settle_ms = None
        self.error = None

    def run(self):
        try:
            settle = StageTimer()
            with settle.stage("capture"):
                read_iq(self.sdr, self.settle_samples)  # discarded
            self.settle_ms = settle.ms["capture"]
            while not self.stop_event.is_set():
                capture = StageTimer()
                with capture.stage("capture"):
                    iq = read_iq(self.sdr, self.block_samples)
                self.blocks_read += 1
                block = (datetime.now(), iq, capture.ms["capture"])
                try:
                    self.blocks.put_nowait(block)
                except queue.Full:
                    # Analysis is behind: drop the oldest block rather than build up lag
                    try:
                        self.blocks.get_nowait()
                        self.blocks_dropped += 1
                    except queue.Empty:
                        pass
                    self.blocks.put_nowait(block)
        except Exception as e:
            self.error = e  # re-raised by the main thread

    def stop(self, timeout=2.0):
        """Stop reading and wait for the thread. Returns True if it has exited."""
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout)
        return not self.is_alive()


class SignalTracker:
    """
    Groups detections from successive stream blocks into per-signal tracks.

    Burstiness comes from the duty cycle the burst detector measured inside
    each block (on/off times of the signal's bursts): a track's duty cycle is
    the median over the blocks it was seen in, bursty below CONTINUOUS_DUTY.
    Presence (blocks seen / blocks spanned) is shown separately and does not
    affect burstiness.

    A frequency-hopping emitter is one track: FHSS detections join the FHSS
    track that shares a channel with them, single-channel detections on one of
    its known channels count towards it, and plain tracks created on its
    channels before it was recognised are absorbed into it.
    """

    def __init__(self):
        self.tracks = []
        self.blocks_analyzed = 0

    def find(self, freq_mhz):
        """Nearest plain (non-FHSS) track within TRACK_TOLERANCE_KHZ of freq_mhz, or None."""
        plain = [t for t in self.tracks if t["fhss"] is None]
        nearest = min(plain, key=lambda t: abs(t["freq_mhz"] - freq_mhz), default=None)
        if nearest is not None and abs(nearest["freq_mhz"] - freq_mhz) * 1000 <= TRACK_TOLERANCE_KHZ:
            return nearest
        return None

    @staticmethod
    def _on_channel(freq_mhz, channels_mhz):
        return any(abs(freq_mhz - c) * 1000 <= FHSS_CHANNEL_TOLERANCE_KHZ for c in channels_mhz)

    def find_fhss(self, channels_mhz):
        """FHSS track sharing at least one channel with channels_mhz, or None."""
        for t in self.tracks:
            if t["fhss"] is not None and any(self._on_channel(c, t["fhss"]["channels_mhz"])
                                             for c in channels_mhz):
                return t
        return None

    def blocks_spanned(self, track):
        return self.blocks_analyzed - track["first_block"] + 1

    def hits(self, track):
        return len(track["blocks"])

    def presence(self, track):
        return self.hits(track) / self.blocks_spanned(track)

    def duty(self, track):
        """Median measured duty cycle over the blocks the track was seen in, or None."""
        return float(np.median(track["duties"])) if track["duties"] else None

    def bursty(self, track):
        """True/False from the measured duty cycle, or None if it was never measured."""
        duty = self.duty(track)
        return None if duty is None else duty < CONTINUOUS_DUTY

    def _new_track(self, freq_mhz, seen_at, block):
        track = {"freq_mhz": freq_mhz, "first_seen": seen_at, "last_seen": seen_at,
                 "first_block": block, "blocks": set(), "max_threat": -1, "best": None,
                 "fhss": None, "duties": [], "lora_hits": 0,
                 "bursts_seen": 0, "lora": None, "dechirp_max": None, "ml_classes": {}}
        self.tracks.append(track)
        return track

    def _fhss_track_for(self, d, seen_at, block):
        """The FHSS track for an FHSS detection: existing, or new with plain tracks on its channels absorbed."""
        info = d["fhss"]
        track = self.find_fhss(info["channels_mhz"])
        if track is None:
            track = self._new_track(d["freq_mhz"], seen_at, block)
            track["fhss"] = {"channels_mhz": [], "hops": 0, "hop_rates_hz": [], "dwells_ms": []}
        fh = track["fhss"]
        for c in info["channels_mhz"]:
            if not self._on_channel(c, fh["channels_mhz"]):
                fh["channels_mhz"].append(c)
        fh["channels_mhz"].sort()
        fh["hops"] += info["hops"]
        fh["hop_rates_hz"].append(info["hop_rate_hz"])
        fh["dwells_ms"].append(info["dwell_ms"])
        track["freq_mhz"] = float(np.mean(fh["channels_mhz"]))
        for t in [t for t in self.tracks if t["fhss"] is None
                  and self._on_channel(t["freq_mhz"], fh["channels_mhz"])]:
            self._absorb(track, t)
        return track

    def _absorb(self, track, other):
        """Merge a plain track into an FHSS track and drop it."""
        track["blocks"] |= other["blocks"]
        track["lora_hits"] += other["lora_hits"]
        track["bursts_seen"] += other["bursts_seen"]
        track["lora"] = track["lora"] or other["lora"]
        if other["dechirp_max"] is not None:
            track["dechirp_max"] = (other["dechirp_max"] if track["dechirp_max"] is None
                                    else max(track["dechirp_max"], other["dechirp_max"]))
        for cls, n in other["ml_classes"].items():
            track["ml_classes"][cls] = track["ml_classes"].get(cls, 0) + n
        track["first_seen"] = min(track["first_seen"], other["first_seen"])
        track["first_block"] = min(track["first_block"], other["first_block"])
        track["last_seen"] = max(track["last_seen"], other["last_seen"])
        if other["max_threat"] > track["max_threat"]:
            track["max_threat"], track["best"] = other["max_threat"], other["best"]
        self.tracks.remove(other)

    def update(self, detections, seen_at):
        """Fold one analyzed block's detections into the tracks."""
        self.blocks_analyzed += 1
        block = self.blocks_analyzed
        for d in detections:
            if d.get("fhss"):
                track = self._fhss_track_for(d, seen_at, block)
            else:
                track = next((t for t in self.tracks if t["fhss"] is not None and
                              self._on_channel(d["freq_mhz"], t["fhss"]["channels_mhz"])), None)
                track = track or self.find(d["freq_mhz"])
                if track is None:
                    track = self._new_track(d["freq_mhz"], seen_at, block)
                if track["fhss"] is None and block not in track["blocks"]:
                    n = len(track["blocks"]) + 1
                    track["freq_mhz"] += (d["freq_mhz"] - track["freq_mhz"]) / n
            if block not in track["blocks"] or d.get("fhss"):
                # One duty measurement per track per block; an FHSS detection's
                # duty (all hops in the block) supersedes single-channel ones
                if d.get("duty_cycle") is not None:
                    if block in track["blocks"] and track["duties"]:
                        track["duties"][-1] = d["duty_cycle"]
                    else:
                        track["duties"].append(d["duty_cycle"])
            track["lora_hits"] += bool(d.get("lora_rule"))
            track["bursts_seen"] += d.get("bursts", 0)
            track["ml_classes"][d["ml_class"]] = track["ml_classes"].get(d["ml_class"], 0) + 1
            if d.get("dechirp_score_db") is not None:
                track["dechirp_max"] = max(track["dechirp_max"] if track["dechirp_max"] is not None
                                           else d["dechirp_score_db"], d["dechirp_score_db"])
            if d.get("lora_rule") and (track["lora"] is None
                                       or d["dechirp_score_db"] > track["lora"]["score_db"]):
                track["lora"] = {"score_db": d["dechirp_score_db"], "sf": d["lora_sf"],
                                 "bw_khz": d["lora_bw_khz"]}
            track["blocks"].add(block)  # a track counts at most once per block
            track["last_seen"] = seen_at
            if d["threat_score"] > track["max_threat"]:
                track["max_threat"] = d["threat_score"]
                track["best"] = d

    def packet_rate(self, track):
        """Bursts seen per second of analyzed time the track spans."""
        return track["bursts_seen"] / (self.blocks_spanned(track) * BLOCK_SECONDS)

    def verdict(self, track):
        """
        (verdict, reasons) for the track, from evidence accumulated over all its
        blocks: LoRa hits, most common ML label, FHSS hop rate, median duty cycle
        and packet rate. A single block can miss hops (or the whole emitter);
        the track sees the full picture.
        """
        best = track["best"]
        ml_class = max(track["ml_classes"], key=track["ml_classes"].get, default=best["ml_class"])
        fhss = None
        if track["fhss"]:
            stats = fhss_stats(track["fhss"])
            fhss = {"channels_mhz": stats["channels_mhz"], "hop_rate_hz": stats["hop_rate_hz"]}
        return decide_verdict({
            "freq_mhz": track["freq_mhz"], "bandwidth_khz": best["bandwidth_khz"],
            "ml_class": ml_class, "lora": track["lora"], "dechirp_score_db": track["dechirp_max"],
            "fhss": fhss, "is_bursty": self.bursty(track), "duty_cycle": self.duty(track),
            "packet_rate_hz": self.packet_rate(track) if track["bursts_seen"] else None,
            "fingerprint_id": best.get("fingerprint_id"),
        })

    def threat(self, track):
        """Highest threat score seen, bounded by the track's verdict."""
        bounds = VERDICTS[self.verdict(track)[0]]
        return min(max(track["max_threat"], bounds["min_score"]), bounds["max_score"])

    def ranked(self):
        return sorted(self.tracks, key=lambda t: (-self.threat(t), t["freq_mhz"]))


def fhss_stats(fh):
    """Channels seen, total hops, mean hop rate and median dwell of an FHSS track."""
    return {"channels_mhz": fh["channels_mhz"], "hops": fh["hops"],
            "hop_rate_hz": round(float(np.mean(fh["hop_rates_hz"])), 2),
            "dwell_ms": round(float(np.median(fh["dwells_ms"])), 1)}


def fhss_summary(fh):
    s = fhss_stats(fh)
    return (f"{len(s['channels_mhz'])} channels ({', '.join(f'{c:.3f}' for c in s['channels_mhz'])} MHz)  "
            f"|  {s['hop_rate_hz']:.1f} hops/s  |  dwell {s['dwell_ms']:.1f} ms  |  {s['hops']} hops seen")


def format_track_table(tracker, max_rows=None):
    """Render the tracks as a fixed-width table, highest max threat first."""
    lines = [f"  {'Freq MHz':>10}  {'First':>8}  {'Last':>8}  {'Hits/Blk':>9}  {'Present':>7}  "
             f"{'Duty':>8}  {'LoRa':>4}  {'Verdict':<28}  {'Threat':<15}  Fingerprint",
             "  " + "-" * 138]
    rows = tracker.ranked()
    for t in rows[:max_rows]:
        duty = tracker.duty(t)
        duty_text = "?" if duty is None else f"{duty:.0%} {'B' if tracker.bursty(t) else 'C'}"
        hits = f"{tracker.hits(t)}/{tracker.blocks_spanned(t)}"
        score = tracker.threat(t)
        threat = f"{score}/100 {severity_for_score(score)}"
        lines.append(f"  {t['freq_mhz']:10.3f}  {t['first_seen']:%H:%M:%S}  {t['last_seen']:%H:%M:%S}  "
                     f"{hits:>9}  {tracker.presence(t):>7.0%}  {duty_text:>8}  "
                     f"{'yes' if t['lora_hits'] else 'no':>4}  {tracker.verdict(t)[0]:<28}  {threat:<15}  "
                     f"{t['best']['fingerprint_name'] or '-'}")
        if t["fhss"]:
            lines.append(f"  {'':10}  FHSS emitter: {fhss_summary(t['fhss'])}")
    if not rows:
        lines.append("  (no signals yet)")
    elif max_rows is not None and len(rows) > max_rows:
        lines.append(f"  ... {len(rows) - max_rows} more")
    lines.append("  Duty: median measured on-time per block (B = bursty, C = continuous)")
    return "\n".join(lines)


def enable_ansi():
    """Enable ANSI cursor control on an interactive console. Returns False if unavailable."""
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))  # VT processing
    except (AttributeError, OSError):
        return False


def save_stream_summary(tracker, stream, metadata):
    """Write the stream summary as JSON + text reports. Returns the filenames."""
    ranked = tracker.ranked()
    verdicts = {id(t): tracker.verdict(t) for t in ranked}
    max_threat = max((tracker.threat(t) for t in ranked), default=0)
    max_severity = severity_for_score(max_threat)
    verdict_counts = {}
    for verdict, _ in verdicts.values():
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    summary = {
        "overall_assessment": f"{max_severity} - {SEVERITY_LEVELS[max_severity]['action']}",
        "max_threat_score": max_threat,
        "tracks": len(ranked),
        "drone_candidates": sum(1 for t in ranked if t["max_threat"] >= 40),
        "fhss_emitters": sum(1 for t in ranked if t["fhss"]),
        "verdict_counts": verdict_counts,
    }
    tracks = [{
        "verdict": verdicts[id(t)][0],
        "verdict_reasons": verdicts[id(t)][1],
        "freq_mhz": round(t["freq_mhz"], 3),
        "first_seen": t["first_seen"].isoformat(),
        "last_seen": t["last_seen"].isoformat(),
        "hits": tracker.hits(t),
        "blocks_spanned": tracker.blocks_spanned(t),
        "presence": round(tracker.presence(t), 3),
        "duty_cycle": None if tracker.duty(t) is None else round(tracker.duty(t), 3),
        "bursty": tracker.bursty(t),
        "lora_blocks": t["lora_hits"],
        "packet_rate_hz": round(tracker.packet_rate(t), 3),
        "fhss": fhss_stats(t["fhss"]) if t["fhss"] else None,
        "threat_score": tracker.threat(t),
        "threat_severity": severity_for_score(tracker.threat(t)),
        "max_threat_detection": t["best"],
    } for t in ranked]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_name = f"rf_threat_report_stream_{stamp}.json"
    txt_name = f"rf_threat_report_stream_{stamp}.txt"
    with open(json_name, "w") as f:
        json.dump({"metadata": metadata, "stream": stream, "summary": summary,
                   "tracks": tracks}, f, indent=2, default=str)

    with open(txt_name, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RF DRONE THREAT STREAM SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Center Freq:  {metadata['center_freq_mhz']:.3f} MHz\n")
        f.write(f"Sample Rate:  {metadata['sample_rate_hz'] / 1e6:.1f} MS/s\n")
        f.write(f"Gain:         {metadata['gain_db']:.1f} dB\n")
        f.write(f"Started:      {metadata['started']}\n")
        f.write(f"Ended:        {metadata['ended']}\n")
        f.write(f"Duration:     {stream['duration_s']:.1f}s\n")
        f.write(f"Blocks:       {stream['blocks_analyzed']} analyzed, "
                f"{stream['blocks_dropped']} dropped, {stream['blocks_read']} read "
                f"({metadata['block_seconds']:g}s each)\n")
        f.write(f"Processing:   {stream['processing_ms_avg']:.0f} ms avg, "
                f"{stream['processing_ms_max']:.0f} ms max per block\n\n")
        f.write("STAGE TIMINGS PER ANALYZED BLOCK (ms)\n")
        f.write(f"  {'Stage':<32}{'Mean':>9}{'P95':>9}{'Max':>9}\n")
        f.write("  " + "-" * 59 + "\n")
        for key, s in stream["timings"].items():
            f.write(f"  {STREAM_STAT_LABELS[key]:<32}{s['mean_ms']:>9.1f}"
                    f"{s['p95_ms']:>9.1f}{s['max_ms']:>9.1f}\n")
        f.write("  Per-signal stages are summed over the signals in each block.\n\n")
        f.write(f"Overall:           {summary['overall_assessment']}\n")
        f.write(f"Max Threat Score:  {summary['max_threat_score']}/100\n")
        f.write(f"Tracks:            {summary['tracks']}\n")
        f.write(f"Drone Candidates:  {summary['drone_candidates']}\n")
        f.write(f"FHSS Emitters:     {summary['fhss_emitters']}\n")
        f.write("Verdicts:          " + (", ".join(f"{k}: {v}" for k, v in verdict_counts.items())
                                         or "none") + "\n\n")
        f.write("SIGNAL TRACKS\n")
        f.write(format_track_table(tracker) + "\n\n")
        f.write("TRACKS: VERDICT, THEN HIGHEST-THREAT DETECTION\n")
        f.write("-" * 60 + "\n")
        for t in ranked:
            d = t["best"]
            verdict, reasons = verdicts[id(t)]
            score = tracker.threat(t)
            f.write(f"\n  {t['freq_mhz']:.3f} MHz  —  VERDICT: {verdict}  "
                    f"({score}/100 {severity_for_score(score)})\n")
            for reason in reasons:
                f.write(f"    - {reason}\n")
            f.write(f"  Power {d['power_db']:.1f} dB, bandwidth {d['bandwidth_khz']:.1f} kHz, "
                    f"ML {d['ml_class']} ({d['ml_confidence']:.0%}), "
                    f"fingerprint {d['fingerprint_name'] or 'no match'}\n")
            if t["fhss"]:
                f.write(f"  FHSS emitter: {fhss_summary(t['fhss'])}\n")
            for fac in d["threat_factors"]:
                f.write(f"    [{fac['points']:+3d}] {fac['factor']}: {fac['detail']}\n")
        f.write("\n" + "=" * 60 + "\nEND OF REPORT\n" + "=" * 60 + "\n")

    return json_name, txt_name


def run_stream(sdr, reader, center_freq, clf, fingerprinter):
    """Analyze blocks from the reader as they arrive and keep a live track table until Ctrl+C."""
    sample_rate = int(sdr.sample_rate)
    gain = sdr.gain
    # One sample per predict call: a worker pool per call (n_jobs=-1) costs ~1 s per block
    clf.n_jobs = 1
    tracker = SignalTracker()
    processing_ms = []
    block_timings = []  # per analyzed block: {stage: ms, "processing": ms, "latency": ms}
    live = enable_ansi()
    started = datetime.now()

    print(f"\n[*] Streaming {BLOCK_SECONDS:g}s blocks at {center_freq/1e6:.3f} MHz "
          f"- Ctrl+C to stop and write the summary report")
    reader.start()
    try:
        while True:
            try:
                # Short timeout keeps Ctrl+C responsive on Windows
                captured_at, iq, capture_ms = reader.blocks.get(timeout=0.5)
            except queue.Empty:
                if reader.error is not None:
                    raise reader.error
                continue

            timer = StageTimer()  # processing clock starts here
            timer.add("capture", capture_ms)
            detections, noise_floor = analyze_iq(
                iq, sample_rate, center_freq, clf, fingerprinter, ThreatEngine(),
                verbose=False, timer=timer)
            processing_ms.append(timer.elapsed_ms())
            tracker.update(detections, captured_at)

            screen = []
            if live:
                screen.append("\033[H\033[J")  # redraw in place
                screen.append(f"RTL-SDR STREAM  {center_freq/1e6:.3f} MHz  gain {gain:.1f} dB  "
                              f"started {started:%H:%M:%S}  (Ctrl+C to stop)\n\n")
            table = format_track_table(tracker, TABLE_ROWS)
            # Latency is taken once the output is ready, just before it is written
            latency_ms = 1000 * (datetime.now() - captured_at).total_seconds()
            screen.append(
                f"Block {tracker.blocks_analyzed}: {processing_ms[-1]:.0f} ms "
                f"[{timer.compact(STREAM_STAGES)}] "
                f"(avg {np.mean(processing_ms):.0f}, max {max(processing_ms):.0f}; "
                f"budget {BLOCK_SECONDS * 1000:.0f})  |  capture {capture_ms:.0f} ms  |  "
                f"latency {latency_ms / 1000:.2f}s  |  "
                f"dropped {reader.blocks_dropped}/{reader.blocks_read}  |  "
                f"noise {noise_floor:.1f} dB  |  {len(detections)} detections\n")
            screen.append(table + "\n")
            sys.stdout.write("".join(screen))
            sys.stdout.flush()

            block_timings.append({**{key: timer.ms[key] for key in ["capture"] + STREAM_STAGES},
                                  "processing": processing_ms[-1], "latency": latency_ms})
    except KeyboardInterrupt:
        print("\n[*] Stopped by user (Ctrl+C).")
    finally:
        reader.stop()
        ended = datetime.now()
        if tracker.blocks_analyzed == 0:
            print("[*] No blocks were analyzed - no summary report written.")
        else:
            stream = {
                "duration_s": round((ended - started).total_seconds(), 1),
                "blocks_read": reader.blocks_read,
                "blocks_analyzed": tracker.blocks_analyzed,
                "blocks_dropped": reader.blocks_dropped,
                "processing_ms_avg": round(float(np.mean(processing_ms)), 1),
                "processing_ms_max": round(max(processing_ms), 1),
                "settle_ms": None if reader.settle_ms is None else round(reader.settle_ms, 1),
                "timings": stage_stats(block_timings),
            }
            metadata = {
                "tool": "RTL-SDR Drone RF Signal Classifier",
                "mode": "stream",
                "center_freq_mhz": center_freq / 1e6,
                "sample_rate_hz": sample_rate,
                "gain_db": gain,
                "block_seconds": BLOCK_SECONDS,
                "started": started.isoformat(),
                "ended": ended.isoformat(),
            }
            json_name, txt_name = save_stream_summary(tracker, stream, metadata)
            print(f"\n[OK] Stream summary saved:")
            print(f"     {json_name}")
            print(f"     {txt_name}")


def main():
    args = parse_args()
    center_freq = int(args.freq)

    print("=" * 60)
    print("RTL-SDR DRONE RF THREAT SCANNER - LIVE")
    print("=" * 60)

    components = load_components()
    if components is None:
        return 1
    clf, fingerprinter, _ = components

    print("\n[*] Opening RTL-SDR...")
    rtlsdr_api = import_rtlsdr()
    if rtlsdr_api is None:
        return 1
    sdr = open_sdr(*rtlsdr_api)
    if sdr is None:
        return 1

    status = 0
    reader = None
    try:
        sdr.sample_rate = SAMPLE_RATE
        sdr.center_freq = center_freq
        sdr.gain = args.gain          # snaps to nearest tuner step, enables manual gain
        sdr.set_agc_mode(False)       # RTL2832 digital AGC off
        print(f"    Center freq: {center_freq/1e6:.3f} MHz")
        print(f"    Sample rate: {int(sdr.sample_rate)/1e6:.1f} MS/s")
        print(f"    Gain:        {sdr.gain:.1f} dB (requested {args.gain:g} dB; "
              f"tuner steps: {', '.join(f'{g:g}' for g in sdr.valid_gains_db)})")

        if args.stream:
            sample_rate = int(sdr.sample_rate)
            # Whole USB packets per block, so read_iq never discards bytes between blocks
            per_packet = USB_PACKET // 2
            block_samples = -(-int(BLOCK_SECONDS * sample_rate) // per_packet) * per_packet
            reader = BlockReader(sdr, block_samples, int(SETTLE_SECONDS * sample_rate))
            run_stream(sdr, reader, center_freq, clf, fingerprinter)
            return status

        scan = 0
        while True:
            scan += 1
            if args.loop:
                print("\n" + "#" * 60)
                print(f"SCAN {scan}  -  {datetime.now():%Y-%m-%d %H:%M:%S}  (Ctrl+C to stop)")
                print("#" * 60)
            run_scan(sdr, center_freq, args, clf, fingerprinter)
            print("\n" + "=" * 60)
            print("SCAN COMPLETE")
            print("=" * 60)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n[*] Stopped by user (Ctrl+C).")
    except IOError as e:
        print(f"\n[!] RTL-SDR error: {e}")
        print("    The dongle may have been unplugged or grabbed by another program.")
        status = 1
    finally:
        if reader is not None and not reader.stop():
            # Closing while the reader is inside a USB read could crash librtlsdr
            print("[!] Reader thread did not stop; leaving the dongle open. "
                  "Unplug and replug it if the next run can't open it.")
        else:
            sdr.close()
            print("[*] RTL-SDR closed.")

    return status


if __name__ == "__main__":
    sys.exit(main())
