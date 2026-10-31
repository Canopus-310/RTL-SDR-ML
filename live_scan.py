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
import time
from datetime import datetime

import numpy as np
from scipy.io import wavfile

from rf_scanner import analyze_iq, load_components, report_scan
from threat_engine import ThreatEngine, SEVERITY_LEVELS

SAMPLE_RATE = 2400000     # rate the ML model was trained at and SDR++ recorded at
SETTLE_SECONDS = 0.2      # discarded after tuning: PLL settling + stale USB buffer data
CHUNK_BYTES = 2 * 262144  # 256k IQ samples per USB read
USB_PACKET = 512          # librtlsdr sync reads must be a multiple of this

# --stream settings
BLOCK_SECONDS = 0.5       # IQ per analyzed block
QUEUE_BLOCKS = 2          # blocks buffered before the oldest is dropped
TRACK_TOLERANCE_KHZ = 20  # a detection joins the nearest track within this distance
BURST_MIN_BLOCKS = 4      # blocks a track must span before its burstiness is judged
BURST_PRESENCE = 0.8      # present in fewer than this fraction of blocks -> bursty
TABLE_ROWS = 20           # live table shows the top tracks by max threat

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
    parser.add_argument("--duration", type=float, default=5,
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


def run_scan(sdr, center_freq, args, clf, fingerprinter):
    """Capture one block of IQ, analyze it and save the reports."""
    sample_rate = int(sdr.sample_rate)
    num_samples = int(args.duration * sample_rate)

    print(f"\n[*] Capturing {args.duration:g}s at {center_freq/1e6:.3f} MHz...")
    read_iq(sdr, int(SETTLE_SECONDS * sample_rate))  # discarded
    started = datetime.now()
    iq = read_iq(sdr, num_samples)
    print(f"    {len(iq):,} samples, {len(iq)/sample_rate:.1f}s, {sample_rate/1e6:.1f} MS/s")

    source = f"live RTL-SDR capture ({started:%Y-%m-%d %H:%M:%S})"
    if args.save_wav:
        source = save_wav(iq, sample_rate, center_freq, started)
        print(f"    IQ saved: {source}")

    # Fresh engine per scan so the summary only covers this capture
    threat_engine = ThreatEngine()
    detections, noise_floor = analyze_iq(iq, sample_rate, center_freq,
                                         clf, fingerprinter, threat_engine)
    report_scan(source, center_freq, sample_rate, len(iq),
                detections, noise_floor, threat_engine)


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
        self.error = None

    def run(self):
        try:
            read_iq(self.sdr, self.settle_samples)  # discarded
            while not self.stop_event.is_set():
                iq = read_iq(self.sdr, self.block_samples)
                self.blocks_read += 1
                block = (datetime.now(), iq)
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

    Burstiness is measured from how often a track appears across analyzed
    blocks, so it only sees on/off patterns slower than BLOCK_SECONDS — a
    signal that bursts at least once per block looks continuous.
    """

    def __init__(self):
        self.tracks = []
        self.blocks_analyzed = 0

    def find(self, freq_mhz):
        """Nearest track within TRACK_TOLERANCE_KHZ of freq_mhz, or None."""
        nearest = min(self.tracks, key=lambda t: abs(t["freq_mhz"] - freq_mhz), default=None)
        if nearest is not None and abs(nearest["freq_mhz"] - freq_mhz) * 1000 <= TRACK_TOLERANCE_KHZ:
            return nearest
        return None

    def blocks_spanned(self, track):
        return self.blocks_analyzed - track["first_block"] + 1

    def presence(self, track):
        return track["hits"] / self.blocks_spanned(track)

    def bursty(self, track):
        """True/False from presence across blocks, or None until the track spans enough blocks."""
        if track is None or self.blocks_spanned(track) < BURST_MIN_BLOCKS:
            return None
        return self.presence(track) < BURST_PRESENCE

    def is_bursty(self, freq_mhz):
        """burst_lookup for analyze_iq: measured burstiness, False while still unknown."""
        return bool(self.bursty(self.find(freq_mhz)))

    def update(self, detections, seen_at):
        """Fold one analyzed block's detections into the tracks."""
        self.blocks_analyzed += 1
        block = self.blocks_analyzed
        for d in detections:
            track = self.find(d["freq_mhz"])
            if track is None:
                track = {"freq_mhz": d["freq_mhz"], "first_seen": seen_at, "last_seen": seen_at,
                         "first_block": block, "last_block": 0, "hits": 0,
                         "max_threat": -1, "best": None}
                self.tracks.append(track)
            if track["last_block"] != block:  # count each track at most once per block
                track["hits"] += 1
                track["freq_mhz"] += (d["freq_mhz"] - track["freq_mhz"]) / track["hits"]
                track["last_block"] = block
                track["last_seen"] = seen_at
            if d["threat_score"] > track["max_threat"]:
                track["max_threat"] = d["threat_score"]
                track["best"] = d

    def ranked(self):
        return sorted(self.tracks, key=lambda t: (-t["max_threat"], t["freq_mhz"]))


def format_track_table(tracker, max_rows=None):
    """Render the tracks as a fixed-width table, highest max threat first."""
    lines = [f"  {'Freq MHz':>10}  {'First':>8}  {'Last':>8}  {'Hits/Blk':>9}  {'Present':>7}  "
             f"{'Bursty':>6}  {'Max threat':<15}  Fingerprint",
             "  " + "-" * 98]
    rows = tracker.ranked()
    for t in rows[:max_rows]:
        bursty = {None: "?", True: "yes", False: "no"}[tracker.bursty(t)]
        hits = f"{t['hits']}/{tracker.blocks_spanned(t)}"
        threat = f"{t['max_threat']}/100 {t['best']['threat_severity']}"
        lines.append(f"  {t['freq_mhz']:10.3f}  {t['first_seen']:%H:%M:%S}  {t['last_seen']:%H:%M:%S}  "
                     f"{hits:>9}  {tracker.presence(t):>7.0%}  {bursty:>6}  {threat:<15}  "
                     f"{t['best']['fingerprint_name'] or '-'}")
    if not rows:
        lines.append("  (no signals yet)")
    elif max_rows is not None and len(rows) > max_rows:
        lines.append(f"  ... {len(rows) - max_rows} more")
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
    max_threat = max((t["max_threat"] for t in ranked), default=0)
    max_severity = next(level for level, info in SEVERITY_LEVELS.items()
                        if max_threat >= info["min_score"])
    summary = {
        "overall_assessment": f"{max_severity} - {SEVERITY_LEVELS[max_severity]['action']}",
        "max_threat_score": max_threat,
        "tracks": len(ranked),
        "drone_candidates": sum(1 for t in ranked if t["max_threat"] >= 40),
    }
    tracks = [{
        "freq_mhz": round(t["freq_mhz"], 3),
        "first_seen": t["first_seen"].isoformat(),
        "last_seen": t["last_seen"].isoformat(),
        "hits": t["hits"],
        "blocks_spanned": tracker.blocks_spanned(t),
        "presence": round(tracker.presence(t), 3),
        "bursty": tracker.bursty(t),
        "max_threat_score": t["max_threat"],
        "max_threat_severity": t["best"]["threat_severity"],
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
        f.write(f"Overall:           {summary['overall_assessment']}\n")
        f.write(f"Max Threat Score:  {summary['max_threat_score']}/100\n")
        f.write(f"Tracks:            {summary['tracks']}\n")
        f.write(f"Drone Candidates:  {summary['drone_candidates']}\n\n")
        f.write("SIGNAL TRACKS\n")
        f.write(format_track_table(tracker) + "\n\n")
        f.write("HIGHEST-THREAT DETECTION PER TRACK\n")
        f.write("-" * 60 + "\n")
        for t in ranked:
            d = t["best"]
            f.write(f"\n  {t['freq_mhz']:.3f} MHz: {d['threat_score']}/100 ({d['threat_severity']})\n")
            f.write(f"  Power {d['power_db']:.1f} dB, bandwidth {d['bandwidth_khz']:.1f} kHz, "
                    f"ML {d['ml_class']} ({d['ml_confidence']:.0%}), "
                    f"fingerprint {d['fingerprint_name'] or 'no match'}\n")
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
    live = enable_ansi()
    started = datetime.now()

    print(f"\n[*] Streaming {BLOCK_SECONDS:g}s blocks at {center_freq/1e6:.3f} MHz "
          f"- Ctrl+C to stop and write the summary report")
    reader.start()
    try:
        while True:
            try:
                # Short timeout keeps Ctrl+C responsive on Windows
                captured_at, iq = reader.blocks.get(timeout=0.5)
            except queue.Empty:
                if reader.error is not None:
                    raise reader.error
                continue

            t0 = time.perf_counter()
            detections, noise_floor = analyze_iq(
                iq, sample_rate, center_freq, clf, fingerprinter, ThreatEngine(),
                verbose=False, burst_lookup=tracker.is_bursty)
            processing_ms.append(1000 * (time.perf_counter() - t0))
            tracker.update(detections, captured_at)
            lag = (datetime.now() - captured_at).total_seconds()

            if live:
                sys.stdout.write("\033[H\033[J")  # redraw in place
                print(f"RTL-SDR STREAM  {center_freq/1e6:.3f} MHz  gain {gain:.1f} dB  "
                      f"started {started:%H:%M:%S}  (Ctrl+C to stop)\n")
            print(f"Block {tracker.blocks_analyzed}: {processing_ms[-1]:.0f} ms "
                  f"(avg {np.mean(processing_ms):.0f}, max {max(processing_ms):.0f}; "
                  f"budget {BLOCK_SECONDS * 1000:.0f})  |  lag {lag:.1f}s  |  "
                  f"dropped {reader.blocks_dropped}/{reader.blocks_read}  |  "
                  f"noise {noise_floor:.1f} dB  |  {len(detections)} detections")
            print(format_track_table(tracker, TABLE_ROWS))
            sys.stdout.flush()
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
            reader = BlockReader(sdr, int(BLOCK_SECONDS * sample_rate),
                                 int(SETTLE_SECONDS * sample_rate))
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
