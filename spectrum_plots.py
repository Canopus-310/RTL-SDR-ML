"""
Saved spectrum and waterfall plots for captures, stream sessions and reports.

    save_capture_plots(regions, detections, center_freq, gain_db, when, ...)
        one capture (single scan / --loop): <ts>_<freq>Hz_spectrum and _waterfall
        in results/spectra/, optionally a combined two-panel figure elsewhere
    SessionPlotter    --stream: keeps the session max-hold, a mean Welch spectrum
                      and a ring buffer of the last ~10 s of blocks; snapshot()
                      copies what a plot needs, save_session_plots() draws it
    PlotWorker        draws snapshots on a background thread; submit() never
                      blocks (a snapshot is skipped if the previous one is still
                      being drawn), so plotting cannot delay block processing

Spectrum panel: Welch average and max-hold (4 ms slices) PSD, the two CFAR-style
thresholds the pipeline uses (Welch: median noise + 10 dB; bursts: per-bin noise
+ the burst threshold), and every detection shaded over its bandwidth (each hop
channel for an FHSS emitter). Waterfall panel: the short-time spectrogram with a
box around each detected burst and FHSS channel labels.

Uses the matplotlib Figure API only (no pyplot): headless and thread-safe.
"""

import os
import queue
import threading
from collections import deque
from datetime import datetime

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from plot_style import DOUBLE_COL, apply_style, save_figure

SPECTRA_DIR = os.path.join("results", "spectra")
VERDICT_COLORS = {
    "DRONE_CONTROL_TELEMETRY_LINK": "#D55E00", "FSK_TELEMETRY": "#E69F00",
    "POSSIBLE_IOT_LORA": "#56B4E9", "NOT_DRONE": "#999999",
}
apply_style()


def file_stem(when, center_freq):
    return f"{when:%Y%m%d_%H%M%S}_{int(round(center_freq))}Hz"


def headline_verdict(detections):
    if not detections:
        return "no detections"
    return max(detections, key=lambda d: d["threat_score"])["verdict"]


def title_for(center_freq, gain_db, verdict, when):
    gain = "gain n/a" if gain_db is None else f"gain {gain_db:.1f} dB"
    return f"{center_freq / 1e6:.3f} MHz  |  {gain}  |  {verdict}  |  {when:%Y-%m-%d %H:%M:%S}"


def capture_view(regions, center_freq, t_offset=0.0):
    """What the waterfall needs from one capture/block: power (dB), axes, burst boxes, FHSS channels."""
    spec = regions["spectrum"]
    n_slices = len(spec["power"])
    return {
        "power_db": 10 * np.log10(spec["power"] + 1e-20) if n_slices else np.zeros((0, len(spec["freqs"]))),
        "freqs_mhz": (spec["freqs"] + center_freq) / 1e6,
        "t0": t_offset, "t1": t_offset + n_slices * spec["slice_s"],
        "boxes": [((b["f_lo_hz"] + center_freq) / 1e6, (b["f_hi_hz"] + center_freq) / 1e6,
                   t_offset + b["start_s"], b["duration_s"]) for b in regions["bursts"]],
        "channels": [(ch["center_hz"] + center_freq) / 1e6
                     for e in regions["emitters"] for ch in e["channels"]],
    }


def marks_from(detections):
    """(freq_mhz, bw_khz, power_db, label, verdict) per detection; one per hop channel for FHSS."""
    marks = []
    for i, d in enumerate(detections, start=1):
        if d.get("fhss"):
            for c in d["fhss"]["channels_mhz"]:
                marks.append((c, d["fhss"]["channel_bandwidth_khz"], d["power_db"], str(i), d.get("verdict")))
        else:
            marks.append((d["freq_mhz"], d["bandwidth_khz"], d["power_db"], str(i), d.get("verdict")))
    return marks


def draw_spectrum(ax, welch_mhz, welch_db, maxhold_mhz, maxhold_db, welch_thr_db, burst_thr_db, marks):
    ax.plot(welch_mhz, welch_db, color="#0072B2", lw=0.8, label="Welch average")
    if maxhold_db is not None:
        ax.plot(maxhold_mhz, maxhold_db, color="#D55E00", lw=0.8, alpha=0.85, label="Max-hold (4 ms slices)")
    ax.axhline(welch_thr_db, color="black", ls="--", lw=0.9, label="Welch CFAR threshold (median + 10 dB)")
    if burst_thr_db is not None:
        ax.plot(maxhold_mhz, burst_thr_db, color="#009E73", ls=":", lw=1.0,
                label="Burst CFAR threshold (per-bin noise + threshold)")
    for f, bw, p, label, verdict in marks:
        color = VERDICT_COLORS.get(verdict, "#CC79A7")
        ax.axvspan(f - bw / 2e3, f + bw / 2e3, color=color, alpha=0.18, lw=0)
        ax.plot([f], [p], marker="v", color=color, markersize=7, zorder=4)
        ax.annotate(label, (f, p), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("PSD (dB/Hz, rel. full scale)")
    ax.set_xlim(welch_mhz[0], welch_mhz[-1])
    lo = np.percentile(welch_db, 5) - 8
    hi = max(np.max(welch_db), np.max(maxhold_db) if maxhold_db is not None else -np.inf) + 8
    ax.set_ylim(lo, hi)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10, frameon=False)


def draw_waterfall(ax, views):
    """Spectrogram (time down) of one or more consecutive views, burst boxes and FHSS channel labels."""
    views = [v for v in views if len(v["power_db"])]
    if not views:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return
    allp = np.concatenate([v["power_db"].ravel() for v in views])
    vmin, vmax = np.percentile(allp, 50) - 3, np.percentile(allp, 99.9)
    for v in views:
        im = ax.imshow(v["power_db"], aspect="auto", origin="upper", cmap="viridis", vmin=vmin, vmax=vmax,
                       extent=[v["freqs_mhz"][0], v["freqs_mhz"][-1], v["t1"], v["t0"]], interpolation="nearest")
        for f_lo, f_hi, t, dur in v["boxes"]:
            ax.add_patch(Rectangle((f_lo, t), f_hi - f_lo, dur, fill=False, ec="white", lw=0.8))
    channels = sorted({round(c, 3) for v in views for c in v["channels"]})
    for k, c in enumerate(channels, start=1):
        ax.annotate(f"ch{k} {c:.3f}", (c, views[0]["t0"]), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color="black", annotation_clip=False)
    ax.set_xlim(views[0]["freqs_mhz"][0], views[0]["freqs_mhz"][-1])
    ax.set_ylim(max(v["t1"] for v in views), min(v["t0"] for v in views))
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Time (s)")
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, label="PSD (dB/Hz)", pad=0.01)


def _spectrum_args(regions, center_freq):
    spec = regions["spectrum"]
    mh_mhz = (spec["freqs"] + center_freq) / 1e6
    burst_thr = 10 * np.log10(spec["noise"]) + spec["threshold_db"] if len(spec["power"]) else None
    return (regions["freqs_mhz"], regions["psd_db"], mh_mhz, spec["maxhold_db"], regions["threshold"], burst_thr)


def save_capture_plots(regions, detections, center_freq, gain_db, when, out_dir=SPECTRA_DIR,
                       combined_path=None, combined_title=None):
    """Spectrum + waterfall for one capture; returns the saved paths (PNG + PDF each)."""
    stem = os.path.join(out_dir, file_stem(when, center_freq))
    title = title_for(center_freq, gain_db, headline_verdict(detections), when)
    marks = marks_from(detections)
    view = capture_view(regions, center_freq)
    paths = []

    fig = Figure(figsize=(DOUBLE_COL, 4.0))
    ax = fig.add_subplot()
    draw_spectrum(ax, *_spectrum_args(regions, center_freq), marks)
    ax.set_title(title)
    paths += save_figure(fig, stem + "_spectrum")

    fig = Figure(figsize=(DOUBLE_COL, 3.6))
    ax = fig.add_subplot()
    draw_waterfall(ax, [view])
    ax.set_title(title, pad=16)
    paths += save_figure(fig, stem + "_waterfall")

    if combined_path:
        fig = Figure(figsize=(DOUBLE_COL, 8.0))
        ax1, ax2 = fig.subplots(2, 1)
        draw_spectrum(ax1, *_spectrum_args(regions, center_freq), marks)
        draw_waterfall(ax2, [view])
        fig.suptitle(combined_title or title, fontsize=10)
        fig.subplots_adjust(hspace=0.75)
        paths += save_figure(fig, combined_path)
    return paths


class SessionPlotter:
    """Accumulates --stream data for a session summary: max-hold, mean Welch, last ~10 s of blocks."""

    def __init__(self, center_freq, block_seconds, history_s=10.0):
        self.center_freq = center_freq
        self.maxhold_db = None
        self.welch_sum = None
        self.blocks = 0
        self.ring = deque(maxlen=max(1, int(round(history_s / block_seconds))))
        self.started = datetime.now()

    def update(self, regions, captured_at):
        spec = regions["spectrum"]
        if spec["maxhold_db"] is not None:
            self.maxhold_db = (spec["maxhold_db"].copy() if self.maxhold_db is None
                               else np.maximum(self.maxhold_db, spec["maxhold_db"]))
        lin = 10 ** (regions["psd_db"] / 10)
        self.welch_sum = lin.copy() if self.welch_sum is None else self.welch_sum + lin
        self.blocks += 1
        self.welch_mhz = regions["freqs_mhz"]
        self.maxhold_mhz = (spec["freqs"] + self.center_freq) / 1e6
        self.threshold = regions["threshold"]
        self.burst_thr = 10 * np.log10(spec["noise"]) + spec["threshold_db"]
        view = capture_view(regions, self.center_freq)
        self.ring.append((captured_at, view))

    def snapshot(self, detections, gain_db, when):
        """Copy everything a plot needs, so drawing can happen on another thread."""
        if not self.blocks:
            return None
        last = self.ring[-1][0]
        views = []
        for at, v in self.ring:
            shift = (at - last).total_seconds() - (v["t1"] - v["t0"])
            views.append({**v, "t0": v["t0"] + shift, "t1": v["t1"] + shift,
                          "boxes": [(a, b, t + shift, d) for a, b, t, d in v["boxes"]]})
        return {"welch_mhz": self.welch_mhz, "welch_db": 10 * np.log10(self.welch_sum / self.blocks),
                "maxhold_mhz": self.maxhold_mhz, "maxhold_db": self.maxhold_db.copy(),
                "threshold": self.threshold, "burst_thr": self.burst_thr.copy(), "views": views,
                "marks": marks_from(detections), "verdict": headline_verdict(detections),
                "gain_db": gain_db, "when": when, "center_freq": self.center_freq}


def save_session_plots(snap, out_dir=SPECTRA_DIR, suffix="session"):
    """Session max-hold spectrum + waterfall of the ring buffer; returns the saved paths."""
    stem = os.path.join(out_dir, file_stem(snap["when"], snap["center_freq"]))
    title = title_for(snap["center_freq"], snap["gain_db"], snap["verdict"], snap["when"])
    paths = []
    fig = Figure(figsize=(DOUBLE_COL, 4.0))
    ax = fig.add_subplot()
    draw_spectrum(ax, snap["welch_mhz"], snap["welch_db"], snap["maxhold_mhz"], snap["maxhold_db"],
                  snap["threshold"], snap["burst_thr"], snap["marks"])
    ax.set_title(title + "  (session max-hold)")
    paths += save_figure(fig, f"{stem}_{suffix}_spectrum")
    fig = Figure(figsize=(DOUBLE_COL, 4.0))
    ax = fig.add_subplot()
    draw_waterfall(ax, snap["views"])
    ax.set_ylabel("Time relative to last block (s)")
    ax.set_title(title + "  (last ~10 s)", pad=16)
    paths += save_figure(fig, f"{stem}_{suffix}_waterfall")
    return paths


class PlotWorker(threading.Thread):
    """Draws snapshot plots in the background. submit() never blocks; busy -> snapshot skipped."""

    def __init__(self, out_dir=SPECTRA_DIR):
        super().__init__(daemon=True)
        self.jobs = queue.Queue(maxsize=1)
        self.out_dir = out_dir
        self.saved = []
        self.skipped = 0

    def submit(self, snap):
        try:
            self.jobs.put_nowait(snap)
        except queue.Full:
            self.skipped += 1

    def run(self):
        while True:
            snap = self.jobs.get()
            if snap is None:
                return
            self.saved += save_session_plots(snap, self.out_dir, suffix="snapshot")

    def stop(self, timeout=30):
        """Finish the snapshot in progress (if any), then exit."""
        try:
            self.jobs.put(None, timeout=timeout)
        except queue.Full:
            pass
        self.join(timeout)
