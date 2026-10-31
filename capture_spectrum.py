import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from scipy.io import wavfile

from plot_style import apply_style, save_figure

apply_style()

if len(sys.argv) < 2:
    print("Usage: python capture_spectrum.py <recording.wav> [center_freq_hz]")
    print("Example: python capture_spectrum.py recording.wav 103959385")
    sys.exit(1)

recording = sys.argv[1]
center_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 100e6

sample_rate, data = wavfile.read(recording)

# SDR++ baseband WAV has two channels: I and Q
iq = data[:, 0] + 1j * data[:, 1]
iq = iq / 32768.0   # normalize Int16 to -1 to +1

print(f"Loaded {len(iq)} IQ samples at {sample_rate} Hz")

plt.figure(figsize=(10, 6))
plt.psd(iq, NFFT=1024, Fs=sample_rate / 1e6, Fc=center_freq / 1e6)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Relative power (dB)")
plt.title(f"Spectrum around {center_freq/1e6} MHz")
plt.grid(True)
plt.tight_layout()
# Also save the figure (300 dpi PNG + PDF) before showing the window
stem = os.path.join("results", "spectra", f"{datetime.now():%Y%m%d_%H%M%S}_{int(round(center_freq))}Hz_spectrum")
for path in save_figure(plt.gcf(), stem):
    print(f"Saved {os.path.normpath(path)}")
plt.show()