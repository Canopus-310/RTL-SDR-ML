"""
Security Scan Report Generator

Produces timestamped JSON and human-readable text reports
from RF scan results, ML classifications, fingerprint matches,
and threat assessments.
"""

import json
from datetime import datetime


class ScanReport:
    """Generate security assessment reports from RF scan data."""

    def __init__(self, recording, center_freq_mhz, sample_rate):
        self.recording = recording
        self.center_freq_mhz = center_freq_mhz
        self.sample_rate = sample_rate
        self.scan_time = datetime.now()
        self.environment = {}
        self.detections = []

    def set_environment(self, noise_floor_db, num_signals, duration_s, num_samples):
        """Set scan environment metadata."""
        self.environment = {
            "noise_floor_db": noise_floor_db,
            "signals_detected": num_signals,
            "duration_s": round(duration_s, 1),
            "num_samples": num_samples,
        }

    def add_detection(self, detection):
        """Add a fully assessed detection (with ML, fingerprint, and threat data)."""
        self.detections.append(detection)

    def _build_report(self):
        """Build the full report data structure."""
        threat_counts = {}
        for d in self.detections:
            sev = d.get("threat_severity", "NONE")
            threat_counts[sev] = threat_counts.get(sev, 0) + 1
        verdict_counts = {}
        for d in self.detections:
            v = d.get("verdict", "UNKNOWN")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

        drone_detections = [d for d in self.detections
                           if d.get("threat_score", 0) >= 40]
        max_threat = max((d.get("threat_score", 0) for d in self.detections),
                        default=0)

        if max_threat >= 80:
            overall = "CRITICAL — high-confidence drone activity detected"
        elif max_threat >= 60:
            overall = "ELEVATED — signals consistent with drone activity"
        elif max_threat >= 40:
            overall = "MODERATE — possible drone signals, monitoring advised"
        elif max_threat >= 20:
            overall = "LOW — some signals in drone bands, likely benign"
        else:
            overall = "CLEAR — no drone threat indicators"

        return {
            "metadata": {
                "tool": "RTL-SDR Drone RF Signal Classifier",
                "version": "1.0",
                "recording": self.recording,
                "center_freq_mhz": self.center_freq_mhz,
                "sample_rate_hz": self.sample_rate,
                "scan_date": self.scan_time.isoformat(),
            },
            "environment": self.environment,
            "summary": {
                "overall_assessment": overall,
                "max_threat_score": max_threat,
                "total_signals": len(self.detections),
                "drone_candidates": len(drone_detections),
                "threat_counts": threat_counts,
                "verdict_counts": verdict_counts,
            },
            "detections": self.detections,
        }

    def save_json(self, filename):
        """Save report as JSON."""
        report = self._build_report()
        with open(filename, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return report

    def save_text(self, filename):
        """Save report as formatted text."""
        report = self._build_report()
        meta = report["metadata"]
        env = report["environment"]
        summary = report["summary"]

        with open(filename, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("RF DRONE THREAT ASSESSMENT REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("SCAN DETAILS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Tool:         {meta['tool']}\n")
            f.write(f"Recording:    {meta['recording']}\n")
            f.write(f"Center Freq:  {meta['center_freq_mhz']:.3f} MHz\n")
            f.write(f"Sample Rate:  {meta['sample_rate_hz'] / 1e6:.1f} MS/s\n")
            f.write(f"Duration:     {env.get('duration_s', 'N/A')}s\n")
            f.write(f"Samples:      {env.get('num_samples', 'N/A'):,}\n")
            f.write(f"Noise Floor:  {env.get('noise_floor_db', 'N/A'):.1f} dB\n")
            f.write(f"Date:         {self.scan_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("THREAT SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Overall:           {summary['overall_assessment']}\n")
            f.write(f"Max Threat Score:  {summary['max_threat_score']}/100\n")
            f.write(f"Signals Found:     {summary['total_signals']}\n")
            f.write(f"Drone Candidates:  {summary['drone_candidates']}\n")
            if summary["threat_counts"]:
                counts = ", ".join(f"{k}: {v}" for k, v in
                                 summary["threat_counts"].items())
                f.write(f"By Severity:       {counts}\n")
            if summary["verdict_counts"]:
                verdicts = ", ".join(f"{k}: {v}" for k, v in summary["verdict_counts"].items())
                f.write(f"Verdicts:          {verdicts}\n")
            f.write("\n")

            f.write("DETAILED DETECTIONS\n")
            f.write("-" * 60 + "\n")

            for i, d in enumerate(self.detections):
                sev = d.get("threat_severity", "NONE")
                score = d.get("threat_score", 0)
                f.write(f"\n  Signal {i+1}: {d['freq_mhz']:.3f} MHz  —  "
                        f"VERDICT: {d.get('verdict', 'UNKNOWN')}\n")
                for reason in d.get("verdict_reasons", []):
                    f.write(f"    - {reason}\n")
                f.write(f"  {'.'*50}\n")
                f.write(f"  Power:          {d.get('power_db', 'N/A'):.1f} dB\n")
                f.write(f"  Bandwidth:      {d.get('bandwidth_khz', 'N/A'):.1f} kHz\n")
                f.write(f"  ML Class:       {d.get('ml_class', 'N/A')}")
                f.write(f" ({d.get('ml_confidence', 0):.0%})\n")

                fp = d.get("fingerprint_name")
                if fp:
                    f.write(f"  Fingerprint:    {fp}")
                    f.write(f" ({d.get('fingerprint_score', 0)}/100)\n")
                else:
                    f.write(f"  Fingerprint:    No match\n")

                if "dechirp_score_db" in d:
                    lora = (f"yes (SF{d['lora_sf']}, {d['lora_bw_khz']:.0f} kHz)"
                            if d.get("lora_rule") else "no")
                    f.write(f"  LoRa:           {lora}, dechirp score {d['dechirp_score_db']:.1f} dB\n")
                    duty = d.get("duty_cycle")
                    f.write(f"  Duty Cycle:     "
                            + ("unknown" if duty is None else
                               f"{duty:.0%} ({'bursty' if d.get('is_bursty') else 'continuous'})")
                            + "\n")

                fhss = d.get("fhss")
                if fhss:
                    channels = ", ".join(f"{c:.3f}" for c in fhss["channels_mhz"])
                    f.write(f"  FHSS:           {len(fhss['channels_mhz'])} channels ({channels} MHz), "
                            f"{fhss['hop_rate_hz']:.1f} hops/s, dwell {fhss['dwell_ms']:.1f} ms\n")

                f.write(f"  Threat Score:   {score}/100 ({sev})\n")

                factors = d.get("threat_factors", [])
                if factors:
                    f.write(f"  Risk Factors:\n")
                    for fac in factors:
                        f.write(f"    [{fac['points']:+3d}] {fac['factor']}: "
                               f"{fac['detail']}\n")

                recs = d.get("recommendations", [])
                if recs:
                    f.write(f"  Recommendations:\n")
                    for r in recs:
                        f.write(f"    - {r}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")

        return report

    def save(self):
        """Save both JSON and text reports. Returns filenames."""
        stamp = self.scan_time.strftime("%Y%m%d_%H%M%S")
        json_name = f"rf_threat_report_{stamp}.json"
        txt_name = f"rf_threat_report_{stamp}.txt"
        self.save_json(json_name)
        self.save_text(txt_name)
        return json_name, txt_name