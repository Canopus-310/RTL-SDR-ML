"""
Threat Assessment Engine

Evaluates detected RF signals for drone-related security threats.
Scores each detection across multiple risk factors and produces
a severity-graded threat assessment with reasoning.
"""


# Frequency bands of interest for drone threat assessment
DRONE_BANDS = [
    {"name": "433 MHz ISM", "min": 433.0, "max": 435.0, "risk_weight": 1.0,
     "reason": "Primary band for drone telemetry (SiK radios, DragonLink)"},
    {"name": "868 MHz ISM", "min": 860.0, "max": 870.0, "risk_weight": 0.9,
     "reason": "EU drone control band (Crossfire, ELRS, FrSky R9)"},
    {"name": "915 MHz ISM", "min": 902.0, "max": 928.0, "risk_weight": 0.9,
     "reason": "US/intl drone control band (Crossfire, ELRS, SiK)"},
]

# ML classes that indicate drone activity
DRONE_ML_CLASSES = ["DRONE_TELEMETRY", "DRONE_VIDEO"]
SUSPECT_ML_CLASSES = ["NARROWBAND_SIGNAL", "INTERFERENCE"]

SEVERITY_LEVELS = {
    "CRITICAL": {"min_score": 80, "color": "RED",
                 "action": "Immediate investigation required. High-confidence drone signal detected."},
    "HIGH":     {"min_score": 60, "color": "ORANGE",
                 "action": "Investigate promptly. Signal characteristics consistent with drone activity."},
    "MEDIUM":   {"min_score": 40, "color": "YELLOW",
                 "action": "Monitor and log. Possible drone-related signal, further analysis needed."},
    "LOW":      {"min_score": 20, "color": "BLUE",
                 "action": "Noted. Low probability of drone activity but signal is in a relevant band."},
    "NONE":     {"min_score": 0,  "color": "GREEN",
                 "action": "No drone threat indicators. Signal appears to be non-drone RF activity."},
}


class ThreatEngine:
    """Assess RF detections for drone-related security threats."""

    def __init__(self):
        self.assessments = []

    def assess(self, detection):
        """
        Assess a single detected signal for drone threat.

        Args:
            detection: dict with keys:
                - freq_mhz: float, center frequency
                - bandwidth_khz: float, measured bandwidth
                - power_db: float, signal power
                - ml_class: str, ML classification result
                - ml_confidence: float, ML confidence 0-1
                - fingerprint_match: tuple (profile, score, reasons) or None
                - is_bursty: bool (optional)
                - duty_cycle: float (optional)

        Returns:
            dict with threat assessment results
        """
        score = 0
        factors = []
        recommendations = []

        freq = detection.get("freq_mhz", 0)
        bw = detection.get("bandwidth_khz", 0)
        power = detection.get("power_db", -100)
        ml_class = detection.get("ml_class", "UNKNOWN")
        ml_conf = detection.get("ml_confidence", 0)
        fp_match = detection.get("fingerprint_match")
        is_bursty = detection.get("is_bursty", False)

        # --- Factor 1: Frequency band (0-25 points) ---
        in_drone_band = False
        for band in DRONE_BANDS:
            if band["min"] <= freq <= band["max"]:
                points = int(25 * band["risk_weight"])
                score += points
                factors.append({
                    "factor": "FREQUENCY",
                    "points": points,
                    "detail": f"Signal at {freq:.1f} MHz is in {band['name']} band — "
                              f"{band['reason']}"
                })
                in_drone_band = True
                break

        if not in_drone_band:
            factors.append({
                "factor": "FREQUENCY",
                "points": 0,
                "detail": f"Signal at {freq:.1f} MHz is not in a known drone band"
            })

        # --- Factor 2: ML classification (0-25 points) ---
        if ml_class in DRONE_ML_CLASSES:
            ml_points = int(25 * ml_conf)
            score += ml_points
            factors.append({
                "factor": "ML_CLASSIFICATION",
                "points": ml_points,
                "detail": f"ML classifier identified signal as {ml_class} "
                          f"({ml_conf:.0%} confidence)"
            })
        elif ml_class in SUSPECT_ML_CLASSES:
            ml_points = int(10 * ml_conf)
            score += ml_points
            factors.append({
                "factor": "ML_CLASSIFICATION",
                "points": ml_points,
                "detail": f"ML classifier returned {ml_class} ({ml_conf:.0%}) — "
                          f"not definitively drone but warrants attention"
            })
        else:
            factors.append({
                "factor": "ML_CLASSIFICATION",
                "points": 0,
                "detail": f"ML classifier identified signal as {ml_class} "
                          f"({ml_conf:.0%}) — not consistent with drone activity"
            })

        # --- Factor 3: Fingerprint match (0-30 points) ---
        if fp_match is not None:
            profile, fp_score, fp_reasons = fp_match
            # Scale fingerprint score (0-100) to threat points (0-30)
            fp_points = int(30 * fp_score / 100)
            score += fp_points
            factors.append({
                "factor": "FINGERPRINT",
                "points": fp_points,
                "detail": f"Matches known drone profile: {profile['name']} "
                          f"(match score {fp_score}/100)",
                "profile_id": profile["id"],
                "profile_name": profile["name"],
                "match_reasons": fp_reasons,
            })

            # Add protocol-specific recommendations
            if not profile.get("encrypted", False):
                recommendations.append(
                    f"WARNING: {profile['name']} uses unencrypted {profile['protocol']} "
                    f"protocol — vulnerable to interception and spoofing")
            if profile.get("protocol") == "MAVLink":
                recommendations.append(
                    f"MAVLink telemetry link detected — drone flight data, GPS position, "
                    f"and commands may be exposed if link is unencrypted")
        else:
            factors.append({
                "factor": "FINGERPRINT",
                "points": 0,
                "detail": "No matching drone RF profile found"
            })

        # --- Factor 4: Signal characteristics (0-15 points) ---
        char_points = 0

        # Narrow bandwidth is more suspicious (drone-like)
        if bw < 50:
            char_points += 5
            factors.append({
                "factor": "BANDWIDTH",
                "points": 5,
                "detail": f"Narrow bandwidth ({bw:.0f} kHz) consistent with "
                          f"telemetry/control link"
            })
        elif bw < 200:
            char_points += 2
            factors.append({
                "factor": "BANDWIDTH",
                "points": 2,
                "detail": f"Medium bandwidth ({bw:.0f} kHz) — could be data link"
            })

        # Bursty pattern typical of telemetry
        if is_bursty:
            char_points += 5
            factors.append({
                "factor": "BURST_PATTERN",
                "points": 5,
                "detail": "Bursty transmission pattern typical of telemetry links"
            })

        # Strong signal = closer proximity
        if power > -50:
            char_points += 5
            factors.append({
                "factor": "PROXIMITY",
                "points": 5,
                "detail": f"Strong signal ({power:.0f} dB) suggests transmitter "
                          f"is in close proximity"
            })
        elif power > -70:
            char_points += 2
            factors.append({
                "factor": "PROXIMITY",
                "points": 2,
                "detail": f"Moderate signal ({power:.0f} dB) — transmitter may be "
                          f"within a few hundred meters"
            })

        score += char_points

        # --- Factor 5: Combined risk multiplier (0-5 bonus) ---
        # If signal is in a drone band AND matches a fingerprint, extra concern
        if in_drone_band and fp_match is not None:
            score += 5
            factors.append({
                "factor": "COMBINED_RISK",
                "points": 5,
                "detail": "Signal is in a drone band AND matches a known drone profile "
                          "— high correlation"
            })

        # --- Determine severity ---
        score = min(score, 100)
        severity = "NONE"
        for level, info in SEVERITY_LEVELS.items():
            if score >= info["min_score"]:
                severity = level
                break

        # --- Generate recommendations ---
        if severity in ("CRITICAL", "HIGH"):
            recommendations.append("Conduct visual scan of the area for drone activity")
            recommendations.append("Log signal details for incident report")
            if in_drone_band:
                recommendations.append(
                    "Consider continuous monitoring of this frequency band")
        elif severity == "MEDIUM":
            recommendations.append("Continue monitoring — re-scan in 5 minutes")
            recommendations.append("Check if authorized drone operations are scheduled")
        elif severity == "LOW":
            recommendations.append("No immediate action required — log for baseline")

        assessment = {
            "freq_mhz": freq,
            "bandwidth_khz": bw,
            "power_db": power,
            "ml_class": ml_class,
            "threat_score": score,
            "severity": severity,
            "severity_info": SEVERITY_LEVELS[severity],
            "factors": factors,
            "recommendations": recommendations,
        }

        self.assessments.append(assessment)
        return assessment

    def print_assessment(self, assessment):
        """Pretty-print a single threat assessment."""
        sev = assessment["severity"]
        info = assessment["severity_info"]

        print(f"\n  {'='*56}")
        print(f"  THREAT SCORE: {assessment['threat_score']}/100  |  "
              f"Severity: {sev} ({info['color']})")
        print(f"  {'='*56}")
        print(f"  Signal: {assessment['freq_mhz']:.1f} MHz  |  "
              f"{assessment['bandwidth_khz']:.0f} kHz  |  "
              f"{assessment['power_db']:.0f} dB  |  ML: {assessment['ml_class']}")
        print(f"  Action: {info['action']}")

        print(f"\n  Risk Factors:")
        for f in assessment["factors"]:
            print(f"    [{f['points']:+3d}] {f['factor']:20s} {f['detail']}")

        if assessment["recommendations"]:
            print(f"\n  Recommendations:")
            for r in assessment["recommendations"]:
                print(f"    - {r}")

    def summary(self):
        """Print overall threat summary."""
        if not self.assessments:
            print("\nNo signals assessed.")
            return

        print("\n" + "=" * 60)
        print("THREAT ASSESSMENT SUMMARY")
        print("=" * 60)
        print(f"Signals assessed: {len(self.assessments)}")

        by_severity = {}
        for a in self.assessments:
            sev = a["severity"]
            by_severity[sev] = by_severity.get(sev, 0) + 1

        for level in SEVERITY_LEVELS:
            count = by_severity.get(level, 0)
            if count > 0:
                color = SEVERITY_LEVELS[level]["color"]
                print(f"  {color:6s}: {count}")

        max_threat = max(self.assessments, key=lambda a: a["threat_score"])
        print(f"\nHighest threat: {max_threat['threat_score']}/100 "
              f"({max_threat['severity']}) at {max_threat['freq_mhz']:.1f} MHz")

        if max_threat["severity"] in ("CRITICAL", "HIGH"):
            print(f"\n  ** ALERT: {SEVERITY_LEVELS[max_threat['severity']]['action']} **")

        for a in self.assessments:
            self.print_assessment(a)

        print("\n" + "=" * 60)


def main():
    """Demo: test threat assessment on different signal types."""
    from drone_fingerprints import DroneFingerprinter

    engine = ThreatEngine()
    fp = DroneFingerprinter()

    print("=" * 60)
    print("THREAT ENGINE - DEMO ASSESSMENTS")
    print("=" * 60)

    # Test 1: SiK drone telemetry in 433 MHz band
    print("\n--- Test 1: Drone telemetry signal (433 MHz) ---")
    fp_result = fp.get_best_match(freq_mhz=433.5, bandwidth_khz=25,
                                   is_bursty=True, duty_cycle=0.15)
    engine.assess({
        "freq_mhz": 433.5,
        "bandwidth_khz": 25,
        "power_db": -48,
        "ml_class": "DRONE_TELEMETRY",
        "ml_confidence": 0.85,
        "fingerprint_match": fp_result,
        "is_bursty": True,
    })

    # Test 2: FM broadcast station (should be low/no threat)
    print("\n--- Test 2: FM broadcast station ---")
    fp_result_fm = fp.get_best_match(freq_mhz=104.5, bandwidth_khz=172)
    engine.assess({
        "freq_mhz": 104.5,
        "bandwidth_khz": 172,
        "power_db": -55,
        "ml_class": "FM_BROADCAST",
        "ml_confidence": 0.65,
        "fingerprint_match": fp_result_fm,
        "is_bursty": False,
    })

    # Test 3: LoRa signal in 915 MHz band (could be Crossfire/ELRS)
    print("\n--- Test 3: LoRa-like signal at 915 MHz ---")
    fp_result_lora = fp.get_best_match(freq_mhz=915.0, bandwidth_khz=250,
                                        is_bursty=False, duty_cycle=0.5)
    engine.assess({
        "freq_mhz": 915.0,
        "bandwidth_khz": 250,
        "power_db": -62,
        "ml_class": "DRONE_TELEMETRY",
        "ml_confidence": 0.72,
        "fingerprint_match": fp_result_lora,
        "is_bursty": False,
    })

    # Test 4: Unknown narrowband signal at 433 MHz
    print("\n--- Test 4: Unknown narrowband at 433 MHz ---")
    fp_result_unk = fp.get_best_match(freq_mhz=433.92, bandwidth_khz=8)
    engine.assess({
        "freq_mhz": 433.92,
        "bandwidth_khz": 8,
        "power_db": -72,
        "ml_class": "NARROWBAND_SIGNAL",
        "ml_confidence": 0.55,
        "fingerprint_match": fp_result_unk,
        "is_bursty": True,
    })

    # Print full summary
    engine.summary()


if __name__ == "__main__":
    main()