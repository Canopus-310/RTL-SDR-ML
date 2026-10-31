"""
Drone RF Fingerprint Database

Known RF profiles for common drone communication systems.
Profiles are based on published specifications and documentation.
Each profile defines the RF characteristics used to match detected signals.

Sources:
- SiK: ArduPilot docs, Holybro/3DR specifications
- TBS Crossfire: Team BlackSheep documentation
- ExpressLRS: ELRS GitHub, community documentation
- DJI: Published FCC filings and teardown analyses
"""


# Each profile defines the RF signature of a known drone system
DRONE_PROFILES = [
    {
        "id": "sik_433",
        "name": "SiK Telemetry Radio (433 MHz)",
        "manufacturer": "3DR / Holybro / mRo",
        "purpose": "MAVLink telemetry link",
        "freq_min_mhz": 433.050,
        "freq_max_mhz": 434.790,
        "bandwidth_min_khz": 10,
        "bandwidth_max_khz": 40,
        "modulation": "GFSK",
        "hopping": True,
        "hop_channels": 50,
        "burst_pattern": "bursty",
        "duty_cycle_typical": 0.1,
        "data_rates_kbps": [19.2, 57.6, 125, 250],
        "tx_power_dbm": [10, 17, 20, 27],
        "protocol": "MAVLink",
        "encrypted": False,
        "detectable_by_rtlsdr": True,
        "notes": "Open-source firmware. FHSS with adaptive TDM. Common on ArduPilot/PX4 drones.",
    },
    {
        "id": "sik_915",
        "name": "SiK Telemetry Radio (915 MHz)",
        "manufacturer": "3DR / Holybro / mRo",
        "purpose": "MAVLink telemetry link",
        "freq_min_mhz": 902.000,
        "freq_max_mhz": 928.000,
        "bandwidth_min_khz": 10,
        "bandwidth_max_khz": 40,
        "modulation": "GFSK",
        "hopping": True,
        "hop_channels": 50,
        "burst_pattern": "bursty",
        "duty_cycle_typical": 0.1,
        "data_rates_kbps": [19.2, 57.6, 125, 250],
        "tx_power_dbm": [10, 17, 20, 27],
        "protocol": "MAVLink",
        "encrypted": False,
        "detectable_by_rtlsdr": True,
        "notes": "US variant of SiK radio. Same firmware, different band.",
    },
    {
        "id": "crossfire_868",
        "name": "TBS Crossfire (868 MHz)",
        "manufacturer": "Team BlackSheep",
        "purpose": "RC control link",
        "freq_min_mhz": 850.000,
        "freq_max_mhz": 870.000,
        "bandwidth_min_khz": 125,
        "bandwidth_max_khz": 500,
        "modulation": "LoRa",
        "hopping": True,
        "hop_channels": 12,
        "burst_pattern": "periodic",
        "duty_cycle_typical": 0.5,
        "data_rates_kbps": [4.39],
        "tx_power_dbm": [14, 23, 27, 30, 33],
        "protocol": "CRSF",
        "encrypted": True,
        "detectable_by_rtlsdr": True,
        "notes": "EU variant. LoRa at 50Hz, FSK at 150Hz. Proprietary encrypted link. Semtech SX1272 chipset.",
    },
    {
        "id": "crossfire_915",
        "name": "TBS Crossfire (915 MHz)",
        "manufacturer": "Team BlackSheep",
        "purpose": "RC control link",
        "freq_min_mhz": 902.000,
        "freq_max_mhz": 927.000,
        "bandwidth_min_khz": 125,
        "bandwidth_max_khz": 500,
        "modulation": "LoRa",
        "hopping": True,
        "hop_channels": 12,
        "burst_pattern": "periodic",
        "duty_cycle_typical": 0.5,
        "data_rates_kbps": [4.39],
        "tx_power_dbm": [14, 23, 27, 30, 33],
        "protocol": "CRSF",
        "encrypted": True,
        "detectable_by_rtlsdr": True,
        "notes": "US/international variant. Up to 2W output. Professional long-range standard.",
    },
    {
        "id": "elrs_868",
        "name": "ExpressLRS (868 MHz)",
        "manufacturer": "Open Source (various: BetaFPV, Happymodel, etc.)",
        "purpose": "RC control link",
        "freq_min_mhz": 863.000,
        "freq_max_mhz": 870.000,
        "bandwidth_min_khz": 125,
        "bandwidth_max_khz": 500,
        "modulation": "LoRa",
        "hopping": True,
        "hop_channels": 13,
        "burst_pattern": "periodic",
        "duty_cycle_typical": 0.5,
        "data_rates_kbps": [4.39],
        "tx_power_dbm": [10, 17, 20, 27],
        "protocol": "CRSF",
        "encrypted": False,
        "detectable_by_rtlsdr": True,
        "notes": "Open-source. LoRa at all packet rates. EU variant.",
    },
    {
        "id": "elrs_915",
        "name": "ExpressLRS (915 MHz)",
        "manufacturer": "Open Source (various: BetaFPV, Happymodel, etc.)",
        "purpose": "RC control link",
        "freq_min_mhz": 902.000,
        "freq_max_mhz": 928.000,
        "bandwidth_min_khz": 125,
        "bandwidth_max_khz": 500,
        "modulation": "LoRa",
        "hopping": True,
        "hop_channels": 80,
        "burst_pattern": "periodic",
        "duty_cycle_typical": 0.5,
        "data_rates_kbps": [4.39],
        "tx_power_dbm": [10, 17, 20, 27],
        "protocol": "CRSF",
        "encrypted": False,
        "detectable_by_rtlsdr": True,
        "notes": "Open-source. US variant. Up to 1000Hz packet rate on 2.4GHz version.",
    },
    {
        "id": "frsky_r9_868",
        "name": "FrSky R9 (868 MHz)",
        "manufacturer": "FrSky",
        "purpose": "RC control link",
        "freq_min_mhz": 860.000,
        "freq_max_mhz": 870.000,
        "bandwidth_min_khz": 125,
        "bandwidth_max_khz": 500,
        "modulation": "LoRa",
        "hopping": True,
        "hop_channels": 24,
        "burst_pattern": "periodic",
        "duty_cycle_typical": 0.5,
        "data_rates_kbps": [4.39],
        "tx_power_dbm": [14, 20, 27],
        "protocol": "FrSky S.Port",
        "encrypted": False,
        "detectable_by_rtlsdr": True,
        "notes": "LoRa-based. Semtech SX1272. EU variant.",
    },
    {
        "id": "frsky_r9_915",
        "name": "FrSky R9 (915 MHz)",
        "manufacturer": "FrSky",
        "purpose": "RC control link",
        "freq_min_mhz": 902.000,
        "freq_max_mhz": 928.000,
        "bandwidth_min_khz": 125,
        "bandwidth_max_khz": 500,
        "modulation": "LoRa",
        "hopping": True,
        "hop_channels": 24,
        "burst_pattern": "periodic",
        "duty_cycle_typical": 0.5,
        "data_rates_kbps": [4.39],
        "tx_power_dbm": [14, 20, 27],
        "protocol": "FrSky S.Port",
        "encrypted": False,
        "detectable_by_rtlsdr": True,
        "notes": "LoRa-based. US variant.",
    },
    {
        "id": "dji_ocusync_24",
        "name": "DJI OcuSync / O3 (2.4 GHz)",
        "manufacturer": "DJI",
        "purpose": "RC control + HD video downlink",
        "freq_min_mhz": 2400.000,
        "freq_max_mhz": 2483.500,
        "bandwidth_min_khz": 10000,
        "bandwidth_max_khz": 20000,
        "modulation": "OFDM",
        "hopping": True,
        "hop_channels": 40,
        "burst_pattern": "continuous",
        "duty_cycle_typical": 0.9,
        "data_rates_kbps": [50000],
        "tx_power_dbm": [20, 26, 30, 33],
        "protocol": "DJI proprietary",
        "encrypted": True,
        "detectable_by_rtlsdr": False,
        "notes": "NOT detectable by RTL-SDR (2.4 GHz exceeds tuner range). Included for reference. Wideband OFDM.",
    },
    {
        "id": "dji_ocusync_58",
        "name": "DJI OcuSync / O3 (5.8 GHz)",
        "manufacturer": "DJI",
        "purpose": "HD video downlink",
        "freq_min_mhz": 5725.000,
        "freq_max_mhz": 5850.000,
        "bandwidth_min_khz": 10000,
        "bandwidth_max_khz": 40000,
        "modulation": "OFDM",
        "hopping": True,
        "hop_channels": 8,
        "burst_pattern": "continuous",
        "duty_cycle_typical": 0.95,
        "data_rates_kbps": [50000],
        "tx_power_dbm": [20, 25, 30],
        "protocol": "DJI proprietary",
        "encrypted": True,
        "detectable_by_rtlsdr": False,
        "notes": "NOT detectable by RTL-SDR. Video downlink. Wideband OFDM.",
    },
]


class DroneFingerprinter:
    """Match detected RF signals against known drone profiles."""

    def __init__(self):
        self.profiles = DRONE_PROFILES
        # Only match against profiles the RTL-SDR can actually detect
        self.detectable = [p for p in self.profiles if p["detectable_by_rtlsdr"]]

    def match(self, freq_mhz, bandwidth_khz, is_bursty=False, duty_cycle=None, hopping=False,
              lora=False):
        """
        Score a detected signal against all known drone profiles.

        Args:
            freq_mhz: Center frequency of detected signal in MHz
            bandwidth_khz: Measured bandwidth in kHz
            is_bursty: Whether the signal appears bursty vs continuous
            duty_cycle: Estimated duty cycle (0-1), None if unknown
            hopping: Whether the signal was seen frequency hopping (FHSS)
            lora: Whether the dechirp rule identified LoRa chirp modulation

        Returns:
            List of (profile, score, reasons) tuples, sorted by score descending.
            Score is 0-100. Only profiles scoring > 0 are returned.
        """
        matches = []

        for profile in self.detectable:
            score = 0
            reasons = []

            # --- Frequency match (most important, 0-40 points) ---
            if profile["freq_min_mhz"] <= freq_mhz <= profile["freq_max_mhz"]:
                score += 40
                reasons.append(f"frequency {freq_mhz:.1f} MHz is within {profile['name']} band "
                             f"({profile['freq_min_mhz']}-{profile['freq_max_mhz']} MHz)")
            else:
                # Check proximity (within 5 MHz)
                dist = min(abs(freq_mhz - profile["freq_min_mhz"]),
                          abs(freq_mhz - profile["freq_max_mhz"]))
                if dist < 5:
                    proximity_score = max(0, 15 - dist * 3)
                    score += proximity_score
                    reasons.append(f"frequency {freq_mhz:.1f} MHz is near {profile['name']} band "
                                 f"({dist:.1f} MHz away)")
                else:
                    continue  # too far away, skip this profile entirely

            # --- Bandwidth match (0-30 points) ---
            bw_min = profile["bandwidth_min_khz"]
            bw_max = profile["bandwidth_max_khz"]
            if bw_min <= bandwidth_khz <= bw_max:
                score += 30
                reasons.append(f"bandwidth {bandwidth_khz:.0f} kHz matches expected range "
                             f"({bw_min}-{bw_max} kHz)")
            elif bandwidth_khz < bw_min:
                ratio = bandwidth_khz / bw_min
                if ratio > 0.3:
                    bw_score = int(20 * ratio)
                    score += bw_score
                    reasons.append(f"bandwidth {bandwidth_khz:.0f} kHz is narrower than expected "
                                 f"({bw_min}-{bw_max} kHz) but plausible")
            elif bandwidth_khz > bw_max:
                ratio = bw_max / bandwidth_khz
                if ratio > 0.3:
                    bw_score = int(20 * ratio)
                    score += bw_score
                    reasons.append(f"bandwidth {bandwidth_khz:.0f} kHz is wider than expected "
                                 f"({bw_min}-{bw_max} kHz) but plausible")

            # --- Burst pattern match (0-20 points) ---
            if is_bursty and profile["burst_pattern"] == "bursty":
                score += 20
                reasons.append("bursty transmission pattern matches telemetry link behavior")
            elif not is_bursty and profile["burst_pattern"] == "continuous":
                score += 15
                reasons.append("continuous transmission matches video/control link")
            elif not is_bursty and profile["burst_pattern"] == "periodic":
                score += 10
                reasons.append("steady signal consistent with periodic control packets")

            # --- Duty cycle match (0-10 points) ---
            if duty_cycle is not None and profile.get("duty_cycle_typical"):
                dc_diff = abs(duty_cycle - profile["duty_cycle_typical"])
                if dc_diff < 0.2:
                    score += 10
                    reasons.append(f"duty cycle ~{duty_cycle:.0%} matches expected "
                                 f"~{profile['duty_cycle_typical']:.0%}")
                elif dc_diff < 0.4:
                    score += 5

            # --- Frequency hopping (0-10 points, total capped at 100) ---
            if hopping and profile["hopping"]:
                score = min(score + 10, 100)
                reasons.append("frequency hopping observed, matching this FHSS link")

            # --- LoRa modulation (0-10 points, total capped at 100) ---
            if lora and profile["modulation"] == "LoRa":
                score = min(score + 10, 100)
                reasons.append("LoRa chirp modulation confirmed by dechirping")

            if score > 0:
                matches.append((profile, score, reasons))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def get_best_match(self, freq_mhz, bandwidth_khz, is_bursty=False, duty_cycle=None,
                       hopping=False, lora=False):
        """Return the single best matching profile, or None."""
        matches = self.match(freq_mhz, bandwidth_khz, is_bursty, duty_cycle, hopping, lora)
        if matches and matches[0][1] >= 30:
            return matches[0]
        return None

    def list_profiles(self):
        """Print all known profiles."""
        print("=" * 70)
        print("DRONE RF FINGERPRINT DATABASE")
        print(f"Profiles: {len(self.profiles)} total, "
              f"{len(self.detectable)} detectable by RTL-SDR")
        print("=" * 70)

        for p in self.profiles:
            rtlsdr = "YES" if p["detectable_by_rtlsdr"] else "NO (out of range)"
            print(f"\n  [{p['id']}] {p['name']}")
            print(f"    Manufacturer:  {p['manufacturer']}")
            print(f"    Purpose:       {p['purpose']}")
            print(f"    Frequency:     {p['freq_min_mhz']}-{p['freq_max_mhz']} MHz")
            print(f"    Bandwidth:     {p['bandwidth_min_khz']}-{p['bandwidth_max_khz']} kHz")
            print(f"    Modulation:    {p['modulation']}")
            print(f"    Hopping:       {'FHSS' if p['hopping'] else 'Fixed'} "
                  f"({p.get('hop_channels', '?')} channels)")
            print(f"    Protocol:      {p['protocol']}")
            print(f"    Encrypted:     {'Yes' if p['encrypted'] else 'No'}")
            print(f"    RTL-SDR:       {rtlsdr}")

        print("\n" + "=" * 70)


def main():
    """Demo: list all profiles and test matching."""
    fp = DroneFingerprinter()
    fp.list_profiles()

    print("\n\n--- MATCHING TESTS ---\n")

    # Test 1: Signal in 433 MHz band with narrow bandwidth (should match SiK)
    print("Test 1: 433.5 MHz, 25 kHz bandwidth, bursty")
    matches = fp.match(freq_mhz=433.5, bandwidth_khz=25, is_bursty=True, duty_cycle=0.15)
    for profile, score, reasons in matches:
        print(f"  [{score}/100] {profile['name']}")
        for r in reasons:
            print(f"    - {r}")

    # Test 2: Signal in 915 MHz band with LoRa-like bandwidth
    print("\nTest 2: 915.0 MHz, 250 kHz bandwidth, periodic")
    matches = fp.match(freq_mhz=915.0, bandwidth_khz=250, is_bursty=False, duty_cycle=0.5)
    for profile, score, reasons in matches:
        print(f"  [{score}/100] {profile['name']}")
        for r in reasons:
            print(f"    - {r}")

    # Test 3: FM broadcast (should NOT match any drone profile)
    print("\nTest 3: 104.5 MHz, 172 kHz bandwidth (FM station)")
    matches = fp.match(freq_mhz=104.5, bandwidth_khz=172, is_bursty=False)
    if not matches:
        print("  No drone profile match (correct -- this is broadcast FM)")
    else:
        for profile, score, reasons in matches:
            print(f"  [{score}/100] {profile['name']}")

    print("\n--- MATCHING TESTS COMPLETE ---")


if __name__ == "__main__":
    main()