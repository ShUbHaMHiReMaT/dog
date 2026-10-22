"""
Live microphone check for the PRANIIVA collar (INMP441 on the ESP32-S3). Shows the sound level as a bar,
asks you to stay quiet, then to clap, and tells you whether the microphone really responds to sound.

    python tools/check_mic.py --port COM15

The firmware only ever sends sound LEVELS (dBFS, twice a second), never audio.
"""
from __future__ import annotations

import argparse
import sys
import time

import serial


def levels(ser, seconds, label):
    out = []
    t_end = time.time() + seconds
    ser.reset_input_buffer()
    while time.time() < t_end:
        line = ser.readline().decode(errors="replace").strip()
        if not line.startswith("#A,"):
            continue
        p = line[3:].split(",")
        if len(p) < 4:
            continue
        try:
            rms, peak, ok = float(p[1]), float(p[2]), int(p[3])
        except ValueError:
            continue
        out.append((rms, peak, ok))
        bar = "#" * max(0, min(60, int((peak + 90) / 1.5)))
        left = max(0, t_end - time.time())
        print(f"\r  {label:<12} {left:4.1f}s  peak {peak:6.1f} dBFS  |{bar:<60}|", end="", flush=True)
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM15")
    a = ap.parse_args()
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = a.port, 921600, 0.5
    ser.dtr = False
    ser.rts = False
    ser.open()
    time.sleep(0.5)
    print("Microphone check - watch the bar.\n")
    input("STEP 1: stay QUIET for 8 seconds. Press Enter to start... ")
    quiet = levels(ser, 8, "quiet")
    input("STEP 2: press Enter, then CLAP LOUDLY next to the microphone 4-5 times (8 seconds)... ")
    loud = levels(ser, 8, "clap!")
    ser.close()
    if not quiet or not loud:
        sys.exit("No sound-level messages received - is firmware 0.4.0 (with USE_MIC 1) flashed?")
    if all(ok == 0 for _, _, ok in quiet + loud):
        sys.exit("FAIL: the microphone sends no data at all -> check SD->GPIO7, WS->GPIO6, SCK->GPIO5, VDD->3V3, GND.")
    q_peak = sorted(p for _, p, _ in quiet)[len(quiet) // 2]        # typical quiet peak
    l_peak = max(p for _, p, _ in loud)                             # loudest clap
    q_rms = sorted(r for r, _, _ in quiet)[len(quiet) // 2]
    print(f"\n  quiet: typical level {q_rms:.1f} dBFS (peak {q_peak:.1f})   claps: loudest peak {l_peak:.1f} dBFS")
    rise = l_peak - q_peak
    if rise >= 15:
        print(f"  PASS: claps are {rise:.0f} dB above the quiet room - the microphone hears sound.")
    elif rise >= 6:
        print(f"  WEAK: claps only {rise:.0f} dB above quiet - clap closer, or check the mic hole is not covered.")
    else:
        print(f"  FAIL: no response to claps (+{rise:.0f} dB). Most likely: L/R pin not tied to GND, or WS and SCK "
              f"swapped (WS->GPIO6, SCK->GPIO5). Tell Claude the result.")


if __name__ == "__main__":
    main()
