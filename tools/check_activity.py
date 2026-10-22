"""
Guided hands-on test of live activity + posture detection (no dog needed). Close the dashboard's live session first.

    python tools/check_activity.py --port COM15

Tape / hold the device flat against YOUR chest (or hold it the way it will sit on the dog's chest) and follow the
prompts. Every step is recorded to data/raw/POSETEST/ with a marker, so the result can be inspected afterwards.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from praniiva.live import ACTIVITIES, LiveCollar  # noqa: E402

CAL = [("standing", "Hold the device as on a STANDING dog's chest (you: stand upright, device flat on chest), keep STILL"),
       ("sitting", "Now as a SITTING dog: lean your upper body BACK about 45 deg (or tilt the device top backwards 45 deg), keep STILL")]
STEPS = [("standing", "Back to STANDING posture, keep still", 8),
         ("sitting", "SITTING posture again (leaning back 45 deg), keep still", 8),
         ("lying_side", "LYING ON SIDE: tilt the device 90 deg to the left or right (as if the dog rolled over), keep still", 8),
         ("walking", "WALK around normally holding the device on your chest", 10),
         ("running", "JOG on the spot / run (strong bouncing)", 8),
         ("shaking", "SHAKE the device fast in your hand, like a dog shaking water off (short bursts)", 8),
         ("small", "Small gentle movements: sway slowly, small shifts (like sniffing / eating)", 10)]


def wait_labels(live, seconds, settle=2.0):
    time.sleep(settle)
    t0 = time.time()
    n0 = len(live.labels)
    while time.time() - t0 < seconds:
        time.sleep(0.5)
        if live.labels:
            _, a, _, i = live.labels[-1]
            print(f"\r   now: {ACTIVITIES[a]:<40} motion {i['motion_g']:.3f} g  {i['dom_hz']:4.1f} Hz  "
                  f"tilt {i['tilt_deg'] if i['tilt_deg'] is not None else '-'}   ", end="", flush=True)
    print()
    return [a for _, a, _, _ in list(live.labels)[n0:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM15")
    a = ap.parse_args()
    live = LiveCollar(a.port, "POSETEST", mount="chest")
    time.sleep(4)
    if not live.connected or live.n < 50:
        live.stop()
        sys.exit(f"No data from {a.port}: {live.error or 'not streaming'} (is the dashboard live session still running?)")
    print("Device streaming. Follow each step; press Enter when you are IN position.\n")
    for kind, text in CAL:
        while True:
            input(f"CALIBRATE {kind.upper()}: {text}. Press Enter, then hold 4 s... ")
            time.sleep(4)
            ok = live.calibrate_standing() if kind == "standing" else live.calibrate_sitting()
            ok = ok if kind == "standing" else ok is None
            if ok:
                print("   calibrated OK\n")
                break
            print("   failed (moving, or sitting not different enough from standing) - try again")
    results = []
    for want, text, secs in STEPS:
        input(f"STEP: {text}. Press Enter when ready...")
        live.mark(f"test_{want}")
        got = wait_labels(live, secs)
        c = Counter(got)
        share = c[want] / max(1, len(got))
        top = c.most_common(1)[0][0] if got else "-"
        verdict = "PASS" if share >= 0.6 else "CHECK"
        results.append((want, share, top, verdict))
        print(f"   {verdict}: {share:.0%} of windows = {ACTIVITIES[want]}  (most common: {ACTIVITIES.get(top, top)})\n")
    live.stop()
    print("=" * 70)
    for want, share, top, verdict in results:
        print(f"  {verdict:<6} {ACTIVITIES[want]:<28} {share:4.0%}   most common: {ACTIVITIES.get(top, top)}")
    print(f"\nRecording saved: {live.base}_*.csv  - tell Claude 'check POSETEST' to review it.")


if __name__ == "__main__":
    main()
