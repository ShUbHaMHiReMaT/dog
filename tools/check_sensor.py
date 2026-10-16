"""
Bench acceptance test: does the assembled collar deliver what the PRANIIVA pipeline needs?

Live, guided (collar on USB):      python tools/check_sensor.py --port COM5
Live, auto-detect port:            python tools/check_sensor.py
Offline on a recorded raw CSV:     python tools/check_sensor.py --file data/raw/ID0001/ID0001_20261013_raw.csv

Guided steps: (1) collar still on a table, (2) collar x-axis (toward the dog's HEAD) pointing at the ceiling,
(3) shake hard. Each requirement comes from DEVICE_SPEC.md / praniiva/params.py / praniiva/ingest.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from praniiva.ingest import IngestError, IngestReport, load_raw     # noqa: E402

FS = 50


def record(ser, seconds, settle=0.5, live=False):
    """Record `seconds` of FRESH data. The serial backlog that piled up while the user was reading the prompt
    (and moving the collar) is discarded, plus `settle` s after Enter, so each step only contains its own posture."""
    ser.reset_input_buffer()
    rows = []
    t_start = time.time() + settle
    t_end = t_start + seconds
    last_print = 0
    while time.time() < t_end:
        line = ser.readline().decode(errors="replace").strip()
        if not (line and line[0].isdigit()) or time.time() < t_start:
            continue
        p = line.split(",")
        if len(p) >= 4:
            try:
                rows.append([int(p[0]), float(p[1]), float(p[2]), float(p[3])])
            except ValueError:
                continue
            if live and time.time() - last_print > 0.5:
                print(f"\r   live: ax={rows[-1][1]:+.2f}  ay={rows[-1][2]:+.2f}  az={rows[-1][3]:+.2f} g   ", end="", flush=True)
                last_print = time.time()
    if live:
        print()
    return pd.DataFrame(rows, columns=["timestamp_ms", "ax", "ay", "az"])


def live_until_enter(ser, prompt):
    """Show live ax/ay/az so the user can find the right posture, until Enter is pressed."""
    import threading
    done = threading.Event()
    threading.Thread(target=lambda: (input(prompt), done.set()), daemon=True).start()
    ser.reset_input_buffer()
    last = 0
    while not done.is_set():
        line = ser.readline().decode(errors="replace").strip()
        if line and line[0].isdigit() and time.time() - last > 0.3:
            p = line.split(",")
            if len(p) >= 4:
                print(f"\r   live: ax={float(p[1]):+.2f}  ay={float(p[2]):+.2f}  az={float(p[3]):+.2f} g   (want ax ~ +1.00)   ",
                      end="", flush=True)
                last = time.time()
    print()


def check_rate(df):
    dt = np.diff(df.timestamp_ms.values) / 1000
    fs = 1 / np.median(dt) if len(dt) else 0
    expected = (df.timestamp_ms.iloc[-1] - df.timestamp_ms.iloc[0]) / 1000 * FS + 1
    missing = max(0.0, 1 - len(df) / expected)
    jitter = np.percentile(np.abs(dt - 1 / FS), 95) * 1000 if len(dt) else np.nan
    return [("sample rate (Hz)", fs, abs(fs - FS) <= 1, "50 +/- 1 Hz - models were built at 50 Hz"),
            ("missing samples (%)", 100 * missing, missing < 0.005, "< 0.5 %"),
            ("timestamp jitter p95 (ms)", jitter, jitter < 5, "< 5 ms (data-ready interrupt timing)")]


def check_still(df):
    """Sensor noise floor = median noise of 2 s windows (a knock on the desk is vibration, not sensor noise).
    Windows with movement are reported separately."""
    a = df[["ax", "ay", "az"]].values
    vm = np.linalg.norm(a, axis=1)
    out = [("|a| at rest (g)", vm.mean(), abs(vm.mean() - 1) < 0.05, "1.00 +/- 0.05 g - scale & range correct")]
    w = 2 * FS
    win_sd = np.array([(a[i:i + w] - a[i:i + w].mean(0)).std(0).max() for i in range(0, len(a) - w + 1, w)]) * 1000
    if len(win_sd) == 0:
        win_sd = np.array([(a - a.mean(0)).std(0).max() * 1000])
    noise_rms = float(np.median(win_sd))
    bumps = int((win_sd > 3 * max(noise_rms, 0.3)).sum())
    out.append(("noise RMS at rest, worst axis (mg)", noise_rms, noise_rms < 3.0,
                f"< 3 mg - breathing micro-motion is only ~1-3 mg (median of 2 s windows; {bumps} bump(s) ignored)"))
    quiet = np.argsort(win_sd)[: max(1, len(win_sd) // 2)]          # quietest half of the windows
    seg = np.concatenate([a[i * w:(i + 1) * w, 2] - a[i * w:(i + 1) * w, 2].mean() for i in sorted(quiet)])
    f, pxx = sps.welch(seg, fs=FS, nperseg=min(256, len(seg)))
    band = (f > 1) & (f < 20)
    nd = np.sqrt(np.median(pxx[band])) * 1e6 if band.any() else np.nan     # g/sqrt(Hz) -> ug/sqrt(Hz)
    out.append(("noise density z (ug/sqrt(Hz))", nd, nd <= 400, "<= 400 (spec 150-400; quietest windows)"))
    steps = np.diff(np.unique(np.round(a[:, 2], 6)))
    lsb = steps[steps > 0].min() * 1000 if (steps > 0).any() else np.nan
    out.append(("resolution / LSB (mg)", lsb, lsb < 1.0, "< 1 mg (16-bit at +/-8 g = 0.244 mg)"))
    return out


def check_orientation(df):
    m = df[["ax", "ay", "az"]].mean().values
    ok = m[0] > 0.85 and abs(m[1]) < 0.3 and abs(m[2]) < 0.3
    k = int(np.argmax(np.abs(m)))
    hint = "" if ok else (f" -> gravity is on {'xyz'[k]} ({'+' if m[k] > 0 else '-'}): set MAP_X_SRC {k} and "
                          f"MAP_X_SGN {'+1' if m[k] > 0 else '-1'} in firmware config.h")
    return [("x-axis toward head points up: ax (g)", m[0], ok, "ax ~ +1 g, ay/az ~ 0" + hint)]


def check_shake(df):
    """Information only: a gentle shake is fine (jumper-wired prototypes should not be shaken hard)."""
    vm = np.linalg.norm(df[["ax", "ay", "az"]].values, axis=1)
    clip = (np.abs(df[["ax", "ay", "az"]].values) > 7.9).mean() * 100
    return [("peak |a| while shaking (g) [info]", vm.max(), True, "information only (dog head shakes reach 2-3 g)"),
            ("samples clipped at +/-8 g (%)", clip, clip < 1.0, "< 1 %")]


def check_config(ser):
    """Ask the firmware for the configuration it READ BACK from the BMI270 chip."""
    ser.reset_input_buffer()
    ser.write(b"S\n")
    t_end = time.time() + 3
    while time.time() < t_end:
        line = ser.readline().decode(errors="replace").strip()
        if line.startswith("#I,status"):
            kv = dict(x.split("=", 1) for x in line.split(",")[2:] if "=" in x)
            rg, odr = float(kv.get("range_g", "nan")), float(kv.get("odr_hz", "nan"))
            return [("accel range read back from chip (g)", rg, rg == 8, "+/-8 g configured (head shakes reach 2-3 g)"),
                    ("accel output rate read back from chip (Hz)", odr, abs(odr - FS) < 0.5, "50 Hz")]
    return [("accel config read back from chip", float("nan"), False, "no status reply - flash firmware >= this version")]


def check_pipeline(path):
    rep = IngestReport()
    try:
        load_raw(path, rep)
        return [("pipeline ingest (praniiva/ingest.py)", 1.0, True,
                 f"accepted; fixes={rep.fixes or 'none'}; warnings={rep.warnings or 'none'}")]
    except IngestError as e:
        return [("pipeline ingest (praniiva/ingest.py)", 0.0, False, f"REJECTED: {e}")]


def report(rows, out_json):
    print("\n" + "=" * 100)
    print(f"{'CHECK':<42}{'VALUE':>12}   RESULT   REQUIREMENT")
    print("-" * 100)
    for name, val, ok, req in rows:
        v = f"{val:.3f}" if isinstance(val, (float, np.floating)) and np.isfinite(val) else str(val)
        print(f"{name:<42}{v:>12}   {'PASS' if ok else 'FAIL':<6}   {req}")
    n_fail = sum(not r[2] for r in rows)
    print("-" * 100)
    print("ALL CHECKS PASSED - collar output is pipeline-ready." if n_fail == 0 else f"{n_fail} check(s) FAILED - fix before dog trials.")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps([dict(check=r[0], value=float(r[1]) if np.isfinite(r[1]) else None,
                                               passed=bool(r[2]), requirement=r[3]) for r in rows], indent=2))
    print(f"report saved: {out_json}")
    return n_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--file")
    ap.add_argument("--baud", type=int, default=921600)
    a = ap.parse_args()
    out_dir = ROOT / "data" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if a.file:
        df = pd.read_csv(a.file)
        rows = check_rate(df) + check_still(df.iloc[: FS * 20]) + check_pipeline(a.file)
        print("(offline mode: still-noise checks assume the first 20 s were recorded with the collar still)")
        sys.exit(report(rows, out_dir / f"bench_{stamp}.json"))

    import serial
    from collar_logger import find_port
    port = a.port or find_port()
    ser = serial.Serial(port, a.baud, timeout=1)
    time.sleep(2.0); ser.reset_input_buffer()
    ser.write(f"T{int(time.time() * 1000)}\n".encode())
    time.sleep(0.5); ser.reset_input_buffer()
    input("STEP 1/3: put the collar flat on a table, hands OFF, then press Enter (records 20 s - do not touch it) ")
    still = record(ser, 20, settle=1.0)
    print("STEP 2/3: hold the collar so its HEAD-direction (the module's X arrow) points straight at the CEILING.")
    live_until_enter(ser, "   adjust until ax ~ +1.00, hold still, press Enter (records 5 s) ")
    up = record(ser, 5, live=True)
    input("STEP 3/3 (optional info): press Enter and shake the collar firmly but gently - mind the jumper wires (5 s) ")
    shake = record(ser, 5, settle=0.2, live=True)
    cfg = check_config(ser)
    ser.close()
    allrec = pd.concat([still, up, shake]).sort_values("timestamp_ms")
    f = out_dir / f"bench_{stamp}_raw.csv"
    allrec.to_csv(f, index=False)
    if len(still) < FS * 5:
        sys.exit(f"Only {len(still)} samples received in 20 s - is the firmware streaming? (STREAM_TO_SERIAL 1)")
    rows = (check_rate(still) + check_still(still) + check_orientation(up) + cfg + check_shake(shake) + check_pipeline(f))
    sys.exit(report(rows, out_dir / f"bench_{stamp}.json"))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
