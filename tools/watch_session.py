"""
Watchdog for a running recording: every 15 s checks the recorder status, that files keep growing, device health
(read errors, sensor restarts, dropped samples, loose fit) and the microphone. Prints OK / ISSUE lines.

    python tools/watch_session.py data/raw/MD_WRIST reports_md_wrist_session.log 30     # 30 minutes
"""
import glob
import os
import re
import sys
import time
from datetime import datetime

import pandas as pd

folder, status_log, minutes = sys.argv[1], sys.argv[2], float(sys.argv[3])
base = sorted(glob.glob(f"{folder}/*_session.json"))[-1].replace("_session.json", "")
t_end = time.time() + minutes * 60
prev = None
checks = problems = 0


def last_status():
    lines = [l for l in open(status_log, encoding="utf-8") if "s conn=" in l]
    return lines[-1] if lines else ""


while time.time() < t_end:
    checks += 1
    issues = []
    m = re.search(r"conn=(\w+) rate=([\d.]+) n=(\d+) missing=(\d+) reconn=(\d+)", last_status())
    if not m:
        issues.append("recorder status line missing")
    else:
        conn, rate, n, miss, rc = m.group(1) == "True", float(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        if not conn:
            issues.append("DEVICE DISCONNECTED")
        if rate < 49:
            issues.append(f"low rate {rate}")
        if prev is not None and n <= prev:
            issues.append("sample count not increasing")
        prev = n
    age = time.time() - os.path.getmtime(f"{base}_raw.csv")
    if age > 10:
        issues.append(f"raw file not written for {age:.0f}s")
    try:
        q = pd.read_csv(f"{base}_qc.csv").iloc[-1]
        if float(q.read_errors) > 0 or float(q.recoveries) > 0 or float(q.dropped) > 0:
            issues.append(f"last minute: read_errors={q.read_errors} recoveries={q.recoveries} dropped={q.dropped}")
        if float(q.rest_jitter_mg) > 5:
            issues.append(f"high rest jitter {q.rest_jitter_mg} mg (loose strap?)")
    except Exception as e:  # noqa: BLE001
        issues.append(f"qc read: {e}")
    try:
        s = pd.read_csv(f"{base}_sound.csv").tail(30)
        if len(s) and s.mic_ok.mean() < 1:
            issues.append("mic reported no data")
    except Exception as e:  # noqa: BLE001
        issues.append(f"sound read: {e}")
    problems += bool(issues)
    kb = os.path.getsize(f"{base}_raw.csv") // 1024
    print(datetime.now().strftime("%H:%M:%S"), "OK   " if not issues else "ISSUE",
          f"n={prev} missing={miss if m else '-'} reconn={rc if m else '-'} raw={kb}KB", "; ".join(issues), flush=True)
    time.sleep(15)
print(f"WATCHDOG DONE: {checks} checks, {problems} with issues", flush=True)
