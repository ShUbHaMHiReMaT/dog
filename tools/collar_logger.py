"""
USB logger for the PRANIIVA collar (ESP32-S3 + BMI270 firmware in firmware/praniiva_collar).

  * finds the collar's serial port, sends the current UTC time (T<epoch_ms>) and dog id (I<id>)
  * writes files the pipeline reads directly (praniiva/ingest.py), split by LOCAL date (IST):
        data/raw/<dog>/<dog>_<YYYYMMDD>_raw.csv      timestamp_ms,ax,ay,az
        data/raw/<dog>/<dog>_<YYYYMMDD>_qc.csv       per-minute collar telemetry
        data/raw/<dog>/<dog>_<YYYYMMDD>_events.csv   marker presses / labels
  * re-syncs the collar clock every hour (ESP32 crystal drifts ~1-2 s/day)
  * prints live health: sample rate, |a|, dropped samples
  * type a label + Enter while logging to write a marker event (e.g. "walk_start")

    python tools/collar_logger.py --dog ID0001            # auto-detect port
    python tools/collar_logger.py --dog ID0001 --port COM5
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import serial
import serial.tools.list_ports

IST = timezone(timedelta(hours=5, minutes=30))
ROOT = Path(__file__).resolve().parents[1]
ESP_VIDS = {0x303A, 0x10C4, 0x1A86, 0x0403}      # Espressif native USB, CP210x, CH34x, FTDI


def find_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.vid in ESP_VIDS:
            return p.device
    names = ", ".join(f"{p.device} ({p.description})" for p in ports) or "none"
    sys.exit(f"No ESP32 serial port found. Ports seen: {names}. Use --port.")


class DayFiles:
    def __init__(self, out: Path, dog: str):
        self.dir = out / dog
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dog, self.day, self.f = dog, None, {}

    def _open(self, day: str):
        for f in self.f.values():
            f.close()
        heads = {"raw": "timestamp_ms,ax,ay,az", "qc": "timestamp_ms,vm_std_mg,rest_jitter_mg,temp_c,samples,"
                 "dropped,read_errors,battery_v,recoveries", "events": "timestamp_ms,label"}
        self.f = {}
        for k, h in heads.items():
            p = self.dir / f"{self.dog}_{day}_{k}.csv"
            new = not p.exists()
            self.f[k] = open(p, "a", buffering=1 << 16, encoding="utf-8")
            if new:
                self.f[k].write(h + "\n")
        self.day = day
        print(f"[logger] writing {self.dir / (self.dog + '_' + day + '_*.csv')}")

    def write(self, kind: str, ts_ms: int, line: str):
        day = datetime.fromtimestamp(ts_ms / 1000, IST).strftime("%Y%m%d")
        if day != self.day:
            self._open(day)
        self.f[kind].write(line + "\n")

    def flush(self):
        for f in self.f.values():
            f.flush()

    def close(self):
        for f in self.f.values():
            f.close()


def sync(ser):
    ser.write(f"T{int(time.time() * 1000)}\n".encode())


def stdin_reader(q: queue.Queue):
    for line in sys.stdin:
        q.put(line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", required=True, help="dog id, e.g. ID0001")
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--minutes", type=float, default=0, help="stop after N minutes (0 = until Ctrl+C)")
    a = ap.parse_args()
    port = a.port or find_port()
    ser = serial.Serial(port, a.baud, timeout=1)
    time.sleep(2.0)                                         # board may reset on open
    ser.reset_input_buffer()
    ser.write(f"I{a.dog}\n".encode())
    sync(ser)
    files = DayFiles(Path(a.out), a.dog)
    q: queue.Queue = queue.Queue()
    threading.Thread(target=stdin_reader, args=(q,), daemon=True).start()
    t0 = last_stat = last_sync = time.time()
    n = 0; n_win = 0; vm_sum = 0.0; last_ts = None; gaps = 0; synced = False
    print(f"[logger] {port}  dog={a.dog}  (type a label + Enter to mark an event; Ctrl+C to stop)")
    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            if line[0].isdigit():
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                ts = int(parts[0])
                if ts < 10**11:                                 # device not synced yet -> ignore, re-sync
                    if time.time() - last_sync > 2:
                        sync(ser); last_sync = time.time()
                    continue
                synced = True
                files.write("raw", ts, ",".join(parts[:4]))
                ax, ay, az = map(float, parts[1:4])
                vm_sum += (ax * ax + ay * ay + az * az) ** 0.5
                if last_ts is not None and ts - last_ts > 30:
                    gaps += (ts - last_ts) // 20 - 1
                last_ts = ts; n += 1; n_win += 1
            elif line.startswith("#Q,"):
                body = line[3:]
                ts = int(body.split(",")[0])
                if ts > 10**11:
                    files.write("qc", ts, body)
            elif line.startswith("#M,"):
                body = line[3:]
                ts = int(body.split(",")[0])
                if ts > 10**11:
                    files.write("events", ts, body)
                print(f"[marker] {body}")
            elif line.startswith(("#I", "#W", "#E")):
                print(f"[collar] {line}")
            now = time.time()
            while not q.empty():
                ser.write(f"M{q.get()}\n".encode())
            if now - last_sync > 3600:
                sync(ser); last_sync = now
            if now - last_stat >= 10:
                rate = n_win / (now - last_stat)
                print(f"[logger] {rate:5.1f} Hz | mean |a| {vm_sum / max(n_win, 1):.3f} g | "
                      f"missing samples so far {gaps} | total {n} | {'synced' if synced else 'NOT SYNCED'}")
                files.flush(); n_win = 0; vm_sum = 0.0; last_stat = now
            if a.minutes and now - t0 > a.minutes * 60:
                break
    except KeyboardInterrupt:
        pass
    finally:
        files.close()
        ser.close()
        print(f"[logger] stopped. {n} samples written.")


if __name__ == "__main__":
    main()
