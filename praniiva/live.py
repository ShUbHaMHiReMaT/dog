"""
Live collar session for the dashboard: reads the PRANIIVA collar over USB serial in a background thread,
classifies behaviour every 2 s, estimates resting respiratory rate when the dog has been lying still,
tracks collar health, and records the raw stream in the normal pipeline format (data/raw/<dog>/...).

Only one program can open a serial port at a time: close tools/collar_logger.py before starting a live session.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np

from .imu import estimate_rr, features_matrix
from .params import BEHAVIOURS, FS

IST = timezone(timedelta(hours=5, minutes=30))
WIN = 2 * FS
ROOT = Path(__file__).resolve().parents[1]


class LiveCollar:
    def __init__(self, port: str, dog: str, baud: int = 921600, record: bool = True, keep_s: int = 900):
        self.port, self.dog, self.baud, self.record = port, dog, baud, record
        self.samples = deque(maxlen=keep_s * FS)              # (ts_ms, ax, ay, az)
        self.labels = deque(maxlen=keep_s // 2)               # (ts_ms, behaviour, confidence)
        self.budget = Counter()                               # seconds per behaviour since start
        self.messages = deque(maxlen=30)
        self.qc = deque(maxlen=60)
        self.rr = deque(maxlen=120)                           # (ts_ms, rr_bpm)
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.error = None
        self.started = time.time()
        self.n = 0
        self.gaps = 0
        self.connected = False
        self._model = joblib.load(ROOT / "models" / "behaviour_model.joblib")["model"]
        self._pending = []
        self._files = {}
        self._day = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------ recording
    def _write(self, kind, ts, line):
        if not self.record:
            return
        day = datetime.fromtimestamp(ts / 1000, IST).strftime("%Y%m%d")
        if day != self._day:
            for f in self._files.values():
                f.close()
            d = ROOT / "data" / "raw" / self.dog
            d.mkdir(parents=True, exist_ok=True)
            heads = {"raw": "timestamp_ms,ax,ay,az",
                     "qc": "timestamp_ms,vm_std_mg,rest_jitter_mg,temp_c,samples,dropped,read_errors,battery_v,recoveries",
                     "events": "timestamp_ms,label"}
            self._files = {}
            for k, h in heads.items():
                p = d / f"{self.dog}_{day}_{k}.csv"
                new = not p.exists()
                self._files[k] = open(p, "a", buffering=1 << 16, encoding="utf-8")
                if new:
                    self._files[k].write(h + "\n")
            self._day = day
        self._files[kind].write(line + "\n")

    def mark(self, label: str):
        """Owner/demo marker, written by the collar into the events stream."""
        if getattr(self, "_ser", None):
            self._ser.write(f"M{label}\n".encode())

    # ------------------------------------------------------------------ analysis
    def _classify(self, block):
        W = np.asarray(block, float)[None, :, 1:4]
        P = self._model.predict_proba(features_matrix(W))[0]
        k = int(P.argmax())
        self.labels.append((block[-1][0], BEHAVIOURS[k], float(P[k])))
        self.budget[BEHAVIOURS[k]] += 2
        # resting RR: last 30 windows (60 s) all lying -> spectral estimate on that minute (EXPERIMENTAL)
        recent = list(self.labels)[-30:]
        if len(recent) == 30 and all(b == "lying_rest" for _, b, _ in recent):
            seg = np.asarray(list(self.samples)[-60 * FS:], float)[:, 1:4]
            if len(seg) == 60 * FS:
                rr, q = estimate_rr(seg)
                if np.isfinite(rr):
                    self.rr.append((block[-1][0], rr))

    # ------------------------------------------------------------------ serial loop
    def _run(self):
        import serial
        try:
            ser = serial.Serial()
            ser.port, ser.baudrate, ser.timeout = self.port, self.baud, 0.5
            ser.dtr = False                                    # do not auto-reset the ESP32 when opening
            ser.rts = False
            ser.open()
            self._ser = ser
            time.sleep(0.5)
            ser.reset_input_buffer()
            ser.write(f"I{self.dog}\n".encode())
            ser.write(f"T{int(time.time() * 1000)}\n".encode())
            self.connected = True
            last_ts = None
            last_sync = time.time()
            while not self.stop_evt.is_set():
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                if line[0].isdigit():
                    p = line.split(",")
                    if len(p) < 4:
                        continue
                    try:
                        ts = int(p[0]); ax, ay, az = float(p[1]), float(p[2]), float(p[3])
                    except ValueError:
                        continue
                    if ts < 10**11:                                    # not synced yet
                        if time.time() - last_sync > 2:
                            ser.write(f"T{int(time.time() * 1000)}\n".encode()); last_sync = time.time()
                        continue
                    with self.lock:
                        if last_ts is not None and ts - last_ts > 30 and self.n > 5 * FS:   # ignore start-up/sync
                            self.gaps += int((ts - last_ts) // 20) - 1
                            self._pending = []
                        last_ts = ts
                        self.samples.append((ts, ax, ay, az))
                        self.n += 1
                        self._pending.append((ts, ax, ay, az))
                        if len(self._pending) == WIN:
                            self._classify(self._pending)
                            self._pending = []
                    self._write("raw", ts, ",".join(p[:4]))
                elif line.startswith("#Q,"):
                    body = line[3:]
                    with self.lock:
                        self.qc.append(body.split(","))
                    self._write("qc", int(body.split(",")[0]) if body.split(",")[0].isdigit() else 0, body)
                elif line.startswith("#M,"):
                    body = line[3:]
                    with self.lock:
                        self.messages.append(f"marker: {body}")
                    ts0 = body.split(",")[0]
                    if ts0.isdigit() and int(ts0) > 10**11:
                        self._write("events", int(ts0), body)
                elif line.startswith(("#W", "#E")):
                    with self.lock:
                        self.messages.append(line)
                if time.time() - last_sync > 3600:
                    ser.write(f"T{int(time.time() * 1000)}\n".encode()); last_sync = time.time()
        except Exception as e:                                         # surfaced on the page
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.connected = False
            for f in self._files.values():
                try:
                    f.close()
                except Exception:
                    pass
            try:
                self._ser.close()
            except Exception:
                pass

    def stop(self):
        self.stop_evt.set()
        self.thread.join(timeout=3)

    # ------------------------------------------------------------------ snapshot for the UI
    def snapshot(self, seconds=30):
        with self.lock:
            s = np.asarray(list(self.samples)[-seconds * FS:], float) if self.samples else np.zeros((0, 4))
            labels = list(self.labels)
            elapsed = time.time() - self.started
            return dict(samples=s, labels=labels, budget=dict(self.budget), rr=list(self.rr), qc=list(self.qc),
                        messages=list(self.messages), n=self.n, gaps=self.gaps, elapsed=elapsed,
                        connected=self.connected, error=self.error,
                        rate=(len(self.samples) and self._rate()))

    def _rate(self):
        if len(self.samples) < 2 * FS:
            return 0.0
        recent = list(self.samples)[-10 * FS:]
        span = (recent[-1][0] - recent[0][0]) / 1000
        return (len(recent) - 1) / span if span > 0 else 0.0
