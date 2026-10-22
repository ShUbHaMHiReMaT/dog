"""
Live session for the PRANIIVA prototype (ESP32-S3 + BMI270 + INMP441) - used by the dashboard's Live page.

What it does, continuously, in a background thread:
  * reads the device over USB serial; AUTO-RECONNECTS if the cable is pulled (dogs pull cables)
  * every 2 s classifies the dog's activity with TRANSPARENT MOTION RULES that work at any mounting position
    (chest harness or neck collar) - plus the simulation-trained model as a secondary "experimental" opinion
  * posture (standing / sitting / lying on belly / lying on side) from two one-tap calibrations:
    "dog is standing" and "dog is sitting" - this works for any mounting orientation
  * estimates resting breathing rate after 60 s of continuous lying (EXPERIMENTAL)
  * RECORDS EVERYTHING to data/raw/<dog>/ in the pipeline format, flushed every few seconds:
        <dog>_<session>_raw.csv      timestamp_ms,ax,ay,az            (50 Hz motion)
        <dog>_<session>_qc.csv       per-minute device health
        <dog>_<session>_events.csv   markers typed on the website / pressed on the button
        <dog>_<session>_sound.csv    sound LEVELS twice a second (never audio)
        <dog>_<session>_labels.csv   the live activity label for every 2 s window (rules + model)
        <dog>_<session>_session.json mount position, calibration, device info, counts

Only one program can open a serial port at a time: close collar_logger.py / check_*.py before starting.
"""
from __future__ import annotations

import json
import os
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
FLUSH_S = 5
POSE_FILE = ROOT / "static" / "live_pose.json"     # read by the 3D dog on the Live page

# ---------------------------------------------------------------- activity rules (mount-independent)
# PROVISIONAL thresholds - set from first principles, to be tuned on the first labelled real recordings.
RULES = dict(still_std_g=0.015,       # total dynamic acceleration (3-axis SD) over 2 s below this = not moving
             small_std_g=0.045,       # below this = small movements (eating, sniffing, shifting)
             run_std_g=0.35,          # above this = running / playing
             shake_hf_frac=0.55,      # share of motion power at 4-12 Hz above this = shaking / scratching
             shake_min_hz=4.5,        # ... and the dominant frequency at least this (strides are slower)
             run_min_hz=1.8,          # running needs a stride rhythm; big SLOW movements are play / rolling
                                      # (learned 2026-10-22: slow arm swings on the wrist were called running)
             band_hz=(0.3, 12.0),     # body-movement band; car / engine vibration above 12 Hz is ignored
             spike_g=0.5,             # one-sample jumps bigger than this (not seen in neighbours) = glitch
             flatline_g=0.0003,       # total motion below this is below the sensor's own noise = sensor stuck
             # fit / placement check (measured 2026-10-22: device on a desk 1.1-4.4 mg at rest, on a wrist 3.8-14 mg)
             nonwear_g=0.0025,        # median resting motion over 2 min below this = probably not on the body
             loose_hi_frac=0.25,      # share of walking motion above 12 Hz above this = rattling = loose fit
             shifted_deg=25.0,        # walking trunk direction this far from the calibrated standing = device moved
             gravity_tol=0.06,        # resting |a| must be 1 g +- this, else sensor / calibration fault
             upright_deg=25.0,        # within this of the standing posture = trunk level
             side_roll_deg=45.0,      # rolled more than this = lying on the side
             flat_tilt_deg=60.0,      # tilted more than this (and not sitting) = lying on side / back
             belly_after_s=120)       # trunk level and motionless this long = lying on belly (standing dogs sway)
ACTIVITIES = {
    "standing": "Standing",
    "sitting": "Sitting",
    "lying_belly": "Lying on belly (likely)",
    "lying_side": "Lying on side",
    "tilted": "Sitting or lying (calibrate 'sitting')",
    "still": "Still (calibrate posture)",
    "small": "Small movements",
    "walking": "Walking",
    "running": "Running / playing",
    "large": "Large slow movements (play / rolling)",
    "shaking": "Shaking / scratching",
}
RESTING = ("lying_belly", "lying_side", "still")


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def posture_angles(g, cal: dict) -> dict:
    """Angles of the current gravity direction g relative to the calibrated standing (and sitting) posture.
    pitch > 0 = trunk tilted the way it tilts when sitting; roll = rotation to the side."""
    u, s = _unit(g), cal["stand"]
    tilt = float(np.degrees(np.arccos(np.clip(u @ s, -1, 1))))
    out = dict(tilt_deg=round(tilt, 1), pitch_deg=None, roll_deg=None)
    if cal.get("sit") is not None:
        a = _unit(np.cross(s, cal["sit"]))                         # sideways axis the trunk rotates about
        out["roll_deg"] = round(float(np.degrees(np.arcsin(np.clip(u @ a, -1, 1)))), 1)
        up = _unit(u - (u @ a) * a)
        out["pitch_deg"] = round(float(np.degrees(np.arctan2(np.cross(s, up) @ a, s @ up))), 1)
    return out


def classify_posture(ang: dict, cal: dict, level_still_s: float) -> str:
    if ang["pitch_deg"] is None:                                  # only 'standing' calibrated
        if ang["tilt_deg"] < RULES["upright_deg"]:
            return "lying_belly" if level_still_s >= RULES["belly_after_s"] else "standing"
        return "lying_side" if ang["tilt_deg"] > 75 else "tilted"
    sit_pitch = cal["sit_pitch"]
    if abs(ang["roll_deg"]) > RULES["side_roll_deg"]:
        return "lying_side"
    if ang["pitch_deg"] > max(12.0, 0.5 * sit_pitch) and ang["pitch_deg"] < sit_pitch + 40:
        return "sitting"
    if ang["tilt_deg"] > RULES["flat_tilt_deg"]:
        return "lying_side"
    return "lying_belly" if level_still_s >= RULES["belly_after_s"] else "standing"


def despike(w: np.ndarray) -> np.ndarray:
    """Replace isolated one-sample glitches (a jump not shared by either neighbour) by the neighbours' mean.
    Real impacts last several samples at 50 Hz, so they are kept."""
    w = np.array(w, float, copy=True)
    if len(w) < 3:
        return w
    a, b, c = w[:-2], w[1:-1], w[2:]
    nb = (a + c) / 2
    bad = ((np.linalg.norm(b - nb, axis=1) > RULES["spike_g"])        # the whole 3-axis jump, not per axis
           & (np.linalg.norm(a - c, axis=1) < RULES["spike_g"] / 2))
    b[bad] = nb[bad]
    return w


def load_device_calibration():
    """Per-axis offset/gain from tools/calibrate_device.py (GGIR-style autocalibration), or None."""
    f = ROOT / "models" / "device_calibration.json"
    try:
        c = json.loads(f.read_text(encoding="utf-8"))
        return np.asarray(c["offset_g"], float), np.asarray(c["gain"], float), c
    except Exception:
        return None


def window_activity(w: np.ndarray, cal: dict | None, level_still_s: float = 0.0) -> tuple[str, dict]:
    """w: (100, 3) in g. cal: {'stand': unit vector, 'sit': unit vector or None, 'sit_pitch': deg} or None.
    level_still_s: how long the dog has already been motionless with a level trunk.
    Returns (activity key, the numbers behind the decision)."""
    w = despike(w)
    x = w - w.mean(0)                                         # dynamic acceleration on all 3 axes
    f = np.fft.rfftfreq(len(x), 1 / FS)
    p = (np.abs(np.fft.rfft(x, axis=0)) ** 2).sum(1)
    body = (f >= RULES["band_hz"][0]) & (f <= RULES["band_hz"][1])
    dyn = float(np.sqrt(2 * p[body].sum()) / len(x))         # RMS motion in the body band, same for any mounting
    band = (f >= 0.5) & (f <= 12)
    hf = float(p[(f >= 4) & (f <= 12)].sum() / (p[band].sum() + 1e-12))
    g = w.mean(0)
    ang = posture_angles(g, cal) if cal and np.linalg.norm(g) > 0.5 else dict(tilt_deg=None, pitch_deg=None,
                                                                               roll_deg=None)
    dom = float(f[band][np.argmax(p[band])])
    hi12 = float(p[f > 12].sum() / (p[f >= 0.3].sum() + 1e-12))
    info = dict(motion_g=round(dyn, 4), hf_frac=round(hf, 2), dom_hz=round(dom, 1), hi12_frac=round(hi12, 3),
                g=[round(float(v), 4) for v in g], **ang)
    if dyn < RULES["still_std_g"]:
        if ang["tilt_deg"] is None:
            return "still", info
        return classify_posture(ang, cal, level_still_s), info
    if hf > RULES["shake_hf_frac"] and info["dom_hz"] >= RULES["shake_min_hz"] and dyn > RULES["small_std_g"]:
        return "shaking", info                                # faster than any running stride
    if dyn > RULES["run_std_g"]:
        return ("running" if dom >= RULES["run_min_hz"] else "large"), info
    if dyn >= RULES["small_std_g"]:
        return "walking", info
    return "small", info


class LiveCollar:
    def __init__(self, port: str, dog: str, mount: str = "chest", baud: int = 921600, record: bool = True,
                 keep_s: int = 900):
        self.port, self.dog, self.mount, self.baud, self.record = port, dog, mount, baud, record
        self.samples = deque(maxlen=keep_s * FS)              # (ts_ms, ax, ay, az)
        self.labels = deque(maxlen=keep_s // 2)               # (ts_ms, activity, model_label, info)
        self.budget = Counter()
        self.messages = deque(maxlen=40)
        self.qc = deque(maxlen=60)
        self.rr = deque(maxlen=120)
        self.sound = deque(maxlen=240)                        # (wall_s, rms_dbfs, peak_dbfs)
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.error = None
        self.started = time.time()
        self.n = 0
        self.gaps = 0
        self.connected = False
        self.reconnects = 0
        self.cal = None                                       # posture calibration
        self.level_still_s = 0.0
        self.device = {}
        self.rows = Counter()
        self.last_write = None
        self._ser = None
        self._model = joblib.load(ROOT / "models" / "behaviour_model.joblib")["model"]
        self.devcal = load_device_calibration()                # raw files stay raw; analysis uses calibrated g
        self.flat_windows = 0
        self._rest, self._walk = deque(maxlen=60), deque(maxlen=15)
        self.fit = dict(worn="checking", loose="checking", shifted="calibrate posture to check",
                        gravity="checking", summary="checking fit…")
        self._fit_said = set()
        self._pending = []
        self._files = {}
        self._last_flush = time.time()
        stamp = datetime.now(IST).strftime("%Y%m%d-%H%M%S")
        self.dir = ROOT / "data" / "raw" / dog
        self.base = self.dir / f"{dog}_{stamp}"
        if record:
            self._open_files()
            self._write_session(status="recording")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------ recording
    def _open_files(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        heads = {"raw": "timestamp_ms,ax,ay,az",
                 "qc": "timestamp_ms,vm_std_mg,rest_jitter_mg,temp_c,samples,dropped,read_errors,battery_v,recoveries",
                 "events": "timestamp_ms,label",
                 "sound": "timestamp_ms,rms_dbfs,peak_dbfs,mic_ok",
                 "labels": "timestamp_ms,activity,model_label,motion_g,hf_frac,tilt_deg,pitch_deg,roll_deg"}
        for k, h in heads.items():
            f = open(f"{self.base}_{k}.csv", "a", buffering=1 << 15, encoding="utf-8")
            f.write(h + "\n")
            self._files[k] = f

    def _write(self, kind, line):
        if not self.record:
            return
        self._files[kind].write(line + "\n")
        self.rows[kind] += 1
        self.last_write = time.time()
        if time.time() - self._last_flush > FLUSH_S:          # never more than a few seconds of data at risk
            for f in self._files.values():
                f.flush()
            self._last_flush = time.time()

    def _write_session(self, status):
        if not self.record:
            return
        meta = dict(dog_id=self.dog, mount=self.mount, port=self.port, status=status,
                    started_ist=datetime.fromtimestamp(self.started, IST).isoformat(),
                    updated_ist=datetime.now(IST).isoformat(), device=self.device,
                    posture_calibration=None if self.cal is None else {
                        k: (round(float(v), 1) if np.isscalar(v) else [round(float(x), 4) for x in v])
                        for k, v in self.cal.items() if v is not None},
                    activity_rules=RULES, rows=dict(self.rows),
                    device_calibration=None if self.devcal is None else self.devcal[2], samples=self.n, missing_samples=self.gaps,
                    reconnects=self.reconnects, fit_check=getattr(self, "fit", None),
                    note="activity labels are rule-based and PROVISIONAL; model labels come from a simulation-trained "
                         "neck-collar model and are experimental")
        Path(f"{self.base}_session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ controls
    def mark(self, label: str):
        label = label.replace(",", " ").strip()
        if self._ser is not None and self.connected:
            try:
                self._ser.write(f"M{label}\n".encode())       # the device timestamps it and echoes it back
                return
            except Exception:
                pass
        self._write("events", f"{int(time.time() * 1000)},{label}")   # device offline: stamp it here

    def _still_gravity(self, seconds):
        with self.lock:
            s = np.asarray(list(self.samples)[-int(seconds * FS):], float)
        if len(s) < seconds * FS * 0.8:
            return None
        w = s[:, 1:4]
        if np.linalg.norm(w, axis=1).std() > 0.05:            # dog was moving - reference would be wrong
            return None
        return _unit(w.mean(0))

    def calibrate_standing(self, seconds: float = 3.0) -> bool:
        """Use the last few seconds as the 'standing still' posture reference (resets the sitting reference)."""
        g = self._still_gravity(seconds)
        if g is None:
            return False
        self.cal = dict(stand=g, sit=None, sit_pitch=None)
        self.mark("calibrated_standing")
        self._write_session(status="recording")
        return True

    def calibrate_sitting(self, seconds: float = 3.0) -> str | None:
        """Use the last few seconds as the 'sitting' reference. Returns None on success, else the reason."""
        if self.cal is None:
            return "calibrate standing first"
        g = self._still_gravity(seconds)
        if g is None:
            return "the device was moving or no data yet"
        ang = float(np.degrees(np.arccos(np.clip(g @ self.cal["stand"], -1, 1))))
        if ang < 12:
            return f"sitting looks the same as standing (only {ang:.0f} deg apart) - is the device fixed to the chest?"
        self.cal = dict(stand=self.cal["stand"], sit=g, sit_pitch=ang)
        self.mark("calibrated_sitting")
        self._write_session(status="recording")
        return None

    def _publish_pose(self, act=None, info=None):
        """Latest activity for the 3D dog (static/live_pose.json). Never allowed to disturb recording."""
        try:
            d = dict(wall=time.time(), dog=self.dog, mount=self.mount, connected=self.connected,
                     activity=act, label=ACTIVITIES.get(act, ""), info=info or {})
            tmp = POSE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(d), encoding="utf-8")
            os.replace(tmp, POSE_FILE)
        except Exception:
            pass

    # ------------------------------------------------------------------ analysis
    def _fit_check(self, act, info):
        """Is the device on the body, firm, still where it was calibrated, and reading gravity correctly?"""
        if info["motion_g"] < RULES["small_std_g"]:
            self._rest.append((info["motion_g"], float(np.linalg.norm(info["g"]))))
        if act == "walking":
            self._walk.append((info["hi12_frac"], np.asarray(info["g"])))
        fit = self.fit
        if len(self._rest) >= 60:
            med = float(np.median([m for m, _ in self._rest]))
            fit["worn"] = ("NO - no body micro-motion for 2 min (on a table?)" if med < RULES["nonwear_g"]
                           else f"yes ({med * 1000:.1f} mg resting motion)")
        if len(self._rest) >= 10:
            gm = float(np.median([gm for _, gm in self._rest]))
            fit["gravity"] = (f"OK ({gm:.3f} g)" if abs(gm - 1) < RULES["gravity_tol"]
                              else f"FAULT - reads {gm:.3f} g at rest (should be 1.000)")
        if len(self._walk) >= 10:
            hf = float(np.median([h for h, _ in self._walk]))
            fit["loose"] = ("LOOSE - rattling while walking, tighten the strap" if hf > RULES["loose_hi_frac"]
                            else f"firm ({hf:.0%} rattle)")
            if self.cal is not None:
                gw = _unit(np.mean([g for _, g in self._walk], 0))
                ang = float(np.degrees(np.arccos(np.clip(gw @ self.cal["stand"], -1, 1))))
                fit["shifted"] = (f"MOVED {ang:.0f} deg since calibration - re-calibrate posture"
                                  if ang > RULES["shifted_deg"] else f"in place ({ang:.0f} deg)")
        bad = [v for k, v in fit.items() if k != "summary" and v.split(" ")[0] in ("NO", "FAULT", "LOOSE", "MOVED")]
        fit["summary"] = "OK" if not bad else " | ".join(bad)
        for v in bad:
            if v.split(" ")[0] not in self._fit_said:
                self._fit_said.add(v.split(" ")[0])
                with self.lock:
                    self.messages.append(f"#W fit check: {v}")
                self.mark(f"fit_check_{v.split(' ')[0].lower()}")

    def _classify(self, block):
        arr = np.asarray(block, float)
        act, info = window_activity(arr[:, 1:4], self.cal, self.level_still_s)
        self.flat_windows = self.flat_windows + 1 if info["motion_g"] < RULES["flatline_g"] else 0
        self._fit_check(act, info)
        if self.flat_windows == 15:
            with self.lock:
                self.messages.append("#W sensor flat-line for 30 s (no noise at all) - sensor may be stuck; check device")
        self.level_still_s = self.level_still_s + 2 if act in ("standing", "lying_belly") else 0
        P = self._model.predict_proba(features_matrix(arr[None, :, 1:4]))[0]
        model_label = BEHAVIOURS[int(P.argmax())]
        ts = int(arr[-1, 0])
        self.labels.append((ts, act, model_label, info))
        self.budget[act] += 2
        self._publish_pose(act, info)
        self._write("labels", f"{ts},{act},{model_label},{info['motion_g']},{info['hf_frac']}," +
                    ",".join("" if info[k] is None else str(info[k]) for k in ("tilt_deg", "pitch_deg", "roll_deg")))
        recent = list(self.labels)[-30:]
        if len(recent) == 30 and all(a in RESTING for _, a, _, _ in recent):
            seg = np.asarray(list(self.samples)[-60 * FS:], float)[:, 1:4]
            if len(seg) == 60 * FS:
                rr, q = estimate_rr(seg)
                if np.isfinite(rr):
                    self.rr.append((ts, rr))

    # ------------------------------------------------------------------ serial loop with auto-reconnect
    def _open_serial(self):
        import serial
        ser = serial.Serial()
        ser.port, ser.baudrate, ser.timeout = self.port, self.baud, 0.5
        ser.dtr = False                                        # do not auto-reset the ESP32 when opening
        ser.rts = False
        ser.open()
        time.sleep(0.3)
        ser.reset_input_buffer()
        ser.write(f"I{self.dog}\n".encode())
        ser.write(f"T{int(time.time() * 1000)}\n".encode())
        ser.write(b"S\n")
        return ser

    def _run(self):
        last_ts = None
        while not self.stop_evt.is_set():
            try:
                self._ser = ser = self._open_serial()
                if self.n:                                     # this is a re-connection
                    self.reconnects += 1
                    with self.lock:
                        self.messages.append(f"reconnected ({self.reconnects})")
                    self._pending = []
                    last_ts = None
                self.connected, self.error = True, None
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
                        if ts < 10**11:                        # device clock not set yet
                            if time.time() - last_sync > 2:
                                ser.write(f"T{int(time.time() * 1000)}\n".encode()); last_sync = time.time()
                            continue
                        if last_ts is not None and ts <= last_ts:   # burst after a short device stall: keep order
                            ts = last_ts + 1
                        if self.devcal is not None:
                            ax, ay, az = (np.array([ax, ay, az]) - self.devcal[0]) * self.devcal[1]
                        with self.lock:
                            if last_ts is not None and ts - last_ts > 30 and self.n > 5 * FS:
                                self.gaps += int((ts - last_ts) // 20) - 1
                                self._pending = []
                            last_ts = ts
                            self.samples.append((ts, ax, ay, az))
                            self.n += 1
                            self._pending.append((ts, ax, ay, az))
                            if len(self._pending) == WIN:
                                self._classify(self._pending)
                                self._pending = []
                        self._write("raw", f"{ts},{p[1]},{p[2]},{p[3]}")
                    elif line.startswith("#A,"):
                        pa = line[3:].split(",")
                        if len(pa) >= 4 and pa[0].isdigit():
                            try:
                                with self.lock:
                                    self.sound.append((time.time(), float(pa[1]), float(pa[2])))
                            except ValueError:
                                continue
                            if int(pa[0]) > 10**11:
                                self._write("sound", ",".join(pa[:4]))
                    elif line.startswith("#Q,"):
                        body = line[3:]
                        with self.lock:
                            self.qc.append(body.split(","))
                        if body.split(",")[0].isdigit() and int(body.split(",")[0]) > 10**11:
                            self._write("qc", body)
                    elif line.startswith("#M,"):
                        body = line[3:]
                        with self.lock:
                            self.messages.append(f"marker: {body}")
                        if body.split(",")[0].isdigit() and int(body.split(",")[0]) > 10**11:
                            self._write("events", body)
                    elif line.startswith("#I,status"):
                        kv = dict(x.split("=", 1) for x in line.split(",")[2:] if "=" in x)
                        self.device = kv
                        self._write_session(status="recording")
                    elif line.startswith(("#W", "#E")):
                        with self.lock:
                            self.messages.append(line)
                    if time.time() - last_sync > 3600:
                        ser.write(f"T{int(time.time() * 1000)}\n".encode()); last_sync = time.time()
            except Exception as e:                             # cable pulled, port busy, device reset ...
                self.connected = False
                self._publish_pose()
                self.error = f"{type(e).__name__}: {e}"
                with self.lock:
                    self.messages.append("USB connection lost - reconnecting every 2 s (recording continues on reconnect)")
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
                for f in self._files.values():
                    f.flush()
                self.stop_evt.wait(2.0)
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
        self.thread.join(timeout=4)
        self.connected = False
        self._publish_pose()
        self._write_session(status="stopped")

    # ------------------------------------------------------------------ snapshot for the UI
    def snapshot(self, seconds=30):
        with self.lock:
            s = np.asarray(list(self.samples)[-seconds * FS:], float) if self.samples else np.zeros((0, 4))
            rate = 0.0
            if len(self.samples) > 2 * FS:
                recent = list(self.samples)[-10 * FS:]
                span = (recent[-1][0] - recent[0][0]) / 1000
                rate = (len(recent) - 1) / span if span > 0 else 0.0
            return dict(samples=s, labels=list(self.labels), budget=dict(self.budget), rr=list(self.rr),
                        qc=list(self.qc), sound=list(self.sound), messages=list(self.messages), n=self.n,
                        gaps=self.gaps, elapsed=time.time() - self.started, connected=self.connected,
                        error=self.error, rate=rate, calibrated=("none" if self.cal is None else "standing+sitting" if self.cal["sit"] is not None else "standing"), rows=dict(self.rows),
                        last_write=self.last_write, reconnects=self.reconnects, files=str(self.base) + "_*.csv",
                        device=self.device, fit=dict(self.fit))
