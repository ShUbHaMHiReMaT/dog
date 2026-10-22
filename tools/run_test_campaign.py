"""
Large randomized test campaign for the live system (no hardware needed - runs alongside a real recording).

    python tools/run_test_campaign.py            # ~10,000 cases, results in reports/test_campaign/

Families
  A  motion classifier     - still / small / walking / running / shaking at random device orientations, with noise
                             and harmonics (wrist, chest or any mount)
  B  posture classifier    - standing / sitting / lying belly / lying side / on back / head down at random mountings
  C  live pipeline (fake device) - complete LiveCollar sessions fed with valid data mixed with garbage lines, truncated
                             lines, bad numbers, unsynced clocks, sample gaps and cable disconnects; checks that nothing
                             crashes, every valid row is recorded exactly once, gaps and reconnects are counted right
  D  recording files       - every file a session writes re-read and validated against the pipeline format
Every case is written to cases.csv (inputs, expected, got, pass) so any failure can be replayed.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import praniiva.live as live  # noqa: E402
from praniiva.live import _unit, window_activity  # noqa: E402

OUT = ROOT / "reports" / "test_campaign"
T = np.arange(100) / 50
rows = []


def rec(family, case, expected, got, ok, **kw):
    rows.append(dict(family=family, case=case, expected=str(expected), got=str(got), passed=bool(ok), **kw))


# ------------------------------------------------------------------ A. motion
def family_a(n_per=800, seed=11):
    rng = np.random.default_rng(seed)
    # class bands are defined in RMS motion (what the rule measures), with a margin around each threshold
    specs = {"still": ((1, 1), (0.0, 0.0)), "small": ((0.4, 1.5), (0.019, 0.040)),
             "walking": ((1.4, 2.6), (0.06, 0.28)), "running": ((2.4, 4.0), (0.42, 1.0)),
             "shaking": ((5.0, 9.0), (0.2, 1.3))}
    for want, ((f0, f1), (a0, a1)) in specs.items():
        for k in range(n_per):
            m = Rot.random(random_state=int(rng.integers(1e9)))
            g = m.inv().apply([0, 0, 1])
            f, a = rng.uniform(f0, f1), rng.uniform(a0, a1)
            u1, u2 = _unit(rng.normal(size=3)), _unit(rng.normal(size=3))
            noise = rng.uniform(0.0005, 0.003)
            ph = rng.uniform(0, 2 * np.pi)
            dyn = (np.outer(np.sin(2 * np.pi * f * T + ph), u1)
                   + np.outer(rng.uniform(0, 0.35) * np.sin(4 * np.pi * f * T + ph + 1), u2))
            dyn -= dyn.mean(0)
            rms = np.sqrt((dyn ** 2).mean(0).sum())
            x = g + (a / rms if rms > 0 else 0) * dyn + rng.normal(0, noise, (100, 3))
            got, info = window_activity(x, None)
            rec("A_motion", f"{want}_{k}", want, got, got == want, freq_hz=round(f, 2), amp_g=round(a, 3),
                noise_g=round(noise, 4), motion_g=info["motion_g"], dom_hz=info["dom_hz"])


# ------------------------------------------------------------------ B. posture
def _grav(pitch=0.0, roll=0.0):
    return Rot.from_euler("yx", [-pitch, roll], degrees=True).inv().apply([0, 0, 1])


def family_b(n=400, seed=12):
    rng = np.random.default_rng(seed)
    for k in range(n):
        m = Rot.random(random_state=int(rng.integers(1e9)))
        w = lambda p=0.0, r=0.0: m.inv().apply(_grav(p, r)) + rng.normal(0, 0.002, (100, 3))
        sit_p = rng.uniform(30, 70)
        stand, sit = _unit(w().mean(0)), _unit(w(sit_p).mean(0))
        cal = dict(stand=stand, sit=sit, sit_pitch=float(np.degrees(np.arccos(stand @ sit))))
        cases = [("standing", dict(p=rng.uniform(-10, 10), r=rng.uniform(-10, 10)), 0),
                 ("standing", dict(p=rng.uniform(-30, -15)), 0),
                 ("sitting", dict(p=sit_p + rng.uniform(-10, 10), r=rng.uniform(-10, 10)), 0),
                 ("lying_side", dict(r=rng.choice([-1, 1]) * rng.uniform(65, 110)), 0),
                 ("lying_side", dict(r=rng.uniform(165, 195)), 0),
                 ("lying_belly", dict(p=rng.uniform(-8, 8)), rng.uniform(120, 3000))]
        for j, (want, kw, still_s) in enumerate(cases):
            got, info = window_activity(w(**kw), cal, still_s)
            rec("B_posture", f"{k}_{j}", want, got, got == want, sit_pitch=round(sit_p, 1), **{k2: round(v, 1)
                for k2, v in kw.items()}, still_s=round(still_s), tilt=info["tilt_deg"])


# ------------------------------------------------------------------ F. worst cases from the literature / real use
def family_f(n=150, seed=15):
    """Each scenario: a known failure mode of body-worn accelerometers and what the system must do.
    Sources: sensor displacement/rotation (Banos 2014, Sensors), car rides / play / collar rotation and tightness
    (Chambers 2021, Animals), static postures hardest (Kumpulainen 2021), rare short behaviours (wolf collar study,
    Anim Biotelemetry), saturation / spikes / stuck sensors / calibration drift (device practice, GGIR)."""
    rng = np.random.default_rng(seed)

    def base(m):
        return m.inv().apply([0, 0, 1])

    def osc(f, rms, u, ph=0.0, n=100):
        return np.outer(np.sqrt(2) * rms * np.sin(2 * np.pi * f * np.arange(n) / 50 + ph), u)

    for k in range(n):
        m = Rot.random(random_state=int(rng.integers(1e9)))
        g, u = base(m), _unit(rng.normal(size=3))
        noise = rng.normal(0, 0.0015, (100, 3))
        still = g + noise
        cases = []
        # 1 single-sample glitch (I2C read error) while the dog lies still -> must stay still
        x = still.copy(); x[int(rng.integers(5, 95))] += rng.choice([-1, 1]) * rng.uniform(1, 7) * _unit(rng.normal(size=3))
        cases.append(("spike_while_still", x, {"still"}))
        # 2 car ride: engine/road vibration 13-24 Hz while lying still -> still, not 'moving'
        x = still + osc(rng.uniform(13, 24), rng.uniform(0.02, 0.08), u)
        cases.append(("car_vibration_lying", x, {"still"}))
        # 3 short head shake (0.8 s burst, 5-8 Hz) inside a 2 s window -> shaking (rare behaviour)
        x = still.copy(); st = int(rng.integers(0, 55)); L = 40
        x[st:st + L] += osc(rng.uniform(5, 8), rng.uniform(0.6, 1.5), u, n=L)
        cases.append(("short_head_shake", x, {"shaking"}))
        # 4 saturation: hard play hitting the +-8 g limit -> not still / not small
        x = g + osc(rng.uniform(2.5, 3.5), rng.uniform(2.5, 5), u) + noise; x = np.clip(x, -8, 8)
        cases.append(("clipped_hard_play", x, {"running", "large", "shaking"}))
        # 5 calibration drift (temperature): +-100 mg offset and +-5 % gain -> motion labels unchanged
        off, gain = rng.uniform(-0.1, 0.1, 3), rng.uniform(0.95, 1.05, 3)
        f = rng.uniform(1.5, 2.5); r = rng.uniform(0.08, 0.25)
        x = ((g + osc(f, r, u) + noise) + off) * gain
        cases.append(("walking_with_drifted_sensor", x, {"walking"}))
        # 6 collar slowly rotating round the neck while walking (sensor displacement) -> still walking
        rot = Rot.from_rotvec(np.outer(np.linspace(0, rng.uniform(0.2, 0.6), 100), _unit(rng.normal(size=3))))
        x = rot.apply(g) + osc(rng.uniform(1.5, 2.5), rng.uniform(0.08, 0.25), u) + noise
        cases.append(("walking_collar_rotating", x, {"walking"}))
        # 7 slow big movements (rolling / arm waving, 0.4-1.5 Hz, large) -> large, NOT running (learned today)
        x = g + osc(rng.uniform(0.5, 1.5), rng.uniform(0.45, 0.9), u) + noise
        cases.append(("slow_large_movement", x, {"large"}))
        # 8 upside-down / any mounting while running -> running
        x = g + osc(rng.uniform(2.5, 4), rng.uniform(0.45, 1.0), u) + osc(rng.uniform(5, 7), 0.1, _unit(rng.normal(size=3))) + noise
        cases.append(("running_any_mounting", x, {"running"}))
        # 9 loose collar rattle while lying (tiny clicks) -> still or small, never walking/running
        x = still.copy(); idx = rng.choice(100, 6, replace=False); x[idx] += rng.normal(0, 0.02, (6, 3))
        cases.append(("loose_collar_rattle", x, {"still", "small"}))
        # 10 stuck sensor (constant output, no noise at all) -> flagged as flat-line
        x = np.repeat(g[None, :], 100, 0)
        cases.append(("stuck_sensor", x, "FLAT"))
        for name, x, want in cases:
            got, info = window_activity(x, None)
            if want == "FLAT":
                ok = info["motion_g"] < live.RULES["flatline_g"]
                rec("F_worst_case", f"{name}_{k}", "flatline flagged", f"motion={info['motion_g']}", ok, scenario=name)
            else:
                rec("F_worst_case", f"{name}_{k}", "|".join(sorted(want)), got, got in want, scenario=name,
                    motion_g=info["motion_g"], dom_hz=info["dom_hz"])


# ------------------------------------------------------------------ G. fit / placement check
def _fit_probe(cal=None):
    lc = live.LiveCollar.__new__(live.LiveCollar)
    from collections import deque
    lc._rest, lc._walk, lc.cal, lc._fit_said, lc.lock = deque(maxlen=60), deque(maxlen=15), cal, set(), threading.Lock()
    lc.messages, lc.mark = deque(maxlen=40), (lambda label: None)
    lc.fit = dict(worn="checking", loose="checking", shifted="calibrate posture to check", gravity="checking", summary="")
    return lc


def family_g(n=60, seed=16):
    """Scenarios measured on the real device 2026-10-22: desk 1.1-4.4 mg at rest, wrist 3.8-14 mg."""
    rng = np.random.default_rng(seed)
    for k in range(n):
        m = Rot.random(random_state=int(rng.integers(1e9)))
        g, u = m.inv().apply([0, 0, 1]), _unit(rng.normal(size=3))

        def walk(gv, rattle=0.0):
            x = gv + np.outer(np.sqrt(2) * rng.uniform(0.08, 0.2) * np.sin(2 * np.pi * rng.uniform(1.6, 2.4) * T), u)
            x += rng.normal(0, 0.0015, (100, 3))
            if rattle:                                                  # loose: clicks / buzz above 12 Hz
                x += np.outer(np.sqrt(2) * rattle * np.sin(2 * np.pi * rng.uniform(14, 22) * T), _unit(rng.normal(size=3)))
            return x

        scen = {}
        # 1 on a table: only sensor noise for 2+ min -> not worn
        lc = _fit_probe()
        for _ in range(65):
            lc._fit_check(*window_activity(g + rng.normal(0, 0.0007, (100, 3)), None))
        scen["on_table_not_worn"] = (lc.fit["worn"].startswith("NO"), lc.fit["worn"])
        # the summary must list each problem once, however long the problem lasts (bug found 2026-10-22)
        scen["summary_not_repeating"] = (lc.fit["summary"] == lc.fit["worn"], lc.fit["summary"][:80])
        # 2 worn, resting: body micro-motion 4-12 mg -> worn
        lc = _fit_probe()
        for _ in range(65):
            amp = rng.uniform(0.004, 0.012)
            x = g + np.outer(np.sqrt(2) * amp * np.sin(2 * np.pi * rng.uniform(0.3, 3) * T), u) + rng.normal(0, 0.0015, (100, 3))
            lc._fit_check(*window_activity(x, None))
        scen["worn_resting"] = (lc.fit["worn"].startswith("yes"), lc.fit["worn"])
        # 3 firm strap while walking -> firm ; 4 loose strap rattling -> loose
        lc = _fit_probe()
        for _ in range(15):
            lc._fit_check(*window_activity(walk(g), None))
        scen["firm_walking"] = (lc.fit["loose"].startswith("firm"), lc.fit["loose"])
        lc = _fit_probe()
        for _ in range(15):
            lc._fit_check(*window_activity(walk(g, rattle=rng.uniform(0.08, 0.2)), None))
        scen["loose_rattling"] = (lc.fit["loose"].startswith("LOOSE"), lc.fit["loose"])
        # 5 device rotated 35-80 deg after posture calibration -> moved ; 6 not rotated -> in place
        cal = dict(stand=_unit(g), sit=None, sit_pitch=None)
        # a TILT (axis perpendicular to gravity); a spin about the vertical leaves gravity unchanged and is invisible
        # to any accelerometer - it also does not affect posture, which is computed from gravity
        tilt_axis = _unit(np.cross(g, rng.normal(size=3)))
        moved = Rot.from_rotvec(np.radians(rng.uniform(35, 80)) * tilt_axis).apply(g)
        lc = _fit_probe(cal)
        for _ in range(15):
            lc._fit_check(*window_activity(walk(moved), None))
        scen["device_shifted"] = (lc.fit["shifted"].startswith("MOVED"), lc.fit["shifted"])
        lc = _fit_probe(cal)
        for _ in range(15):
            lc._fit_check(*window_activity(walk(Rot.from_rotvec(np.radians(rng.uniform(0, 12)) * _unit(rng.normal(size=3))).apply(g)), None))
        scen["device_in_place"] = (lc.fit["shifted"].startswith("in place"), lc.fit["shifted"])
        # 7 sensor gain fault (reads 0.85 or 1.15 g) -> gravity fault ; 8 healthy -> OK
        for nm, gain in (("gravity_fault", rng.choice([0.85, 1.15])), ("gravity_ok", rng.uniform(0.99, 1.01))):
            lc = _fit_probe()
            for _ in range(12):
                lc._fit_check(*window_activity(g * gain + rng.normal(0, 0.003, (100, 3)), None))
            want = "FAULT" if nm == "gravity_fault" else "OK"
            scen[nm] = (lc.fit["gravity"].startswith(want), lc.fit["gravity"])
        for nm, (ok, got) in scen.items():
            rec("G_fit_check", f"{nm}_{k}", nm, got, ok, scenario=nm)


# ------------------------------------------------------------------ C/D. live pipeline with a fake device
class FakeSerial:
    def __init__(self, lines, disconnect_at_end):
        self.lines, self.i, self.disc, self.sent = lines, 0, disconnect_at_end, []
        self.done = threading.Event()

    def readline(self):
        if self.i < len(self.lines):
            self.i += 1
            return self.lines[self.i - 1]
        if self.disc:
            self.done.set()
            raise OSError("fake cable pulled")
        self.done.set()
        time.sleep(0.01)
        return b""

    def write(self, b):
        self.sent.append(b)

    def close(self):
        pass


GARBAGE = [b"\x7f\x7f\x7f\x00\xff\n", b"12345\n", b"1792000000000,0.1\n", b"1792000000000,abc,0.2,0.3\n",
           b"#A,xyz\n", b"#A,1792000000000,nan?,-20,1\n", b"#Q\n", b"#Q,\n", b"#M,\n", b"#I,status,=,==,a=b=c\n",
           b"#W,something\n", b"#E,imu_not_ready\n", b",,,,\n", b"\n", b"   \n", "éè中\n".encode(),
           b"1e12,0,0,1\n", b"-5,0,0,1\n", b"17920000000001792,0.1,0.2\n", b"#A,1792000000000,-30\n"]


def build_session(rng, t0):
    """Returns per-connection line lists and the exact expected counts."""
    n_conn = int(rng.integers(1, 4))
    conns, exp = [], dict(raw=0, sound=0, qc=0, events=0, gaps=0, reconnects=n_conn - 1)
    ts = t0
    total = 0
    for c in range(n_conn):
        L = []
        if rng.random() < 0.3:                                           # device clock not yet synced
            for k in range(int(rng.integers(5, 40))):
                L.append(f"{1000 + 20 * k},0.0,0.0,1.0\n".encode())
        n = int(rng.integers(150, 900))
        for k in range(n):
            if total > 5 * 50 and rng.random() < 0.002:                  # device stall: burst with repeated stamps
                for _ in range(int(rng.integers(3, 25))):
                    L.append(f"{ts},0.0,0.0,1.0\n".encode())
                    exp["raw"] += 1
                    total += 1
            if total > 5 * 50 and rng.random() < 0.004:                  # radio/I2C gap of 2..60 samples
                skip = int(rng.integers(2, 60))
                ts += 20 * skip
                exp["gaps"] += skip - 1
            else:
                ts += 20
            ax, ay, az = rng.normal(0, 0.3, 3)
            L.append(f"{ts},{ax:.5f},{ay:.5f},{az + 1:.5f}\n".encode())
            exp["raw"] += 1
            total += 1
            r = rng.random()
            if r < 0.02:
                L.append(f"#A,{ts},{rng.uniform(-80, -5):.1f},{rng.uniform(-60, 0):.1f},1\n".encode())
                exp["sound"] += 1
            elif r < 0.023:
                L.append(f"#Q,{ts},1.2,1.5,nan,3000,0,0,nan,0\n".encode())
                exp["qc"] += 1
            elif r < 0.025:
                L.append(f"#M,{ts},label_{k}\n".encode())
                exp["events"] += 1
            elif r < 0.06:
                L.append(GARBAGE[int(rng.integers(len(GARBAGE)))])
            elif r < 0.065:
                L.append(f"{ts}".encode()[:int(rng.integers(1, 8))] + b"\n")          # truncated line
        conns.append(L)
        if c < n_conn - 1:
            ts += 20 * 50 * 3                                            # 3 s while the cable was out
    return conns, exp


def run_fake_session(idx, rng, tmp):
    conns, exp = build_session(rng, 1792000000000 + idx * 10_000_000)
    fakes = [FakeSerial(L, disconnect_at_end=(i < len(conns) - 1)) for i, L in enumerate(conns)]
    it = iter(fakes)
    lc = live.LiveCollar.__new__(live.LiveCollar)
    lc._open_serial = lambda: next(it)                                  # instance override, no hardware
    orig_init_thread = threading.Thread.start
    live.LiveCollar.__init__(lc, "FAKE", f"S{idx:04d}", mount="test")
    t_end = time.time() + 30
    while not fakes[-1].done.is_set() and time.time() < t_end:
        time.sleep(0.02)
    lc.stop()
    got = dict(raw=lc.rows["raw"], sound=lc.rows["sound"], qc=lc.rows["qc"], events=lc.rows["events"],
               gaps=lc.gaps, reconnects=lc.reconnects)
    for key in exp:                                                     # gaps after a reconnect are reset by design
        e, g = exp[key], got[key]
        ok = (g <= e) if key == "gaps" else (g == e)
        rec("C_pipeline", f"session{idx}_{key}", e, g, ok, connections=len(conns))
    # D. files on disk
    base = Path(str(lc.base))
    try:
        r = pd.read_csv(f"{base}_raw.csv")
        ok = (list(r.columns) == ["timestamp_ms", "ax", "ay", "az"] and len(r) == exp["raw"]
              and r.timestamp_ms.is_monotonic_increasing and r.timestamp_ms.min() > 10**11 and not r.isna().any().any())
        rec("D_files", f"session{idx}_raw", exp["raw"], len(r), ok)
        lb = pd.read_csv(f"{base}_labels.csv")
        ok = len(lb) > 0 and lb.activity.isin(list(live.ACTIVITIES)).all() and len(lb) <= exp["raw"] // 100 + 1
        rec("D_files", f"session{idx}_labels", "valid", len(lb), ok)
        for k in ("sound", "qc", "events"):
            d = pd.read_csv(f"{base}_{k}.csv")
            rec("D_files", f"session{idx}_{k}", exp[k], len(d), len(d) == exp[k])
        meta = json.loads(Path(f"{base}_session.json").read_text())
        rec("D_files", f"session{idx}_session_json", "stopped", meta["status"], meta["status"] == "stopped"
            and meta["rows"]["raw"] == exp["raw"])
    except Exception as e:                                              # noqa: BLE001
        rec("D_files", f"session{idx}_read", "readable", repr(e), False)


def family_cd(n_sessions=250, seed=13):
    rng = np.random.default_rng(seed)
    tmp = Path(tempfile.mkdtemp(prefix="praniiva_campaign_"))
    live.ROOT = tmp                                                     # recordings go to a temp folder
    live.POSE_FILE = tmp / "live_pose.json"
    (tmp / "models").mkdir()
    import shutil
    shutil.copy(ROOT / "models" / "behaviour_model.joblib", tmp / "models" / "behaviour_model.joblib")
    threads = []
    for i in range(n_sessions):
        th = threading.Thread(target=run_fake_session, args=(i, np.random.default_rng(rng.integers(1e9)), tmp))
        th.start()
        threads.append(th)
        if len(threads) >= 16:
            for t in threads:
                t.join()
            threads = []
    for t in threads:
        t.join()
    shutil.rmtree(tmp, ignore_errors=True)


def family_e(seed=14):
    """Boundary sweep (informational, not pass/fail): where does each label switch as motion grows?"""
    rng = np.random.default_rng(seed)
    out = []
    for f in (1.0, 2.0, 3.0, 6.0):
        for a in np.round(np.geomspace(0.003, 1.5, 60), 4):
            g = Rot.random(random_state=int(rng.integers(1e9))).inv().apply([0, 0, 1])
            u = _unit(rng.normal(size=3))
            x = g + np.outer(np.sqrt(2) * a * np.sin(2 * np.pi * f * T), u) + rng.normal(0, 0.001, (100, 3))
            out.append(dict(freq_hz=f, rms_g=a, label=window_activity(x, None)[0]))
    pd.DataFrame(out).to_csv(OUT / "boundary_sweep.csv", index=False)
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    family_a()
    print(f"A done ({len(rows)} cases, {time.time() - t0:.0f}s)", flush=True)
    family_b()
    family_f()
    family_g()
    print(f"B+F done ({len(rows)} cases, {time.time() - t0:.0f}s)", flush=True)
    family_cd()
    print(f"C/D done ({len(rows)} cases, {time.time() - t0:.0f}s)", flush=True)
    sw = family_e()
    print("boundary sweep - first motion level (RMS g) at which each label appears:")
    print(sw.groupby(["freq_hz", "label"]).rms_g.min().unstack().to_string())
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cases.csv", index=False)
    summ = df.groupby("family").passed.agg(cases="size", passed="sum")
    summ["pass_rate"] = (summ.passed / summ.cases).round(4)
    summ.to_csv(OUT / "summary.csv")
    fails = df[~df.passed]
    fails.to_csv(OUT / "failures.csv", index=False)
    conf = (df[df.family.isin(["A_motion", "B_posture"])].assign(exp=lambda d: d.family + ":" + d.expected)
            .pipe(lambda d: pd.crosstab(d.exp, d.got)))
    conf.to_csv(OUT / "confusion.csv")
    print(summ.to_string())
    print(f"\nTOTAL {len(df)} cases, {int(df.passed.sum())} passed, {len(fails)} failed, {time.time() - t0:.0f}s")
    print(conf.to_string())


if __name__ == "__main__":
    main()
