"""
Tier 1 - raw collar IMU simulation, window features and resting-respiratory-rate estimation.

Device frame convention (what the firmware must also use):
    x = along the neck (cranial +), y = lateral, z = dorso-ventral.
The collar is free to ROTATE about x (Westgarth 2017: dorsal placement rotates to ventral),
so features use only rotation-about-x invariant quantities: |a|, a_x, sqrt(a_y^2+a_z^2),
and eigenvalues of the window covariance.

The same `window_features` and `estimate_rr` functions are used on real hardware data
(see praniiva/ingest.py), so whatever is validated here is what will run on the device data.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .params import BEHAVIOURS, DEFAULT_SENSOR, FS, WIN_S


# ======================================================================================
# Dog-specific motion idiosyncrasy - this is what produces the leave-one-dog-out drop
# ======================================================================================
def dog_motion_profile(weight_kg: float, coat: str, rng: np.random.Generator,
                       cfr: float = 0.5) -> dict:
    walk_f = 1.9 * (weight_kg / 20.0) ** -0.17         # stride frequency, smaller dogs faster
    coat_coupling = {"short": 1.0, "medium": 0.8, "long": 0.6}[coat]
    return dict(
        weight=weight_kg,
        walk_f=walk_f * rng.normal(1.0, 0.08),
        amp=float(np.exp(rng.normal(0.0, 0.25))),         # gait vigour
        scratch_f=rng.uniform(4.5, 7.0),
        shake_f=rng.uniform(5.5, 9.0),
        chew_f=rng.uniform(1.4, 2.6),
        lap_f=rng.uniform(2.8, 4.2),
        pant_f=rng.uniform(3.3, 6.0),
        breath_amp=0.0022 * coat_coupling * rng.uniform(0.6, 1.4) * (20.0 / max(weight_kg, 2)) ** 0.15,
        fit_jitter=rng.uniform(0.002, 0.012),              # loose collar -> more jitter
        posture_bias=rng.normal(0, 8),                     # deg
        brachy=cfr < 0.2,
        coat=coat,
        # per-dog, per-behaviour idiosyncrasy (how THIS dog eats, scratches, pants...)
        beh_amp={b: float(np.exp(rng.normal(0, 0.55))) for b in BEHAVIOURS},
        beh_freq={b: float(np.exp(rng.normal(0, 0.2))) for b in BEHAVIOURS},
        eat_pitch=rng.uniform(-70, -10),                   # some dogs eat lying down / bowl raised
        sniff_rate=rng.uniform(0.1, 0.5),                  # share of 'standing' spent sniffing
        lick_rate=rng.uniform(0.0, 0.25),                  # share of 'lying' spent licking / grooming
        device_damping=float(np.clip(1.0 - 0.01 * (30.0 / max(weight_kg, 2)), 0.6, 1.0)),
    )


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _gravity(pitch_deg, roll_deg):
    p, r = np.deg2rad(pitch_deg), np.deg2rad(roll_deg)
    return np.array([np.sin(p), np.cos(p) * np.sin(r), np.cos(p) * np.cos(r)])


def _quantise(a, sensor):
    lsb = 2 * sensor["range_g"] / (2 ** sensor["bits"])
    a = np.clip(a, -sensor["range_g"], sensor["range_g"])
    return np.round(a / lsb) * lsb


def _core(behaviour, prof, rng, n, fs, rr_bpm, lameness):
    """Body-frame acceleration (gravity + dynamics) for one behaviour, before rotation/noise."""
    t = np.arange(n) / fs
    A = prof["amp"] * prof["beh_amp"][behaviour] * prof["device_damping"]
    fm = prof["beh_freq"][behaviour]
    ph = rng.uniform(0, 2 * np.pi, 6)
    dyn = np.zeros((n, 3))
    pitch, roll = 15.0 + prof["posture_bias"], 0.0

    if behaviour == "lying_rest":
        pitch, roll = rng.uniform(-30, 20), rng.choice([rng.uniform(-90, -50), rng.uniform(-15, 15), rng.uniform(50, 90)])
        rr = (rr_bpm or 16.0) / 60.0
        # breath-to-breath variability (OU process, sd ~6%), non-sinusoidal waveform
        ou = sps.lfilter([1.0], [1, -0.998], rng.normal(0, 0.0028, n))
        phase = 2 * np.pi * np.cumsum(rr * (1 + ou)) / fs + ph[1]
        wave = np.sin(phase) + 0.35 * np.sin(2 * phase + ph[3])
        br = prof["breath_amp"] * (1 + 0.4 * np.sin(2 * np.pi * 0.02 * t + ph[0]))
        dyn[:, 2] += br * wave
        dyn[:, 1] += 0.4 * br * wave
        # cardiac ballistic component (1-2 Hz) and slow postural drift overlapping the RR band
        hr = rng.uniform(55, 120) / 60.0
        dyn[:, 2] += 0.35 * prof["breath_amp"] * np.sin(2 * np.pi * hr * t + ph[2])
        drift = sps.lfilter([1.0], [1, -0.995], rng.normal(0, 1, (n, 3)), axis=0)
        dyn += drift / (np.abs(drift).max() + 1e-9) * prof["breath_amp"] * rng.uniform(0.3, 1.5)
        if n > 10 * fs and rng.random() < 0.35:          # twitch / REM movement
            k = rng.integers(0, n - fs)
            L = int(fs * rng.uniform(0.3, 2.0))
            dyn[k:k + L] += rng.normal(0, rng.uniform(0.01, 0.08), (min(L, n - k), 3))
        elif n <= 10 * fs and rng.random() < prof["lick_rate"]:   # self-grooming looks like chewing
            dyn[:, 2] += rng.uniform(0.02, 0.07) * np.sin(2 * np.pi * rng.uniform(1.5, 3.5) * t + ph[4])
    elif behaviour == "standing":
        dyn[:, 0] += 0.015 * A * np.sin(2 * np.pi * rng.uniform(0.3, 0.8) * t + ph[0])
        dyn[:, 1] += 0.02 * A * np.sin(2 * np.pi * rng.uniform(0.3, 0.8) * t + ph[1])
        dyn += rng.normal(0, 0.008, (n, 3))
        if rng.random() < prof["sniff_rate"]:              # sniffing: head down, irregular small motion
            pitch = rng.uniform(-60, -20)
            dyn += rng.normal(0, rng.uniform(0.02, 0.08), (n, 3)) * (rng.random((n, 1)) < 0.3)
            dyn[:, 2] += rng.uniform(0.01, 0.05) * np.sin(2 * np.pi * rng.uniform(1, 4) * t)
    elif behaviour in ("walking", "trotting"):
        f = prof["walk_f"] * fm * (1.0 if behaviour == "walking" else rng.uniform(1.35, 1.8))
        a = (0.25 if behaviour == "walking" else 0.6) * A * rng.uniform(0.7, 1.3)
        dyn[:, 2] += a * np.sin(2 * np.pi * 2 * f * t + ph[0]) + 0.3 * a * np.sin(2 * np.pi * 4 * f * t + ph[1])
        dyn[:, 2] += lameness * a * 0.8 * np.sin(2 * np.pi * f * t + ph[2])       # head-bob asymmetry
        dyn[:, 1] += 0.45 * a * np.sin(2 * np.pi * f * t + ph[3])
        dyn[:, 0] += 0.35 * a * np.sin(2 * np.pi * 2 * f * t + ph[4])
        dyn += rng.normal(0, 0.05 * a + 0.01, (n, 3))
        pitch = rng.uniform(-30, 25) + prof["posture_bias"]          # head carriage varies (sniffing walk)
    elif behaviour == "scratching":
        f = prof["scratch_f"] * fm
        env = np.ones(n)
        env[int(rng.uniform(0.4, 1.0) * n):] = 0.15
        a = 0.35 * A
        dyn[:, 1] += env * a * np.sin(2 * np.pi * f * t + ph[0])
        dyn[:, 2] += env * 0.6 * a * np.sin(2 * np.pi * f * t + ph[1])
        dyn += rng.normal(0, 0.04, (n, 3))
        pitch, roll = rng.uniform(-20, 30), rng.uniform(-60, 60)
    elif behaviour == "head_shaking":
        f = prof["shake_f"] * fm
        L = int(rng.uniform(0.4, 1.2) * fs)
        k = rng.integers(0, max(1, n - L))
        burst = np.zeros(n)
        burst[k:k + L] = np.hanning(L)
        a = rng.uniform(0.8, 3.0) * prof["beh_amp"][behaviour] ** 0.5
        dyn[:, 1] += burst * a * np.sin(2 * np.pi * f * t + ph[0])
        dyn[:, 2] += burst * 0.7 * a * np.sin(2 * np.pi * f * t + ph[1])
        dyn += rng.normal(0, 0.01, (n, 3))
    elif behaviour == "eating":
        f = prof["chew_f"] * fm
        a = rng.uniform(0.03, 0.10) * A
        dyn[:, 2] += a * np.sin(2 * np.pi * f * t + ph[0])
        dyn[:, 0] += 0.5 * a * np.sin(2 * np.pi * f * t + ph[1])
        jerks = rng.random(n) < 0.03
        dyn[jerks] += rng.normal(0, 0.15, (jerks.sum(), 3))
        dyn += rng.normal(0, 0.012, (n, 3))
        pitch = prof["eat_pitch"] + rng.normal(0, 10)
    elif behaviour == "drinking":
        f = prof["lap_f"] * fm
        a = rng.uniform(0.03, 0.08) * A
        dyn[:, 2] += a * np.sin(2 * np.pi * f * t + ph[0])
        dyn[:, 0] += 0.4 * a * np.sin(2 * np.pi * f * t + ph[1])
        dyn += rng.normal(0, 0.008, (n, 3))
        pitch = prof["eat_pitch"] - 10 + rng.normal(0, 10)
    elif behaviour == "panting":
        f = prof["pant_f"] * fm
        a = rng.uniform(0.02, 0.07) * A * (1.5 if prof["brachy"] else 1.0)
        dyn[:, 2] += a * np.sin(2 * np.pi * f * t + ph[0])
        dyn[:, 1] += 0.3 * a * np.sin(2 * np.pi * f * t + ph[1])
        dyn += rng.normal(0, 0.006, (n, 3))
        if rng.random() < 0.5:
            pitch, roll = rng.uniform(-20, 10), rng.uniform(-80, 80)
    else:
        raise ValueError(behaviour)

    g = _gravity(pitch + rng.normal(0, 3), roll + rng.normal(0, 3))
    acc = g[None, :] + dyn
    if behaviour != "lying_rest":
        acc += rng.normal(0, prof["fit_jitter"], (n, 3))
    return acc


_TRANSITION_P = 0.2   # share of 2 s windows that straddle two behaviours (label = majority)


def synth_segment(behaviour: str, prof: dict, rng: np.random.Generator, seconds: float = WIN_S,
                  rr_bpm: float | None = None, lameness: float = 0.0, sensor: dict | None = None,
                  collar_angle: float | None = None, transitions: bool = True) -> np.ndarray:
    """Return (n, 3) acceleration in g, in the (rotated) device frame, for one behaviour segment."""
    sensor = sensor or DEFAULT_SENSOR
    fs = sensor["fs"]
    n = int(round(seconds * fs))
    acc = _core(behaviour, prof, rng, n, fs, rr_bpm, lameness)
    if transitions and seconds <= 4 and rng.random() < _TRANSITION_P:
        other = BEHAVIOURS[rng.integers(len(BEHAVIOURS))]
        k = int(n * rng.uniform(0.55, 0.85))              # majority stays the labelled behaviour
        acc2 = _core(other, prof, rng, n, fs, rr_bpm, 0.0)
        if rng.random() < 0.5:
            acc[k:] = acc2[k:]
        else:
            acc[:n - k] = acc2[:n - k]
    theta = rng.uniform(-np.pi, np.pi) if collar_angle is None else collar_angle
    acc = acc @ _rot_x(theta).T                            # collar rotation about neck axis
    sd = sensor["noise_ug_rthz"] * 1e-6 * np.sqrt(fs / 2.0)
    acc += rng.normal(0, sd, acc.shape)
    return _quantise(acc, sensor)


# ======================================================================================
# Window features (orientation-invariant about the neck axis)
# ======================================================================================
_BANDS = [(0.1, 1.0), (1.0, 2.5), (2.5, 4.0), (4.0, 6.0), (6.0, 9.0), (9.0, 15.0)]

FEATURE_NAMES = (
    ["vm_mean", "vm_std", "vm_min", "vm_max", "vm_p10", "vm_p90", "vm_range",
     "ax_mean", "ax_std", "ayz_mean", "ayz_std", "ev1", "ev2", "ev3",
     "jerk_mean", "jerk_p90", "dom_freq", "spec_entropy", "zcr", "ac_peak", "peak_frac"]
    + [f"bp_{lo:g}_{hi:g}" for lo, hi in _BANDS]
)


def window_features(w: np.ndarray, fs: int = FS) -> np.ndarray:
    vm = np.linalg.norm(w, axis=1)
    ax = w[:, 0]
    ayz = np.hypot(w[:, 1], w[:, 2])
    d = vm - vm.mean()
    ev = np.sort(np.linalg.eigvalsh(np.cov(w.T)))[::-1]
    ev = np.sqrt(np.clip(ev, 0, None))
    jerk = np.abs(np.diff(vm)) * fs
    f, p = sps.periodogram(d, fs=fs)
    p = p + 1e-12
    pn = p / p.sum()
    bp = [np.log(p[(f >= lo) & (f < hi)].sum() + 1e-10) for lo, hi in _BANDS]
    zcr = np.mean(np.diff(np.sign(d)) != 0)
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = int(fs / 15)
    ac_peak = ac[lo:len(ac) // 2].max() if len(ac) // 2 > lo else 0.0
    peak_frac = np.mean(np.abs(d) > 3 * (np.median(np.abs(d)) + 1e-6))
    return np.array([
        vm.mean(), vm.std(), vm.min(), vm.max(), np.percentile(vm, 10), np.percentile(vm, 90), np.ptp(vm),
        ax.mean(), ax.std(), ayz.mean(), ayz.std(), ev[0], ev[1], ev[2],
        jerk.mean(), np.percentile(jerk, 90), f[np.argmax(p[1:]) + 1], -(pn * np.log(pn)).sum(), zcr,
        ac_peak, peak_frac, *bp,
    ])


def features_matrix(windows: np.ndarray, fs: int = FS) -> np.ndarray:
    return np.vstack([window_features(w, fs) for w in windows])


# ======================================================================================
# Resting respiratory rate from collar micro-motion (ballistocardiography-style)
# ======================================================================================
def estimate_rr(seg: np.ndarray, fs: int = FS, min_quality: float = 4.0) -> tuple[float, float]:
    """
    Estimate breathing rate (breaths/min) from a >=30 s lying segment.
    Returns (rr_bpm or nan, quality). Quality = spectral peak / median band power;
    segments under `min_quality` are rejected -> this rejection rate is the 'yield'.
    """
    b, a = sps.butter(2, [0.1, 1.5], btype="band", fs=fs)
    x = sps.filtfilt(b, a, seg - seg.mean(axis=0), axis=0)
    # project on first principal component (unknown collar rotation)
    u, s, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    pc = x @ vt[0]
    nper = min(len(pc), fs * 30)
    f, p = sps.welch(pc, fs=fs, nperseg=nper, noverlap=nper // 2, nfft=max(4096, nper))
    band = (f >= 0.13) & (f <= 1.0)                     # 8-60 breaths/min
    if not band.any():
        return np.nan, 0.0
    fb, pb = f[band], p[band]
    k = np.argmax(pb)
    quality = float(pb[k] / (np.median(pb) + 1e-18))
    if quality < min_quality:
        return np.nan, quality
    # parabolic interpolation of the peak
    if 0 < k < len(pb) - 1:
        y0, y1, y2 = np.log(pb[k - 1:k + 2] + 1e-30)
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        fpk = fb[k] + delta * (fb[1] - fb[0])
    else:
        fpk = fb[k]
    return float(fpk * 60.0), quality


def make_windows(prof, behaviour, n, rng, sensor=None, lameness=0.0):
    return np.stack([synth_segment(behaviour, prof, rng, sensor=sensor, lameness=lameness) for _ in range(n)])


__all__ = ["BEHAVIOURS", "dog_motion_profile", "synth_segment", "window_features",
           "features_matrix", "estimate_rr", "FEATURE_NAMES", "make_windows"]
