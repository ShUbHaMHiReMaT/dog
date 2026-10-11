"""
Real-hardware ingest: raw collar CSV  ->  daily measured row (same schema the simulator produces).

Expected raw file (one per dog per day, or longer - it is split by local date):
    timestamp_ms, ax, ay, az            (g, device frame: x along the neck)
optional columns:
    env_temp_c, env_rh, cough_event     (from on-collar env sensor / mic event detector)

Pipeline (identical maths to the simulation, so validation results transfer):
  1. resample to 50 Hz, cut into 2 s windows
  2. window features  -> behaviour classifier (models/behaviour_model.joblib)
  3. sticky-HMM bout smoothing (+ head-shake event bypass)
  4. daily minutes per behaviour, night restlessness bouts, gait asymmetry from walking windows
  5. resting RR from 60 s lying segments at night (rejecting low-quality segments)

When real labelled data exists, retrain the behaviour model on it with
praniiva.behaviour.run_behaviour_study-style grouped validation before trusting outputs.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import signal as sps

from .behaviour import viterbi_smooth
from .imu import estimate_rr, features_matrix
from .params import BEHAVIOURS, FS

WIN = int(2 * FS)


def load_raw(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"timestamp_ms", "ax", "ay", "az"}
    if not need <= set(df.columns):
        raise ValueError(f"raw file needs columns {need}, got {list(df.columns)}")
    df["t"] = pd.to_datetime(df.timestamp_ms, unit="ms")
    return df.sort_values("t").reset_index(drop=True)


def resample_50hz(df: pd.DataFrame) -> pd.DataFrame:
    t0 = df.t.iloc[0]
    ts = (df.t - t0).dt.total_seconds().values
    grid = np.arange(0, ts[-1], 1 / FS)
    out = pd.DataFrame({"t": t0 + pd.to_timedelta(grid, unit="s")})
    for c in ("ax", "ay", "az"):
        out[c] = np.interp(grid, ts, df[c].values)
    # flag gaps > 1 s (device off / BLE loss) so they are not counted as behaviour
    gap = np.zeros(len(grid), bool)
    big = np.where(np.diff(ts) > 1.0)[0]
    for i in big:
        gap[(grid > ts[i]) & (grid < ts[i + 1])] = True
    out["gap"] = gap
    return out


def classify_stream(acc: np.ndarray, model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    bundle = joblib.load(model_path)
    n = len(acc) // WIN
    W = acc[: n * WIN].reshape(n, WIN, 3)
    P = bundle["model"].predict_proba(features_matrix(W))
    return viterbi_smooth(P), W


def night_bouts(labels: np.ndarray, times: pd.Series) -> int:
    """Number of transitions from rest to activity between 22:00 and 06:00."""
    hour = times.dt.hour.values
    night = (hour >= 22) | (hour < 6)
    rest = labels == BEHAVIOURS.index("lying_rest")
    return int(np.sum(night[1:] & rest[:-1] & ~rest[1:]))


def gait_asymmetry(W: np.ndarray, labels: np.ndarray) -> float:
    """Head-bob asymmetry: power at stride frequency vs step frequency in |a| during walking (EMERGING)."""
    idx = np.where(labels == BEHAVIOURS.index("walking"))[0]
    if len(idx) < 10:
        return np.nan
    vals = []
    for i in idx:
        vm = np.linalg.norm(W[i], axis=1)
        f, p = sps.periodogram(vm - vm.mean(), fs=FS)
        k2 = np.argmax(p[(f > 2) & (f < 6)]) + np.searchsorted(f, 2)
        f_step = f[k2]
        k1 = np.argmin(np.abs(f - f_step / 2))
        vals.append(p[k1] / (p[k2] + 1e-12))
    return float(np.median(np.sqrt(vals)))


def night_rr(acc: np.ndarray, labels: np.ndarray, times: pd.Series, seg_s=60) -> tuple[float, int]:
    hour = times.dt.hour.values
    seg_w = seg_s // 2
    lying = BEHAVIOURS.index("lying_rest")
    ests = []
    for s in range(0, len(labels) - seg_w, seg_w):
        if np.all(labels[s:s + seg_w] == lying) and ((hour[s] >= 0) & (hour[s] < 6)):
            rr, q = estimate_rr(acc[s * WIN:(s + seg_w) * WIN])
            if np.isfinite(rr):
                ests.append(rr)
    return (float(np.median(ests)) if len(ests) >= 3 else np.nan), len(ests)


def daily_from_raw(path: Path | str, dog_id: str, model_dir: Path) -> pd.DataFrame:
    raw = load_raw(path)
    rs = resample_50hz(raw)
    rows = []
    for date, day in rs.groupby(rs.t.dt.date):
        acc = day[["ax", "ay", "az"]].values
        labels, W = classify_stream(acc, model_dir / "behaviour_model.joblib")
        wt = day.t.iloc[::WIN].iloc[: len(labels)].reset_index(drop=True)
        valid = ~day.gap.values[::WIN][: len(labels)]
        labels_v = labels[valid]
        wear_frac = valid.sum() * 2 / 86400
        mins = {b: (labels_v == i).sum() * 2 / 60 / max(wear_frac, 1e-6) for i, b in enumerate(BEHAVIOURS)}
        rr, n_seg = night_rr(acc, labels, wt)
        hs = BEHAVIOURS.index("head_shaking")
        row = dict(dog_id=dog_id, date=pd.Timestamp(date), wear_frac=min(wear_frac, 1.0),
                   active_min=mins["walking"] + mins["trotting"], high_int_min=mins["trotting"],
                   standing_min=mins["standing"], rest_min=mins["lying_rest"], scratch_min=mins["scratching"],
                   headshake_count=float((labels_v == hs).sum()) / max(wear_frac, 1e-6),
                   eat_min=mins["eating"], drink_min=mins["drinking"], pant_min=mins["panting"],
                   night_bouts=night_bouts(labels, wt), stiffness_latency=np.nan,
                   gait_asym=gait_asymmetry(W, labels), sleep_rr=rr, rr_valid_segments=n_seg)
        for opt, col in (("env_temp_c", "env_temp"), ("cough_event", "cough_count")):
            if opt in raw.columns:
                sub = raw[raw.t.dt.date == date][opt]
                row[col] = sub.mean() if col == "env_temp" else sub.sum()
            else:
                row[col] = np.nan
        row["skin_temp"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_demo_raw(path: Path, hours=3.0, seed=0):
    """Write a synthetic raw CSV in the hardware format (lets you test ingest end-to-end today)."""
    from .imu import dog_motion_profile, synth_segment
    rng = np.random.default_rng(seed)
    prof = dog_motion_profile(28, "short", rng)
    plan = [("lying_rest", 1800), ("walking", 600), ("scratching", 8), ("standing", 120), ("eating", 240),
            ("drinking", 40), ("lying_rest", 3600), ("trotting", 300), ("head_shaking", 2), ("panting", 400),
            ("lying_rest", 3600)]
    angle = rng.uniform(-np.pi, np.pi)
    segs = [synth_segment(b, prof, rng, seconds=s, rr_bpm=15, collar_angle=angle, transitions=False) for b, s in plan]
    acc = np.concatenate(segs)[: int(hours * 3600 * FS)]
    t0 = pd.Timestamp("2026-10-10 00:00:00")
    ts = (t0.value // 10**6) + (np.arange(len(acc)) * 1000 / FS).astype(np.int64)
    pd.DataFrame(dict(timestamp_ms=ts, ax=acc[:, 0], ay=acc[:, 1], az=acc[:, 2])).to_csv(path, index=False)
    return path
