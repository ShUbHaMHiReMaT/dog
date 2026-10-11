"""
Real-hardware ingest: raw collar CSV  ->  validated stream  ->  daily measured row (same schema as the simulator).

Expected raw file:
    timestamp_ms, ax, ay, az            (epoch milliseconds UTC; acceleration in g; device frame x along the neck)
optional columns:
    env_temp_c, env_rh, cough_event

Guarantees (v0.3):
  * Input is VALIDATED before anything is computed: empty files, NaN/inf, duplicate or non-monotonic timestamps,
    clock resets, wrong timestamp units (s / us instead of ms), wrong acceleration units (m/s^2 instead of g),
    implausible sample rates. Fatal problems raise IngestError; recoverable ones are fixed and reported.
  * Gaps are never turned into motion: a 2 s window is valid only if EVERY sample in it is real (no gap > 1 s
    was interpolated across it), and the same validity mask is applied to behaviour minutes, night restlessness,
    gait and respiratory rate.
  * Days are split on LOCAL date (default Asia/Kolkata), not UTC.
  * Observed minutes and wear coverage are kept. Full-day estimates are only produced when coverage is adequate
    (>= MIN_DAY_COVERAGE of the day, >= MIN_NIGHT_COVERAGE of 00-06 h for night metrics); otherwise the
    day/metric is marked insufficient and left empty - an alert must never come from a 2-hour recording.

Respiratory rate from the collar is EXPERIMENTAL until validated against synchronised reference counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import signal as sps

from .behaviour import viterbi_smooth
from .imu import estimate_rr, features_matrix
from .params import BEHAVIOURS, FS

WIN = int(2 * FS)
MAX_GAP_S = 1.0
MIN_DAY_COVERAGE = 0.75        # share of 24 h with valid data before full-day estimates are produced
MIN_NIGHT_COVERAGE = 0.5       # share of 00-06 h needed for night restlessness / sleeping RR
MIN_FS, MAX_FS = 20.0, 200.0
LOCAL_TZ = "Asia/Kolkata"


class IngestError(ValueError):
    """Raised when a raw file cannot be interpreted safely. Better a clear error than a confident wrong score."""


@dataclass
class IngestReport:
    rows_in: int = 0
    rows_used: int = 0
    fixes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    est_fs: float = float("nan")


# ======================================================================================
# 1. validation
# ======================================================================================
def load_raw(path: Path | str, report: IngestReport | None = None) -> pd.DataFrame:
    rep = report if report is not None else IngestReport()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        raise IngestError(f"{path}: file is empty") from e
    need = {"timestamp_ms", "ax", "ay", "az"}
    if not need <= set(df.columns):
        raise IngestError(f"raw file needs columns {sorted(need)}, got {list(df.columns)}")
    rep.rows_in = len(df)
    if len(df) == 0:
        raise IngestError(f"{path}: no samples")
    for c in ("timestamp_ms", "ax", "ay", "az"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = ~np.isfinite(df[["timestamp_ms", "ax", "ay", "az"]].to_numpy(dtype=float)).all(axis=1)
    if bad.any():
        rep.fixes.append(f"dropped {int(bad.sum())} rows with NaN/inf/non-numeric values")
        df = df[~bad]
    if len(df) < WIN:
        raise IngestError(f"{path}: fewer than {WIN} valid samples after cleaning")

    # timestamp units: epoch ms today is ~1.7e12; seconds ~1.7e9; microseconds ~1.7e15
    ts = df.timestamp_ms.to_numpy(dtype=float)
    med = np.median(ts)
    if 1e9 <= med < 1e11:
        df["timestamp_ms"] = ts * 1000; rep.fixes.append("timestamps looked like seconds -> converted to ms")
    elif 1e14 <= med < 1e17:
        df["timestamp_ms"] = ts / 1000; rep.fixes.append("timestamps looked like microseconds -> converted to ms")
    elif not (1e11 <= med < 1e14):
        raise IngestError(f"cannot interpret timestamp units (median {med:.3g}); expected epoch milliseconds")

    # clock resets: a large backwards jump in file order means two sessions glued together
    d = np.diff(df.timestamp_ms.to_numpy())
    if (d < -60_000).any():
        rep.warnings.append(f"{int((d < -60_000).sum())} device clock reset(s) detected; samples re-ordered by time")
    dup = df.timestamp_ms.duplicated()
    if dup.any():
        rep.fixes.append(f"dropped {int(dup.sum())} duplicate timestamps")
        df = df[~dup]
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    # sample rate
    dt = np.diff(df.timestamp_ms.to_numpy()) / 1000
    dt = dt[(dt > 0) & (dt < MAX_GAP_S)]
    if len(dt) == 0:
        raise IngestError("no consecutive samples - cannot estimate sample rate")
    rep.est_fs = float(1 / np.median(dt))
    if not (MIN_FS <= rep.est_fs <= MAX_FS):
        raise IngestError(f"sample rate {rep.est_fs:.1f} Hz outside supported {MIN_FS}-{MAX_FS} Hz")
    if abs(rep.est_fs - FS) > 2:
        rep.warnings.append(f"sample rate {rep.est_fs:.1f} Hz differs from the {FS} Hz the models were trained at "
                            f"- resampled; validate accuracy at this rate")

    # acceleration units: a resting collar reads |a| ~ 1 g (or ~9.81 if the firmware sends m/s^2)
    vm = np.sqrt(df.ax ** 2 + df.ay ** 2 + df.az ** 2)
    med_vm = float(np.median(vm))
    if 7.5 < med_vm < 12.5:
        df[["ax", "ay", "az"]] = df[["ax", "ay", "az"]] / 9.80665
        rep.fixes.append("acceleration looked like m/s^2 -> converted to g")
    elif not (0.6 < med_vm < 1.6):
        raise IngestError(f"median |a| = {med_vm:.2f}: not g or m/s^2 - check sensor scaling/axis config")
    if (np.abs(df[["ax", "ay", "az"]]) > 15).any().any():
        rep.warnings.append("samples beyond ±15 g found - possible saturation / impact / corrupt packets")

    df["t"] = pd.to_datetime(df.timestamp_ms, unit="ms", utc=True).dt.tz_convert(LOCAL_TZ)
    rep.rows_used = len(df)
    return df


def resample_50hz(df: pd.DataFrame) -> pd.DataFrame:
    """Uniform 50 Hz grid. Samples inside gaps > MAX_GAP_S are interpolated only so arrays line up -
    they are flagged `gap` and every window touching them is invalid."""
    t0 = df.t.iloc[0]
    ts = (df.t - t0).dt.total_seconds().to_numpy()
    grid = np.arange(0, ts[-1], 1 / FS)
    out = pd.DataFrame({"t": t0 + pd.to_timedelta(grid, unit="s")})
    for c in ("ax", "ay", "az"):
        out[c] = np.interp(grid, ts, df[c].to_numpy())
    gap = np.zeros(len(grid), bool)
    for i in np.where(np.diff(ts) > MAX_GAP_S)[0]:
        gap[(grid > ts[i]) & (grid < ts[i + 1])] = True
    out["gap"] = gap
    return out


# ======================================================================================
# 2. per-window processing with ONE validity mask used everywhere
# ======================================================================================
def window_validity(gap: np.ndarray) -> np.ndarray:
    n = len(gap) // WIN
    return ~gap[: n * WIN].reshape(n, WIN).any(axis=1)          # whole window must be real data


def classify_stream(acc: np.ndarray, valid: np.ndarray, model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Classify valid windows; invalid windows get label -1 and are excluded from every metric."""
    bundle = joblib.load(model_path)
    n = len(valid)
    W = acc[: n * WIN].reshape(n, WIN, 3)
    labels = np.full(n, -1)
    # smooth each contiguous valid run separately so the HMM never bridges a gap
    idx = np.where(valid)[0]
    if len(idx):
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for r in runs:
            P = bundle["model"].predict_proba(features_matrix(W[r]))
            labels[r] = viterbi_smooth(P) if len(r) > 1 else P.argmax(1)
    return labels, W


def night_bouts(labels: np.ndarray, hours: np.ndarray) -> int:
    """Rest -> activity transitions 22:00-06:00, counted only between two adjacent VALID windows."""
    night = (hours >= 22) | (hours < 6)
    rest = labels == BEHAVIOURS.index("lying_rest")
    active = (labels >= 0) & ~rest
    return int(np.sum(night[1:] & rest[:-1] & active[1:]))


def gait_asymmetry(W: np.ndarray, labels: np.ndarray) -> float:
    """Head-bob asymmetry during walking (EMERGING evidence)."""
    idx = np.where(labels == BEHAVIOURS.index("walking"))[0]
    if len(idx) < 10:
        return np.nan
    vals = []
    for i in idx:
        vm = np.linalg.norm(W[i], axis=1)
        f, p = sps.periodogram(vm - vm.mean(), fs=FS)
        band = (f > 2) & (f < 6)
        k2 = np.where(band)[0][np.argmax(p[band])]
        k1 = np.argmin(np.abs(f - f[k2] / 2))
        vals.append(p[k1] / (p[k2] + 1e-12))
    return float(np.median(np.sqrt(vals)))


def night_rr(acc: np.ndarray, labels: np.ndarray, hours: np.ndarray, seg_s=60) -> tuple[float, int, float]:
    """EXPERIMENTAL. Median of 60 s segments that are entirely valid, entirely lying, between 00-06 h.
    Returns (rr or nan, n_segments, within-night IQR of segment estimates)."""
    seg_w = seg_s // 2
    lying = BEHAVIOURS.index("lying_rest")
    ests = []
    for s in range(0, len(labels) - seg_w + 1, seg_w):
        if np.all(labels[s:s + seg_w] == lying) and np.all(hours[s:s + seg_w] < 6):
            rr, q = estimate_rr(acc[s * WIN:(s + seg_w) * WIN])
            if np.isfinite(rr):
                ests.append(rr)
    if len(ests) < 3:
        return np.nan, len(ests), np.nan
    q75, q25 = np.percentile(ests, [75, 25])
    return float(np.median(ests)), len(ests), float(q75 - q25)


# ======================================================================================
# 3. daily summary
# ======================================================================================
def daily_from_raw(path: Path | str, dog_id: str, model_dir: Path, report: IngestReport | None = None) -> pd.DataFrame:
    rep = report if report is not None else IngestReport()
    raw = load_raw(path, rep)
    rs = resample_50hz(raw)
    rows = []
    for date, day in rs.groupby(rs.t.dt.date):
        acc = day[["ax", "ay", "az"]].to_numpy()
        valid = window_validity(day.gap.to_numpy())
        labels, W = classify_stream(acc, valid, model_dir / "behaviour_model.joblib")
        hours = day.t.dt.hour.to_numpy()[::WIN][: len(labels)]
        observed_s = valid.sum() * 2
        coverage = observed_s / 86400
        night_cov = (valid & (hours < 6)).sum() * 2 / (6 * 3600)
        obs_min = {b: (labels == i).sum() * 2 / 60 for i, b in enumerate(BEHAVIOURS)}
        ok_day = coverage >= MIN_DAY_COVERAGE
        ok_night = night_cov >= MIN_NIGHT_COVERAGE
        scale = 1 / coverage if ok_day else np.nan            # full-day estimate only with adequate coverage
        est = {b: v * scale for b, v in obs_min.items()}
        rr, n_seg, rr_iqr = night_rr(acc, labels, hours) if ok_night else (np.nan, 0, np.nan)
        hs = BEHAVIOURS.index("head_shaking")
        win_valid = labels >= 0
        vm = np.linalg.norm(W[win_valid], axis=2) if win_valid.any() else np.zeros((1, WIN))
        row = dict(
            dog_id=dog_id, date=pd.Timestamp(date), wear_frac=min(coverage, 1.0),
            observed_hours=observed_s / 3600, night_coverage=night_cov,
            day_status="ok" if ok_day else "insufficient_coverage",
            # full-day estimates (NaN when coverage is inadequate)
            active_min=est["walking"] + est["trotting"], high_int_min=est["trotting"],
            standing_min=est["standing"], rest_min=est["lying_rest"], scratch_min=est["scratching"],
            headshake_count=(labels == hs).sum() * scale, eat_min=est["eating"], drink_min=est["drinking"],
            pant_min=est["panting"],
            # observed (un-extrapolated) minutes are always kept for audit
            **{f"obs_{b}_min": v for b, v in obs_min.items()},
            night_bouts=night_bouts(labels, hours) if ok_night else np.nan, stiffness_latency=np.nan,
            gait_asym=gait_asymmetry(W, labels), sleep_rr=rr, rr_valid_segments=n_seg, rr_seg_iqr=rr_iqr,
            vm_std_mg=float(np.median(vm.std(axis=1)) * 1000),
            fit_jitter_mg=np.nan,          # needs the firmware's resting-jitter metric; not derivable post hoc
        )
        for opt, col, agg in (("env_temp_c", "env_temp", "mean"), ("cough_event", "cough_count", "sum")):
            if opt in raw.columns:
                sub = raw[raw.t.dt.date == date][opt]
                row[col] = getattr(sub, agg)() if ok_day else np.nan
            else:
                row[col] = np.nan
        row["skin_temp"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_demo_raw(path: Path, hours=3.0, seed=0):
    """Synthetic raw CSV in the hardware format (lets you test ingest end-to-end today)."""
    from .imu import dog_motion_profile, synth_segment
    rng = np.random.default_rng(seed)
    prof = dog_motion_profile(28, "short", rng)
    plan = [("lying_rest", 1800), ("walking", 600), ("scratching", 8), ("standing", 120), ("eating", 240),
            ("drinking", 40), ("lying_rest", 3600), ("trotting", 300), ("head_shaking", 2), ("panting", 400),
            ("lying_rest", 3600)]
    angle = rng.uniform(-np.pi, np.pi)
    segs = [synth_segment(b, prof, rng, seconds=s, rr_bpm=15, collar_angle=angle, transitions=False) for b, s in plan]
    acc = np.concatenate(segs)[: int(hours * 3600 * FS)]
    t0 = pd.Timestamp("2026-10-10 00:00:00", tz=LOCAL_TZ).tz_convert("UTC")
    ts = (t0.value // 10**6) + (np.arange(len(acc)) * 1000 / FS).astype(np.int64)
    pd.DataFrame(dict(timestamp_ms=ts, ax=acc[:, 0], ay=acc[:, 1], az=acc[:, 2])).to_csv(path, index=False)
    return path
