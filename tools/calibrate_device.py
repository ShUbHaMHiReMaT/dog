"""
Device calibration from ordinary wear data (no special rig needed).

    python tools/calibrate_device.py data/raw/MD_WRIST          # newest session in the folder

1. Accelerometer offset + gain per axis - GGIR-style autocalibration (van Hees et al. 2014, J Appl Physiol):
   find motionless 10 s windows (SD < 13 mg on every axis), then fit offset and gain so that gravity reads exactly
   1 g in every orientation. Only accepted when the motionless windows cover enough orientations (each axis must
   see at least +-0.3 g), otherwise the fit is not trustworthy and nothing is saved.
2. Sensor noise floor (median motion in the quietest windows) - sets the 'sensor stuck' and 'still' margins.
3. Sample rate accuracy of the BMI270 (samples per second of device clock).
4. Microphone background level (5th percentile of the sound level).
Writes models/device_calibration.json (used by the live system) and reports/sessions/<name>/calibration.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]


def still_windows(A, fs=50, win_s=10, sd_max=0.013):
    n = win_s * fs
    out = []
    for i in range(0, len(A) - n, n // 2):
        w = A[i:i + n]
        if (w.std(0) < sd_max).all():
            out.append(w.mean(0))
    return np.array(out)


def fit(means):
    def res(p):
        off, gain = p[:3], p[3:]
        return np.linalg.norm((means - off) * gain, axis=1) - 1.0
    r = least_squares(res, x0=[0, 0, 0, 1, 1, 1])
    return r.x[:3], r.x[3:], res(r.x)


def main(folder):
    folder = Path(folder)
    base = str(sorted(folder.glob("*_raw.csv"))[-1]).replace("_raw.csv", "")
    raw = pd.read_csv(f"{base}_raw.csv")
    A = raw[["ax", "ay", "az"]].to_numpy()
    means = still_windows(A)
    cov = dict(n_still_windows=len(means),
               axis_min=means.min(0).round(3).tolist() if len(means) else None,
               axis_max=means.max(0).round(3).tolist() if len(means) else None)
    enough = len(means) >= 10 and (means.min(0) < -0.3).all() and (means.max(0) > 0.3).all()
    before = np.abs(np.linalg.norm(means, axis=1) - 1) if len(means) else np.array([np.nan])
    result = dict(source_session=Path(base).name, made=datetime.now().isoformat(timespec="seconds"),
                  method="GGIR-style autocalibration (van Hees 2014): still 10 s windows, SD<13 mg per axis",
                  orientation_coverage=cov, enough_orientations=bool(enough),
                  error_before_mg=round(float(np.nanmean(before)) * 1000, 2))
    if len(means) >= 10:
        off, gain, r = fit(means)
        result.update(offset_g=off.round(5).tolist(), gain=gain.round(5).tolist(),
                      error_after_mg=round(float(np.abs(r).mean()) * 1000, 2))
    # noise floor, rate, mic
    vm = np.linalg.norm(A - pd.DataFrame(A).rolling(100, center=True, min_periods=1).mean().to_numpy(), axis=1)
    win_sd = np.array([A[i:i + 100].std(0).sum() for i in range(0, len(A) - 100, 100)])
    result["noise_floor_mg"] = round(float(np.percentile(win_sd, 5)) * 1000, 2)
    span = (raw.timestamp_ms.iloc[-1] - raw.timestamp_ms.iloc[0]) / 1000
    result["sample_rate_hz"] = round((len(raw) - 1) / span, 3)
    snd = pd.read_csv(f"{base}_sound.csv")
    if len(snd):
        result["mic_background_dbfs"] = round(float(snd.rms_dbfs.quantile(0.05)), 1)
        result["mic_loud_dbfs"] = round(float(snd.peak_dbfs.quantile(0.99)), 1)
    out = ROOT / "reports" / "sessions" / folder.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if enough and result["error_after_mg"] < result["error_before_mg"]:
        (ROOT / "models" / "device_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["saved_for_live_use"] = True
    else:
        result["saved_for_live_use"] = False
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/MD_WRIST")
