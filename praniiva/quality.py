"""
Data-quality layer (v0.2). Fur, weather, collar fit and failing sensors are permanent facts of life, so every
day of data is graded BEFORE it can influence a baseline or an alert.

Uses only what a real collar reports (never the simulator's ground truth):
  wear_frac          share of the day the collar was on and streaming
  fit_jitter_mg      resting high-frequency jitter      -> loose collar (cf. collar-fit patent: axis coherence)
  vm_std_mg          median window SD of |a|            -> frozen / flat-lining sensor (non-wear heuristics:
                                                           SD < 13 mg and range < 50 mg)
  rr_valid_segments  60 s lying windows passing the spectral quality test -> fur / wet coat / panting nights
  rr_seg_iqr         within-night spread of segment RR estimates (wet fur scatters them; a real CHF rise
                     moves the whole night up together, so it is NOT masked)
  physiological plausibility of each channel           -> spikes, saturation, firmware glitches

Outputs per dog-day: dq_score in [0, 1], flags, a cleaned copy of the data (bad channels set to NaN so they
neither trigger alerts nor corrupt the dog's baseline), and owner-facing maintenance messages
("collar seems loose") instead of fake health alerts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MOTION = ["active_min", "high_int_min", "standing_min", "rest_min", "scratch_min", "headshake_count",
          "eat_min", "drink_min", "pant_min", "night_bouts", "gait_asym"]
LIMITS = dict(active_min=480, high_int_min=240, standing_min=600, scratch_min=120, headshake_count=600,
              eat_min=120, drink_min=90, pant_min=900, night_bouts=80, sleep_rr=90)
MIN_RR_SEGMENTS = 5


def _robust_dev(s: pd.Series, lag=1, win=28):
    base = s.shift(lag).rolling(win, min_periods=7).median()
    mad = (s.shift(lag) - base).abs().rolling(win, min_periods=7).median() * 1.4826
    return (s - base) / np.maximum(mad, 1e-3), base


def assess(meas: pd.DataFrame) -> pd.DataFrame:
    """Return meas + dq columns. Does not modify sensor values (see clean())."""
    m = meas.sort_values(["dog_id", "day"]).copy()
    g = m.groupby("dog_id", sort=False)
    worn = m.wear_frac.fillna(0) > 0
    m["dq_nonwear"] = m.wear_frac.fillna(0) < 0.5
    m["dq_flatline"] = worn & (m.vm_std_mg < 3)
    jz = g["fit_jitter_mg"].transform(lambda s: _robust_dev(s)[0])
    m["dq_loose"] = worn & ((jz > 4) | (m.fit_jitter_mg > 12))
    impl = np.zeros(len(m), bool)
    for c, lim in LIMITS.items():
        if c in m:
            impl |= (m[c] > lim).fillna(False).values
    hz = g["high_int_min"].transform(lambda s: _robust_dev(s)[0])
    m["dq_spike"] = worn & (impl | ((hz > 8) & (m.high_int_min > 3 * g["high_int_min"].transform("median"))))
    m["dq_rr_low_yield"] = worn & ((m.rr_valid_segments.fillna(0) < MIN_RR_SEGMENTS) |
                                   (m.get("rr_seg_iqr", pd.Series(0, index=m.index)).fillna(0) > 3.0))
    score = np.ones(len(m))
    score *= np.where(m.dq_nonwear, 0.0, np.clip(m.wear_frac.fillna(0), 0, 1) ** 0.5)
    score *= np.where(m.dq_flatline, 0.0, 1.0)
    score *= np.where(m.dq_spike, 0.2, 1.0)
    score *= np.where(m.dq_loose, 0.5, 1.0)
    score *= np.where(m.dq_rr_low_yield, 0.85, 1.0)
    m["dq_score"] = score
    return m


def clean(m: pd.DataFrame) -> pd.DataFrame:
    """Mask channels that cannot be trusted on a given day."""
    m = m.copy()
    bad_all = m.dq_nonwear | m.dq_flatline
    m.loc[bad_all, MOTION + ["sleep_rr", "cough_count"]] = np.nan
    m.loc[m.dq_spike, ["active_min", "high_int_min", "standing_min", "rest_min"]] = np.nan
    m.loc[m.dq_loose, ["scratch_min", "headshake_count", "gait_asym"]] = np.nan
    m.loc[m.dq_rr_low_yield, "sleep_rr"] = np.nan
    return m


def maintenance_messages(m: pd.DataFrame, min_run=2) -> pd.DataFrame:
    """Owner-facing device messages (debounced runs of >= min_run days)."""
    rows = []
    msgs = {"dq_loose": "Collar seems loose - tighten so two fingers fit underneath",
            "dq_flatline": "Sensor not responding - restart or contact support",
            "dq_nonwear": "Collar not worn / not charged",
            "dq_rr_low_yield": "Breathing-rate readings unreliable (wet or matted fur, heat, panting)"}
    for dog, g in m.sort_values("day").groupby("dog_id", sort=False):
        for flag, text in msgs.items():
            f = g[flag].fillna(False).values.astype(bool)
            run = 0
            for day, v in zip(g.day.values, f):
                run = run + 1 if v else 0
                if run == min_run:
                    rows.append(dict(dog_id=dog, day=int(day), issue=flag.replace("dq_", ""), message=text))
    return pd.DataFrame(rows, columns=["dog_id", "day", "issue", "message"])


def detection_report(m: pd.DataFrame) -> pd.DataFrame:
    """Simulation only: how well do the QC flags find the injected faults?"""
    pairs = {"loose_collar": "dq_loose", "sensor_flatline": "dq_flatline", "spike_glitch": "dq_spike",
             "wet_fur": "dq_rr_low_yield"}
    rows = []
    worn = m.wear_frac.fillna(0) > 0
    for fault, flag in pairs.items():
        truth = (m._fault == fault) & worn
        pred = m[flag].fillna(False) & worn
        tp = int((truth & pred).sum())
        rows.append(dict(fault=fault, flag=flag, fault_days=int(truth.sum()), flagged_days=int(pred.sum()),
                         recall=tp / max(truth.sum(), 1), precision=tp / max(pred.sum(), 1)))
    return pd.DataFrame(rows)
