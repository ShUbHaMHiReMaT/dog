"""
Daily feature engineering: every channel is expressed relative to THIS dog's own baseline
(dossier: population models explain almost none of individual variance, R^2 = 0.045).

For each measured channel x and day t (only data up to and including t is used):
    base   = median of x over days [t-35, t-8]          (the dog's recent normal)
    scale  = 1.4826 * MAD over the same window (floored)
    z3/z7  = (mean of last 3/7 days - base) / scale     (acute / sub-acute change)
    d7     = (mean7 - mean7 one week earlier) / scale   (trend)
    zlong  = (mean14 - median of days [t-120, t-60]) / scale  (slow drift: OA, PU/PD)
    cusum  = one-sided CUSUM of standardised deviations (change-point statistic)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import CONDITIONS, positive_window

CHANNELS = ["active_min", "high_int_min", "standing_min", "rest_min", "scratch_min", "headshake_count",
            "eat_min", "drink_min", "pant_min", "night_bouts", "stiffness_latency", "gait_asym",
            "sleep_rr", "cough_count", "env_temp", "skin_temp"]
ABS_CHANNELS = ["sleep_rr", "gait_asym", "skin_temp", "env_temp", "cough_count"]   # absolute level matters too
SCALE_FLOOR = {"active_min": 5, "high_int_min": 2, "standing_min": 5, "rest_min": 20, "scratch_min": 0.3,
               "headshake_count": 3, "eat_min": 0.8, "drink_min": 0.5, "pant_min": 3, "night_bouts": 2,
               "stiffness_latency": 0.4, "gait_asym": 0.02, "sleep_rr": 0.8, "cough_count": 0.7,
               "env_temp": 0.8, "skin_temp": 0.3}


def _cusum(z: np.ndarray, k=0.5, h_cap=20.0):
    """Two one-sided CUSUMs on a standardised series (NaN-tolerant). Returns (up, down)."""
    up = np.zeros(len(z)); dn = np.zeros(len(z))
    u = d = 0.0
    for i, v in enumerate(z):
        if np.isfinite(v):
            u = min(max(0.0, u + v - k), h_cap)
            d = min(max(0.0, d - v - k), h_cap)
        up[i], dn[i] = u, d
    return up, dn


def build_features(meas: pd.DataFrame, cohort: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    meas = meas.sort_values(["dog_id", "day"]).reset_index(drop=True)
    g = meas.groupby("dog_id", sort=False)
    F = {}
    avail = [c for c in CHANNELS if meas[c].notna().any()]
    for c in avail:
        x = meas[c]
        gx = g[c]
        base = gx.transform(lambda s: s.shift(8).rolling(28, min_periods=10).median())
        mad = gx.transform(lambda s: (s.shift(8) - s.shift(8).rolling(28, min_periods=10).median()).abs()
                           .rolling(28, min_periods=10).median())
        scale = np.maximum(1.4826 * mad, SCALE_FLOOR[c])
        m3 = gx.transform(lambda s: s.rolling(3, min_periods=1).mean())
        m7 = gx.transform(lambda s: s.rolling(7, min_periods=3).mean())
        m14 = gx.transform(lambda s: s.rolling(14, min_periods=5).mean())
        longb = gx.transform(lambda s: s.shift(60).rolling(60, min_periods=20).median())
        F[f"{c}__z3"] = (m3 - base) / scale
        F[f"{c}__z7"] = (m7 - base) / scale
        F[f"{c}__d7"] = (m7 - m7.groupby(meas.dog_id).shift(7)) / scale
        F[f"{c}__zlong"] = (m14 - longb) / scale
        zday = ((x - base) / scale).clip(-6, 6)
        cu = np.zeros(len(x)); cd = np.zeros(len(x))
        for _, idx in meas.groupby("dog_id", sort=False).indices.items():
            u, d = _cusum(zday.values[idx])
            cu[idx], cd[idx] = u, d
        F[f"{c}__cusum_up"] = cu
        F[f"{c}__cusum_dn"] = cd
        if c in ABS_CHANNELS:
            F[f"{c}__m7"] = m7
            F[f"{c}__m3"] = m3
    F = pd.concat([meas[["dog_id", "day", "date"]], pd.DataFrame(F, index=meas.index)], axis=1)
    F["wear_7d"] = g["wear_frac"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    if "dq_score" in meas:                       # v0.2 data-quality context
        F["dq_7d"] = g["dq_score"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        F["dq_today"] = meas["dq_score"].values
    F["wx_hi_3d"] = g["wx_hi"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    F["wx_tmax"] = meas["wx_tmax"]
    doy = pd.to_datetime(meas["date"]).dt.dayofyear
    F["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    F["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    co = cohort.set_index("dog_id")
    F["age"] = co.loc[F.dog_id, "age0"].values + F.day.values / 365
    F["weight"] = co.loc[F.dog_id, "weight"].values
    F["brachy"] = co.loc[F.dog_id, "brachy"].values.astype(int)
    F["cfr"] = co.loc[F.dog_id, "cfr"].values
    F["coat_long"] = (co.loc[F.dog_id, "coat"].values == "long").astype(int)
    F["double_coat"] = co.loc[F.dog_id, "double_coat"].values.astype(int)
    F["mmvd_dx"] = co.loc[F.dog_id, "mmvd"].values.astype(int)          # known, vet-diagnosed
    if events is not None and len(events):
        # prior pruritic history known to the clinic at time t (no look-ahead)
        pr = events[events.condition == "pruritic_flare"]
        hist = np.zeros(len(F))
        for dog_id, grp in pr.groupby("dog_id"):
            idx = np.where(F.dog_id.values == dog_id)[0]
            hist[idx] = np.searchsorted(np.sort(grp.event_day.values), F.day.values[idx], side="left")
        F["prior_flares"] = hist
    else:
        F["prior_flares"] = 0
    return F


def add_labels(F: pd.DataFrame, events: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """y_<cond> = 1 if an event of that condition occurs in (t, t+H]. eval_<cond> masks rows usable
    for that condition (baseline formed, data worn recently, not in post-event treatment window,
    CHF only in diagnosed MMVD dogs)."""
    F = F.copy()
    mmvd = cohort.set_index("dog_id")["mmvd"]
    for c, info in CONDITIONS.items():
        H = info["horizon"]
        y = np.zeros(len(F), dtype=int)
        excl = np.zeros(len(F), dtype=bool)
        ev = events[events.condition == c]
        for dog_id, grp in ev.groupby("dog_id"):
            idx = np.where(F.dog_id.values == dog_id)[0]
            days = F.day.values[idx]
            for e, o in zip(grp.event_day.values, grp.onset_day.values):
                a, b = positive_window(c, e, o)
                y[idx[(days >= a) & (days < b)]] = 1
                excl[idx[(days >= e) & (days <= e + 21)]] = True
                if info.get("chronic"):
                    excl[idx[days >= e]] = True        # once diagnosed, no longer a detection target
        F[f"y_{c}"] = y
        ok = (F.day.values >= 35) & ~excl & (F.wear_7d.values > 0.2)
        if c == "chf_decomp":
            ok &= mmvd.loc[F.dog_id].values
        F[f"eval_{c}"] = ok
    return F


def feature_columns(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if not (c.startswith("y_") or c.startswith("eval_")
                                         or c in ("dog_id", "day", "date"))]
