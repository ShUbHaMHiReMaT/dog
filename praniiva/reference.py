"""
v0.2 additions inspired by competitors:

  * "Dogs like yours" population reference (Tractive compares scratching with millions of dogs, so it is
    useful from day one). Percentiles per channel by size class x age band. Used during the 35-day learning
    period, before a personal baseline exists, and as context for vets. Display only - alerts still come from
    the personal-baseline models.
  * Alert explanations ("why"): the channels that moved most against the dog's own normal - vets will not act
    on a black-box score (dossier: explainability is unmet need #13).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .params import size_class

REF_CHANNELS = ["active_min", "high_int_min", "standing_min", "scratch_min", "headshake_count", "eat_min",
                "drink_min", "pant_min", "night_bouts", "sleep_rr"]
AGE_BANDS = [0, 2, 7, 10, 30]
AGE_LABELS = ["<2 y", "2-7 y", "7-10 y", "10+ y"]
LEARNING_DAYS = 35

COND_DRIVERS = {
    "oa_mobility": ["high_int_min", "active_min", "standing_min", "gait_asym", "night_bouts"],
    "pruritic_flare": ["scratch_min", "headshake_count", "night_bouts"],
    "chf_decomp": ["sleep_rr", "cough_count", "active_min", "night_bouts"],
    "acute_gi": ["eat_min", "active_min", "drink_min"],
    "tick_fever": ["active_min", "eat_min", "sleep_rr", "skin_temp"],
    "ckd_pupd": ["drink_min", "night_bouts", "eat_min"],
}
NICE = {"active_min": "activity", "high_int_min": "running/play", "standing_min": "standing time",
        "gait_asym": "gait asymmetry", "night_bouts": "night restlessness", "scratch_min": "scratching",
        "headshake_count": "head shaking", "sleep_rr": "sleeping breathing rate", "cough_count": "coughing",
        "eat_min": "eating", "drink_min": "drinking", "skin_temp": "skin temperature"}


def _bands(df):
    return (df.weight.map(size_class) + " / " +
            pd.cut(df.age, AGE_BANDS, labels=AGE_LABELS, right=False).astype(str))


def build_reference(meas_clean: pd.DataFrame, cohort: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    m = meas_clean.merge(cohort[["dog_id", "weight", "age0"]], on="dog_id")
    m["age"] = m.age0 + m.day / 365
    m["band"] = _bands(m)
    # use each dog's weekly means so heavy-data dogs do not dominate
    m["week"] = m.day // 7
    wk = m.groupby(["dog_id", "week", "band"])[REF_CHANNELS].mean().reset_index()
    q = wk.groupby("band")[REF_CHANNELS].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    q.index.names = ["band", "q"]
    ref = q.reset_index()
    if out is not None:
        ref.to_parquet(out)
    return ref


def percentile_vs_similar(ref: pd.DataFrame, weight: float, age: float, channel: str, value: float) -> float | None:
    band = size_class(weight) + " / " + AGE_LABELS[int(np.searchsorted(AGE_BANDS, age, side="right") - 1)]
    r = ref[ref.band == band]
    if r.empty or not np.isfinite(value):
        return None
    qs, vs = r.q.values, r[channel].values
    if not np.isfinite(vs).all():
        return None
    return float(np.interp(value, vs, qs * 100, left=2.5, right=97.5))


def explain(F_row: pd.Series, cond: str, top=3) -> list[str]:
    """Human-readable drivers behind an alert, from personal-baseline deviations."""
    out = []
    for ch in COND_DRIVERS[cond]:
        best = None
        for suf, label in (("__z7", "past week"), ("__zlong", "past 2 weeks vs 2-4 months ago")):
            v = F_row.get(f"{ch}{suf}", np.nan)
            if pd.notna(v) and (best is None or abs(v) > abs(best[0])):
                best = (float(v), label)
        if best is not None and abs(best[0]) >= 1.5:
            out.append((abs(best[0]), f"{NICE.get(ch, ch)} {'up' if best[0] > 0 else 'down'} "
                                      f"{abs(best[0]):.1f}x its normal spread ({best[1]})"))
    if "sleep_rr__m3" in F_row and cond == "chf_decomp" and pd.notna(F_row["sleep_rr__m3"]):
        out.append((99, f"sleeping breathing rate {F_row['sleep_rr__m3']:.0f} br/min (3-day mean; vets use 30 as a flag)"))
    return [t for _, t in sorted(out, reverse=True)[:top]]
