"""Ingest must refuse or repair bad input - never silently produce a confident wrong number."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from praniiva.ingest import (FS, LOCAL_TZ, WIN, IngestError, IngestReport, daily_from_raw, load_raw,
                             resample_50hz, window_validity)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "behaviour_model.joblib"
T0 = pd.Timestamp("2026-10-10 10:00:00", tz=LOCAL_TZ).tz_convert("UTC").value // 10**6


def make(tmp_path, seconds=60, fs=FS, unit_scale=1.0, ts_scale=1.0, name="r.csv", start=T0):
    n = int(seconds * fs)
    rng = np.random.default_rng(0)
    acc = np.column_stack([rng.normal(0, .01, n), rng.normal(0, .01, n), 1 + rng.normal(0, .01, n)]) * unit_scale
    ts = (start + np.arange(n) * 1000 / fs) * ts_scale
    df = pd.DataFrame(dict(timestamp_ms=ts, ax=acc[:, 0], ay=acc[:, 1], az=acc[:, 2]))
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p, df


def test_empty_file_raises(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("")
    with pytest.raises(IngestError):
        load_raw(p)


def test_missing_columns_raise(tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame(dict(t=[1], x=[1])).to_csv(p, index=False)
    with pytest.raises(IngestError):
        load_raw(p)


def test_nan_and_inf_rows_dropped(tmp_path):
    p, df = make(tmp_path)
    df.loc[5, "ax"] = np.nan
    df.loc[6, "ay"] = np.inf
    df.loc[7, "az"] = "garbage"
    df.to_csv(p, index=False)
    rep = IngestReport()
    out = load_raw(p, rep)
    assert len(out) == len(df) - 3
    assert any("NaN/inf" in f for f in rep.fixes)


def test_mps2_converted_to_g(tmp_path):
    p, _ = make(tmp_path, unit_scale=9.80665)
    rep = IngestReport()
    out = load_raw(p, rep)
    assert abs(np.median(np.sqrt(out.ax ** 2 + out.ay ** 2 + out.az ** 2)) - 1) < 0.05
    assert any("m/s^2" in f for f in rep.fixes)


def test_absurd_units_rejected(tmp_path):
    p, _ = make(tmp_path, unit_scale=1000)
    with pytest.raises(IngestError):
        load_raw(p)


def test_seconds_timestamps_converted(tmp_path):
    p, _ = make(tmp_path, ts_scale=1 / 1000)
    rep = IngestReport()
    load_raw(p, rep)
    assert any("seconds" in f for f in rep.fixes)
    assert abs(rep.est_fs - FS) < 1


def test_duplicates_and_clock_reset(tmp_path):
    p, df = make(tmp_path)
    dup = df.iloc[:100]
    back = df.iloc[:50].assign(timestamp_ms=df.timestamp_ms.iloc[:50] - 3_600_000)   # earlier session appended
    pd.concat([df, dup, back]).to_csv(p, index=False)
    rep = IngestReport()
    out = load_raw(p, rep)
    assert out.timestamp_ms.is_monotonic_increasing and out.timestamp_ms.is_unique
    assert any("duplicate" in f for f in rep.fixes)
    assert any("clock reset" in w for w in rep.warnings)


def test_bad_sample_rate_rejected(tmp_path):
    p, _ = make(tmp_path, fs=5, seconds=200)
    with pytest.raises(IngestError):
        load_raw(p)


def test_gap_invalidates_whole_window(tmp_path):
    p, df = make(tmp_path, seconds=30)
    # remove 1.5 s starting in the MIDDLE of a 2 s window (the old code only looked at the first sample)
    t = df.timestamp_ms.values
    cut = (t >= t[0] + 10_700) & (t < t[0] + 12_200)
    df[~cut].to_csv(p, index=False)
    rs = resample_50hz(load_raw(p))
    v = window_validity(rs.gap.values)
    assert (~v).sum() >= 1
    assert not v[5] and not v[6]          # windows covering 10-12 s and 12-14 s touch the gap
    assert v[0] and v[-1]


def test_days_split_on_local_date(tmp_path):
    # 23:59:30 -> 00:00:30 India time must give two local dates
    start = pd.Timestamp("2026-10-10 23:59:30", tz=LOCAL_TZ).tz_convert("UTC").value // 10**6
    p, _ = make(tmp_path, seconds=60, start=start)
    out = load_raw(p)
    assert set(out.t.dt.date.astype(str)) == {"2026-10-10", "2026-10-11"}


@pytest.mark.skipif(not MODEL.exists(), reason="behaviour model not trained (run scripts/run_simulation.py)")
def test_short_recording_is_not_extrapolated(tmp_path):
    p, _ = make(tmp_path, seconds=600)           # 10 minutes is NOT a day
    d = daily_from_raw(p, "X", MODEL.parent)
    r = d.iloc[0]
    assert r.day_status == "insufficient_coverage"
    assert np.isnan(r.active_min) and np.isnan(r.sleep_rr) and np.isnan(r.night_bouts)
    assert r.observed_hours == pytest.approx(10 / 60, rel=0.05)
    obs = sum(r[c] for c in d.columns if c.startswith("obs_"))
    assert obs == pytest.approx(10, rel=0.05)     # the observed minutes are still reported
