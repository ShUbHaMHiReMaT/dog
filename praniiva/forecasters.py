"""
Per-dog expected-value forecasters used as abnormality detectors.

  * Chronos-Bolt (amazon/chronos-bolt-tiny, Hugging Face) - pretrained time-series foundation model.
    Weekly forecast origins; each day gets the forecast made from data up to the previous origin.
  * ARIMA (statsmodels SARIMAX(1,0,1)+c) - fitted per dog on its first 60 days, then run forward
    as a filter (parameters frozen, state updated) giving one-step-ahead predictions.

Both return a 'surprise' z = (observed - expected) / expected spread, so they are comparable to the
rolling-median z used in features.py.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_CHRONOS = None


def chronos_pipeline():
    global _CHRONOS
    if _CHRONOS is None:
        import torch
        from chronos import BaseChronosPipeline
        _CHRONOS = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu",
                                                       torch_dtype=torch.float32)
    return _CHRONOS


def chronos_surprise(meas: pd.DataFrame, channels, ctx_len=120, step=7, first=28, batch=512) -> pd.DataFrame:
    import torch
    pipe = chronos_pipeline()
    meas = meas.sort_values(["dog_id", "day"]).reset_index(drop=True)
    dogs = meas.dog_id.unique()
    D = int(meas.day.max()) + 1
    out = {}
    for c in channels:
        M = meas.pivot(index="dog_id", columns="day", values=c).reindex(index=dogs, columns=range(D)).values
        Mf = pd.DataFrame(M).T.ffill().bfill().T.values          # forward-fill gaps for the context
        q10 = np.full_like(M, np.nan); q50 = np.full_like(M, np.nan); q90 = np.full_like(M, np.nan)
        for o in range(first, D, step):
            ctx = Mf[:, max(0, o - ctx_len):o]
            ok = np.isfinite(ctx).all(1)
            if not ok.any():
                continue
            h = min(step, D - o)
            idx = np.where(ok)[0]
            for b in range(0, len(idx), batch):
                ii = idx[b:b + batch]
                qs, _ = pipe.predict_quantiles(torch.tensor(ctx[ii], dtype=torch.float32), prediction_length=h,
                                               quantile_levels=[0.1, 0.5, 0.9])
                qs = qs.numpy()
                q10[ii, o:o + h], q50[ii, o:o + h], q90[ii, o:o + h] = qs[..., 0], qs[..., 1], qs[..., 2]
        spread = np.maximum((q90 - q10) / 2.563, 1e-3 + 0.05 * np.abs(q50))
        z = (M - q50) / spread
        zdf = pd.DataFrame(z, index=dogs).stack(dropna=False).rename(f"{c}__chronos_z").reset_index()
        zdf.columns = ["dog_id", "day", f"{c}__chronos_z"]
        out[c] = zdf
        q = pd.DataFrame(q50, index=dogs).stack(dropna=False).reset_index()
        q.columns = ["dog_id", "day", f"{c}__chronos_q50"]
        out[c] = out[c].merge(q, on=["dog_id", "day"])
    res = meas[["dog_id", "day"]]
    for c, zdf in out.items():
        res = res.merge(zdf, on=["dog_id", "day"], how="left")
        res[f"{c}__chronos_z3"] = res.groupby("dog_id")[f"{c}__chronos_z"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return res


def chronos_forecast_one(series: np.ndarray, horizon=14):
    """Forecast for the dashboard: returns (q10, q50, q90) arrays of length `horizon`."""
    import torch
    s = pd.Series(series).ffill().bfill().values[-120:]
    qs, _ = chronos_pipeline().predict_quantiles(torch.tensor(s[None], dtype=torch.float32), prediction_length=horizon,
                                                 quantile_levels=[0.1, 0.5, 0.9])
    q = qs.numpy()[0]
    return q[:, 0], q[:, 1], q[:, 2]


def arima_surprise(meas: pd.DataFrame, channels, fit_days=60) -> pd.DataFrame:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    warnings.filterwarnings("ignore")
    meas = meas.sort_values(["dog_id", "day"]).reset_index(drop=True)
    res = meas[["dog_id", "day"]].copy()
    for c in channels:
        z = np.full(len(meas), np.nan)
        for dog_id, idx in meas.groupby("dog_id", sort=False).indices.items():
            y = meas[c].values[idx].astype(float)
            if np.isfinite(y[:fit_days]).sum() < 30:
                continue
            try:
                m = SARIMAX(y[:fit_days], order=(1, 0, 1), trend="c").fit(disp=False, maxiter=50)
                full = m.apply(y)                       # frozen params, filter over the whole series
                pred = full.get_prediction().predicted_mean
                sd = np.sqrt(max(full.params[-1], 1e-6))
                z[idx] = (y - pred) / sd
            except Exception:
                continue
        res[f"{c}__arima_z"] = z
        res[f"{c}__arima_z3"] = res.groupby("dog_id")[f"{c}__arima_z"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return res
