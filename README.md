# PRANIIVA canine health - simulation platform v0.1

A working, end-to-end **simulation** of the PRANIIVA collar: raw accelerometer signals, behaviour
recognition, resting-respiratory-rate extraction, a 400-dog × 400-day Indian cohort with diseases and
look-alike events, per-dog baselines, risk prediction, calibrated alerts and a clinician dashboard.
Real hardware data plugs into the same pipeline through `praniiva/ingest.py`.

> Everything here is simulated from parameters in the feasibility dossier (09 Oct 2026). It tests the
> **pipeline and the choice of algorithms and sensors**. It is not evidence that the device detects disease.
> The dossier's claim discipline applies: Level 1-2 (monitoring, per-dog anomaly) only.

## Quick start

```powershell
pip install -r requirements.txt
python scripts/run_bakeoff.py          # algorithm bake-off -> reports/ALGORITHM_BAKEOFF.md (30-60 min)
python scripts/run_simulation.py       # trains production models with the winners, writes data/
streamlit run app.py                   # dashboard
```

## How it fits together

```
Tier 1  raw IMU (50 Hz, 3-axis)    praniiva/imu.py        synthetic signals per behaviour, per-dog idiosyncrasy,
                                                          collar rotation, coat damping, sensor noise/quantisation
        behaviour + RR             praniiva/behaviour.py  classifier, dogs-held-out validation, learning curve,
                                                          sticky-HMM bout smoothing, RR algorithm sweep
                     │  measured error models (confusion matrix, RR yield/bias by coat)
                     ▼
Tier 2  daily cohort               praniiva/daily.py      Belagavi weather, 13 breeds incl. Indian Pariah,
                                                          6 conditions + heat illness, 6 look-alike episode types,
                                                          5 sensor configurations (measurement layer)
        features                   praniiva/features.py   per-dog baseline z-scores, trends, slow drift, CUSUM
        detectors / forecasters    praniiva/forecasters.py Chronos-Bolt (Hugging Face), ARIMA
        bake-off                   praniiva/bakeoff*.py   15 algorithms, ADOPT / REJECT
        evaluation                 praniiva/evaluation.py event-level precision/recall, threshold calibration
        production                 praniiva/pipeline.py   winners -> calibrated risk %, debounced alerts
Real    hardware ingest            praniiva/ingest.py     raw CSV -> same daily schema
UI      dashboard                  app.py                 clinic overview, dog timeline, live simulator,
                                                          validation, device design, parameter registry
```

## Conditions modelled (and the honest ceiling)

| Condition | Horizon | What the collar sees | Dossier evidence |
|---|---|---|---|
| OA / mobility decline | undiagnosed progressive phase | activity, high-intensity, standing, gait | ESTABLISHED for monitoring |
| Allergic skin / ear flare | 14 d | scratching, head shakes, night restlessness | EMERGING |
| CHF decompensation (**diagnosed MMVD only**) | 7 d | sleeping RR, cough, activity | EMERGING |
| Acute GI | 3 d | eating, activity | EMERGING - small headroom |
| Tick fever (India) | 7 d | lethargy, inappetence (fever needs temperature) | research |
| PU/PD drift (CKD/endocrine) | progressive phase | drinking, night restlessness | research |

## Replacing simulation with reality

1. Phase 1 prototype -> raw CSVs (`timestamp_ms, ax, ay, az`) + video labels.
2. Retrain the behaviour model on real labelled windows (grouped validation, `behaviour.py`).
3. Replace `ASSUMPTION` / `SIM-ENRICHED` values in `params.py` with field-survey and pilot numbers.
4. `ingest.daily_from_raw()` -> daily rows -> `features.build_features()` -> `pipeline.score()`.
5. Re-run the bake-off on real data before trusting any verdict.

Try the ingest path today with a synthetic raw file:

```python
from pathlib import Path
from praniiva.ingest import write_demo_raw, daily_from_raw
p = write_demo_raw(Path("data/demo_raw/D9999_raw.csv"))
print(daily_from_raw(p, "D9999", Path("models")).T)
```

## v0.2 - real-world robustness (fur, weather, fit, failing sensors)

* `praniiva/daily.py` now injects real failure modes: loose collar, frozen sensor, impact/firmware spikes,
  wet/matted fur, plus realistic per-dog RR error and heterogeneous CHF onsets.
* `praniiva/quality.py` grades every day from observable telemetry only, masks untrustworthy channels and
  turns faults into **owner maintenance messages** instead of health alerts.
* `praniiva/reference.py` - "dogs like yours" percentiles (useful from day 1) and a plain-language **why** for
  every alert.
* Deployment gate (`pipeline.py`): a condition only sends alerts if, on unseen dogs, >= 30 % of alerts were real
  and >= 30 % of events were caught. Acute GI and tick fever currently run in **monitor-only** mode.
* `scripts/run_robustness.py` -> `reports/robustness.csv`: with faults, skin-flare detection fell 63 % -> 32 %
  without QC and recovered to 59 % with QC; fault-caused false alerts 85 -> 24.
* `docs/RESEARCH_NOTES.md` - competitor methods and data-quality literature, with sources.
* `scripts/audit_plots.py` -> `reports/audit/*.png` - visual audit (event-aligned signals, alert causes, sanity
  checks against the literature).
