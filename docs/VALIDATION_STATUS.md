# Validation status - what is and is not established (v0.3)

> **For:** anyone reading results from this repository, including investors, vets and engineers.
> PRANIIVA is positioned as a **canine health-monitoring and anomaly-alert prototype**, not a device that
> diagnoses or reliably predicts disease. Nothing below is a clinical performance claim.

## Established (in software, on simulated data)
| Item | Evidence | Where |
|---|---|---|
| Ingest rejects or repairs bad input (empty, NaN/inf, duplicates, clock resets, s/µs timestamps, m/s², bad sample rate) | 11 unit tests | `tests/test_ingest.py` |
| Gaps never become motion; one window-validity mask used for every metric | unit test (mid-window gap) | `ingest.window_validity` |
| Short recordings are not extrapolated to a full day (< 75 % coverage -> insufficient) | unit test | `ingest.daily_from_raw` |
| No score without its key measurement / personal baseline; missing sensors flagged | unit tests | `pipeline.score` |
| No alerts after a vet diagnosis; NaN is never an alert | unit tests | `pipeline.debounce_alerts` |
| Model choice, threshold and calibration fixed on development dogs, then evaluated **once** on a separate 300-dog cohort with dog-level 95 % CIs | `reports/final_test.csv` | `pipeline.train_production` |

## NOT established - required before any real-world claim
| Item | Why it matters | How to establish |
|---|---|---|
| **Any real-world detection performance** | The simulator encodes our assumptions; models can learn those assumptions. | Phase 1-4 pilot with real dogs and vet-confirmed outcomes. |
| Behaviour classification on real collars | Orientation, fit, breed and sampling differences distort every feature. | Synchronised video, dual-observer labels, leave-dog-out / leave-breed-out. |
| Collar respiratory rate | **Experimental.** Collar motion, panting, posture can mimic breathing. | Synchronised reference counts (video/ECG belt), vet-reviewed, by coat type. |
| Loose-collar detection on hardware | Needs the firmware's resting-jitter metric; post-hoc CSVs cannot compute it. | Firmware feature + bench test with known fits. |
| Calibrated "risk" as a probability | Calibration was fitted on simulated outcomes. The dashboard therefore shows a **concern level** (score / alert threshold), not a percentage. | Recalibrate on real labelled outcomes; report reliability curves. |
| Alert gate (precision >= 0.30, sensitivity >= 0.30, >= 10 events) | A research filter, **not** a veterinary standard. | Targets set with vets; prospective evaluation. |
| PU/PD result (100 % sensitivity) | Only 18 test events and a very clean simulated drinking signal - not credible as a real number. | Real water-intake validation. |
| Clean-install runtime | Install time, memory and latency not yet measured on a fresh machine. | `pip install -r requirements.txt` in a new venv; time `run_simulation.py` and dashboard start-up. |

## Response to the external code review (Oct 2026)
| Review point | Verdict | Change |
|---|---|---|
| Synthetic data may mislead | Agree | Positioning + this document; dashboard banner |
| Real collar data may be misinterpreted | Agree | Validated ingest, tests |
| Risk scores are not probabilities | Agree | "Concern level" replaces "risk %" in the UI |
| 30 %/30 % gate insufficient for vet alerts | Agree - it is a research filter | Gate now uses the untouched test cohort + minimum events; documented as research-only |
| A. Extrapolating partial days | Agree | Coverage thresholds, observed vs estimated minutes |
| B. Gap mask inconsistently applied / first-sample only | Agree (real bug) | Whole-window mask, applied everywhere; HMM never bridges gaps |
| C. Input validation | Agree | Units, timestamps, duplicates, clock resets, NaN, sample rate, local-date split |
| D. RR needs independent validation | Agree | Marked experimental |
| E. Gate evaluated on data used for thresholding | Agree (real flaw) | Separate untouched test cohort, frozen evaluation, bootstrap CIs |
| F. Silent NaN imputation of missing features | Agree | Status per condition: ok / insufficient_data / reduced_inputs / not_applicable |
| G. Heavy dependency stack | Agree | Core vs research requirements, pinned; core runs without torch/Chronos |
| "Prefer a simple model over a sophisticated ensemble" | Already the case | Stacking and the CNN were rejected in the bake-off; production uses single tree models or rules |
