# PRANIIVA collar v0 - device specification derived from the simulation

> **Who this is for:** the hardware/firmware team choosing parts for the first prototype.
> **What it is based on:** the feasibility dossier's validated-claims list, plus PRANIIVA's own simulation
> (`reports/ALGORITHM_BAKEOFF.md`, `models/rr_validation_summary.csv`). Every simulated number should be
> re-measured on the Phase 1 prototype (20-30 dogs, Labrador + Indian Pariah).

## 1. Parameters the device must observe, and why

| # | Parameter (daily) | Derived from | Condition(s) it serves | Evidence |
|---|---|---|---|---|
| 1 | Active minutes, high-intensity minutes | accelerometer behaviour classifier | OA / mobility, tick fever, GI, CHF | ESTABLISHED (Brown 2010 RCT) |
| 2 | Daytime standing minutes | accelerometer (static posture) | OA / mobility | EMERGING (Rowlison 2022: 6.4% -> 3.5%) |
| 3 | Scratching minutes | accelerometer, 4-7 Hz bursts | allergic skin flare | EMERGING (Griffies 2018) |
| 4 | Head-shake count | accelerometer, 5-9 Hz bursts | otitis | EMERGING, weak (PPV ~0.4 in our sim) |
| 5 | Eating / drinking minutes | accelerometer (+ microphone) | GI, PU/PD, appetite | EMERGING (Chambers 2021) |
| 6 | Sleeping respiratory rate | accelerometer micro-motion, 60 s lying windows, 00-06 h | CHF in **diagnosed** MMVD; fever | EMERGING (Rishniw 2012 threshold <30) |
| 7 | Night restlessness bouts | accelerometer | CHF, pruritus, OA pain, PU/PD | HYPOTHESIS |
| 8 | Panting minutes (respiratory mode) | accelerometer 3-6 Hz | heat stress; must be classified *before* RR | dossier Part 7 |
| 9 | Gait asymmetry (head bob) | accelerometer during walking | OA / lameness | EMERGING, collar-limited |
| 10 | Cough events | microphone, on-device detector | CHF, kennel cough | NO wearable evidence - research |
| 11 | Collar micro-climate temp / RH | env sensor | heat risk, explains hot-day activity drops | context |
| 12 | Wear time, valid-RR-segment count | firmware | data-quality gating | mandatory |

**Not on the device (dossier Table ES-1):** optical heart rate (PPG through fur), HRV, sleep staging,
pain score, GPS / cellular, electrodermal sensing. Skin temperature is **research-only** - see section 4.

## 2. Accelerometer requirements

| Spec | Value | Reason (simulation) |
|---|---|---|
| Axes / range | 3-axis, ±8 g | head shakes reach 2-3 g; impacts clip below ±4 g |
| Sample rate | **50 Hz** (configurable 25-100) | head-shake & scratch bands up to 9 Hz plus harmonics; RR alone works at 25 Hz but long-coat dogs degraded (LoA ±9 br/min at 25 Hz vs ±1.5 at 50 Hz) |
| Resolution | 16-bit | breathing micro-motion is ~1-3 mg; at ±8 g a 12-bit LSB is 3.9 mg |
| Noise density | ≤ 150-400 µg/√Hz | RR error was insensitive across this range in simulation - **verify on bench** |
| Mounting | rigid to enclosure, x-axis along the neck | features are invariant to collar *rotation* about the neck, not to arbitrary mounting |
| Candidate parts | Bosch BMI270 / BMA400, ST LIS2DW12 / LSM6DSO | low-power, FIFO, 16-bit |

## 3. Other sensors - decided by the simulation's sensor ablation

AUPRC per condition, dogs held out (from `reports/sensor_ablation.csv`):

| Condition | A: accel only | B: + mic | C: + mic + env | D: + realistic skin temp | E: perfect sensing |
|---|---|---|---|---|---|
| OA / mobility | 0.878 | 0.876 | 0.865 | 0.880 | 0.997 |
| Skin / ear flare | 0.285 | 0.288 | 0.291 | 0.288 | 0.319 |
| CHF (diagnosed MMVD) | 0.829 | 0.832 | 0.827 | 0.830 | 0.840 |
| PU/PD drift | 0.918 | 0.929 | 0.909 | 0.915 | 0.903 |
| Tick fever | 0.038 | 0.042 | 0.040 | 0.033 | 0.237 |
| Acute GI | 0.003 | 0.003 | 0.003 | 0.003 | 0.004 |

**Reading it:** in the simulation the accelerometer alone carries almost all the value. Microphone and
on-collar temp/humidity add nothing measurable; a skin thermometer with the *published* error does not rescue
tick fever. Only near-perfect temperature (E) would - and no collar achieves that today.

| Sensor | Decision for v0 production | Notes |
|---|---|---|
| Accelerometer (16-bit, 50 Hz) | **Include - the core sensor** | carries the mobility, scratching and RR claims |
| MEMS microphone | **Research units only** | no simulated gain; its assumed benefit (eat/drink fusion, cough) is unproven - measure on real dogs before paying for it |
| Ambient temp + RH | **Leave out; use phone weather API** | no simulated gain over city weather |
| Skin / contact temperature | **Research units only** | published accuracy too poor; the one thing that would unlock tick fever - worth a dedicated R&D track |
| Optical PPG | **Exclude** | no canine validation through fur; data loss by pigmentation |
| GPS / LTE-M | **Exclude** | location is commoditised; cellular inherits the carrier-sunset clock that killed Whistle |

Caveat: the microphone was modelled conservatively. If real chewing/cough audio proves strong, re-run the
ablation with real data before ruling it out.

## 4. Biggest lever is not a sensor

Perfect sensing (E) lifts OA from 0.88 to 0.997 - i.e. **measurement noise and wear compliance**, not missing
sensor types, are what limit the accelerometer channels. Spend effort on fit, battery life (fewer charging gaps)
and the behaviour model trained on real labelled Indian dogs.

## 5. System architecture

```
 collar MCU (nRF52840 class, BLE 5)          phone app                     PRANIIVA cloud / clinic
 ─────────────────────────────────          ─────────────                  ─────────────────────────
 50 Hz IMU ─► 2 s window features ─┐                                         per-dog baselines
 mic ─► event detector (counts) ───┼─► behaviour classifier ─► daily  ─BLE─► features + risk models
 env sensor ───────────────────────┘    + HMM bout smoothing     summary     calibrated alerts
                                         60 s night RR estimator  (~2 KB/day) clinician view first
 on-device flash: >= 14 days of daily summaries + 48 h raw ring buffer (for re-labelling / validation)
```

* **Offline-first:** the collar keeps working with no phone in range; it syncs when it can.
* **No mandatory carrier dependency:** Bluetooth only, ₹0 connectivity cost (dossier Part 14).
* **Raw-data mode:** research units log raw 50 Hz IMU to flash so models can be retrained on real dogs.

## 6. Mechanical / power targets

| Item | Target |
|---|---|
| Mass | ≤ 30 g for dogs ≥ 10 kg; a small-dog variant ≤ 15 g; report mass + dimensions (not just "3%") |
| Ingress | IP67 minimum (monsoon, swimming) |
| Battery | ≥ 14 days measured (not claimed) with 50 Hz IMU + 2 s on-device features; publish the measured figure |
| Mount | collar first, harness adapter tested in Phase 1 (published 16-point accuracy gap) |
| Orientation | design for ventral rotation; features are rotation-invariant about the neck axis |

## 7. What the prototype must measure (feeds `praniiva` re-validation)

1. Raw IMU + video for ≥ 20 min labelled per behaviour per dog (Phase 1 protocol).
2. RR: 60 s lying segments vs video breath counts, by coat length and colour (yield **and** error).
3. Collar rotation over 24 h.
4. Battery life under real use; wear hours per day.
5. Skin temperature vs rectal (research units only).

Drop the CSVs into `ingest.py`'s format and re-run `scripts/run_bakeoff.py` with real data.
