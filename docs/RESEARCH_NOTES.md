# Research notes - real-world data quality & competitor methods (v0.2 upgrade)

> **For:** the PRANIIVA product/firmware team. Compiled 11 Oct 2026 from web sources; company claims are marked as
> such - almost none of the competitor methods are independently validated.

## 1. How competitors predict

| Company | What they do (as published) | What PRANIIVA took | Evidence quality |
|---|---|---|---|
| **Invoxia** (Biotracker / AI-collar) | Seismocardiography: heart & respiratory rate from neck micro-vibrations, recorded **only when the dog is lying immobile ≥ 40 s**. Case report: a rising **40-day moving average** of daytime HR preceded a change in RR by ~60 days in an MMVD dog. | Opportunistic "still-dog" measurement windows (already: 60 s lying segments); long-window drift features (`zlong`, CUSUM); per-dog baselines. | 703-dog reference-range study, 2 of 4 authors from Invoxia; no independent accuracy study found. |
| **Tractive** (DOG 6) | Scratch monitoring compared against **data from millions of dogs** - no baseline period needed, data from day one. Vitals compared against the dog's **own baseline**. Explicitly "not a medical device". | **"Dogs like yours" reference** for the learning period (`praniiva/reference.py`); personal baseline for alerts. | Company documentation only. |
| **PetPace** | Multi-sensor (temperature, pulse, respiration, HRV, activity, posture); "pain score" = probability of joint pain from ML; insights unlock after weeks-months of data; **owner-adjustable alert thresholds**. | Explicit learning period; **owner feedback loop** that relaxes a dog's threshold after dismissed alerts. | Company claims; independent Holter comparison found wide HR limits of agreement (dossier). |
| **Maven Pet** | Resting measurements -> personal baseline -> trend display over days/weeks/months. | Same trend philosophy. | Dossier cites a peer-reviewed RR validation; **this search could not locate it** - verify before quoting. |

## 2. Data quality - what the literature/patents say

* **Motion artifacts are the most prevalent corruption** in wearable physiological data (arXiv 2401.13518).
* **Non-wear is the main cause of missing data.** Heuristic: a window is non-wear when, for ≥ 2 axes, SD < 13 mg
  and range < 50 mg; heuristics work best for episodes > 60 min, decision trees for shorter ones
  (Sci Rep 2023 via PMC9925815).
* **Spectral signal-quality index:** quality ∝ energy at the target frequency / energy of motion components
  (wearable patents) - this is what `imu.estimate_rr` already uses to reject RR segments.
* **Collar-fit metric:** animal-worn device patent checks coherence of each axis with the total acceleration
  vector and alerts on poor fit (US 9872481) -> our `fit_jitter_mg` telemetry.
* **Imputation:** zero-inflated models exist (R `accelmissing`), but for alerting we **mask, don't impute** -
  an imputed value must never trigger or suppress a health alert.

## 3. What v0.2 implements

| Real-world problem | Detection (observable telemetry) | Action |
|---|---|---|
| Collar not worn / not charged | `wear_frac` < 0.5 | data masked; owner message |
| Loose collar (jitter -> fake scratching/shakes) | `fit_jitter_mg` jump vs dog's own history | scratch/shake/gait masked; "tighten collar" message |
| Frozen / flat-lining sensor (looks like a lethargic dog!) | `vm_std_mg` < 3 mg | all motion masked; "sensor not responding" |
| Impact spikes / firmware glitches | physiological plausibility limits + robust z | activity masked |
| Wet / matted fur, panting nights (RR unreliable) | `rr_valid_segments` < 5 or within-night `rr_seg_iqr` > 3 | sleeping RR masked; owner message |

A real CHF rise moves the **whole night** up together (low within-night spread), so it is deliberately **not**
masked by the RR-consistency check.

## Sources
- [Invoxia - AI collar science page](https://www.invoxia.com/ldg/science/ai-collar/)
- [Chetboul et al. 2025, Frontiers in Veterinary Science (AI-COLLAR, 703 dogs)](https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2025.1667355/abstract) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12477687)
- [Invoxia - monitoring heart health at home](https://www.invoxia.com/blog/petcare/how-to-monitor-your-dogs-heart-health-at-home/)
- [Tractive - health monitoring](https://tractive.com/en/fp/health-monitoring-for-dogs-and-cats) · [Tractive help centre](https://help.tractive.com/hc/gen/articles/34400151311506)
- [PetPace - how it works](https://petpace.com/how-it-works/) · [PetPace pain score](https://petpace.com/how-petpace-detects-pain-in-dogs-and-how-to-use-the-pain-score/) · [PetPace 2.0 press release](https://www.businesswire.com/news/home/20240304225777/en/PetPace-Launches-PetPace-2.0-the-Only-Vet-Grade-AI-Powered-Canine-Pet-Collar-Providing-Continuous-Life-Saving-Medical-Insights-for-Vets-Researchers-and-Pet-Owners)
- [Webvet - Maven respiratory-rate trends](https://www.webvet.com/dog-respiratory-rate-trends/) · [IEEE Access 2024 - wearable RR in dogs](https://doaj.org/article/2b63750a16e74706b2d985bfb04b2b6d)
- [Data quality in wrist-worn wearables (arXiv 2401.13518)](https://arxiv.org/pdf/2401.13518) · [Generalized signal-quality estimation for IoT sensors (arXiv 2002.01279)](https://arxiv.org/pdf/2002.01279)
- [Non-wear detection comparison, Sci Rep 2023 (PMC9925815)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9925815) · [R package accelmissing](https://cran.mi2.ai/web/packages/accelmissing/index.html)
- [Patent US 9872481 - collar fit quality metrics](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9872481) · [Patent US 10382839 - signal quality in wearables](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10382839)
