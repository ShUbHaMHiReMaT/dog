"""
Parameter registry for the PRANIIVA canine simulation.

Every number here is tagged with where it comes from, using the evidence classes of the
PRANIIVA feasibility dossier (09 Oct 2026):

    ESTABLISHED / EMERGING  - published canine evidence (source named)
    ASSUMPTION              - our modelling choice, stated so it can be replaced
    SIM-ENRICHED            - deliberately inflated so the simulator produces enough
                              positive cases to train on. NOT a prevalence claim.

When real hardware data arrives, the ASSUMPTION and SIM-ENRICHED values are the ones to
replace first. The ESTABLISHED values define what "physiologically plausible" means.
"""
from __future__ import annotations

import numpy as np

SIM_START = "2025-01-01"
SIM_DAYS = 400
FS = 50                 # Hz, accelerometer sample rate (dossier: 30-50 Hz validated range)
WIN_S = 2.0             # s, behaviour window (dossier: 1-2 s windows)

# --------------------------------------------------------------------------------------
# Behaviour classes the collar must recognise (Tier 1, raw IMU)
# --------------------------------------------------------------------------------------
BEHAVIOURS = [
    "lying_rest", "standing", "walking", "trotting", "scratching",
    "head_shaking", "eating", "drinking", "panting",
]

# --------------------------------------------------------------------------------------
# Breeds. Shares: Ahmedabad municipal registration (n=19,756) for pedigree breeds,
# plus Indian Pariah / indie as an ASSUMPTION (no published share exists).
# cfr = craniofacial ratio (Packer 2015: <0.1 -> >80% BOAS, >=0.5 -> none).
# --------------------------------------------------------------------------------------
BREEDS = {
    #  name               share  kg(lo,hi)   cfr   coat    dbl   group          risk multipliers
    "Labrador Retriever": dict(share=3939, kg=(25, 36), cfr=0.55, coat="short", double=True,  group="retriever",
                               oa=2.0, atopic=2.0, otitis=1.3, mmvd=0.6, ckd=1.0, tick=1.2),
    "German Shepherd":    dict(share=1527, kg=(25, 40), cfr=0.60, coat="medium", double=True, group="shepherd",
                               oa=1.8, atopic=1.8, otitis=1.2, mmvd=0.6, ckd=1.0, tick=2.0),
    "Shih Tzu":           dict(share=1497, kg=(4, 8),   cfr=0.20, coat="long", double=False,  group="toy_small",
                               oa=0.6, atopic=1.4, otitis=1.5, mmvd=2.5, ckd=1.2, tick=0.6),
    "Golden Retriever":   dict(share=1416, kg=(25, 34), cfr=0.55, coat="long", double=True,   group="retriever",
                               oa=2.2, atopic=2.2, otitis=2.2, mmvd=0.6, ckd=1.0, tick=1.2),
    "Pomeranian":         dict(share=1138, kg=(2, 4),   cfr=0.35, coat="long", double=True,   group="toy_small",
                               oa=0.5, atopic=1.0, otitis=1.0, mmvd=2.5, ckd=1.0, tick=0.6),
    "Beagle":             dict(share=571,  kg=(9, 14),  cfr=0.50, coat="short", double=False, group="hound",
                               oa=1.0, atopic=1.2, otitis=2.5, mmvd=1.2, ckd=1.0, tick=1.0),
    "Siberian Husky":     dict(share=476,  kg=(18, 27), cfr=0.55, coat="medium", double=True, group="spitz",
                               oa=0.8, atopic=1.0, otitis=1.0, mmvd=0.6, ckd=1.0, tick=1.0),
    "Pug":                dict(share=800,  kg=(6, 9),   cfr=0.08, coat="short", double=False, group="brachy",
                               oa=1.0, atopic=1.8, otitis=1.3, mmvd=1.5, ckd=1.0, tick=0.6),
    "Rottweiler":         dict(share=356,  kg=(38, 55), cfr=0.50, coat="short", double=True,  group="giant_large",
                               oa=3.1, atopic=1.0, otitis=1.0, mmvd=0.6, ckd=1.0, tick=1.5),
    "French Bulldog":     dict(share=400,  kg=(9, 13),  cfr=0.10, coat="short", double=False, group="brachy",
                               oa=1.0, atopic=2.0, otitis=1.5, mmvd=1.2, ckd=1.0, tick=0.6),
    "Cocker Spaniel":     dict(share=150,  kg=(12, 15), cfr=0.45, coat="long", double=False,  group="hound",
                               oa=1.0, atopic=1.3, otitis=3.0, mmvd=2.0, ckd=6.39, tick=1.0),
    "Indian Pariah":      dict(share=3000, kg=(15, 30), cfr=0.55, coat="short", double=True,  group="indie",
                               oa=1.0, atopic=1.0, otitis=1.0, mmvd=0.8, ckd=1.0, tick=1.5),
    "Indie / mixed":      dict(share=2500, kg=(8, 28),  cfr=0.50, coat="short", double=True,  group="indie",
                               oa=1.0, atopic=1.0, otitis=1.0, mmvd=1.0, ckd=1.0, tick=1.3),
}

# --------------------------------------------------------------------------------------
# Conditions the platform tries to anticipate. horizon = days ahead a prediction targets.
# `annual_p` is SIM-ENRICHED (real prevalences are far lower; see dossier Part 5).
# --------------------------------------------------------------------------------------
CONDITIONS = {
    "oa_mobility": dict(
        label="Osteoarthritis / mobility decline", horizon=30, chronic=True, annual_p=0.10,
        evidence="ESTABLISHED for monitoring (Brown 2010 RCT; Rowlison 2022). Prediction = HYPOTHESIS.",
        dossier_ladder="Level 1-2", sensors=["accelerometer"]),
    "pruritic_flare": dict(
        label="Allergic skin / ear flare", horizon=14, annual_p=0.22,
        evidence="EMERGING (Griffies 2018 sens 76.9%/spec 99.7%; AJVR 2025 scratching rises before visit).",
        dossier_ladder="Level 1-2", sensors=["accelerometer"]),
    "chf_decomp": dict(
        label="CHF decompensation (diagnosed MMVD only)", horizon=7, annual_p=0.60,
        evidence="EMERGING (Rishniw 2012: sleeping RR <30; one device <1 bpm bias in healthy dogs).",
        dossier_ladder="Level 1-2, diagnosed patients only", sensors=["accelerometer", "microphone"]),
    "acute_gi": dict(
        label="Acute gastrointestinal episode", horizon=3, annual_p=0.12,
        evidence="EMERGING (Chambers 2021 eat/drink detection). UK incidence 8.18%/yr.",
        dossier_ladder="Level 1", sensors=["accelerometer", "microphone"]),
    "tick_fever": dict(
        label="Tick-borne febrile illness (ehrlichiosis-like, India)", horizon=7, annual_p=0.07,
        evidence="Indian signature: lethargy 100%, fever 75% (Patra 2024). Collar temperature UNVALIDATED.",
        dossier_ladder="Research", sensors=["accelerometer", "skin_temperature?"]),
    "ckd_pupd": dict(
        label="PU/PD drift (CKD / endocrine) - exploratory", horizon=30, chronic=True, annual_p=0.08,
        evidence="WEAK: only 3 of 9 CKD presenting signs are wearable-visible (O'Neill 2013).",
        dossier_ladder="Research", sensors=["accelerometer", "microphone"]),
}

# --------------------------------------------------------------------------------------
# Physiology reference values
# --------------------------------------------------------------------------------------
# Active minutes vs age: O'Rourke 2025 (n=28,562): 68.6 @1.5y, 39.5 @5y, 35.4 @10y. ESTABLISHED
ACTIVITY_AGE_CURVE = ([0.4, 1.5, 5.0, 10.0, 15.0], [80.0, 68.6, 39.5, 35.4, 26.0])
ACTIVITY_DOG_SIGMA = 0.40       # between-dog SD ~ the mean itself (O'Rourke) -> lognormal sigma. ASSUMPTION-calibrated
# Night sleeping RR: Chetboul 2025 (n=703) night 14.2 [12.1-17.0]; heart disease 18.0. ESTABLISHED
NIGHT_RR_MEDIAN = 14.2
NIGHT_RR_DOG_SD = 1.8
RR_CHF_THRESHOLD = 30.0         # Rishniw 2012 home-monitoring threshold. ESTABLISHED
# Daytime standing: Rowlison 2022: healthy 6.4% of time vs OA 3.5%.  ESTABLISHED (n=45, industry)
STANDING_MIN_HEALTHY = 0.064 * 960
# Night sleep: 10-32 bouts per night (dossier Part 7.5)
NIGHT_BOUTS_MEAN = 18
CORE_TEMP_MEAN = 38.6           # rectal 37.5-39.2 C
# Collar/skin temperature error: Warhoe 2025 LoA -2.3..+5.2 F vs +-0.5 F threshold.
SKIN_TEMP_BIAS_C = 0.74         # +1.34 F
SKIN_TEMP_NOISE_C = 1.05        # ~ (5.2+2.3)/2 F / 1.96 -> 1.9 F -> 1.05 C

# Belagavi climate (monthly mean Tmax C, RH %). ASSUMPTION from climate normals.
BELAGAVI_TMAX = [29, 31, 34, 35, 33, 27, 24, 25, 27, 28, 28, 28]
BELAGAVI_RH = [45, 40, 42, 50, 62, 85, 90, 88, 82, 72, 60, 52]
DIWALI_DOY = 293                # fireworks -> anxiety confounder (approx. date; varies yearly)

# --------------------------------------------------------------------------------------
# Accelerometer hardware assumptions used by Tier 1 (these ARE the design variables)
# --------------------------------------------------------------------------------------
DEFAULT_SENSOR = dict(
    range_g=8.0,               # +-8 g: head shaking reaches 2-3 g, impacts higher
    bits=16,                   # 16-bit -> 0.24 mg LSB at +-8 g
    noise_ug_rthz=150.0,       # typical low-power MEMS (e.g. BMI270/LIS2DW12 class)
    fs=FS,
)


def breed_names():
    return list(BREEDS.keys())


def breed_probs():
    s = np.array([b["share"] for b in BREEDS.values()], dtype=float)
    return s / s.sum()


def size_class(weight_kg: float) -> str:
    if weight_kg <= 10:
        return "small"
    if weight_kg <= 25:
        return "medium"
    if weight_kg <= 45:
        return "large"
    return "giant"


def positive_window(cond: str, event_day: int, onset_day: int) -> tuple[int, int]:
    """Days [start, end) on which an alert counts as a true early warning.
    Acute conditions: the `horizon` days before the event.
    Chronic conditions (OA, PU/PD): the undiagnosed progressive phase - from 40% of the way between
    onset and clinical recognition, up to recognition (the dossier's 'recognition gap')."""
    c = CONDITIONS[cond]
    if c.get("chronic"):
        return int(onset_day + 0.4 * (event_day - onset_day)), int(event_day)
    return int(event_day - c["horizon"]), int(event_day)
