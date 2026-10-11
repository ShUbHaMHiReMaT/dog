"""
Tier 2 - longitudinal daily simulation of a dog cohort (ground truth), plus the
measurement layer that turns truth into what a given sensor configuration would report.

Truth  : what is physiologically happening (never visible to the models)
Measured: what the collar + phone would actually record, with
          - behaviour-classifier confusion (from the Tier 1 LODO study)
          - RR algorithm error & yield by coat (from the Tier 1 RR study)
          - wear compliance gaps, charging, missing nights
          - channels that simply do not exist without a given sensor (NaN)

Disease trajectories AND non-disease look-alikes (minor injury, boarding, kennel cough,
flea season, Diwali fireworks, heat waves) are both simulated - look-alikes are what create
false alerts in the real world, so a model that never meets them would be dishonestly good.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .params import (ACTIVITY_AGE_CURVE, ACTIVITY_DOG_SIGMA, BEHAVIOURS, BELAGAVI_RH, BELAGAVI_TMAX,
                     BREEDS, CONDITIONS, CORE_TEMP_MEAN, DIWALI_DOY, NIGHT_BOUTS_MEAN, NIGHT_RR_DOG_SD,
                     NIGHT_RR_MEDIAN, SIM_DAYS, SIM_START, SKIN_TEMP_BIAS_C, SKIN_TEMP_NOISE_C,
                     STANDING_MIN_HEALTHY, breed_names, breed_probs)

TRUTH_BUDGET = ["walk_min", "trot_min", "standing_min", "scratch_min", "headshake_min",
                "eat_min", "drink_min", "pant_min"]           # lying = remainder of 1440

SENSOR_CONFIGS = {
    "A_accel":          dict(label="Accelerometer only", mic=False, env=False, skin=False, ideal=False),
    "B_accel_mic":      dict(label="Accelerometer + microphone", mic=True, env=False, skin=False, ideal=False),
    "C_accel_mic_env":  dict(label="Accel + mic + on-collar temp/humidity", mic=True, env=True, skin=False, ideal=False),
    "D_plus_skin_temp": dict(label="C + skin temperature (unvalidated)", mic=True, env=True, skin=True, ideal=False),
    "E_ideal":          dict(label="Perfect sensing (upper bound)", mic=True, env=True, skin=True, ideal=True),
}
DEFAULT_CONFIG = "C_accel_mic_env"


# ======================================================================================
# Weather
# ======================================================================================
def heat_index_c(t, rh):
    """Rothfusz heat index (input C, %), returns C."""
    f = t * 9 / 5 + 32
    hi = (-42.379 + 2.04901523 * f + 10.14333127 * rh - 0.22475541 * f * rh - 6.83783e-3 * f * f
          - 5.481717e-2 * rh * rh + 1.22874e-3 * f * f * rh + 8.5282e-4 * f * rh * rh - 1.99e-6 * f * f * rh * rh)
    hi = np.where(f < 80, f, hi)
    return (hi - 32) * 5 / 9


def make_weather(days=SIM_DAYS, start=SIM_START, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    doy = dates.dayofyear.values
    mid = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    tmax = np.interp(doy, np.r_[mid - 365, mid, mid + 365], np.tile(BELAGAVI_TMAX, 3))
    rh = np.interp(doy, np.r_[mid - 365, mid, mid + 365], np.tile(BELAGAVI_RH, 3))
    # synoptic variability (AR(1)) + a couple of heat waves in Apr-May
    ar = np.zeros(days)
    for i in range(1, days):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 1.0)
    tmax = tmax + 1.2 * ar
    for _ in range(2):
        c = rng.integers(95, 140)
        tmax[max(0, c - 4):c + 5] += np.hanning(min(9, days - max(0, c - 4)))[:len(tmax[max(0, c - 4):c + 5])] * rng.uniform(3, 5)
    rh = np.clip(rh + rng.normal(0, 6, days), 20, 98)
    return pd.DataFrame(dict(date=dates, doy=doy, wx_tmax=tmax, wx_rh=rh, wx_hi=np.clip(heat_index_c(tmax, rh), None, 50)))


# ======================================================================================
# Cohort
# ======================================================================================
def make_cohort(n_dogs=400, seed=0, prefix="D"):
    rng = np.random.default_rng(seed)
    names = breed_names()
    rows = []
    for i in range(n_dogs):
        b = rng.choice(names, p=breed_probs())
        info = BREEDS[b]
        w = float(rng.uniform(*info["kg"]) * rng.lognormal(0, 0.08))
        overweight = rng.random() < 0.45               # ~59% scored above ideal in clinic samples
        if overweight:
            w *= rng.uniform(1.08, 1.25)
        age0 = float(np.clip(rng.gamma(2.2, 2.3), 0.5, 15.5))
        brachy = info["cfr"] < 0.2
        small = w <= 10
        p_mmvd = 0.10 * info["mmvd"]   # SIM-ENRICHED x2.5
        p_mmvd = p_mmvd * (2.5 if small else 1.0) * (3.0 if age0 > 8 else (1.4 if age0 > 5 else 0.3))
        rows.append(dict(
            dog_id=f"{prefix}{i:04d}", breed=b, group=info["group"], weight=round(w, 1), age0=round(age0, 2),
            sex=rng.choice(["M", "F"]), neutered=bool(rng.random() < 0.45),
            coat=info["coat"], double_coat=info["double"], cfr=info["cfr"], brachy=brachy,
            coat_colour=rng.choice(["light", "mixed", "dark"], p=[0.35, 0.35, 0.30]),
            overweight=overweight,
            indoor_ac=bool(rng.random() < 0.3), outdoor=bool(rng.random() < 0.25),
            act_mult=float(rng.lognormal(0, ACTIVITY_DOG_SIGMA)),
            compliance=float(rng.beta(8, 1.5)),
            mmvd=bool(rng.random() < min(p_mmvd, 0.6)),
            atopic=bool(rng.random() < min(0.12 * info["atopic"], 0.6)),
            otitis_prone=bool(rng.random() < min(0.10 * info["otitis"], 0.6)),
            anxiety=float(rng.beta(2, 4)),
        ))
    return pd.DataFrame(rows)


# ======================================================================================
# Per-dog truth simulation
# ======================================================================================
def _ramp(days, start, end, power=1.0):
    r = np.clip((days - start) / max(end - start, 1), 0, 1) ** power
    r[days < start] = 0
    return r


def _bump(days, start, length):
    m = (days >= start) & (days < start + length)
    return m.astype(float)


def simulate_dog(dog: pd.Series, wx: pd.DataFrame, rng: np.random.Generator, inject: dict | None = None):
    """inject: optional {condition: event_day} - forces exactly these conditions (used by the live simulator)."""
    D = len(wx)
    d = np.arange(D, dtype=float)
    doy = wx.doy.values
    info = BREEDS[dog.breed]
    age = dog.age0 + d / 365.0
    micro_t = wx.wx_tmax.values + (-4 if dog.indoor_ac else 0) + (2 if dog.outdoor else 0) + rng.normal(0, 0.8, D)
    micro_hi = np.clip(heat_index_c(micro_t, wx.wx_rh.values), None, 50)

    mult = {k: np.ones(D) for k in TRUTH_BUDGET + ["night_bouts", "cough"]}
    add = {k: np.zeros(D) for k in ["night_bouts", "stiffness", "gait_asym", "sleep_rr", "cough", "core_temp"]}
    events, episodes = [], []

    # ---------------------------- diseases ------------------------------------------
    sim_p = {c: v["annual_p"] * D / 365 for c, v in CONDITIONS.items()}

    # Osteoarthritis / mobility decline
    p_oa = sim_p["oa_mobility"] * info["oa"] * (2.3 if dog.overweight else 1.0) * (0.15 if dog.age0 < 4 else 1.0 if dog.age0 < 8 else 1.8)
    if (inject is None and rng.random() < min(p_oa, 0.7)) or (inject is not None and "oa_mobility" in inject):
        ev = int(rng.integers(45, D - 5)) if inject is None else int(inject["oa_mobility"])
        dur = int(rng.uniform(60, 180))
        s = _ramp(d, ev - dur, ev, 1.6)
        s[d >= ev] = 1.0 - 0.4 * _ramp(d, ev, ev + 21)[d >= ev]           # NSAID response (Brown 2010)
        mult["walk_min"] *= 1 - 0.25 * s
        mult["trot_min"] *= 1 - 0.65 * s
        mult["standing_min"] *= 1 - 0.45 * s                               # Rowlison 2022
        add["night_bouts"] += 6 * s
        add["stiffness"] += 6 * s                                          # HYPOTHESIS channel - kept modest
        add["gait_asym"] += 0.12 * s                                       # EMERGING (head-bob asymmetry)
        events.append(("oa_mobility", ev, ev - dur))

    # Pruritic flares (atopic / otitis), seasonal (monsoon & post-monsoon)
    if inject is not None:
        if "pruritic_flare" in inject:
            for st in [int(inject["pruritic_flare"]) - 10]:
                L = 10; ev = st + L
                r = _ramp(d, st, ev, 1.5)
                decay = np.exp(-(d - ev) / 5); post = d >= ev
                r[post] = decay[post] + (1 - decay[post]) * 0.3 * 0.3
                mult["scratch_min"] *= 1 + 5 * r
                mult["headshake_min"] *= 1 + (4 * r if dog.otitis_prone else 0)
                add["night_bouts"] += 4 * r
                events.append(("pruritic_flare", int(ev), int(st)))
    elif dog.atopic or dog.otitis_prone:
        season_w = np.where((doy >= 160) & (doy <= 320), 2.0, 1.0)
        n_fl = rng.poisson(sim_p["pruritic_flare"] / 0.22 * 1.6)
        starts = rng.choice(np.arange(20, D - 20), size=min(n_fl, 4), replace=False, p=(season_w[20:D - 20] / season_w[20:D - 20].sum()))
        last = -999
        for st in sorted(starts):
            if st - last < 40:
                continue
            L = int(rng.uniform(6, 15))
            ev = st + L
            if ev >= D:
                continue
            r = _ramp(d, st, ev, 1.5)
            resp = rng.choice([0.52, 0.69, 0.77])            # antibody / immunosuppressant / shampoo (AJVR 2025)
            post = (d >= ev)
            decay = np.exp(-(d - ev) / rng.uniform(3, 7))
            r[post] = decay[post] + (1 - decay[post]) * (1 - resp) * 0.3
            peak = rng.uniform(2.5, 7)
            mult["scratch_min"] *= 1 + peak * r
            if dog.otitis_prone:
                mult["headshake_min"] *= 1 + rng.uniform(2.5, 7) * r
            add["night_bouts"] += 4 * r
            events.append(("pruritic_flare", int(ev), int(st)))
            last = ev

    # CHF decompensation - only in dogs with diagnosed MMVD (claim discipline)
    if dog.mmvd or (inject is not None and "chf_decomp" in inject):
        add["sleep_rr"] += 2.0                                              # heart-disease baseline (Chetboul 2025)
        if (inject is None and rng.random() < sim_p["chf_decomp"] * (1.5 if dog.age0 > 10 else 1.0)) or                 (inject is not None and "chf_decomp" in inject):
            ev = int(rng.integers(40, D - 3)) if inject is None else int(inject["chf_decomp"])
            L = int(rng.choice([rng.uniform(2, 6), rng.uniform(7, 21)], p=[0.3, 0.7]))   # some are sudden
            r = _ramp(d, ev - L, ev, 2.0)
            r[d >= ev] = 0.0
            peak = rng.uniform(34, 52) if rng.random() > 0.2 else rng.uniform(24, 30)    # ~20% muted RR rise
            base_rr = NIGHT_RR_MEDIAN + 4
            add["sleep_rr"] += (peak - base_rr) * r + 3.0 * (d >= ev)        # treated afterwards, slightly higher
            mult["walk_min"] *= (1 - 0.35 * r) * np.where(d >= ev, 0.9, 1.0)
            mult["trot_min"] *= 1 - 0.6 * r
            add["night_bouts"] += 8 * r
            add["cough"] += 8 * r
            mult["eat_min"] *= 1 - 0.2 * r
            events.append(("chf_decomp", ev, ev - L))

    # Acute GI episodes (puppies & young more often)
    n_gi = rng.poisson(sim_p["acute_gi"] * (1.6 if dog.age0 < 2 else 1.0)) if inject is None else int("acute_gi" in inject)
    for _ in range(n_gi):
        ev = int(rng.integers(40, D - 2)) if inject is None else int(inject["acute_gi"])
        L = int(rng.integers(1, 4))
        r = _ramp(d, ev - L, ev, 1.0)
        rec = (d > ev) & (d <= ev + 3)
        r[rec] = 1 - (d[rec] - ev) / 4
        r[d > ev + 3] = 0
        mult["eat_min"] *= 1 - 0.7 * r
        mult["walk_min"] *= 1 - 0.3 * r
        mult["trot_min"] *= 1 - 0.5 * r
        mult["drink_min"] *= 1 + 0.3 * r
        add["night_bouts"] += 5 * r
        events.append(("acute_gi", ev, ev - L))

    # Tick-borne febrile illness (clinical cases only; most infections are silent - Abd Rani 2011)
    p_tick = sim_p["tick_fever"] * info["tick"] * (1.8 if dog.outdoor else 1.0) * (1.4 if dog.weight > 20 else 0.8)
    if (inject is None and rng.random() < min(p_tick, 0.5)) or (inject is not None and "tick_fever" in inject):
        ev = int(rng.integers(40, D - 2)) if inject is None else int(inject["tick_fever"])
        L = int(rng.uniform(3, 8))
        r = _ramp(d, ev - L, ev, 1.2)
        rec = d >= ev
        r[rec] = np.exp(-(d[rec] - ev) / 4)
        mult["walk_min"] *= 1 - 0.45 * r
        mult["trot_min"] *= 1 - 0.6 * r
        mult["standing_min"] *= 1 - 0.3 * r
        mult["eat_min"] *= 1 - 0.4 * r
        add["core_temp"] += rng.uniform(1.0, 1.8) * r * (rng.random() < 0.75 or 0.2)   # fever in ~75%
        add["sleep_rr"] += 3 * r
        events.append(("tick_fever", ev, ev - L))

    # PU/PD drift (CKD / endocrine) - older dogs
    p_ckd = sim_p["ckd_pupd"] * info["ckd"] * (3.0 if dog.age0 > 10 else 1.0 if dog.age0 > 6 else 0.1)
    if (inject is None and rng.random() < min(p_ckd, 0.5)) or (inject is not None and "ckd_pupd" in inject):
        ev = int(rng.integers(90, D - 5)) if inject is None else int(inject["ckd_pupd"])
        dur = int(rng.uniform(60, 150))
        s = _ramp(d, ev - dur, ev, 1.2)
        s[d >= ev] = 0.7
        mult["drink_min"] *= 1 + 1.3 * s
        mult["eat_min"] *= 1 - 0.2 * s
        mult["walk_min"] *= 1 - 0.15 * s
        add["night_bouts"] += 4 * s
        add["sleep_rr"] += 1.0 * s
        events.append(("ckd_pupd", ev, ev - dur))

    # ---------------------------- look-alikes / confounders --------------------------
    def episode(kind, rate):
        for _ in range(rng.poisson(rate * D / 365)):
            st = int(rng.integers(0, D))
            yield kind, st

    for kind, st in episode("minor_injury", 1.0):
        L = int(rng.uniform(3, 10)); m = _bump(d, st, L) * np.exp(-(d - st).clip(0) / L)
        add["gait_asym"] += rng.uniform(0.15, 0.3) * m
        mult["trot_min"] *= 1 - 0.6 * m; mult["walk_min"] *= 1 - 0.25 * m
        episodes.append((kind, st, L))
    for kind, st in episode("boarding_or_owner_away", 0.8):
        L = int(rng.uniform(3, 10)); m = _bump(d, st, L); f = rng.choice([0.6, 1.3])
        mult["walk_min"] *= 1 + (f - 1) * m; mult["eat_min"] *= 1 - 0.2 * m
        add["night_bouts"] += 4 * m; mult["pant_min"] *= 1 + 0.3 * m
        episodes.append((kind, st, L))
    for kind, st in episode("kennel_cough", 0.3):
        L = int(rng.uniform(7, 14)); m = _bump(d, st, L)
        add["cough"] += rng.uniform(4, 10) * m; add["sleep_rr"] += 2 * m
        episodes.append((kind, st, L))
    flea_w = 1.6 if dog.outdoor else 0.8
    for kind, st in episode("flea_tick_infestation", flea_w):
        L = int(rng.uniform(5, 15)); m = _bump(d, st, L)
        mult["scratch_min"] *= 1 + rng.uniform(1, 3) * m
        episodes.append((kind, st, L))
    for kind, st in episode("dietary_indiscretion", 1.0):
        m = _bump(d, st, 1); mult["eat_min"] *= 1 - 0.5 * m
        episodes.append((kind, st, 1))
    dw = np.where(np.abs(doy - DIWALI_DOY) <= 2, 1.0, 0.0) * dog.anxiety            # fireworks
    add["night_bouts"] += 10 * dw; mult["pant_min"] *= 1 + 1.5 * dw; mult["eat_min"] *= 1 - 0.3 * dw
    if dw.any():
        episodes.append(("diwali_fireworks", int(np.argmax(dw > 0)), 5))

    # Heat-related illness (acute, same-day), dossier: brachy OR 2.10, >50 kg OR 3.42
    hz = 1 / (1 + np.exp(-(-15.0 + 0.4 * (micro_hi - 30) + 1.4 * dog.brachy + 1.2 * (dog.weight > 45)
                           + 0.5 * dog.double_coat + 0.4 * (dog.age0 > 10))))
    heat_days = np.where(rng.random(D) < hz)[0]
    if inject is not None:
        heat_days = np.array([int(inject["heat_illness"])]) if "heat_illness" in inject else np.array([], int)
    for ev in heat_days[:1]:
        m = _bump(d, ev, 1); mm = _bump(d, ev + 1, 2)
        mult["pant_min"] *= 1 + 4 * m; mult["walk_min"] *= 1 - 0.5 * m - 0.3 * mm
        events.append(("heat_illness", int(ev), int(ev)))

    # ---------------------------- baselines -----------------------------------------
    act = np.interp(age, *ACTIVITY_AGE_CURVE) * dog.act_mult
    act *= np.where(wx.date.dt.dayofweek.values >= 5, 1.1, 1.0)
    act *= 1 - 0.025 * np.clip(micro_hi - 30, 0, 12)                       # hot days -> less activity
    act *= np.where((doy >= 160) & (doy <= 270), 0.9, 1.0)                  # monsoon rain
    act *= (0.8 if dog.brachy else 1.0)
    act *= rng.lognormal(0, 0.2, D)
    trot_frac = np.clip(0.28 - 0.012 * age, 0.06, 0.3)
    walk = act * (1 - trot_frac)
    trot = act * trot_frac
    standing = STANDING_MIN_HEALTHY * np.exp(rng.normal(0, 0.25)) * rng.lognormal(0, 0.15, D)
    scratch = np.exp(rng.normal(np.log(1.0), 0.5)) * (1.4 if dog.atopic else 1.0) * rng.lognormal(0, 0.35, D)
    headshake = np.exp(rng.normal(np.log(0.12), 0.4)) * rng.lognormal(0, 0.35, D)
    eat = 8 * (dog.weight / 20) ** 0.3 * np.exp(rng.normal(0, 0.2)) * rng.lognormal(0, 0.15, D)
    drink = 3 * (dog.weight / 20) ** 0.5 * np.exp(rng.normal(0, 0.25)) * (1 + 0.03 * np.clip(micro_t - 26, 0, None)) * rng.lognormal(0, 0.18, D)
    pant = (12 * (1 + 0.12 * np.clip(micro_hi - 27, 0, None)) * (1.8 if dog.brachy else 1.0)
            * (1.25 if dog.double_coat else 1.0) * (dog.weight / 20) ** 0.2 + 0.25 * trot) * rng.lognormal(0, 0.25, D)

    base = dict(walk_min=walk, trot_min=trot, standing_min=standing, scratch_min=scratch,
                headshake_min=headshake, eat_min=eat, drink_min=drink, pant_min=pant)
    truth = {k: base[k] * mult[k] for k in TRUTH_BUDGET}
    occupied = sum(truth.values())
    over = np.clip(occupied / 900, 1, None)                                   # never exceed the day
    for k in TRUTH_BUDGET:
        truth[k] = truth[k] / over
    truth["lying_min"] = 1440 - sum(truth[k] for k in TRUTH_BUDGET)

    nb_dog = NIGHT_BOUTS_MEAN + rng.normal(0, 3) + 0.4 * np.clip(age - 6, 0, None)
    truth["night_bouts"] = rng.poisson(np.clip(nb_dog + add["night_bouts"], 1, None)).astype(float)
    truth["stiffness"] = np.clip(1.5 + 0.2 * np.clip(age - 8, 0, None) + add["stiffness"] + rng.normal(0, 0.4, D), 0.2, None)
    truth["gait_asym"] = np.clip(rng.normal(0.05, 0.015) + add["gait_asym"] + rng.normal(0, 0.01, D), 0, None)
    rr_dog = (NIGHT_RR_MEDIAN + rng.normal(0, NIGHT_RR_DOG_SD) + (1.2 if dog.weight <= 10 else 0)
              + (3.0 if dog.brachy else 0) + 1.0 * (age > 10))
    truth["sleep_rr"] = rr_dog + add["sleep_rr"] + 0.06 * np.clip(micro_t - 30, 0, None) + rng.normal(0, 1.0, D)
    cough_base = 0.2 + (0.5 if dog.brachy else 0) + (0.4 if dog.weight < 5 else 0)
    truth["cough"] = rng.poisson(np.clip(cough_base + add["cough"], 0, None)).astype(float)
    truth["core_temp"] = CORE_TEMP_MEAN + rng.normal(0, 0.2) + add["core_temp"] + rng.normal(0, 0.12, D)
    truth["micro_temp"] = micro_t
    truth["micro_hi"] = micro_hi
    df = pd.DataFrame(truth)
    df.insert(0, "day", np.arange(D))
    df.insert(0, "dog_id", dog.dog_id)
    return df, events, episodes


def simulate_cohort(n_dogs=400, days=SIM_DAYS, seed=0, verbose=True, prefix="D"):
    wx = make_weather(days, seed=seed)
    cohort = make_cohort(n_dogs, seed=seed, prefix=prefix)
    rng = np.random.default_rng(seed + 1)
    frames, ev_rows, ep_rows = [], [], []
    for _, dog in cohort.iterrows():
        df, evs, eps = simulate_dog(dog, wx, rng)
        frames.append(df)
        ev_rows += [dict(dog_id=dog.dog_id, condition=c, event_day=e, onset_day=o) for c, e, o in evs]
        ep_rows += [dict(dog_id=dog.dog_id, kind=k, start_day=s, length=L) for k, s, L in eps]
    truth = pd.concat(frames, ignore_index=True)
    truth = truth.merge(wx.reset_index().rename(columns={"index": "day"}), on="day", how="left")
    events = pd.DataFrame(ev_rows)
    episodes = pd.DataFrame(ep_rows)
    if verbose:
        print(f"  cohort {n_dogs} dogs x {days} days; events:",
              events.condition.value_counts().to_dict())
    return cohort, truth, events, episodes, wx


# ======================================================================================
# Measurement layer
# ======================================================================================
def _load_error_models(model_dir: Path):
    val = json.loads((model_dir / "behaviour_validation.json").read_text())
    cm = np.array(val.get("confusion_for_measurement", val["confusion_lodo_row_normalised"]))  # HMM-smoothed, unseen dogs
    rr = json.loads((model_dir / "rr_error_profile.json").read_text())
    return cm, rr


def _mic_cm(cm):
    """Microphone fusion: halve confusions into/out of eating/drinking/scratching. ASSUMPTION."""
    cm = cm.copy()
    for b in ("eating", "drinking", "scratching", "panting"):
        i = BEHAVIOURS.index(b)
        off = cm[i].copy(); off[i] = 0
        cm[i] -= off * 0.5; cm[i, i] += off.sum() * 0.5
        col = cm[:, i].copy(); col[i] = 0
        cm[:, i] -= col * 0.5
        for r in range(len(BEHAVIOURS)):
            if r != i:
                cm[r, r] += col[r] * 0.5
    return cm / cm.sum(1, keepdims=True)


def measure(truth: pd.DataFrame, cohort: pd.DataFrame, config: str, model_dir: Path, seed=0,
            faults: bool = True) -> pd.DataFrame:
    cfg = SENSOR_CONFIGS[config]
    rng = np.random.default_rng(seed + 17)
    cm, rr_prof = _load_error_models(model_dir)
    if cfg["mic"]:
        cm = _mic_cm(cm)
    if cfg["ideal"]:
        cm = np.eye(len(BEHAVIOURS))
    dogs = cohort.set_index("dog_id")
    n = len(truth)
    t = truth
    out = pd.DataFrame({"dog_id": t.dog_id.values, "day": t.day.values, "date": t.date.values,
                        "wx_tmax": t.wx_tmax.values, "wx_rh": t.wx_rh.values, "wx_hi": t.wx_hi.values})

    comp = dogs.loc[t.dog_id, "compliance"].values
    worn = rng.random(n) < (0.55 + 0.43 * comp) if not cfg["ideal"] else np.ones(n, bool)
    wear_frac = np.where(worn, rng.uniform(0.7, 1.0, n), 0.0) if not cfg["ideal"] else np.ones(n)
    out["wear_frac"] = wear_frac

    # behaviour budget through the classifier confusion matrix
    cols = {"lying_rest": "lying_min", "standing": "standing_min", "walking": "walk_min", "trotting": "trot_min",
            "scratching": "scratch_min", "head_shaking": "headshake_min", "eating": "eat_min",
            "drinking": "drink_min", "panting": "pant_min"}
    T = np.column_stack([t[cols[b]].values for b in BEHAVIOURS])
    M = T @ cm
    if not cfg["ideal"]:
        M *= rng.lognormal(0, 0.06, M.shape)
    meas = {b: M[:, i] for i, b in enumerate(BEHAVIOURS)}
    out["active_min"] = meas["walking"] + meas["trotting"]
    out["high_int_min"] = meas["trotting"]
    out["standing_min"] = meas["standing"]
    out["rest_min"] = meas["lying_rest"]
    out["scratch_min"] = meas["scratching"]
    out["headshake_count"] = meas["head_shaking"] * 60 / 0.8
    out["eat_min"] = meas["eating"]
    out["drink_min"] = meas["drinking"]
    out["pant_min"] = meas["panting"]
    ideal = cfg["ideal"]
    out["night_bouts"] = t.night_bouts.values + (0 if ideal else rng.normal(0, 2, n).round())
    # morning stiffness has no validated collar algorithm (ingest.py cannot compute it) -> only in E_ideal
    out["stiffness_latency"] = t.stiffness.values if ideal else np.nan
    wm = np.clip(t.walk_min.values, 1, None)
    # collar head-bob asymmetry is EMERGING: per-dog fit bias + day noise that shrinks with walking time
    g_bias = pd.Series(t.dog_id.values).map({d: rng.normal(0, 0.03) for d in dogs.index}).values
    out["gait_asym"] = t.gait_asym.values + (0 if ideal else g_bias + rng.normal(0, 0.08 / np.sqrt(wm / 10), n))

    # sleeping RR - nightly median of valid 60 s lying segments
    coats = dogs.loc[t.dog_id, "coat"].values
    yld = np.array([rr_prof[c]["yield_"] for c in coats])
    bias = np.array([rr_prof[c]["bias"] for c in coats])
    sd = np.array([rr_prof[c]["sd"] for c in coats])
    hot_night = np.clip((t.micro_hi.values - 33) / 8, 0, 0.6)                # panting nights lose segments
    n_valid = rng.binomial(25, np.clip(yld * (1 - hot_night), 0, 1))
    # Bench-simulated segment error (sd/sqrt(n)) is optimistic; in the field add per-dog fit/coat bias,
    # night-to-night noise (position, dreams, room temp) and ~5% corrupted nights (unlabelled panting/movement).
    rr_bias_dog = pd.Series(t.dog_id.values).map({d: rng.normal(0, 1.5) for d in dogs.index}).values
    outlier = rng.random(n) < 0.05
    rr = (t.sleep_rr.values + bias + rr_bias_dog + rng.normal(0, 1, n) * sd / np.sqrt(np.clip(n_valid, 1, None))
          + rng.normal(0, 1.5, n) + outlier * rng.normal(4, 4, n))
    out["sleep_rr"] = np.where(ideal, t.sleep_rr.values, np.where(n_valid >= 3, rr, np.nan))
    out["rr_valid_segments"] = n_valid

    out["cough_count"] = (t.cough.values if ideal else rng.binomial(t.cough.values.astype(int), 0.7) + rng.poisson(0.3, n)) if cfg["mic"] else np.nan
    out["env_temp"] = (t.micro_temp.values + (0 if ideal else rng.normal(0, 0.8, n))) if cfg["env"] else np.nan
    if cfg["skin"]:
        long = (coats == "long") * -0.5
        dog_off = dogs.loc[t.dog_id, "weight"].values * 0 + pd.Series(t.dog_id.values).map(
            {d: rng.normal(0, 0.5) for d in dogs.index}).values
        skin = (t.core_temp.values - 3.2 + 0.25 * (t.micro_temp.values - 28) + long + SKIN_TEMP_BIAS_C + dog_off
                + rng.normal(0, 0.6, n) + rng.normal(0, SKIN_TEMP_NOISE_C, n) / np.sqrt(20))
        out["skin_temp"] = t.core_temp.values if ideal else skin
    else:
        out["skin_temp"] = np.nan

    # ---------------- real-world failure modes (observable only through QC telemetry) ----------------
    # QC telemetry the firmware reports every day (computed from raw IMU on-device):
    #   fit_jitter_mg : high-frequency jitter while resting  (loose collar -> high)
    #   vm_std_mg     : median window SD of |a|              (frozen/flat-lining sensor -> ~0)
    jit_dog = pd.Series(t.dog_id.values).map({d: rng.normal(4, 1) for d in dogs.index}).values
    out["fit_jitter_mg"] = np.clip(jit_dog + rng.normal(0, 0.8, n), 0.5, None)
    out["vm_std_mg"] = np.clip(25 + 0.2 * out["active_min"].fillna(0) + rng.normal(0, 5, n), 3, None)
    #   rr_seg_iqr    : spread of the night's per-segment RR estimates (wet/matted fur, movement -> wide)
    out["rr_seg_iqr"] = np.abs(rng.normal(1.0, 0.35, n))
    out["_fault"] = ""                                       # ground truth, never used by the pipeline
    if faults and not ideal:
        dog_idx = pd.Series(np.arange(n)).groupby(t.dog_id.values).indices
        mon = (t.doy.values >= 160) & (t.doy.values <= 270)
        outdoor = dogs.loc[t.dog_id, "outdoor"].values
        for dog, idx in dog_idx.items():
            Dd = len(idx)
            def spans(rate, lo, hi):
                for _ in range(rng.poisson(rate * Dd / 365)):
                    st = int(rng.integers(0, Dd)); L = int(rng.integers(lo, hi + 1))
                    yield idx[st:st + L]
            for k in spans(2.0, 3, 20):                      # loose collar: jitter -> false scratch/shakes/activity
                out.loc[k, "scratch_min"] *= rng.uniform(2, 4); out.loc[k, "headshake_count"] *= rng.uniform(1.5, 3)
                out.loc[k, "active_min"] *= 1.2; out.loc[k, "fit_jitter_mg"] += rng.uniform(8, 15)
                out.loc[k, "_fault"] = "loose_collar"
            for k in spans(0.4, 1, 5):                       # frozen sensor: reads as a motionless dog
                for c in ["active_min", "high_int_min", "standing_min", "scratch_min", "headshake_count",
                          "eat_min", "drink_min", "pant_min"]:
                    out.loc[k, c] = 0.0
                out.loc[k, "rest_min"] = 1440.0; out.loc[k, "vm_std_mg"] = rng.uniform(0.1, 1.0)
                out.loc[k, "sleep_rr"] = np.nan; out.loc[k, "rr_valid_segments"] = 0
                out.loc[k, "night_bouts"] = 0; out.loc[k, "_fault"] = "sensor_flatline"
            for k in spans(0.6, 1, 2):                       # impact/saturation spikes, firmware glitch
                out.loc[k, "active_min"] *= rng.uniform(3, 6); out.loc[k, "high_int_min"] *= rng.uniform(5, 10)
                out.loc[k, "_fault"] = "spike_glitch"
            wet = idx[(rng.random(Dd) < np.where(mon[idx] & outdoor[idx], 0.35, 0.04))]   # wet fur / bath
            sel = wet[np.isfinite(out.loc[wet, "sleep_rr"].values)]
            out.loc[sel, "rr_valid_segments"] = (out.loc[sel, "rr_valid_segments"] * 0.3).astype(int)
            out.loc[sel, "sleep_rr"] += rng.normal(0, 4, len(sel))       # damp fur couples motion badly
            out.loc[sel, "rr_seg_iqr"] += rng.uniform(2.5, 6, len(sel))
            out.loc[sel[out.loc[sel, "_fault"].values == ""], "_fault"] = "wet_fur"

    # non-worn days: no sensor data at all; partially worn days: minutes scale with wear
    sensor_cols = ["active_min", "high_int_min", "standing_min", "rest_min", "scratch_min", "headshake_count",
                   "eat_min", "drink_min", "pant_min", "night_bouts", "stiffness_latency", "gait_asym",
                   "sleep_rr", "cough_count", "env_temp", "skin_temp", "fit_jitter_mg", "vm_std_mg", "rr_seg_iqr"]
    out.loc[~worn, sensor_cols] = np.nan
    if not ideal:
        # firmware reports per-24h-normalised minutes, but partial wear adds sampling noise
        part = (wear_frac < 1) & worn
        for c in ["active_min", "high_int_min", "standing_min", "scratch_min", "eat_min", "drink_min", "pant_min"]:
            out.loc[part, c] = out.loc[part, c] * rng.lognormal(0, 0.15 * (1 - wear_frac[part]) / 0.3)
    return out
