# Algorithm bake-off - PRANIIVA canine simulation

> All numbers come from SIMULATED dogs whose disease effects are parameterised from the published
> literature in the feasibility dossier. They rank algorithms against each other; they are NOT clinical
> performance claims. Re-run this bake-off on real collar data before trusting any verdict.

Validation: whole dogs held out (GroupKFold), alert thresholds calibrated on separate validation dogs,
false-alert budget 0.25 per dog-month, event-level precision/recall reported.

## Verdicts

| Algorithm | Verdict | Evidence (simulation) |
|---|---|---|
| XGBoost | TIE - keep simpler incumbent (behaviour) / ADOPT (risk) | behaviour: macro-F1 0.877 vs incumbent 0.876; risk: best (or simplest within 0.01 AUPRC) for: ckd_pupd |
| Bayesian Network | REJECT | within 0.02 AUPRC of best on only 1/6 conditions |
| ARIMA | ADOPT as feature (single channel) / NOT added to risk model | beats rolling-median z by >=0.02 AUROC on oa_mobility, ckd_pupd; inside the risk model mean AUPRC change -0.001, no condition improved by >=0.02 |
| Random Forest | TIE - keep simpler incumbent (behaviour) / ADOPT (risk) | behaviour: macro-F1 0.868 vs incumbent 0.876; risk: best (or simplest within 0.01 AUPRC) for: chf_decomp, pruritic_flare, tick_fever |
| SVM | REJECT | macro-F1 0.864 vs incumbent 0.876 |
| Stacked Ensemble | TIE - keep simpler incumbent (behaviour) / COMPETITIVE - not needed (risk) | behaviour: macro-F1 0.881 vs incumbent 0.876; risk: within 0.02 AUPRC of the best on 3/6 conditions but never the simplest winner |
| Survival Analysis (Cox PH) | REJECT | within 0.02 AUPRC of best on only 1/6 conditions |
| CNN (1D, raw signal) | REJECT | macro-F1 0.831 vs incumbent 0.876 |
| Change-point detection | ADOPT as feature + HMM bout smoothing ADOPTED | CUSUM: beats rolling-median z by >=0.02 AUROC on ckd_pupd, acute_gi, tick_fever. Sticky-HMM (Viterbi) bout segmentation raised day-level PPV of eating 0.78->0.85 and drinking 0.83->1.00 (see behaviour_validation.json) |
| Feature extraction | ADOPT | engineered features + HistGB macro-F1 0.876 vs raw-signal CNN 0.831; features are explainable to a vet |
| Sensor fusion | WEAK | mean AUPRC gain of config C over accelerometer-only: -0.003 |
| Deterministic rule-based system | ADOPT + always-on safety rules | best (or simplest within 0.01 AUPRC) for: acute_gi; RR>30 rule kept as a transparent safety net for diagnosed cardiac dogs |
| Threshold calibration | ADOPT | compare calibrated vs fixed 0.5 threshold in the table below - fixed thresholds either flood owners with alerts or miss events |
| Linear regression | REJECT (logistic risk) / REJECT (trend) | risk: within 0.02 AUPRC of best on only 1/6 conditions; trend detector: mean AUROC change vs baseline -0.050 |
| (extra) Chronos-Bolt, Hugging Face | REJECT (single channel) / NOT added to risk model | mean AUROC change vs baseline -0.033; inside the risk model mean AUPRC change -0.001, no condition improved by >=0.02 |
| Precision & recall evaluation | ADOPT (mandatory) | random row splits inflate AUROC by +0.109 on average vs dogs held out - every number here is event-level P/R on unseen dogs |

## A. Behaviour recognition (raw collar IMU, dogs held out)

| model | accuracy | macro_f1 | scratch_sens | scratch_ppv | seconds | verdict |
|---|---|---|---|---|---|---|
| HistGB (incumbent) | 0.873 | 0.876 | 0.917 | 0.907 | 199.600 | INCUMBENT |
| XGBoost | 0.874 | 0.877 | 0.916 | 0.914 | 115.100 | TIE - keep simpler incumbent |
| Random Forest | 0.865 | 0.868 | 0.906 | 0.887 | 25.400 | TIE - keep simpler incumbent |
| SVM (RBF) | 0.861 | 0.864 | 0.898 | 0.894 | 18.300 | REJECT |
| Linear (logistic) | 0.818 | 0.824 | 0.839 | 0.825 | 2.700 | REJECT |
| Stacked (HGB+RF+SVM -> LR) | 0.878 | 0.881 | 0.926 | 0.912 | 563.500 | TIE - keep simpler incumbent |
| 1D CNN on raw signal | 0.826 | 0.831 | 0.934 | 0.923 | 87.400 | REJECT |

## B. Per-dog abnormality detectors (single channel)

| condition | channel | detector | auroc | sensitivity | alert_precision | false_alerts_per_dog_month | median_lead_days | compute_s |
|---|---|---|---|---|---|---|---|---|
| chf_decomp | sleep_rr | Rolling-median z (baseline) | 0.966 | 0.870 | 0.952 | 0.003 | 2.500 |  |
| chf_decomp | sleep_rr | CUSUM change-point | 0.947 | 0.826 | 0.792 | 0.015 | 5.000 |  |
| chf_decomp | sleep_rr | Linear-regression trend | 0.968 | 0.739 | 0.944 | 0.003 | 2.000 |  |
| chf_decomp | sleep_rr | ARIMA surprise | 0.904 | 0.913 | 0.600 | 0.043 | 3.000 | 751.200 |
| chf_decomp | sleep_rr | Chronos-Bolt surprise (HF) | 0.945 | 0.739 | 0.810 | 0.012 | 4.000 | 106.500 |
| chf_decomp | sleep_rr | Rule: sleeping RR > 30 (Rishniw) | 0.855 | 0.826 | 0.452 | 0.071 | 3.000 |  |
| pruritic_flare | scratch_min | Rolling-median z (baseline) | 0.750 | 0.432 | 0.213 | 0.046 | 2.000 |  |
| pruritic_flare | scratch_min | CUSUM change-point | 0.726 | 0.676 | 0.168 | 0.100 | 3.000 |  |
| pruritic_flare | scratch_min | Linear-regression trend | 0.725 | 0.351 | 0.200 | 0.040 | 2.000 |  |
| pruritic_flare | scratch_min | ARIMA surprise | 0.769 | 0.757 | 0.127 | 0.176 | 3.000 | 751.200 |
| pruritic_flare | scratch_min | Chronos-Bolt surprise (HF) | 0.731 | 0.541 | 0.235 | 0.051 | 2.000 | 106.500 |
| oa_mobility | active_min | Rolling-median z (baseline) | 0.612 | 0.360 | 0.071 | 0.127 | 41.000 |  |
| oa_mobility | active_min | CUSUM change-point | 0.618 | 0.080 | 0.143 | 0.011 | 15.500 |  |
| oa_mobility | active_min | Linear-regression trend | 0.551 | 0.000 | 0.000 | 0.009 |  |  |
| oa_mobility | active_min | ARIMA surprise | 0.650 | 0.600 | 0.078 | 0.216 | 29.000 | 751.200 |
| oa_mobility | active_min | Chronos-Bolt surprise (HF) | 0.595 | 0.520 | 0.060 | 0.206 | 24.000 | 106.500 |
| ckd_pupd | drink_min | Rolling-median z (baseline) | 0.792 | 0.200 | 0.125 | 0.005 | 46.000 |  |
| ckd_pupd | drink_min | CUSUM change-point | 0.965 | 1.000 | 0.421 | 0.009 | 36.000 |  |
| ckd_pupd | drink_min | Linear-regression trend | 0.653 | 0.000 | 0.000 | 0.008 |  |  |
| ckd_pupd | drink_min | ARIMA surprise | 0.923 | 1.000 | 0.500 | 0.008 | 39.000 | 751.200 |
| ckd_pupd | drink_min | Chronos-Bolt surprise (HF) | 0.628 | 0.400 | 0.013 | 0.175 | 40.500 | 106.500 |
| acute_gi | eat_min | Rolling-median z (baseline) | 0.583 | 0.150 | 0.019 | 0.122 | 1.000 |  |
| acute_gi | eat_min | CUSUM change-point | 0.610 | 0.100 | 0.008 | 0.195 | 1.000 |  |
| acute_gi | eat_min | Linear-regression trend | 0.564 | 0.050 | 0.008 | 0.090 | 2.000 |  |
| acute_gi | eat_min | ARIMA surprise | 0.573 | 0.150 | 0.008 | 0.283 | 1.000 | 751.200 |
| acute_gi | eat_min | Chronos-Bolt surprise (HF) | 0.588 | 0.150 | 0.021 | 0.110 | 1.000 | 106.500 |
| tick_fever | active_min | Rolling-median z (baseline) | 0.592 | 0.056 | 0.011 | 0.071 | 1.000 |  |
| tick_fever | active_min | CUSUM change-point | 0.616 | 0.056 | 0.012 | 0.061 | 2.000 |  |
| tick_fever | active_min | Linear-regression trend | 0.535 | 0.056 | 0.005 | 0.160 | 5.000 |  |
| tick_fever | active_min | ARIMA surprise | 0.573 | 0.111 | 0.009 | 0.171 | 1.000 | 751.200 |
| tick_fever | active_min | Chronos-Bolt surprise (HF) | 0.609 | 0.000 | 0.000 | 0.006 |  | 106.500 |

## C. Risk prediction (all features)

| condition | model | auroc | auprc | prevalence | sensitivity | alert_precision | false_alerts_per_dog_month | median_lead_days | seconds |
|---|---|---|---|---|---|---|---|---|---|
| oa_mobility | Deterministic rules | 0.930 | 0.609 | 0.040 | 0.802 | 0.893 | 0.003 | 20.000 | 14.200 |
| oa_mobility | Logistic regression (linear) | 0.969 | 0.793 | 0.040 | 0.901 | 0.905 | 0.003 | 37.500 | 16.300 |
| oa_mobility | HistGB | 0.983 | 0.865 | 0.040 | 0.877 | 0.895 | 0.005 | 42.000 | 39.700 |
| oa_mobility | XGBoost | 0.986 | 0.870 | 0.040 | 0.938 | 0.768 | 0.013 | 46.000 | 27600.900 |
| oa_mobility | Random Forest | 0.977 | 0.861 | 0.040 | 0.926 | 0.820 | 0.009 | 43.500 | 55.200 |
| oa_mobility | Bayesian (naive-Bayes network) | 0.946 | 0.599 | 0.040 | 0.951 | 0.418 | 0.047 | 43.000 | 29.800 |
| oa_mobility | Stacked ensemble (LR+XGB+HGB -> LR) | 0.975 | 0.847 | 0.040 | 0.889 | 0.895 | 0.004 | 40.000 | 134.700 |
| oa_mobility | Survival analysis (Cox PH) | 0.970 | 0.730 | 0.040 | 0.827 | 0.851 | 0.003 | 20.000 | 21.100 |
| pruritic_flare | Deterministic rules | 0.735 | 0.106 | 0.014 | 0.626 | 0.301 | 0.046 | 2.000 | 17.100 |
| pruritic_flare | Logistic regression (linear) | 0.788 | 0.233 | 0.014 | 0.748 | 0.474 | 0.029 | 3.000 | 19.300 |
| pruritic_flare | HistGB | 0.807 | 0.265 | 0.014 | 0.701 | 0.436 | 0.037 | 4.000 | 44.300 |
| pruritic_flare | XGBoost | 0.786 | 0.243 | 0.014 | 0.687 | 0.403 | 0.037 | 4.000 | 24.200 |
| pruritic_flare | Random Forest | 0.832 | 0.291 | 0.014 | 0.769 | 0.511 | 0.028 | 4.000 | 77.600 |
| pruritic_flare | Bayesian (naive-Bayes network) | 0.707 | 0.080 | 0.014 | 0.728 | 0.282 | 0.136 | 4.000 | 28.500 |
| pruritic_flare | Stacked ensemble (LR+XGB+HGB -> LR) | 0.790 | 0.272 | 0.014 | 0.748 | 0.480 | 0.030 | 4.000 | 162.100 |
| pruritic_flare | Survival analysis (Cox PH) | 0.784 | 0.193 | 0.014 | 0.701 | 0.471 | 0.028 | 2.000 | 52.800 |
| chf_decomp | Deterministic rules | 0.855 | 0.529 | 0.014 | 0.625 | 0.882 | 0.009 | 3.000 | 3.200 |
| chf_decomp | Logistic regression (linear) | 0.935 | 0.752 | 0.014 | 0.975 | 0.822 | 0.014 | 4.000 | 3.200 |
| chf_decomp | HistGB | 0.951 | 0.810 | 0.014 | 0.975 | 0.841 | 0.015 | 5.000 | 9.700 |
| chf_decomp | XGBoost | 0.947 | 0.798 | 0.014 | 0.950 | 0.892 | 0.008 | 4.000 | 6.100 |
| chf_decomp | Random Forest | 0.952 | 0.827 | 0.014 | 0.950 | 0.852 | 0.012 | 4.000 | 7.200 |
| chf_decomp | Bayesian (naive-Bayes network) | 0.892 | 0.191 | 0.014 | 0.950 | 0.224 | 0.407 | 4.000 | 4.700 |
| chf_decomp | Stacked ensemble (LR+XGB+HGB -> LR) | 0.927 | 0.810 | 0.014 | 0.975 | 0.888 | 0.009 | 3.500 | 35.700 |
| chf_decomp | Survival analysis (Cox PH) | 0.949 | 0.531 | 0.014 | 0.925 | 0.722 | 0.026 | 3.000 | 7.900 |
| acute_gi | Deterministic rules | 0.591 | 0.003 | 0.001 | 0.129 | 0.016 | 0.099 | 1.500 | 15.900 |
| acute_gi | Logistic regression (linear) | 0.490 | 0.001 | 0.001 | 0.048 | 0.006 | 0.165 | 2.000 | 16.800 |
| acute_gi | HistGB | 0.545 | 0.002 | 0.001 | 0.081 | 0.006 | 0.198 | 2.000 | 24.200 |
| acute_gi | XGBoost | 0.580 | 0.002 | 0.001 | 0.081 | 0.006 | 0.182 | 1.000 | 26.300 |
| acute_gi | Random Forest | 0.574 | 0.004 | 0.001 | 0.081 | 0.006 | 0.171 | 1.000 | 60.100 |
| acute_gi | Bayesian (naive-Bayes network) | 0.512 | 0.002 | 0.001 | 0.081 | 0.009 | 0.129 | 2.000 | 25.100 |
| acute_gi | Stacked ensemble (LR+XGB+HGB -> LR) | 0.493 | 0.001 | 0.001 | 0.016 | 0.001 | 0.233 | 3.000 | 95.700 |
| acute_gi | Survival analysis (Cox PH) | 0.479 | 0.001 | 0.001 | 0.081 | 0.004 | 0.206 | 2.000 | 15.500 |
| tick_fever | Deterministic rules | 0.607 | 0.007 | 0.003 | 0.264 | 0.017 | 0.117 | 1.000 | 15.000 |
| tick_fever | Logistic regression (linear) | 0.666 | 0.011 | 0.003 | 0.377 | 0.023 | 0.155 | 2.000 | 14.700 |
| tick_fever | HistGB | 0.577 | 0.020 | 0.003 | 0.283 | 0.157 | 0.092 | 1.000 | 35.000 |
| tick_fever | XGBoost | 0.608 | 0.019 | 0.003 | 0.264 | 0.083 | 0.085 | 1.000 | 20.700 |
| tick_fever | Random Forest | 0.566 | 0.040 | 0.003 | 0.321 | 0.343 | 0.007 | 1.000 | 48.900 |
| tick_fever | Bayesian (naive-Bayes network) | 0.639 | 0.005 | 0.003 | 0.547 | 0.016 | 0.403 | 2.000 | 21.600 |
| tick_fever | Stacked ensemble (LR+XGB+HGB -> LR) | 0.624 | 0.013 | 0.003 | 0.358 | 0.038 | 0.115 | 1.000 | 109.400 |
| tick_fever | Survival analysis (Cox PH) | 0.662 | 0.006 | 0.003 | 0.358 | 0.058 | 0.115 | 1.000 | 16.100 |
| ckd_pupd | Deterministic rules | 0.937 | 0.621 | 0.010 | 0.800 | 0.614 | 0.004 | 25.000 | 12.200 |
| ckd_pupd | Logistic regression (linear) | 0.993 | 0.858 | 0.010 | 0.920 | 0.894 | 0.001 | 30.000 | 9.100 |
| ckd_pupd | HistGB | 0.997 | 0.866 | 0.010 | 0.720 | 0.788 | 0.003 | 37.500 | 23.200 |
| ckd_pupd | XGBoost | 0.998 | 0.909 | 0.010 | 0.720 | 0.800 | 0.003 | 33.000 | 18.400 |
| ckd_pupd | Random Forest | 0.995 | 0.876 | 0.010 | 0.840 | 0.912 | 0.001 | 29.000 | 37.600 |
| ckd_pupd | Bayesian (naive-Bayes network) | 0.930 | 0.334 | 0.010 | 0.840 | 0.324 | 0.034 | 41.000 | 21.800 |
| ckd_pupd | Stacked ensemble (LR+XGB+HGB -> LR) | 0.980 | 0.876 | 0.010 | 0.640 | 0.892 | 0.001 | 36.500 | 81.200 |
| ckd_pupd | Survival analysis (Cox PH) | 0.992 | 0.747 | 0.010 | 0.760 | 0.717 | 0.002 | 33.500 | 18.700 |

Winners (AUPRC, ties -> simpler model): **acute_gi**: Deterministic rules, **chf_decomp**: Random Forest, **ckd_pupd**: XGBoost, **oa_mobility**: HistGB, **pruritic_flare**: Random Forest, **tick_fever**: Random Forest

## D. Sensor fusion ablation (AUPRC by sensor configuration)

| condition | A_accel | B_accel_mic | C_accel_mic_env | D_plus_skin_temp | E_ideal |
|---|---|---|---|---|---|
| acute_gi | 0.003 | 0.003 | 0.003 | 0.003 | 0.004 |
| chf_decomp | 0.829 | 0.832 | 0.827 | 0.830 | 0.840 |
| ckd_pupd | 0.918 | 0.929 | 0.909 | 0.915 | 0.903 |
| oa_mobility | 0.878 | 0.876 | 0.865 | 0.880 | 0.997 |
| pruritic_flare | 0.285 | 0.288 | 0.291 | 0.288 | 0.319 |
| tick_fever | 0.038 | 0.042 | 0.040 | 0.033 | 0.237 |

## E. Why dogs must be held out, and why thresholds must be calibrated

| condition | model | auroc_dogs_held_out | auroc_random_split | auprc_dogs_held_out | auprc_random_split | calibrated_sens | calibrated_prec | calibrated_fa | fixed05_sens | fixed05_prec | fixed05_fa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oa_mobility | HistGB | 0.983 | 0.997 | 0.865 | 0.963 | 0.877 | 0.895 | 0.005 | 0.975 | 0.482 | 0.043 |
| pruritic_flare | Random Forest | 0.832 | 0.915 | 0.291 | 0.390 | 0.769 | 0.511 | 0.028 | 0.741 | 0.508 | 0.025 |
| chf_decomp | Random Forest | 0.952 | 0.968 | 0.827 | 0.839 | 0.950 | 0.852 | 0.012 | 1.000 | 0.569 | 0.053 |
| acute_gi | HistGB | 0.545 | 0.805 | 0.002 | 0.143 | 0.081 | 0.006 | 0.198 | 0.000 |  | 0.000 |
| tick_fever | Random Forest | 0.566 | 0.844 | 0.040 | 0.083 | 0.321 | 0.343 | 0.007 | 0.000 |  | 0.000 |
| ckd_pupd | XGBoost | 0.998 | 0.999 | 0.909 | 0.968 | 0.720 | 0.800 | 0.003 | 0.960 | 0.684 | 0.005 |

## F. Do ARIMA / Chronos features help the risk model? (AUPRC, dogs held out)

| condition | + ARIMA | + Chronos | baseline features |
|---|---|---|---|
| acute_gi | 0.002 | 0.002 | 0.002 |
| chf_decomp | 0.808 | 0.800 | 0.810 |
| ckd_pupd | 0.807 | 0.798 | 0.798 |
| oa_mobility | 0.834 | 0.835 | 0.838 |
| pruritic_flare | 0.167 | 0.179 | 0.174 |
| tick_fever | 0.002 | 0.003 | 0.002 |
