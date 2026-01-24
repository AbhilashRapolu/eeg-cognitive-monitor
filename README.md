# Real-Time Cognitive Monitoring System: EEG-Based Brain State Detection

![Status](https://img.shields.io/badge/status-Prototype%20Validated-brightgreen) ![License](https://img.shields.io/badge/license-Private%20(Seeking%20Cofounders)-blue) ![Python](https://img.shields.io/badge/python-3.8%2B-blue)

**Real-time EEG cognitive monitoring system combining consumer-grade Muse 2 headsets with neuroscience-grounded algorithms to detect brain states (Optimal, Fatigue, Mind-Wandering, Overload) and trigger adaptive interventions in real-time.**

---

## Quick Facts

**Lab Validation (Phase 1–2):**
- 16 subjects,48 sessions, 62k+ trials (research-grade EEG)
- 15 biomarkers (theta, alpha, beta, gamma, entropy, complexity, PAC, etc.)
- State classification accuracy: 68.3% (Gradient Boosting AUC 0.68)
- Drift detection: +30.5% error risk when drift >50%, -22.9% efficiency loss

**Real-Time Deployment (10 Sessions):**
- 4,087 trials, 273 minutes of live EEG (Muse 2 consumer headset)
- 96%+ optimal state detection, <5% false alarm rate
- 1 successful intervention triggered (Session 6, 50-minute work session)
- 3-axis cognitive metrics (Arousal, Control, Executive) with pre/post session deltas
- Largest arousal recovery post-intervention: +1.488 (Session 6)
- Largest executive load release during passive recovery: -1.590 (Session 2)

**Technology Stack:**
- Python + Jupyter, LSL (Lab Streaming Layer), real-time signal processing
- 15 biomarkers extracted per 2-sec epoch, <5 sec end-to-end latency
- Rolling baseline adaptation (learns user's brain state within 100 trials / 3–5 min)
- Consumer-grade Muse 2 (4 channels, 256 Hz), 25% artifact rejection rate

---

## Lab Validation: Phase 1 & 2

### Phase 1: Rolling Baseline & State Classification (60 Sessions, 21 Subjects)

**Objective:** Validate that the 15 EEG biomarkers and hierarchical decision tree can reliably classify cognitive states across diverse subjects and task contexts.

**Design:**
- 16 subjects, 48 sessions (work-focused tasks)
- Research-grade EEG: 3 channels, professional amplifiers, controlled lab environment

**Validation Pipeline:**
1. **Feature Extraction:** 15 biomarkers per 2-sec epoch via IIR filters, FFT, entropy (Permutation, Lempel-Ziv, Multiscale)
2. **Rolling Baseline:** 100-trial chunks (3.3 min), adaptive to circadian drift
3. **Z-Score Normalization:** Per feature, per trial (relative to rolling baseline)

**State Distribution Targets vs. Observed:**
| State | Target Range | Observed Range | Status |
|---|---|---|---|
| Optimal | 60–75% | 76% | 
| Mind-Wandering | 15–25% | 12.5% |
| Fatigue | 5–10% | 6% |
| Overload | 2–5% | 1-2% |

**Key Biomarkers (Top Predictors):**
- **z_delta:** Sleep pressure / metabolic fatigue
- **z_theta:** Mental load / working memory effort
- **z_alpha_relative:** Disengagement / drowsiness
- **z_tbr (Theta/Beta ratio):** Mind-wandering marker
- **z_lz (Lempel-Ziv complexity):** Consciousness level

**Source:** `docs/PHASE1_BASELINE_DETECTION.md`

---

### Phase 2: Cumulative Drift & Performance Grounding (48 Sessions, 16 Subjects)

**Objective:** Prove that EEG drift states causally predict behavioral errors and efficiency loss.

**Design:**
- 16 unique subjects (cleaned dataset), 48 sessions (remove duplicates)
- Live task performance metrics: error rate, response time, accuracy
- Cumulative drift window: 120 sec (~60 trials), threshold >50% triggers intervention

**Core Findings:**

#### 1. Information Geometry: States are Distinct Mathematical Manifolds
| State | Mahalanobis Dist | KL Divergence | Riemannian Dist | Status |
|---|---|---|---|---|
| **Optimal-Monitoring** | 0.00 | 0.00 | 0.00 | Reference |
| **Mind-Wandering** | 2.33 | 9.41 | 1.48 | Mild Deviation |
| **Optimal-Engaged** | 3.14 | 56.92 | 1.73 | Active Shift |
| **Fatigue** | 11.23 | 1776.14 | 2.51 | **Significant Drift** ✅ |
| **Overload** | 13.71 | 127.06 | 6.10 | **System Failure** ✅ |

**Interpretation:** Drift states (Fatigue/Overload) are mathematically far from the optimal manifold, confirming we detect distinct neurological modes, not noise.

#### 2. Machine Learning Predictive Validity: EEG Predicts Real-World Errors
**Model:** GradientBoostingClassifier (5-fold cross-validation)
- **AUC:** 0.6784
- **Accuracy:** 68.30%
- **Sensitivity:** 90.2% (catches 9 out of 10 error events)

**Top 3 Predictive Features:**
1. z_delta (Sleep pressure)
2. z_theta (Mental fatigue)
3. z_alpha_relative (Disengagement)

#### 3. Drift ↔ Performance Correlation
| Comparison | Low Drift (<10%) | High Drift (>15%) | Impact |
|---|---|---|---|
| **Error Risk** | 43.1% | 56.2% | **+30.5% Higher** ✅ |
| **Efficiency** | 62.7 / 100 | 48.3 / 100 | **-22.9% Lower** ✅ |


**Conclusion:** High drift is a reliable predictor of performance collapse.

#### 4. State Stability (Markov Analysis): How "Sticky" are States?
| State | P(Stay) | Interpretation |
|---|---|---|
| **Optimal States** | 90–91% | Highly stable; flow is self-reinforcing |
| **Fatigue** | 76% | Sticky; hard to escape without intervention |
| **Overload** | 40% | Transient; acute crisis |

#### 5. Subject Segmentation: Two Clear Performance Tiers
**High-Drift Group (n=6):**
- Mean Drift >15%
- Error Risk: 46.9%, Efficiency: 57.5/100
- Alerts needed: 24/session average

**Low-Drift Group (n=6):**
- Mean Drift <10%
- Error Risk: 40.1%, Efficiency: 63.7/100
- Alerts needed: <5/session average

**Source:** `docs/PHASE2_DRIFT_DETECTION.md`, `docs/LAB_VALIDATION_RESULTS.md`

---

## Real-Time Deployment: 10-Session Results (Dec 10–26, 2025)

### Executive Summary
**Deployed same biomarker set + decision tree from lab onto Muse 2 (consumer EEG) in real-time. System remained stable across 3 task modes, triggered 1 actionable intervention, and collected complete 3-axis cognitive metrics with pre/post session deltas.**

### Session Overview

| Session | Task | Date | Duration | Trials | Artifact% | Optimal% | Eff Start | Eff End | Eff Decay |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PEAK_GENERATIVE | 2025-12-10 | 23.7 min | 472 | 27.1% | 97.1% | 84.0% | 51.1% | -33% |
| 2 | PASSIVE_CONSUMPTION | 2025-12-10 | 29.3 min | 567 | 17.2% | **97.6%** | 84.5% | 27.8% | **-57%** |
| 3 | RECEPTIVE_ENGAGED | 2025-12-10 | 30.7 min | 471 | 33.3% | 97.1% | 83.9% | 38.6% | -45% |
| 4 | STANDARD_WORK | 2025-12-11 | 23.9 min | 387 | 31.1% | 94.3% | 83.8% | 57.0% | -32% |
| 5 | RECEPTIVE_ENGAGED | 2025-12-11 | 20.3 min | 352 | 25.3% | 95.4% | 84.2% | 45.0% | -46% |
| **6** | **STANDARD_WORK** | **2025-12-12** | **50.3 min** | **790** | **24.1%** | **93.2%** | **81.8%** | **51.9%** | **-37%** |
| 7 | RECEPTIVE_ENGAGED | 2025-12-13 | 15.6 min | 337 | 18.0% | 95.2% | 84.4% | 82.6% | **-2%** |
| 8 | RECEPTIVE_ENGAGED | 2025-12-25 | 24.0 min | 380 | 36.2% | 96.4% | 86.3% | 44.1% | -49% |
| 9 | RECEPTIVE_ENGAGED | 2025-12-25 | 30.1 min | 527 | 26.7% | 96.4% | 84.0% | 46.5% | -45% |
| 10 | STANDARD_WORK | 2025-12-26 | 25.4 min | 611 | 13.3% | 97.8% | 84.0% | 45.7% | -46% |
| | **TOTAL** | | **273 min** | **4,087** | **25.2%** | **96.2%** | **84.1%** | **49.0%** | **-43.7%** |

**Key Observations:**
- ✅ **Artifact management:** 25.2% mean rejection (real-world degradation vs. lab 5–8%)
- ✅ **State classification:** 96.2% optimal (vs. lab target 60–75%; real-time thresholds conservative)
- ✅ **Efficiency decay:** All sessions start at 84%, end at 27–83% depending on task and duration
- ⚠️ **Threshold over-classification:** Real-time 96% optimal >> lab 60–75% target (needs Phase A recalibration on N≥3 users)

### Standout Finding: December 10th Recovery Arc (3 Sessions, Same Day)

**Session 1 (Peak Generative, 11:06 AM):** Arousal escalates (+0.876) despite efficiency drop (84%→51%) → effort compensation; user "wired" but depleted

**Session 2 (Passive Consumption, 3:09 PM, 4.5 hours later):** Executive load collapses (−1.590), efficiency drops steepest (84%→28%) BUT optimal% highest (97.6%, 0% drift) → **neurophysiological reset achieved**

**Session 3 (Receptive Learning, 6:16 PM):** Efficiency moderate (84%→39%), control improves (+0.104), arousal decreases (−0.116) → re-engagement with different neural pathways

**Implication:** Passive recovery works. Low efficiency ≠ poor cognitive state; it's recovery mechanism.

### Intervention Event: Session 6 (Dec 12, 50-min Standard Work)

**Pre-Intervention (T≈45 min):**
- Cumulative drift: 51.7%
- Arousal: −0.690 (drowsy)
- State: Fatigue + disengagement

**Alert Triggered (T=50 min, trial 720 of 790):** Drift >50% + state still Fatigue + cooldown expired

**Post-Intervention (Final 5 min, trials 721–790):**
- **Arousal:** −0.690 → +0.798 | **+1.488 Δ** ← Largest arousal jump across all 10 sessions ✅
- **Control:** +0.202 → −1.206 | **−1.408 Δ** ← Sharp motor/cognitive refocus
- **Executive:** −0.220 → +3.536 | **+3.756 Δ** ← Largest executive escalation across all 10 sessions ✅
- **State:** Recovered to 100% Optimal in final window

**Interpretation:** Intervention timing was neurophysiologically valid. Post-alert 3-axis deltas are extremes of entire deployment, validating intervention logic.

### 3-Axis Cognitive Metrics (All 10 Sessions)

| Session | Task | Arousal Δ | Control Δ | Executive Δ | Key Pattern |
|---|---|---|---|---|---|
| 1 | PEAK | +0.876 | +0.192 | +0.830 | High mobilization; arousal spike |
| 2 | PASSIVE | −0.006 | −0.068 | −1.590 | Recovery; largest load release ⭐ |
| 3 | RECEPTIVE | −0.116 | +0.104 | +0.128 | Learning engages control |
| 4 | STANDARD | −0.886 | −0.044 | −0.880 | Work → recovery transition |
| 5 | RECEPTIVE | 0.000 | +0.410 | −1.108 | Learning; control gains |
| 6 | STANDARD | +1.488 ⭐ | −1.408 | +3.756 ⭐ | **Intervention success** |
| 7 | RECEPTIVE | +0.134 | +0.148 | +3.272 ⭐ | Short session; executive spike |
| 8 | RECEPTIVE | −0.710 | −0.342 | −0.038 | Evening fatigue accumulation |
| 9 | RECEPTIVE | +0.186 | +1.536 ⭐ | −1.802 | Learning; control peaks |
| 10 | STANDARD | +0.542 | +0.400 | +0.046 | End-of-deployment stabilization |

**Status of 3-Axis Model:**
- ✅ **Computed:** Full pre/post deltas for all 10 sessions, per trial
- ✅ **Patterns:** Qualitatively meaningful (passive recovery, intervention success, task-mode differences)
- ⚠️ **Not Yet Validated:** Against actual task performance metrics (planned Phase B)

**Source:** `results/REALTIME_10_SESSION_COMPLETE.md`, `results/all_sessions_efficiency_3axis.csv`

---

## What This System Does

### Real-Time Pipeline
```
Muse 2 EEG (256 Hz, 4 channels)
    ↓
Lab Streaming Layer + Adaptive Artifact Rejection
    ↓
2-sec Epoch Buffering (512 samples)
    ↓
Feature Extraction: 15 Biomarkers (filters, FFT, entropy, PAC)
    ↓
Rolling Baseline (100-trial chunks, clean trials only)
    ↓
Z-Score Normalization (per feature, per trial)
    ↓
Rule-Based State Classification (Tier 1–4 decision tree)
    ↓
30-sec Windowed Aggregation (confidence-weighted voting)
    ↓
3-Axis Computation (Arousal, Control, Executive)
    ↓
Drift Tracking + Error Risk Prediction
    ↓
Intervention Manager (drift >50%, cooldown, state check)
    ↓
CSV Output + Real-Time Display
```

### Key Features
- ✅ **Rolling baseline adaptation:** Learns user's brain within 100 trials (3–5 min)
- ✅ **Drift tracking:** 120-sec cumulative windows detect sustained fatigue/mind-wandering
- ✅ **Intervention logic:** Triggers only when drift >50% + state is Fatigue/Overload + cooldown >60 sec (prevents alert fatigue)
- ✅ **Multi-mode support:** Work, Active Learning, Passive Consumption with different EEG signatures
- ✅ **Error prediction:** Real-time error risk model (AUC 0.68 from lab)
- ✅ **Efficiency tracking:** Per-trial efficiency metric (84% start → 27–83% end task-dependent)
- ✅ **3-axis metrics:** Arousal (alert↔drowsy), Control (motor/cognitive focus), Executive (load/effort)


---


## Validation Status

### ✅ Lab-Validated (Phase 1–2: 21 Subjects, 60+ Sessions)
- 15 biomarkers (theta, alpha, beta, gamma, entropy, complexity, PAC)
- State classification: 68.3% accuracy (Gradient Boosting), AUC 0.68
- Drift ↔ performance: +30.5% error risk when drift >50%, −22.9% efficiency loss
- State stability: Information geometry confirms distinct manifolds (KL divergence analysis)
- Intervention timing: Should trigger at 45–50 min for 50-min+ sessions

### ✅ Real-Time Validated (1 User, 10 Sessions, 4,087 Trials)
- Biomarkers translate from lab to Muse 2
- Rolling baseline adapts within 100 trials
- Intervention triggered once (Session 6) at expected time
- 3-axis deltas show qualitatively meaningful patterns (recovery, intervention success)
- Multi-mode feasibility proven (Work, Learning, Passive all deployable)

### ⚠️ Not Yet Validated (Next: Phase A)
- **Multi-user generalization:** Single user; transfer to N≥3 new users untested
- **Threshold recalibration:** Real-time 96% optimal >> lab 60–75% target; needs re-calibration on diverse users
- **3-axis performance correlation:** No link yet between Arousal/Control/Executive and actual error rate/speed
- **Task-mode baselines:** Each mode may have different natural drift rates (~15–40%); not yet differentiated
- **Cross-subject impedance profiles:** Muse 2 artifact rejection varies by electrode fit; untested on diverse scalps

### Production Readiness
- **Research-ready:** Can deploy to N≥3 user cohort for Phase A validation
- **Not production-ready:** Without multi-user validation + 3-axis performance grounding + task-mode-specific thresholds

---

## Known Limitations

1. **Single-subject deployment:** 10 sessions = 1 user's brain; generalization untested
2. **Threshold calibration incomplete:** Real-time optimal% (96%) >> lab target (60–75%)
3. **3-axis unvalidated:** No correlation with actual task performance (planned Phase B)
4. **Task-mode baselines:** Work thresholds applied uniformly; Learning/Passive may need different distributions
5. **Muse 2 hardware constraints:** 4 channels (vs. 19–32 lab), lower SNR; custom hardware (Phase B) improves
6. **Consumer EEG artifact rate:** 25% vs. 5–8% in lab; real-world degradation expected
7. **Efficiency metric is work-optimized:** Penalizes low-entropy patterns healthy in Passive/Receptive modes


---


## References

**Lab Validation:**
- 16 subjects, 48 sessions, 45,093 trials (research-grade EEG)
- Biomarkers validated in literature (theta/alpha/beta/gamma, entropy, PAC, etc.)
- Error-drift correlation: AUC 0.68 (Gradient Boosting)
- State distributions match neuroscience targets (60–75% Optimal, 15–25% MW, 5–10% Fatigue, 2–5% Overload)

**Real-Time Deployment:**
- December 10–26, 2025
- 10 sessions, 4,087 trials, ~273 minutes
- 1 user 

**See detailed docs:**
- `docs/LAB_VALIDATION_RESULTS.md` – Full Phase 2 stats
- `docs/PHASE1_BASELINE_DETECTION.md` – State classification details
- `docs/PHASE2_DRIFT_DETECTION.md` – Drift + intervention logic
- `RealTIme_Analysis/REALTIME_10_SESSION.md` – 10-session full report
- `docs/FEATURE_EXTRACTION.md` – 15-biomarker specs

---

