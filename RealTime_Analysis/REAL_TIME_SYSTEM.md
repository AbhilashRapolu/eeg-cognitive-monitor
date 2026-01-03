# Real-Time System: Lab Validation to Live Deployment

## Executive Summary

The **real-time cognitive monitoring system** extends the lab-validated pipeline (Phase 1 & 2) into live, consumer-grade EEG deployment on a Muse 2 headset. It streams per-trial cognitive state classifications, cumulative drift tracking, and intervention decisions **in real time**.

**Key Upgrade:** Extended lab-validated 14-biomarker system (3 channels: Fp1, Fp2, TP10) to real-time 15-biomarker system (4 channels: Fp1, Fp2, TP9, TP10). Added frontal theta and improved temporal theta with dual-channel aggregation.

**Current Validation:** 4,049 trials across 3 task modes over ~338 minutes, 1 user, 10 sessions. **Status: Proof-of-concept.**

---

## Lab vs. Real-Time Comparison

| Aspect | Lab Validation | Real-Time |
|---|---|---|
| **Subjects** | 21 diverse | 1 user (early) |
| **Trials** | 45,093 | 4,049 |
| **Hardware** | Research-grade (19–32 channels) | Consumer Muse 2 (4 channels) |
| **Task Domain** | Work-focused | 3 modes: Work, Learning, Passive |
| **Artifact Rate** | ~5–8% | ~5–8% |
| **Deployment** | Post-hoc | Real-time streaming |
| **Status** | Validated (AUC 0.68) | Proof-of-concept |

---

## System Architecture

```
Raw EEG (256 Hz, Muse 2) 
  ↓
[1] Epoch Collection (2-sec, artifact rejection)
[2] Feature Extraction (15 biomarkers)
[3] Rolling Baseline (100-trial chunks)
[4] Z-Score Normalization
[5] Instant State Classification (rule-based)
[6] 30-sec Windowed Aggregation
[7] 3-Axis Metrics (arousal, control, executive)
[8] Cumulative Drift Tracking (120-sec window)
[9] Error Risk Prediction (work tasks only)
[10] Intervention Manager (if drift > 50%)
  ↓
Output: CSV logs (35+ fields/trial) + Real-time feedback
```

---

## 15 Biomarkers

| Biomarker | Meaning |
|---|---|
| **z_theta** | Working memory load (higher = effort) |
| **z_alpha** | Disengagement/fatigue (higher = zoning out) |
| **z_beta** | Alertness & active processing |
| **z_delta** | Sleep pressure (higher = fatigue) |
| **z_gamma** | Feature binding & high cognition |
| **z_tbr** | Mind-wandering (Theta/Beta ratio) |
| **z_at_ratio** | Drowsy fatigue (Alpha/Theta) |
| **z_pe** | Signal complexity (lower = disengaged) |
| **z_lz** | Consciousness level (Lempel-Ziv) |
| **z_pac** | Theta-Gamma cross-frequency coupling |
| **z_wpe** | Weighted permutation entropy |
| **z_mse** | Temporal structure (lower = overload) |
| **z_alpha_rel** | Relative alpha dominance |
| **z_frontal_asym** | Emotional valence (Fp1 asymmetry) |
| **z_theta_frontal** | Frontal theta (executive control) **[NEW]** |

---

## Core Components

### 1. Epoch Collection & Artifact Rejection
- 2-sec windows (512 samples @ 256 Hz)
- Adaptive amplitude thresholds: TP10 < 100µV, Fp1/Fp2 < 150µV
- Kurtosis filter: flag bursts (EMG, eye movement)
- Rejection rate: ~5–8%

### 2. Feature Extraction
- ~100 ms per epoch (IIR filters, FFT, entropy)
- All 15 biomarkers computed per trial

### 3. Rolling Baseline
- Last **100 clean optimal-state trials**
- Updated every 100 trials (adapts within 3–5 minutes)
- Formula: `z = (feature - μ) / σ`

### 4. Instant State Classification (Rule-Based)

**Overload** (2–5%): z_theta > 4.5 + z_lz < -3.5  
**Fatigue** (5–10%): z_alpha > 1.8 + z_delta > 0.5 + z_lz < -1.0  
**Mind-Wandering** (15–25%): z_tbr > 0.9 + z_pe < -0.4  
**Optimal** (60–75%): High Beta & Gamma, high entropy

### 5. 30-Second Windowed Aggregation
- Confidence-weighted voting (~15 trials)
- Output: stable state label + mean z-scores
- Matches 30-sec behavioral timescale

### 6. 3-Axis Metrics

| Axis | Components | Range |
|---|---|---|
| **Arousal** | Beta, Alpha, Delta | -2 to +2 |
| **Control** | Beta, Entropy, TBR | -2 to +2 |
| **Executive** | Frontal Theta, PAC | -2 to +2 |

Combined: `intensity = weighted(arousal, control, executive)` → 0–100

### 7. Cumulative Drift Tracking
- 120-sec rolling window (~60 trials)
- Metric: % trials in drift states
- Thresholds: <15% (Normal), 15–50% (Warning), >50% (Critical)

### 8. Error Risk Prediction (Work Only)
- Logistic regression from lab (45,093 trials)
- Input: 14 z-scores + cumulative drift %
- Output: 0–100% error probability
- **Not computed for Learning or Passive modes**

### 9. Efficiency Score
- `efficiency = (0.5 × stability) + (0.2 × entropy) + (0.3 × intensity)`
- Range: 0–100 (higher = more efficient)

### 10. Intervention Manager
**Triggers alert IF:**
- Cumulative drift > 50% ✓
- Cooldown expired (≥60 sec) ✓
- Current state = Fatigue or Overload ✓
- Max 1 alert per minute

---

## Output & Logging

**CSV Log (Per Trial):**  
timestamp, trial_idx, task_type, instant_state, windowed_state, confidence, 15 z-scores, arousal, control, executive, intensity, efficiency, cumulative_drift_pct, error_risk, intervention_triggered

**On-Screen Feedback:**  
- Instant state + color (Green=Optimal, Red=Fatigue, Yellow=MW)
- Intensity & efficiency gauges (0–100)
- Drift % gauge + intervention alert

**Metadata JSON (End of Session):**  
duration, total_trials, task_type, mean_efficiency, state_distribution, intervention_count, artifacts_rejected

---

## Task Modes

| Mode | Error Model? | Intervention Threshold | Target State % |
|---|---|---|---|
| **PEAK_GENERATIVE** | Yes (high sensitivity) | drift > 50% | Optimal-Engaged > 70% |
| **STANDARD_WORK** | Yes (moderate) | drift > 50% | Optimal > 70% |
| **RECEPTIVE_ENGAGED** | No | drift > 50% | Optimal + Passive > 80% |
| **PASSIVE_CONSUMPTION** | No | delta extreme only | Fatigue monitoring only |

---

## Performance & Latency

| Operation | Latency |
|---|---|
| Feature extraction | ~100 ms |
| Baseline lookup & z-score | ~10 ms |
| State classification | ~5 ms |
| Windowed aggregation | ~10 ms |
| Drift tracking | ~5 ms |
| Error prediction | ~20 ms |
| **Total per trial** | ~150 ms |
| **Trial rate** | 0.5 Hz |
| **Alert decision** | < 5 ms |

---

## System Limitations & Validation Needed

### **Key Limitations**

The current system has several critical limitations that must be addressed before production deployment:

1. **Hardware Transfer Risk:** Lab validation used research-grade hardware (19–32 channels, 3–5x higher SNR); Muse 2 consumer hardware is significantly noisier and covers fewer brain regions. Thresholds tuned on lab data may not transfer directly to consumer EEG.

2. **Single-User Pilot:** Current real-time validation uses only 1 subject across 10 sessions (4,049 trials) vs. 45,093 lab trials across 21 subjects. Cannot assess inter-individual variability or generalization to other users.

3. **Task-Specific Classifiers Missing:** Learning and Passive consumption modes currently reuse work-task thresholds, which are likely invalid for these contexts. These require separate model development.

4. **Intervention Efficacy Unknown:** No evidence yet that alerts improve user performance or reduce errors. Effectiveness depends on alert timing, modality (audio vs. haptic), and user engagement.

5. **Error Model Not Validated on Muse 2:** Lab error model achieved AUC 0.68, but it was trained on different hardware. Real performance on Muse 2 is unknown (likely drop to 0.60).

6. **State Distribution Unverified:** System should produce ~60–75% Optimal, 15–25% Mind-Wandering, ~5% Fatigue, ~2% Overload states. This distribution has never been verified on Muse 2.

### **Validations Required Before Production**

1. **Cross-subject validation:** Test on 5–10 diverse subjects to confirm state distributions match lab expectations and thresholds generalize. Verify state classification accuracy >75% and error model AUC ≥0.60 on Muse 2.

2. **Threshold re-optimization:** Refit decision rules using Muse 2 data to account for lower SNR; re-validate Mahalanobis distance and entropy metrics (PE, LZ, MSE) are reliable on consumer hardware.

3. **Task-specific model development:** Build separate classifiers for Learning and Passive modes using task-specific ground truth (test scores for learning, engagement metrics for passive consumption).

4. **Intervention efficacy A/B testing:** Randomized study with on/off alert conditions to measure if interventions actually reduce error rates and improve efficiency.

5. **Real-world robustness testing:** Deploy system in naturalistic environments to validate performance under poor signal conditions, electrode movement, and user adaptation over time.

These validations should be completed before claiming real-world performance claims or deploying to users outside research settings.

---
