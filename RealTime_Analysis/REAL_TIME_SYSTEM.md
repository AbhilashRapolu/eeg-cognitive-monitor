# Real-Time System: Lab Validation to Live Deployment

## Executive Summary

The **real-time cognitive monitoring system** extends the lab-validated pipeline (Phase 1 & 2) into live, consumer-grade EEG deployment on a Muse 2 headset. It streams per-trial cognitive state classifications, cumulative drift tracking, and intervention decisions **in real time**.

**Key Upgrade:** Extended lab-validated 14-biomarker system (3 channels: Fp1, Fp2, TP10) to real-time 15-biomarker system (4 channels: Fp1, Fp2, TP9, TP10). Added frontal theta and improved temporal theta with dual-channel aggregation.

**Current Validation:** 4,049 trials across 3 task modes over ~338 minutes, 1 user, 10 sessions. **Status: Proof-of-concept.**

---

## System Architecture

```
Raw EEG (256 Hz, Muse 2) 
  ↓
[0] Morning Baseline Setup (One-Time, 60 sec)
    ├─ Daily baseline stored in ~/.cognitive_monitor/daily_baselines/
    ├─ Used for all recommendations that day
    └─ Auto-loads next session if same calendar day
  ↓
[1] Epoch Collection (2-sec, artifact rejection)
[2] Feature Extraction (15 biomarkers)
[3] Session Baseline Detector (ChunkedBaselineDetector)
    ├─ Rolling: Last 100 clean optimal-state trials
    ├─ Initialized from morning baseline
    └─ Updates every 100 trials thereafter
[4] Z-Score Normalization (per trial)
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

## Morning Baseline Setup 
**Purpose:** Establish a fresh reference state for the day's recommendations and initial session calibration.

**Mechanism:**
- One-time collection of 60 seconds (~30 epochs) in a rested, neutral state
- User sits comfortably, eyes open, looks at neutral point
- System rejects epochs with artifacts automatically
- Baseline stored in `~/.cognitive_monitor/daily_baselines/{user_id}_baseline_v1.json`
- If file exists and matches today's date, automatically reloaded (no re-collection needed)

**Usage in Session Runner:**
```python
if self.monitor.daily_baseline_mgr.needs_baseline():
    baseline = runner.collect_morning_baseline()
    # Stores: date, time, mean & std for all 15 biomarkers
```

**Benefits:**
- ✅ Personalized reference point (accounts for circadian/fatigue cycles)
- ✅ Fast recommendations at session start (no 100-trial wait)
- ✅ Reuses if same calendar day (no repeat collection needed)
- ✅ Ground-truth validation: post-session 3-axis should return near zero if fresh

---

## Two-Baseline Architecture (Critical Design)

The system uses **two separate baseline managers** to separate concerns:

### 1. `DailyBaselineManager` (Morning)
- **When**: One-time per calendar day, before any session
- **What**: Collects 60 sec of neutral resting state
- **Where**: `~/.cognitive_monitor/daily_baselines/{user_id}_baseline_v1.json`
- **Use case**: Pre-session recommendations, initial 3-axis context
- **Output**: `z_scores = daily_baseline_mgr.compute_z_scores(features)`

### 2. `ChunkedBaselineDetector` (Session)
- **When**: Throughout session, starting from first trial
- **What**: Rolling 100-trial buffer of clean optimal-state trials (or first 100 if initializing)
- **Where**: In-memory during session (reset each session)
- **Use case**: Real-time classification, cumulative drift tracking
- **Output**: `z_scores = baseline_detector.process_trial(features, is_optimal)`

**Why two?**
- Daily baseline provides **stable reference** for arousal/control metrics
- Session baseline provides **adaptive normalization** to within-session changes (fatigue accumulation, context shifts)
- Morning baseline prevents "baseline creep" during long sessions

---

## Baseline Persistence & Cross-Session Continuity

After each session, the system saves:
```python
BaselinePersistence.save_baseline(
    baseline_detector,   # ChunkedBaselineDetector
    ig_computer,         # RealTimeIGComputer (for information geometry)
    session_name="YYYY-MM-DD_task"
)
```

**Stored in:** `~/.cognitive_monitor/session_baselines/{user_id}/{session_name}.pkl`

**Next session:** If same task is re-run within 7 days, baseline can be pre-loaded for instant calibration.

---

## 15 Biomarkers (Real-Time)

| Biomarker | Meaning |
|---|---|
| **z_theta** | Working memory load (higher = effort) |
| **z_alpha** | Disengagement/fatigue (higher = zoning out) |
| **z_beta** | Alertness & active processing |
| **z_delta** | Sleep pressure (higher = fatigue) |
| **z_gamma** | Feature binding & high cognition |
| **z_tbr** | Theta/Beta ratio (mind-wandering marker) |
| **z_at_ratio** | Alpha/Theta ratio (drowsy fatigue) |
| **z_pe** | Permutation Entropy (signal complexity) |
| **z_lz** | Lempel-Ziv Complexity (consciousness level) |
| **z_pac** | Phase-Amplitude Coupling (theta-gamma sync) |
| **z_wpe** | Weighted Permutation Entropy |
| **z_mse** | Multiscale Entropy (temporal structure) |
| **z_alpha_rel** | Relative alpha dominance (idling) |
| **z_frontal_asym** | Frontal asymmetry (emotional valence) |
| **z_theta_frontal** | Frontal theta (executive control) [NEW] |

---

## Core Components

### 1. Epoch Collection & Artifact Rejection
- 2-sec windows (512 samples @ 256 Hz)
- Adaptive amplitude thresholds: TP10 < 500µV, Fp1/Fp2 < 500µV
- Kurtosis filter: flag bursts (EMG, eye movement) > 8.0
- Rejection rate: ~5–8% (lab) → ~25% (real-world Muse 2)

### 2. Feature Extraction
- ~100 ms per epoch (IIR filters, FFT, entropy)
- All 15 biomarkers computed per trial
- Spectral bands: Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), Gamma (30–45 Hz)

### 3. Morning Baseline Initialization
- **First-time users**: Collect 60 sec, store as daily baseline
- **Returning users**: Load from disk if same calendar day
- **Auto-rejection**: Skip epochs with artifacts, re-collect until 20+ clean epochs

### 4. Session Baseline Detector (ChunkedBaselineDetector)
- Last **100 clean optimal-state trials** (or first 100 if initializing)
- Updated every 100 trials (adapts within 3–5 minutes)
- Formula: `z = (feature - μ) / σ`
- Ensures "High Alpha" means "High relative to the last few minutes," adapting to fatigue accumulation

### 5. Instant State Classification (Rule-Based, Tier 1–4)

**Tier 1: Overload Detection (Target: 2–5%)**
- Primary: Extreme Frontal Theta (> 4.5 σ)
- Secondary: Complexity Collapse (MSE < -3.5 σ), Gamma Suppression (< -3.5 σ)
- Condition: If ≥4 markers → **Overload**

**Tier 2: Fatigue Detection (Target: 5–10%)**
- Primary: Posterior Alpha (> 1.8 σ)
- Secondary: Delta increase (> 0.5 σ), Theta increase (> 0.2 σ), Complexity decrease (LZ < -1.0 σ)
- Condition: If ≥2 markers → **Fatigue**

**Tier 3: Mind-Wandering (Target: 15–25%)**
- Path 1: High Theta/Beta Ratio (> 0.25 σ) + Low Entropy (PE < -0.2 σ)
- Path 2: Low Alpha (< -0.15 σ) + High Theta (> 0.3 σ)
- Path 3: Complexity Drop (PE < -1.2 σ)
- Condition: If ≥2 paths met → **Mind-Wandering**

**Tier 4: Optimal States (Target: 60–75%)**
- **Optimal-Engaged**: High engagement markers (Alpha > 0.3, Beta > 0.6, Gamma > 0.5, LZ > 0.4) AND ≥8 markers present
- **Optimal-Monitoring**: Fewer engagement markers but no drift signals

### 6. 30-Second Windowed Aggregation
- Confidence-weighted voting (~15 trials per window)
- Output: Stable state label + mean z-scores
- Matches 30-sec behavioral timescale (reduces jitter from single-trial classification)

### 7. 3-Axis Metrics (Real-Time, Task-Aware)

| Axis | Components | Range |
|---|---|---|
| **Arousal** | Beta, Alpha, Delta (alertness ↔ drowsiness) | -2 to +2 σ |
| **Control** | Beta, Entropy, Theta/Beta ratio (motor/cognitive focus) | -2 to +2 σ |
| **Executive** | Frontal Theta, PAC (working memory load, cognitive demand) | -2 to +2 σ |

**Formula:**
```
Intensity = 100 × Σ(w_i × clip(z_i, [0,1]))
where positive contributors: Beta, Gamma, LZ, PE, PAC, MSE
where negative contributors: Theta, Theta/Beta ratio
```

**Efficiency Score:**
```
Efficiency = (0.5 × stability_score) + (0.2 × entropy_score) + (0.3 × intensity_score)
where stability = 100 / (1 + Mahalanobis_distance)
where entropy = 100 / (1 + KL_divergence / 100)
where intensity = computed above [0-100]
```

### 8. Cumulative Drift Tracking (120-sec Window)
- Deque of last ~60 trials (~120 s)
- Metric: **% of trials in drift states** (Mind-Wandering, Fatigue, Overload)
- Thresholds:
  - Low Drift (< 15%): Normal operation
  - Medium Drift (15–50%): Warning zone
  - High Drift (> 50%): **Critical intervention zone**

### 9. Error Risk Prediction (Work Tasks Only)
- Logistic regression from lab (trained on 45,093 trials)
- Input: 14 z-scores + cumulative drift %
- Output: 0–100% error probability
- **Not computed for Learning or Passive modes** (invalid context)

### 10. Intervention Manager
**Triggers alert IF AND ONLY IF:**
1. Cumulative drift > 50% for 2 consecutive windows
2. Cooldown expired (≥ 5 minutes since last intervention)
3. Current state = Fatigue or Overload (not just transient drift)

**Message generation:** Context-specific (mind-wandering vs. fatigue) with actionable suggestions
**Desktop notification:** Cross-platform (Windows, macOS, Linux)
**Max frequency:** 1 alert per 5 minutes (prevents alert fatigue)

---

## Output & Logging

**CSV Log (Per Trial):**
```
timestamp, trial, instant_state, windowed_state, confidence, 
intensity, signal_quality, drift_strength, drift_label, drift_markers,
drift_risk, cumulative_drift_pct, rolling_drift, arousal_sigma, 
control_sigma, executive_sigma, error_risk, efficiency, mahalanobis,
z_theta, z_alpha, z_beta, z_gamma, z_delta, z_pe, z_lz, z_pac, ...
```

**On-Screen Feedback (Every 10 trials):**
```
[  234] 🎯Optimal-Engaged       → ✅Optimal-Monitoring | Ar:+0.1σ | Ctl:-0.2σ | SQ:0.85 | CumDrift:8.2% | Drift:NONE
```

**Metadata JSON (End of Session):**
```json
{
  "session_name": "learning_session_20250104_143000",
  "task_type": "learning",
  "start_time": "2025-01-04T14:30:00",
  "end_time": "2025-01-04T16:20:00",
  "duration_minutes": 110.0,
  "total_trials": 3300,
  "total_epochs_rejected": 840,
  "artifact_rejection_rate": 20.3,
  "state_distribution": {
    "Optimal-Engaged": 2100,
    "Optimal-Monitoring": 800,
    "Mind-Wandering": 400
  },
  "pre_session": {
    "arousal_sigma": -0.2,
    "control_sigma": 0.1,
    "executive_sigma": -0.3
  },
  "session_averages": {
    "arousal_sigma": 0.0,
    "control_sigma": 0.1,
    "executive_sigma": 0.2,
    "efficiency": 65.2
  },
  "interventions": [
    {
      "trial": 1200,
      "cumulative_drift_pct": 52.1,
      "state": "Fatigue",
      "message": "😴 Fatigue Detected..."
    }
  ]
}
```

---

## Task Modes

| Mode | Error Model? | Drift Threshold | Target State % | Use Case |
|---|---|---|---|---|
| **PEAK_GENERATIVE** | Yes (high sensitivity) | > 50% | Optimal-Engaged > 70% | Complex creation, coding, writing |
| **STANDARD_WORK** | Yes (moderate) | > 50% | Optimal > 70% | Routine work, admin, email |
| **RECEPTIVE_ENGAGED** | No | > 50% | Optimal + Passive > 80% | Active learning, studying, lectures |
| **PASSIVE_CONSUMPTION** | No | Delta extreme only | Fatigue monitoring only | Video, articles, podcasts |

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
| **Trial rate** | 0.5 Hz (2 sec per trial) |
| **Alert decision** | < 5 ms |

---

## System Limitations & Validation Needed

### **Key Limitations**

1. **Hardware Transfer Risk:** Lab validation used research-grade hardware (19–32 channels, 3–5x higher SNR); Muse 2 consumer hardware is significantly noisier and covers fewer brain regions. Thresholds tuned on lab data may not transfer directly to consumer EEG.

2. **Single-User Pilot:** Current real-time validation uses only 1 subject across 10 sessions (4,049 trials) vs. 62,494 lab trials across 16 subjects. Cannot assess inter-individual variability or generalization to other users.

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
