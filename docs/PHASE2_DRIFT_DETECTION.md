# Phase 2: Cumulative Drift & Intervention

While Phase 1 classifies *instantaneous* states (2-second windows), Phase 2 tracks the **accumulation of drift** over longer timescales to trigger interventions only when necessary.

## 1. The "Stability vs. Sensitivity" Problem

A single 2-second classification of "Mind-Wandering" could be a momentary blink or artifact. It is annoying to alert the user for every transient lapse.

**Solution:** Use a **Cumulative Drift Window** (120 seconds / ~60 trials) to smooth the signal and measure the *percentage* of time spent in drift states.

---

## 2. Cumulative Drift Tracker

This engine maintains a rolling history of the last 120 seconds of classifications.

### Core Logic
1. **Window Size:** 120 seconds (~60 trials at 0.5 Hz)
2. **Drift Definition:** Any trial classified as `Mind-Wandering`, `Fatigue`, or `Overload` counts as "Drift" (1). Optimal trials count as "Stable" (0).
3. **Metric:** `Drift %` = (Sum of Drift Trials / Total Trials in Window) * 100

### Thresholds
- **Low Drift (< 15%):** Normal operation. User is generally focused.
- **Medium Drift (15% - 50%):** Warning zone. Efficiency is dropping.
- **High Drift (> 50%):** **Critical Intervention Zone.** The user is effectively offline.

```python
class CumulativeTracker:
    window = Deque(maxlen=60) # ~2 minutes

    def update(state):
        # 1. Track Drift History
        is_drift = state in ["Fatigue", "Overload", "Mind-Wandering"]
        window.append(1 if is_drift else 0)

        # 2. Calculate Percentage
        drift_pct = sum(window) / len(window)
        return drift_pct
```

---

## 3. Intervention Policy

To prevent "alert fatigue," we do not trigger alarms solely based on the drift percentage. We use a **state-dependent intervention manager** with cooldowns.

### Intervention Rules
An intervention (audio/haptic cue) is triggered **IF AND ONLY IF**:
1. **Cumulative Drift > 50%** (The user has been drifting for > 1 minute total in the last 2 minutes).
2. **Cooldown Expired:** At least 30 trials (60 seconds) have passed since the last intervention.
3. **Persistent Drift:** The *current* smoothed state is still a Drift State.

```python
class InterventionManager:
    last_trigger = -999

    def check_intervention(drift_pct, current_state, trial_index):
        # Rule 1: Intensity of Drift
        high_drift = drift_pct > 0.50

        # Rule 2: Cooldown (Prevent alert fatigue)
        cooldown_ok = (trial_index - last_trigger) > 30

        # Rule 3: Immediate State
        currently_drifting = current_state in DRIFT_STATES

        if high_drift and cooldown_ok and currently_drifting:
            last_trigger = trial_index
            return TRIGGER_ALERT
        return NO_ACTION
```

*Why this works:*
- It ignores short bursts of drift (e.g., checking a phone notification).
- It catches sustained "zoning out" or fatigue crashes.
- It ensures alerts are spaced out (maximum 1 alert per minute).

---

## 4. Phase 2 Results (16 Subjects)

We validated this logic on the clean dataset of 16 unique subjects (removing duplicates).

### A. Impact of Drift on Performance
By segmenting data into "High Drift" vs. "Low Drift" windows, we found massive performance penalties:

| Metric | Low Drift Windows | High Drift Windows | Delta | Impact |
|:---|:---|:---|:---|:---|
| **Error Risk** | 43.1% | 56.2% | +13.1% | **30.5% Higher Risk** |
| **Efficiency** | 62.7 | 48.3 | -14.4 | **22.9% Lower Efficiency** |

### B. Subject Segmentation
The cumulative drift metric successfully separated users into clear performance tiers:

1. **High-Drift Group (n=6):**
   - Mean Drift > 15%
   - **Performance:** High Error Risk (46.9%), Low Efficiency (57.5)
   - **Alerts:** 24 alerts/session average (Needs intervention)

2. **Low-Drift Group (n=6):**
   - Mean Drift < 10%
   - **Performance:** Low Error Risk (40.1%), High Efficiency (63.7)
   - **Alerts:** < 5 alerts/session average (Flow state)

### C. Predictive Validity
Machine Learning models trained to predict *behavioral errors* using these drift features achieved an **AUC of 0.68**, confirming that the EEG drift signal is a valid proxy for real-world performance drops.
