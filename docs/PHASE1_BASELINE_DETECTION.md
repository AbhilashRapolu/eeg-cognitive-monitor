# Phase 1: Rolling Baseline & State Classification

## 1. Rolling Baseline Logic

Unlike static baselines that fail as brain state evolves over hours, our system uses a **dynamic rolling baseline** to account for non-stationarity in EEG signals.

### Mechanism
- **Chunk Size:** 100 trials (approx. 3.3 minutes)
- **Minimum Trials:** 15 valid trials required to form a baseline
- **Update Rule:** 
  - The baseline (mean $\mu$, std $\sigma$) is re-calculated every `CHUNK_SIZE` trials.
  - Only "clean" trials are added to the buffer (artifact-free).
  - Z-scores for the current trial are computed against the *previous* chunk's statistics.
  - Trials 1–100 → everyone contributes to the initial baseline.
  - After 100 → only optimal trials update the rolling baseline.

$$
Z_{feature} = \frac{X_{feature} - \mu_{baseline}}{\sigma_{baseline}}
$$

```python
class BaselineDetector:
    def process_trial(features, state):
        # 1. Collect Data
        # Only update baseline using "clean" data (first 100 trials OR optimal states)
        if is_initialization or state in ["Optimal-Engaged", "Optimal-Monitoring"]:
            buffer.add(features)

        # 2. Update Baseline (Periodic)
        if trials_count % CHUNK_SIZE == 0:
            current_baseline.mean = mean(buffer)
            current_baseline.std = std(buffer)
            buffer.clear()

        # 3. Compute Z-Scores (Standardization)
        # Compare current trial against the *previous* valid baseline
        z_scores = (features - current_baseline.mean) / current_baseline.std
        return z_scores
```

This ensures that "High Alpha" means "High relative to the last few minutes," adapting to the user's circadian rhythm and fatigue accumulation.

---

## 2. Classification Logic (Decision Tree)

The classifier uses a hierarchical decision tree grounded in neuroscience literature. It prioritizes detecting **dangerous states** (Overload, Fatigue) before classifying engagement.

### Tier 1: Overload Detection (Target: 2-5%)
*Catastrophic failure of cognitive control.*
- **Primary Marker:** Extreme Frontal Midline Theta (FM$\theta$) > 4.5 $\sigma$
- **Secondary Markers:** 
  - Complexity Collapse ($Z_{mse} < -3.5$)
  - Gamma Suppression ($Z_{gamma} < -3.5$)
- **Condition:** If `Overload Markers` $\ge$ 4, classify as **Overload**.

### Tier 2: Fatigue Detection (Target: 5-10%)
*Metabolic exhaustion and reduced alertness.*
- **Primary Marker:** Posterior Alpha ($Z_{alpha} > 1.8$)
- **Secondary Markers:**
  - Delta increase ($Z_{delta} > 0.5$)
  - Theta increase ($Z_{theta} > 0.3$)
  - Complexity decrease ($Z_{lz} < -1.0$)
- **Condition:** If `Fatigue Markers` $\ge$ 4, classify as **Fatigue**.

### Tier 3: Mind-Wandering (Target: 15-25%)
*Attentional decoupling and drift.*
- **Path 1 (Classic):** High Theta/Beta Ratio ($Z_{tbr} > 0.9$) + Low Entropy ($Z_{pe} < -0.4$)
- **Path 2 (Disengagement):** Low Alpha ($Z_{alpha} < -0.5$) + High Theta ($Z_{theta} > 0.8$)
- **Path 3 (Complexity Drop):** Significant PE drop ($Z_{pe} < -1.2$)

### Tier 4: Optimal States (Target: 60-70%)
If no negative states are detected, the system scores engagement:
- **Engagement Score:** Sum of positive markers (Alpha > 0.3, Beta > 0.6, Gamma > 0.5, LZ > 0.4).
- **Classification:**
  - Score $\ge$ 8 $\rightarrow$ **Optimal-Engaged** (Flow state)
  - Score < 8 $\rightarrow$ **Optimal-Monitoring** (Steady state)

---

## 3. Calibration Targets

To validate the model, we check if the distribution of states matches expected physiological norms for a focused task:

| State | Target Range | Rationale |
|:---|:---|:---|
| **Optimal** | 60% - 75% | User should be focused most of the time. |
| **Mind-Wandering** | 15% - 25% | Natural attentional lapses (10-15 mins/hour). |
| **Fatigue** | 5% - 10% | Accumulates over time; shouldn't dominate early. |
| **Overload** | 2% - 5% | Rare events of high difficulty or stress. |

*Validation Results (Phase 1):* Across 48 sessions, the model produced **state distributions close to** the neuroscience targets (MW, Fatigue, Overload slightly below target ranges but qualitatively consistent).
