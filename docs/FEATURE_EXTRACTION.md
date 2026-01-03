# 14 Validated Biomarkers

I'm using 14 eeg features which are neuroscience literature based regarding attention, fatigue, and cognitive load.

## Feature Overview

Each feature is computed from specific frequency bands or complexity metrics, then normalized (z-scored) against a rolling baseline.

| Feature ID | Name | Frequency / Definition | Cognitive Significance |
|:---:|:---|:---|:---|
| **z_theta** | Frontal Theta | 4–8 Hz (Frontal) | Increases with **working memory load** and mental effort. |
| **z_beta** | Sensorimotor Beta | 13–30 Hz | Marker of **alertness**, active processing, and motor inhibition. |
| **z_alpha** | Posterior Alpha | 8–13 Hz | Increases during **relaxation** or disengagement (mind-wandering). |
| **z_delta** | Delta Power | 0.5–4 Hz | Primary marker of **sleep pressure** and fatigue. |
| **z_gamma** | Gamma Power | 30–45 Hz | Associated with feature binding and **high-level cognitive processing**. |
| **z_tbr** | Theta/Beta Ratio | Ratio ($\theta / \beta$) | Validated marker for **ADHD** and mind-wandering (van Son et al., 2019). |
| **z_pe** | Permutation Entropy | Complexity Metric | Measures signal randomness; drops during **disengagement** or drowsiness. |
| **z_wpe** | Weighted PE | Weighted Complexity | Variation of PE that accounts for amplitude information. |
| **z_mse** | Multiscale Entropy | Temporal Complexity | Measures complexity across multiple time scales; drops in **overload**. |
| **z_lz** | Lempel-Ziv Complexity | Compressibility | Measures information density; correlates with **consciousness level**. |
| **z_pac** | Phase-Amp Coupling | Cross-Frequency | Synchronization between Theta phase and Gamma amplitude. |
| **z_at_ratio** | Alpha/Theta Ratio | Ratio ($\alpha / \theta$) | Distinguishes relaxation from drowsy fatigue. |
| **z_alpha_rel**| Relative Alpha | $\alpha$ / Total Power | Normalized measure of alpha dominance (idling). |
| **z_frontal_asym** | Frontal Asymmetry | (Fp1 - Fp2) / Sum | Index of **emotional valence** (approach vs. withdrawal). |

---

## Intensity Metric (0-100)

The system computes a composite **Intensity Score** to represent overall cognitive engagement/load on a 0–100 scale.

**Formula Structure:**
The score is a weighted sum of normalized z-scores, clipped to the [0, 1] range before weighting:

$$
\text{Intensity} = 100 \times \sum (w_i \cdot \text{norm}(z_i))
$$

Where components include:
- **Positive contributors:** Beta, Gamma, LZ, PE, PAC, MSE (Higher = Higher Intensity)
- **Negative contributors:** Theta, TBR (Higher = Lower Intensity)

*Note: Weights were manually set using neuroscience priors and iteratively adjusted by inspecting how intensity distributions differed between Optimal and drift states in the lab dataset; no automated fitting step was used.*

---

## Neuroscience Grounding

### 1. Mind-Wandering (Distraction)
**Signature:** High Theta/Beta Ratio + Low Entropy
- **van Son et al. (2019):** Demonstrated that frontal Theta/Beta ratio is a reliable detector of mind-wandering during sustained attention tasks.
- **Kam et al. (2011):** Showed reduced sensory evoked potentials and signal complexity during mind-wandering.

### 2. Fatigue (Metabolic Exhaustion)
**Signature:** High Alpha + High Delta
- **Tran et al. (2020):** Identified theta and delta increases in frontal regions as reliable indicators of fatigue transition.
- **Wascher et al. (2014):** Linked posterior alpha increases to "mental fatigue" and resource depletion.

### 3. Overload (System Failure)
**Signature:** Extreme Frontal Theta + Beta Collapse
- **Ishii et al. (2014):** Frontal midline theta scales with task difficulty but collapses ("overload") when capacity is exceeded.
