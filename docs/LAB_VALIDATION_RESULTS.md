# Lab Validation Results (Phase 2)

This document summarizes the comprehensive validation of the cognitive monitoring system on the "Clean-16" dataset (16 unique subjects, 48 sessions).

## 1. Information Geometry Analysis
*Validating that the identified states are distinct mathematical manifolds.*

**File:** `ig_metrics.csv`

| State | Mahalanobis Dist | KL Divergence | Riemannian Dist | Interpretation |
|:---|:---|:---|:---|:---|
| **Optimal-Monitoring** | 0.00 | 0.00 | 0.00 | *Reference State* |
| **Mind-Wandering** | 2.33 | 9.41 | 1.48 | *Mild Deviation* |
| **Optimal-Engaged** | 3.14 | 56.92 | 1.73 | *Active Shift* |
| **Fatigue** | 11.23 | 1776.14 | 2.51 | *Significant Drift* |
| **Overload** | 13.71 | 127.06 | 6.10 | *System Failure* |

**Key Finding:** Drift states (Fatigue/Overload) are mathematically distant from the optimal manifold, confirming that our classifier is detecting distinct neurological modes, not just random noise.

---

## 2. Machine Learning Predictive Validity
*Does the EEG state predict actual task performance?*

We trained a Gradient Boosting model to predict behavioral errors using only the 14 EEG biomarkers.

- **Model:** GradientBoostingClassifier
- **AUC:** 0.6784 (5-fold cross-validation)
- **Accuracy:** 68.30%
- **Sensitivity:** 90.2% (High detection of errors)

**Top Predictive Features:**
1. `z_delta` (Sleep pressure)
2. `z_theta` (Mental fatigue)
3. `z_alpha_relative` (Disengagement)

---

## 3. Impact of Drift on Performance
*Quantifying the cost of drift.*

**File:** `sessions_complete.csv` (aggregated analysis)

| Comparison Level | Metric | Low Drift (<10%) | High Drift (>15%) | Impact |
|:---|:---|:---|:---|:---|
| **Trial-Level** | Error Risk | 44.2% | 49.4% | **+11.7% Risk** |
| **Window-Level** | Efficiency | 62.7 | 48.3 | **-22.9% Efficiency** |
| **Subject-Level** | Error Risk | 40.2% | 46.9% | **+16.9% Risk** |

**Conclusion:** Users in "High Drift" states are significantly more prone to errors and slower reaction times.

---

## 4. State Stability (Markov Analysis)
*How "sticky" are these states?*

**File:** `transition_matrix.csv`

- **Optimal States** are highly stable ($P_{stay} \approx 90-91\%$).
- **Fatigue** is sticky ($P_{stay} \approx 76\%$), meaning once you are fatigued, it is hard to snap out of it without intervention.
- **Overload** is transient ($P_{stay} \approx 40\%$), suggesting it is an acute crisis that resolves quickly (either by recovery or giving up).

---

## 5. Artifacts
Raw data tables supporting this analysis are available in the `docs/data/` folder:
- `comprehensive_summary_report.txt`: Full statistical breakdown.
- `transition_matrix.csv`: State transition probabilities.
- `ig_metrics.csv`: Manifold geometry distances.
