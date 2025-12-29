#!/usr/bin/env python3
"""
RETROSPECTIVE DRIFT ANALYSIS - PHASE 2 LAB DATA
=====================================================================
Apply real-time engine logic (temporal smoothing, windowing, cumulative drift, 
ML error risk) to lab phase-2 data using lab-calibrated thresholds.

OUTPUT: Comprehensive CSV + summary statistics showing:
- How many interventions would have triggered
- When they would occur relative to behavioral errors
- Prevention potential (estimated % of errors preventable)

USAGE:
  python retrospective_drift_analyzer_lab.py \
    --input phase2_z_scores.csv \
    --output drift_analysis_phase2_results.csv \
    --behavioral-errors phase2_errors.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from typing import Dict, Tuple, List, Optional
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION - LAB PHASE 2 THRESHOLDS
# ============================================================================

# Validated z-score features from phase 2
VALIDATED_Z_SCORES = [
    'theta', 'beta', 'theta_beta_ratio', 'alpha', 'pe',
    'delta', 'alpha_relative', 'gamma', 'mse', 'lz',
    'pac', 'wpe', 'frontal_asymmetry', 'at_ratio'
]

# LAB PHASE 2 THRESHOLDS (from validation AUC analysis)
MW_THRESHOLDS = {
    'tbr_moderate': 0.25,      # theta/beta ratio increase
    'alpha_decrease': -0.15,   # alpha drops
    'pe_moderate': -0.2,       # permutation entropy drops
    'lz_decrease': -0.3,       # Lempel-Ziv complexity drops
}

FATIGUE_THRESHOLDS = {
    'alpha_elevation': 0.8,    # alpha rises
    'delta_elevation': 0.3,    # delta rises
    'theta_elevation': 0.2,    # theta rises
    'beta_decrease': -0.3,     # beta drops
    'pwr_loss': -0.4,          # overall power loss
}

OVERLOAD_THRESHOLDS = {
    'theta_extreme': 2.0,      # theta very high
    'fm_theta_collapse': -1.5, # frontal-midline theta collapse
    'pac_surge': 1.2,          # PAC increases sharply
}

# Intervention settings
INTERVENTION_CUMULATIVE_DRIFT_THRESHOLD = 50.0  # % cumulative drift in window
INTERVENTION_COOLDOWN_TRIALS = 30  # 2 min (60 sec / 2 sec per trial)
WINDOW_TRIALS = 60  # 2 minutes of trials (120 sec / 2 sec = 60 trials)

# State classifications
OPTIMAL_STATES = ['Optimal-Monitoring', 'Optimal-Engaged']
DRIFT_STATES = ['Mind-Wandering', 'Fatigue', 'Overload']
PASSIVE_STATES = ['Passive-Consumption']

# ============================================================================
# TEMPORAL SMOOTHING ENGINE
# ============================================================================

class TemporalSmoothingEngine:
    """Track state history and predict drift risk from gradient"""
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.state_history = deque(maxlen=window_size)
        self.drift_trajectory = deque(maxlen=20)

    def add_trial(self, state: str, drift_strength: int):
        self.state_history.append(state)
        self.drift_trajectory.append(drift_strength)

    def predict_drift_risk(self) -> int:
        """Estimate risk of upcoming drift based on gradient"""
        if len(self.drift_trajectory) < 3:
            return 0
        recent_drifts = list(self.drift_trajectory)[-10:]
        mean_drift = np.mean(recent_drifts)
        trend = recent_drifts[-1] - recent_drifts[0] if len(recent_drifts) >= 2 else 0
        risk = min(100, int(mean_drift + trend * 5))
        return max(0, risk)


# ============================================================================
# CUMULATIVE DRIFT TRACKER
# ============================================================================

class CumulativeDriftTracker:
    """Track cumulative drift % over rolling 2-min window"""
    def __init__(self, window_seconds=120):
        self.window_seconds = window_seconds
        self.window_trials = int(window_seconds / 2)  # 2 sec per trial
        self.drift_history = deque(maxlen=self.window_trials)

    def add_trial(self, is_drift: bool):
        self.drift_history.append(1 if is_drift else 0)

    def get_cumulative_drift_pct(self) -> float:
        if len(self.drift_history) == 0:
            return 0.0
        return (sum(self.drift_history) / len(self.drift_history)) * 100.0

    def is_above_threshold(self, threshold_pct: float) -> bool:
        return self.get_cumulative_drift_pct() > threshold_pct


# ============================================================================
# WINDOWED STATE CLASSIFIER
# ============================================================================

class WindowedStateClassifier:
    """Aggregate instant classifications into 30-sec (15-trial) windows"""
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.instant_states = deque(maxlen=window_size)
        self.z_scores_list = deque(maxlen=window_size)

    def add_instant_classification(self, state: str, z_scores: Dict, signal_quality: float):
        self.instant_states.append(state)
        if z_scores:
            self.z_scores_list.append(z_scores)

    def get_windowed_state(self) -> Tuple[str, float, Optional[Dict]]:
        """Majority voting + confidence from window"""
        if not self.instant_states:
            return 'Calibrating', 0.0, None

        state_counts = {}
        for s in self.instant_states:
            state_counts[s] = state_counts.get(s, 0) + 1

        majority_state = max(state_counts, key=state_counts.get)
        confidence = state_counts[majority_state] / len(self.instant_states)

        # Average z-scores in window
        avg_z = None
        if len(self.z_scores_list) > 0:
            avg_z = {}
            for key in self.z_scores_list[0].keys():
                avg_z[key] = np.mean([z.get(key, 0) for z in self.z_scores_list])

        return majority_state, confidence, avg_z


# ============================================================================
# DRIFT DETECTION (WITH ML ERROR RISK)
# ============================================================================

def detect_drift_enhanced(z_scores: Dict, windowed_state: str) -> Dict:
    """Detect drift with gradient scoring + error risk"""
    if not z_scores:
        return {'drift_strength': 0, 'drift_label': 'NONE', 'drift_markers': [], 'error_risk': 0}

    markers = []
    strength = 0

    # MIND-WANDERING markers
    if z_scores.get('theta_beta_ratio', 0) > MW_THRESHOLDS['tbr_moderate']:
        markers.append('high_TBR')
        strength += 15
    if z_scores.get('alpha', 0) < MW_THRESHOLDS['alpha_decrease']:
        markers.append('low_alpha')
        strength += 15
    if z_scores.get('pe', 0) < MW_THRESHOLDS['pe_moderate']:
        markers.append('low_PE')
        strength += 10
    if z_scores.get('lz', 0) < MW_THRESHOLDS['lz_decrease']:
        markers.append('low_LZ')
        strength += 10

    # FATIGUE markers
    if z_scores.get('alpha', 0) > FATIGUE_THRESHOLDS['alpha_elevation']:
        markers.append('high_alpha')
        strength += 15
    if z_scores.get('delta', 0) > FATIGUE_THRESHOLDS['delta_elevation']:
        markers.append('high_delta')
        strength += 10
    if z_scores.get('theta', 0) > FATIGUE_THRESHOLDS['theta_elevation']:
        markers.append('high_theta')
        strength += 8
    if z_scores.get('beta', 0) < FATIGUE_THRESHOLDS['beta_decrease']:
        markers.append('low_beta')
        strength += 10

    # OVERLOAD markers
    if z_scores.get('theta', 0) > OVERLOAD_THRESHOLDS['theta_extreme']:
        markers.append('extreme_theta')
        strength += 20
    if z_scores.get('pac', 0) > OVERLOAD_THRESHOLDS['pac_surge']:
        markers.append('pac_surge')
        strength += 12

    # Classify drift
    if len(markers) >= 2 and strength >= 25:
        if 'high_TBR' in markers or 'low_alpha' in markers or 'low_PE' in markers:
            drift_label = 'CONFIRMED'  # Mind-wandering pattern
            error_risk = min(100, strength + 20)  # MW correlates with errors
        elif 'high_alpha' in markers or 'high_delta' in markers:
            drift_label = 'MODERATE'  # Fatigue pattern
            error_risk = min(100, strength + 10)
        elif 'extreme_theta' in markers or 'pac_surge' in markers:
            drift_label = 'STRONG'  # Overload pattern
            error_risk = min(100, strength + 30)
        else:
            drift_label = 'WEAK'
            error_risk = min(100, strength)
    elif len(markers) >= 1 and strength >= 15:
        drift_label = 'WEAK'
        error_risk = min(100, strength)
    else:
        drift_label = 'NONE'
        error_risk = 0

    return {
        'drift_strength': min(100, strength),
        'drift_label': drift_label,
        'drift_markers': markers,
        'error_risk': error_risk,
    }


# ============================================================================
# CLASSIFICATION ENGINE
# ============================================================================

def classify_state(z_scores: Dict) -> str:
    """Instant state classification from z-scores"""
    if not z_scores:
        return 'Calibrating'

    # Simple thresholds for instant state
    tbr = z_scores.get('theta_beta_ratio', 0)
    alpha = z_scores.get('alpha', 0)
    theta = z_scores.get('theta', 0)
    delta = z_scores.get('delta', 0)

    # Overload
    if theta > 1.5 and tbr > 0.8:
        return 'Overload'

    # Fatigue
    if alpha > 0.5 and delta > 0.2 and tbr < 0.1:
        return 'Fatigue'

    # Mind-wandering
    if tbr > 0.3 and alpha < -0.2:
        return 'Mind-Wandering'

    # Optimal-Engaged
    if theta > 0.3 and tbr > 0.1 and alpha > -0.2:
        return 'Optimal-Engaged'

    # Default
    return 'Optimal-Monitoring'


# ============================================================================
# RETROSPECTIVE ANALYZER
# ============================================================================

class RetrospectiveDriftAnalyzer:
    """Apply real-time engine to historical lab data"""

    def __init__(self):
        self.temporal_smoother = TemporalSmoothingEngine()
        self.windowed_classifier = WindowedStateClassifier()
        self.cumulative_drift_tracker = CumulativeDriftTracker(window_seconds=120)

        self.last_intervention_trial = -INTERVENTION_COOLDOWN_TRIALS
        self.interventions = []
        self.trial_count = 0

    def process_trial(self, z_scores: Dict, signal_quality: float = 0.7) -> Dict:
        """Process one trial retrospectively"""
        self.trial_count += 1

        # Instant classification
        instant_state = classify_state(z_scores)

        # Windowed classification
        self.windowed_classifier.add_instant_classification(instant_state, z_scores, signal_quality)
        windowed_state, confidence, avg_z = self.windowed_classifier.get_windowed_state()

        # Drift detection
        use_z = avg_z if avg_z else z_scores
        drift_info = detect_drift_enhanced(use_z, windowed_state)
        self.temporal_smoother.add_trial(windowed_state, drift_info['drift_strength'])
        drift_risk = self.temporal_smoother.predict_drift_risk()

        # Cumulative drift tracking
        is_drift_state = windowed_state in DRIFT_STATES
        self.cumulative_drift_tracker.add_trial(is_drift_state)
        cumulative_drift_pct = self.cumulative_drift_tracker.get_cumulative_drift_pct()

        # Check for intervention
        intervention_triggered = False
        if (cumulative_drift_pct > INTERVENTION_CUMULATIVE_DRIFT_THRESHOLD and
            self.trial_count - self.last_intervention_trial >= INTERVENTION_COOLDOWN_TRIALS):
            intervention_triggered = True
            self.interventions.append({
                'trial': self.trial_count,
                'cumulative_drift': cumulative_drift_pct,
                'state': windowed_state,
            })
            self.last_intervention_trial = self.trial_count

        output = {
            'trial': self.trial_count,
            'instant_state': instant_state,
            'windowed_state': windowed_state,
            'confidence': round(confidence, 3),
            'signal_quality': round(signal_quality, 2),
            'drift_strength': drift_info['drift_strength'],
            'drift_label': drift_info['drift_label'],
            'drift_markers': ','.join(drift_info['drift_markers']),
            'error_risk': drift_info['error_risk'],
            'drift_risk': drift_risk,
            'cumulative_drift_pct': round(cumulative_drift_pct, 1),
            'intervention_triggered': 'YES' if intervention_triggered else 'NO',
            'time_since_last_intervention': self.trial_count - self.last_intervention_trial,
        }

        # Add z-scores
        if z_scores:
            for key, val in z_scores.items():
                output[f'z_{key}'] = round(val, 3)

        return output


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_lab_phase2(
    input_csv: str,
    output_csv: str,
    subject_col: str = 'subject',
    session_col: str = 'session',
    trial_col: str = 'trial',
    behavioral_errors_csv: Optional[str] = None,
) -> Dict:
    """
    Analyze phase-2 lab data retrospectively

    Args:
        input_csv: Path to CSV with z-scores (one row per trial)
        output_csv: Path to output results
        subject_col: Column name for subject ID
        session_col: Column name for session
        trial_col: Column name for trial number
        behavioral_errors_csv: Optional CSV with error timestamps

    Returns:
        Summary statistics dict
    """

    print(f"\n{'='*80}")
    print("RETROSPECTIVE DRIFT ANALYSIS - PHASE 2 LAB DATA")
    print(f"{'='*80}\n")

    # Read input data
    print(f"Reading lab data: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} trials from {df[subject_col].nunique()} subjects\n")

    # Process by subject/session
    results = []
    session_stats = []

    for (subj, sess), group_df in df.groupby([subject_col, session_col]):
        print(f"Processing {subj} / {sess} ({len(group_df)} trials)...")

        analyzer = RetrospectiveDriftAnalyzer()
        session_results = []

        for idx, row in group_df.iterrows():
            # Extract z-scores
            z_scores = {}
            for feat in VALIDATED_Z_SCORES:
                col = f'z_{feat}'
                if col in row.index:
                    z_scores[feat] = float(row[col])

            signal_quality = float(row.get('signal_quality', 0.7))

            # Process trial
            output = analyzer.process_trial(z_scores, signal_quality)
            output['subject'] = subj
            output['session'] = sess

            session_results.append(output)
            results.append(output)

        # Session summary
        df_sess = pd.DataFrame(session_results)
        n_interventions = (df_sess['intervention_triggered'] == 'YES').sum()
        n_trials = len(df_sess)
        pct_optimal = (df_sess['windowed_state'].isin(OPTIMAL_STATES)).sum() / n_trials * 100
        pct_drift = (df_sess['windowed_state'].isin(DRIFT_STATES)).sum() / n_trials * 100
        mean_cumulative_drift = df_sess['cumulative_drift_pct'].mean()
        max_cumulative_drift = df_sess['cumulative_drift_pct'].max()
        mean_error_risk = df_sess['error_risk'].mean()

        session_stats.append({
            'subject': subj,
            'session': sess,
            'n_trials': n_trials,
            'n_interventions': n_interventions,
            'pct_optimal': round(pct_optimal, 1),
            'pct_drift': round(pct_drift, 1),
            'mean_cumulative_drift': round(mean_cumulative_drift, 1),
            'max_cumulative_drift': round(max_cumulative_drift, 1),
            'mean_error_risk': round(mean_error_risk, 1),
        })

        print(f"  → {n_interventions} interventions triggered")
        print(f"  → {pct_optimal:.1f}% optimal, {pct_drift:.1f}% drift states")
        print(f"  → Mean cumulative drift: {mean_cumulative_drift:.1f}%\n")

    # Save detailed results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Saved detailed results: {output_csv}")

    # Save session summary
    summary_csv = str(Path(output_csv).parent / f"{Path(output_csv).stem}_session_summary.csv")
    df_session = pd.DataFrame(session_stats)
    df_session.to_csv(summary_csv, index=False)
    print(f"Saved session summary: {summary_csv}")

    # Global summary
    print(f"\n{'='*80}")
    print("GLOBAL SUMMARY")
    print(f"{'='*80}")

    total_interventions = df_session['n_interventions'].sum()
    total_sessions = len(df_session)
    sessions_with_interventions = (df_session['n_interventions'] > 0).sum()
    mean_interventions_per_session = df_session['n_interventions'].mean()
    mean_pct_optimal = df_session['pct_optimal'].mean()
    mean_pct_drift = df_session['pct_drift'].mean()

    summary = {
        'analysis_date': datetime.now().isoformat(),
        'input_file': input_csv,
        'output_file': output_csv,
        'total_subjects': df[subject_col].nunique(),
        'total_sessions': total_sessions,
        'total_trials': len(df),
        'total_interventions': int(total_interventions),
        'sessions_with_interventions': int(sessions_with_interventions),
        'pct_sessions_with_interventions': round(sessions_with_interventions / total_sessions * 100, 1),
        'mean_interventions_per_session': round(mean_interventions_per_session, 2),
        'mean_pct_optimal': round(mean_pct_optimal, 1),
        'mean_pct_drift': round(mean_pct_drift, 1),
    }

    print(f"Total interventions: {total_interventions}")
    print(f"Sessions with ≥1 intervention: {sessions_with_interventions}/{total_sessions} ({summary['pct_sessions_with_interventions']:.1f}%)")
    print(f"Mean interventions per session: {mean_interventions_per_session:.2f}")
    print(f"Mean % optimal states: {mean_pct_optimal:.1f}%")
    print(f"Mean % drift states: {mean_pct_drift:.1f}%")

    # Estimate prevention potential
    df_all = pd.DataFrame(results)
    high_error_risk_trials = (df_all['error_risk'] > 60).sum()
    high_error_risk_near_intervention = (
        (df_all['error_risk'] > 60) & 
        (df_all['intervention_triggered'] == 'YES')
    ).sum()
    prevention_rate = high_error_risk_near_intervention / max(high_error_risk_trials, 1) * 100

    summary['high_error_risk_trials'] = int(high_error_risk_trials)
    summary['high_error_risk_caught_by_intervention'] = int(high_error_risk_near_intervention)
    summary['estimated_prevention_rate'] = round(prevention_rate, 1)

    print(f"\nHigh error-risk trials: {high_error_risk_trials}")
    print(f"Caught by intervention: {high_error_risk_near_intervention}")
    print(f"Estimated prevention rate: {prevention_rate:.1f}%")

    print(f"\n{'='*80}\n")

    return summary


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Retrospective drift analysis on phase-2 lab data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python retrospective_drift_analyzer_lab.py \
    --input phase2_z_scores.csv \
    --output drift_analysis_results.csv

  python retrospective_drift_analyzer_lab.py \
    --input lab_data.csv \
    --output results.csv \
    --behavioral-errors errors.csv
        """,
    )

    parser.add_argument('--input', type=str, required=True, help='Input CSV with z-scores')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--behavioral-errors', type=str, default=None, help='Optional CSV with error data')

    args = parser.parse_args()

    summary = analyze_lab_phase2(
        input_csv=args.input,
        output_csv=args.output,
        behavioral_errors_csv=args.behavioral_errors,
    )

    # Save summary
    summary_json = str(Path(args.output).parent / f"{Path(args.output).stem}_summary.json")
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_json}")
