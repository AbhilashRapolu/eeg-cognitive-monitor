#!/usr/bin/env python3
"""
COGNITIVE MONITOR v7.3 - PRODUCTION ENGINE (COMPLETE)
=====================================================================
âœ… 14 EXACT FEATURES + ENTROPY + IG + DRIFT + TASK REC
âœ… WINDOWED CLASSIFICATION (30-sec aggregation)
âœ… REAL CONFIDENCE SCORES + SIGNAL QUALITY
âœ… INTENSITY + ENGAGEMENT METRICS
âœ… ENHANCED DRIFT DETECTION (gradient scoring)
âœ… REAL MUSE INTEGRATION + LSL STREAMING
âœ… ADAPTIVE ARTIFACT REJECTION + BASELINE PERSISTENCE

This is the ENGINE ONLY - stateless, reusable, testable.
Import this in your session runner.

USAGE:
  from real_time_monitor_v7_3 import CognitiveMonitorV7_3, extract_features, is_epoch_clean
  monitor = CognitiveMonitorV7_3(task_type="learning")
  features = extract_features(epoch_data, sfreq=256)
  output = monitor.process_trial(features)
"""

import numpy as np
import pandas as pd
import pickle
from collections import deque
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import joblib
from scipy.signal import welch, butter, filtfilt, hilbert
from scipy.stats import entropy as scipy_entropy, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
import warnings
import time
import json  # ADD THIS at top with other imports

warnings.filterwarnings('ignore')

try:
    from pylsl import StreamInlet, resolve_streams
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_RATE = 256
EPOCH_DURATION = 2.0
AMP_THRESHOLD_UV = 500.0
KURTOSIS_THRESHOLD = 8.0
USE_ADAPTIVE_ARTIFACT = True

VALIDATED_Z_SCORES = [
    'theta', 'beta', 'theta_beta_ratio', 'alpha', 'pe',
    'delta', 'alpha_relative', 'gamma', 'mse', 'lz',
    'pac', 'wpe', 'frontal_asymmetry', 'at_ratio',
    'theta_frontal',  # NEW
]

MW_THRESHOLDS = {
    'tbr_moderate': 0.25,
    'alpha_decrease': -0.15,
    'pe_moderate': -0.2,
}

FATIGUE_THRESHOLDS = {
    'alpha_elevation': 0.8,
    'delta_elevation': 0.3,
    'theta_elevation': 0.2,
    'beta_decrease': -0.3,
}

OVERLOAD_THRESHOLDS = {
    'theta_extreme': 2.0,
    'fm_theta_collapse': -1.5,
}


OPTIMAL_STATES = ['Optimal-Engaged', 'Optimal-Monitoring']
DRIFT_STATES = ['Mind-Wandering', 'Fatigue', 'Overload']
# NEW: Passive states
PASSIVE_STATES = ['Passive-Consumption']
ALL_STATES = OPTIMAL_STATES + DRIFT_STATES + PASSIVE_STATES
PCA_WEIGHTS = {
    'w_mahal': 0.527,
    'w_kl': 0.189,
    'w_intensity': 0.284
}

TEMPORAL_WINDOW_SIZE = 10

# ============================================================================
# ENTROPY ENGINE
# ============================================================================

class EntropyEngine:
    @staticmethod
    def permutation_entropy(signal, order=3, delay=1):
        try:
            signal = np.asarray(signal, dtype=np.float64)
            signal = np.nan_to_num(signal, nan=0.0)
            n = len(signal)
            if n < delay * (order - 1) + 1:
                return 0.0
            L = order * delay
            windows = sliding_window_view(signal, window_shape=L)
            indices = np.arange(0, L, delay)
            sel = windows[:, indices]
            perms = np.argsort(sel, axis=1)
            perms_flat = perms.view(np.int64).reshape(perms.shape[0], -1)
            _, counts = np.unique(perms_flat, axis=0, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            return float(np.clip(entropy, 0, 10))
        except:
            return 0.0

    @staticmethod
    def weighted_permutation_entropy(signal, order=3, delay=1):
        try:
            signal = np.asarray(signal, dtype=np.float64)
            signal = np.nan_to_num(signal, nan=0.0)
            n = len(signal)
            if n < delay * (order - 1) + 1:
                return 0.0
            L = order * delay
            windows = sliding_window_view(signal, window_shape=L)
            indices = np.arange(0, L, delay)
            sel = windows[:, indices]
            perms = np.argsort(sel, axis=1)
            weights = np.std(sel, axis=1)
            wsum = weights.sum()
            if wsum == 0:
                return 0.0
            perms_flat = perms.view(np.int64).reshape(perms.shape[0], -1)
            _, inv = np.unique(perms_flat, axis=0, return_inverse=True)
            weighted_counts = np.zeros(np.max(inv) + 1, dtype=np.float64)
            for k, w in enumerate(weights):
                weighted_counts[inv[k]] += w
            probs = weighted_counts / (wsum + 1e-12)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            return float(np.clip(entropy, 0, 10))
        except:
            return 0.0

    @staticmethod
    def mutual_information(signal_x, signal_y, bins=8):
        try:
            signal_x = np.asarray(signal_x, dtype=np.float64)
            signal_y = np.asarray(signal_y, dtype=np.float64)
            signal_x = np.nan_to_num(signal_x, nan=0.0)
            signal_y = np.nan_to_num(signal_y, nan=0.0)
            if len(signal_x) != len(signal_y) or len(signal_x) < 50:
                return 0.0
            signal_x = (signal_x - np.mean(signal_x)) / (np.std(signal_x) + 1e-10)
            signal_y = (signal_y - np.mean(signal_y)) / (np.std(signal_y) + 1e-10)
            hist_2d, _, _ = np.histogram2d(signal_x, signal_y, bins=bins)
            pxy = hist_2d / (hist_2d.sum() + 1e-12)
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            px_py = px[:, None] * py[None, :]
            nzs = pxy > 0
            if np.sum(nzs) == 0:
                return 0.0
            mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / (px_py[nzs] + 1e-12)))
            return float(np.clip(mi, 0, 2.0))
        except:
            return 0.0

    @staticmethod
    def lempel_ziv_complexity(signal):
        try:
            binary = (signal > np.median(signal)).astype(int)
            binary_str = ''.join(binary.astype(str))
            n = len(binary_str)
            if n <= 1:
                return 1.0
            i, k, l = 0, 1, 1
            c, k_max = 1, 1
            while l + k <= n:
                if binary_str[i + k - 1] == binary_str[l + k - 1]:
                    k += 1
                else:
                    k_max = max(k, k_max)
                    i += 1
                    if i == l:
                        c += 1
                        l += k_max
                        if l + 1 > n:
                            break
                        else:
                            i, k, k_max = 0, 1, 1
                    else:
                        k = 1
            return float(c / (n / np.log2(n)))
        except:
            return 0.0

    @staticmethod
    def compute_pac(signal, sfreq):
        try:
            if len(signal) < 100:
                return 0.0
            nyq = sfreq / 2
            b_theta, a_theta = butter(2, [4 / nyq, 8 / nyq], btype='band')
            b_gamma, a_gamma = butter(2, [30 / nyq, 50 / nyq], btype='band')
            theta_sig = filtfilt(b_theta, a_theta, signal)
            gamma_sig = filtfilt(b_gamma, a_gamma, signal)
            theta_phase = np.angle(hilbert(theta_sig))
            gamma_amp = np.abs(hilbert(gamma_sig))
            pac_value = np.abs(np.corrcoef(np.cos(theta_phase), gamma_amp)[0, 1])
            return 0.0 if np.isnan(pac_value) or np.isinf(pac_value) else pac_value
        except:
            return 0.0

    @staticmethod
    def multiscale_entropy(signal, max_scale=3):
        try:
            signal = np.asarray(signal, dtype=np.float64)
            mse = []
            for scale in range(1, max_scale + 1):
                if len(signal) < scale * 10:
                    mse.append(0.0)
                    continue
                coarse = signal[len(signal) % scale:].reshape(-1, scale).mean(axis=1)
                if len(coarse) < 3:
                    mse.append(0.0)
                else:
                    pe = EntropyEngine.permutation_entropy(coarse, order=3)
                    mse.append(pe)
            return float(np.mean(mse)) if mse else 0.0
        except:
            return 0.0

    @staticmethod
    def compute_frontal_asymmetry(task_data, sfreq):
        try:
            if task_data.shape[0] < 2:
                return 0.0
            ch_fp1, ch_fp2 = task_data[0], task_data[1]
            freqs, psd_fp1 = welch(ch_fp1, fs=sfreq, nperseg=256)
            _, psd_fp2 = welch(ch_fp2, fs=sfreq, nperseg=256)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            alpha_fp1 = psd_fp1[alpha_mask].mean()
            alpha_fp2 = psd_fp2[alpha_mask].mean()
            if alpha_fp1 + alpha_fp2 > 0:
                asym = (alpha_fp1 - alpha_fp2) / (alpha_fp1 + alpha_fp2)
                return float(np.clip(asym, -1, 1))
            else:
                return 0.0
        except:
            return 0.0

    @staticmethod
    def compute_alpha_relative(task_data, sfreq):
        try:
            if task_data.ndim == 1:
                task_data = task_data[np.newaxis, :]
            alpha_power = 0
            total_power = 0
            bands = {'alpha': (8, 13), 'theta': (4, 8), 'beta': (13, 30),
                     'gamma': (30, 45), 'delta': (0.5, 4)}
            freqs, psd = welch(task_data, fs=sfreq, nperseg=256)
            for band_name, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs <= hi)
                band_power = psd[:, mask].mean()
                total_power += band_power
                if band_name == 'alpha':
                    alpha_power = band_power
            if total_power > 0:
                return float(alpha_power / total_power)
            else:
                return 0.0
        except:
            return 0.0

# ============================================================================
# ARTIFACT REJECTION
# ============================================================================

def is_epoch_clean(epoch_data, amp_thresh=500.0, kurt_thresh=8.0, use_adaptive=True):
    if use_adaptive:
        return is_epoch_clean_adaptive(epoch_data)
    else:
        if np.any(np.abs(epoch_data) > amp_thresh):
            return False, "amplitude"
        if np.any(kurtosis(epoch_data, axis=1) > kurt_thresh):
            return False, "kurtosis"
        return True, "clean"

def is_epoch_clean_adaptive(epoch_data, percentile=99):
    try:
        all_amps = np.abs(epoch_data).flatten()
        amp_thresh = np.percentile(all_amps, percentile)
        if np.any(np.abs(epoch_data) > amp_thresh * 2):
            return False, "amplitude_adaptive"
        if np.any(kurtosis(epoch_data, axis=1) > 8.0):
            return False, "kurtosis"
        return True, "clean"
    except:
        return True, "clean"

# ============================================================================
# LSL CONNECTION
# ============================================================================

def connect_muse():
    if not LSL_AVAILABLE:
        raise RuntimeError("âŒ pylsl not installed. Install with: pip install pylsl")
    print("ðŸ” Looking for Muse EEG stream...")
    streams = resolve_streams()
    eeg_streams = [s for s in streams if s.type() == 'EEG']
    if not eeg_streams:
        raise RuntimeError("âŒ No Muse EEG stream found. Is it on and paired?")
    inlet = StreamInlet(eeg_streams[0])
    print(f"âœ… Connected to Muse: {eeg_streams[0].name()}")
    print(f"   Channels: {inlet.channel_count}")
    print(f"   Sample rate: {inlet.info().nominal_srate()} Hz\n")
    return inlet


def collect_epoch(inlet, duration=2.0, sfreq=256, max_retries=2):
    n_samples_needed = int(duration * sfreq)
    min_samples = int(n_samples_needed * 0.85)

    for attempt in range(max_retries + 1):
        inlet.pull_chunk(timeout=0.0, max_samples=10000)

        all_samples = []
        start_time = time.time()

        # ✅ FIXED: Dynamic timeout, no sleep
        while time.time() - start_time < duration:
            remaining = duration - (time.time() - start_time)
            chunk, timestamps = inlet.pull_chunk(
                timeout=min(remaining, 0.2),
                max_samples=256
            )
            if chunk:
                all_samples.extend(chunk)
            # ✅ REMOVED: time.sleep(0.05) - let timeout handle timing

        # Check if we got enough samples
        if len(all_samples) >= min_samples:
            # Trim or pad to exact length
            if len(all_samples) > n_samples_needed:
                epoch_data = all_samples[:n_samples_needed]
            elif len(all_samples) < n_samples_needed:
                # Pad with last sample repeated (better than zeros)
                shortfall = n_samples_needed - len(all_samples)
                last_sample = all_samples[-1] if all_samples else [0.0] * 4
                epoch_data = all_samples + [last_sample] * shortfall
            else:
                epoch_data = all_samples

            return np.array(epoch_data).T  # Shape: (n_channels, n_samples)

        # If insufficient, retry
        if attempt < max_retries:
            print(f"⚠️ Retry {attempt + 1}/{max_retries}: Got {len(all_samples)}/{n_samples_needed} samples")
            time.sleep(0.5)  # Brief pause before retry

    # Failed after all retries
    print(f"❌ Insufficient epoch after {max_retries} retries: {len(all_samples)}/{n_samples_needed} samples")
    return None


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(epoch_data, sfreq, entropy_engine=None):
    try:
        mid_point = epoch_data.shape[1] // 2
        task_data = epoch_data[:, mid_point:]
        features = {}
        n_channels = epoch_data.shape[0]
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
                 'beta': (13, 30), 'gamma': (30, 45)}
        nperseg_task = min(256, max(16, task_data.shape[1] // 2))
        freqs_task, psd_task = welch(task_data, fs=sfreq, nperseg=nperseg_task)

        for band_name, (lo, hi) in bands.items():
            mask_task = (freqs_task >= lo) & (freqs_task <= hi)
            bp_values = []
            for ch_idx in range(n_channels):
                bp = psd_task[ch_idx, mask_task].mean()
                features[f'task_bp_{band_name}_ch{ch_idx}'] = bp
                bp_values.append(bp)
            # âœ… FIXED: Store as just 'theta', 'beta', 'alpha', 'gamma', 'delta'
            features[band_name] = np.mean(bp_values)  # â† Remove 'task_bp_' prefix

        theta_bp = features.get('task_bp_theta', 0.5)
        beta_bp = features.get('task_bp_beta', 0.5)
        features['tbr_raw'] = theta_bp / (beta_bp + 1e-10)

        theta_frontal = np.mean([
            features.get('task_bp_theta_ch0', 0),  # Fp1
            features.get('task_bp_theta_ch1', 0),  # Fp2
        ])

        # Temporal theta (drowsiness, fatigue)
        theta_temporal = np.mean([
            features.get('task_bp_theta_ch2', 0),  # TP9
            features.get('task_bp_theta_ch3', 0),  # TP10
        ])

        features['theta_frontal'] = theta_frontal
        features['theta_temporal'] = theta_temporal
        features['theta'] = theta_temporal  # Keep this for backward compatibility


        for ch_idx in range(n_channels):
            features[f'pac_task_ch{ch_idx}'] = EntropyEngine.compute_pac(task_data[ch_idx], sfreq)
        features['pac'] = np.mean([features.get(f'pac_task_ch{i}', 0) for i in range(n_channels)])

        lz_values = []
        for ch_idx in range(n_channels):
            lz = EntropyEngine.lempel_ziv_complexity(task_data[ch_idx])
            features[f'lz_task_ch{ch_idx}'] = lz
            lz_values.append(lz)
        features['lz'] = np.mean(lz_values)

        alpha_theta_values = []
        for ch_idx in range(n_channels):
            at = features.get(f'task_bp_alpha_ch{ch_idx}', 1e-10) / (features.get(f'task_bp_theta_ch{ch_idx}', 1e-10) + 1e-10)
            features[f'ratio_task_alpha_theta_ch{ch_idx}'] = at
            alpha_theta_values.append(at)
        features['at_ratio'] = np.mean(alpha_theta_values)

        entropy_values = []
        for ch_idx in range(n_channels):
            try:
                ch = task_data[ch_idx]
                ch_norm = (ch - ch.mean()) / (ch.std() + 1e-10)
                _, p = welch(ch_norm, fs=sfreq, nperseg=min(256, len(ch) // 4))
                p /= p.sum() + 1e-12
                ent = scipy_entropy(p)
                features[f'task_entropy_ch{ch_idx}'] = ent
                entropy_values.append(ent)
            except:
                features[f'task_entropy_ch{ch_idx}'] = 0.0
        features['entropy'] = np.mean(entropy_values)

        pe_values = []
        for ch_idx in range(n_channels):
            pe = EntropyEngine.permutation_entropy(task_data[ch_idx], order=3, delay=1)
            features[f'pe_ch{ch_idx}'] = pe
            pe_values.append(pe)
        features['pe'] = np.mean(pe_values)

        wpe_values = []
        for ch_idx in range(n_channels):
            wpe = EntropyEngine.weighted_permutation_entropy(task_data[ch_idx], order=3, delay=1)
            features[f'wpe_task_ch{ch_idx}'] = wpe
            wpe_values.append(wpe)
        features['wpe'] = np.mean(wpe_values)

        mse_values = []
        for ch_idx in range(n_channels):
            mse = EntropyEngine.multiscale_entropy(task_data[ch_idx], max_scale=3)
            features[f'mse_task_ch{ch_idx}'] = mse
            mse_values.append(mse)
        features['mse'] = np.mean(mse_values)

        if n_channels >= 2:
            features['mi_fp1_fp2'] = EntropyEngine.mutual_information(task_data[0], task_data[1])
        else:
            features['mi_fp1_fp2'] = 0.0

        features['frontal_asymmetry'] = EntropyEngine.compute_frontal_asymmetry(task_data, sfreq)
        features['alpha_relative'] = EntropyEngine.compute_alpha_relative(task_data, sfreq)
        features['delta'] = np.mean([features.get(f'task_bp_delta_ch{i}', 0) for i in range(n_channels)])

        return pd.Series(features)
    except Exception as e:
        print(f"âŒ Feature extraction failed: {e}")
        return None

# ============================================================================
# SIGNAL QUALITY TRACKER
# ============================================================================

class SignalQualityTracker:
    """Track signal quality for confidence computation"""
    def __init__(self):
        self.recent_quality = deque(maxlen=20)

    def add_epoch(self, epoch_data, is_artifact):
        if is_artifact:
            quality = 0.0
        else:
            try:
                signal_power = np.var(epoch_data)
                noise_estimate = np.median(np.abs(np.diff(epoch_data, axis=1))) + 1e-10
                snr = signal_power / noise_estimate
                quality = min(1.0, snr / 10.0)
            except:
                quality = 0.5
        self.recent_quality.append(quality)

    def get_mean_quality(self):
        return float(np.mean(self.recent_quality)) if self.recent_quality else 0.5

# ============================================================================
# WINDOWED STATE CLASSIFIER
# ============================================================================

class WindowedStateClassifier:
    """30-second windowed aggregation with real confidence"""
    def __init__(self, window_seconds=30):
        self.window_seconds = window_seconds
        self.window_size = int(window_seconds / 2)  # 2 seconds per trial
        self.instant_states = deque(maxlen=self.window_size)
        self.instant_confidences = deque(maxlen=self.window_size)
        self.instant_z_scores = deque(maxlen=self.window_size)

    def add_instant_classification(self, state, z_scores, signal_quality):
        if z_scores is None or not z_scores:
            instant_conf = 0.0
        else:
            z_values = [abs(z) for z in z_scores.values() if isinstance(z, (int, float))]
            max_z = max(z_values) if z_values else 0
            extremity_conf = np.exp(-abs(max_z - 1.0) / 2.0)
            instant_conf = extremity_conf * signal_quality

        self.instant_states.append(state)
        self.instant_confidences.append(instant_conf)
        self.instant_z_scores.append(z_scores)

    def get_windowed_state(self):
        if len(self.instant_states) < 10:
            return "Calibrating", 0.0, {}

        state_votes = {}
        for state, conf in zip(self.instant_states, self.instant_confidences):
            state_votes[state] = state_votes.get(state, 0) + conf

        if not state_votes:
            return "Calibrating", 0.0, {}

        total_votes = sum(state_votes.values())
        winner = max(state_votes, key=state_votes.get)
        winner_votes = state_votes[winner]

        vote_confidence = winner_votes / (total_votes + 1e-10)
        window_fullness = len(self.instant_states) / self.window_size
        final_confidence = vote_confidence * window_fullness

        winner_indices = [i for i, s in enumerate(self.instant_states) if s == winner]
        winner_z_scores = [self.instant_z_scores[i] for i in winner_indices if self.instant_z_scores[i]]

        if winner_z_scores:
            avg_z = {}
            all_keys = set()
            for z in winner_z_scores:
                all_keys.update(z.keys())
            for key in all_keys:
                values = [z[key] for z in winner_z_scores if key in z]
                avg_z[key] = float(np.mean(values)) if values else 0.0
        else:
            avg_z = {}

        return winner, float(final_confidence), avg_z

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_intensity(z_scores):
    """Overall cognitive load with state-based scaling"""
    if not z_scores:
        return 50

    base_intensity = (
        0.15 * max(0, min(1, (2.0 - z_scores.get('theta', 0)) / 4.0)) +
        0.20 * max(0, min(1, (z_scores.get('beta', 0) + 2.0) / 4.0)) +
        0.18 * max(0, min(1, (z_scores.get('gamma', 0) + 2.0) / 4.0)) +
        0.12 * max(0, min(1, (z_scores.get('lz', 0) + 2.0) / 4.0)) +
        0.10 * max(0, min(1, (z_scores.get('pe', 0) + 2.0) / 4.0)) +
        0.10 * max(0, min(1, (z_scores.get('pac', 0) + 2.0) / 4.0)) +
        0.08 * max(0, min(1, (z_scores.get('entropy', 0) + 2.0) / 4.0)) +
        0.07 * max(0, min(1, (2.0 - z_scores.get('at_ratio', 0)) / 4.0))
    ) * 100

    return int(np.clip(base_intensity, 0, 100))


# def compute_engagement(z_scores):
#     """High-frequency arousal (distinct from intensity)"""
#     if not z_scores:
#         return 50
#
#     z_beta = z_scores.get('beta', 0)
#     z_gamma = z_scores.get('gamma', 0)
#     z_lz = z_scores.get('lz', 0)
#     z_pac = z_scores.get('pac', 0)
#
#     engagement = (
#         0.35 * max(0, min(1, (z_beta + 2.0) / 4.0)) +
#         0.35 * max(0, min(1, (z_gamma + 2.0) / 4.0)) +
#         0.15 * max(0, min(1, (z_lz + 2.0) / 4.0)) +
#         0.15 * max(0, min(1, (z_pac + 2.0) / 4.0))
#     ) * 100
#
#     return int(np.clip(engagement, 0, 100))

# def detect_drift_enhanced(z_scores, state):
#     """Enhanced drift detection with gradient scoring"""
#     if z_scores is None:
#         return {
#             'is_drifting': False,
#             'drift_strength': 0,
#             'drift_label': 'NONE',
#             'drift_markers': [],
#         }
#
#     if state in DRIFT_STATES:
#         return {
#             'is_drifting': True,
#             'drift_strength': 100,
#             'drift_label': 'CONFIRMED',
#             'drift_markers': [f'State: {state}'],
#         }
#
#     drift_score = 0
#     markers = []
#
#     z_alpha = z_scores.get('alpha', 0)
#     z_theta = z_scores.get('theta', 0)
#     z_beta = z_scores.get('beta', 0)
#     z_gamma = z_scores.get('gamma', 0)
#     z_lz = z_scores.get('lz', 0)
#     z_pac = z_scores.get('pac', 0)
#
#     if z_alpha > 1.0:
#         drift_score += 30
#         markers.append('Alphaâ†‘')
#     elif z_alpha > 0.5:
#         drift_score += 15
#
#     if z_theta > 1.5:
#         drift_score += 25
#         markers.append('Thetaâ†‘')
#     elif z_theta > 0.8:
#         drift_score += 12
#
#     if z_beta < -1.0:
#         drift_score += 20
#         markers.append('Betaâ†“')
#     elif z_beta < -0.5:
#         drift_score += 10
#
#     if z_lz < -1.0:
#         drift_score += 20
#         markers.append('LZâ†“')
#
#     is_drifting = drift_score >= 30
#
#     if drift_score >= 70:
#         drift_label = 'SEVERE'
#     elif drift_score >= 50:
#         drift_label = 'STRONG'
#     elif drift_score >= 30:
#         drift_label = 'MODERATE'
#     else:
#         drift_label = 'NONE'
#
#     return {
#         'is_drifting': is_drifting,
#         'drift_strength': min(100, drift_score),
#         'drift_label': drift_label,
#         'drift_markers': markers,
#     }

def detect_drift_enhanced(z_scores, state, task_type="standard"):
    """
    Neuroscience-Grounded Drift Detection (Task-Aware)
    """
    if z_scores is None:
        return {'is_drifting': False, 'drift_strength': 0, 'drift_label': 'NONE', 'drift_markers': []}

    # 1. DYNAMIC THRESHOLDS (The Fix)
    # ---------------------------------------------------------
    if task_type in ['PASSIVE_CONSUMPTION', 'RECEPTIVE_ENGAGED']:
        # INPUT MODE: We tolerate Alpha (Relaxation). We fear Delta (Sleep).
        ALPHA_THRESH = 2.5  # Only flag if Alpha is EXTREME (Drowsiness)
        THETA_THRESH = 2.0  # High Theta in passive = Sleep onset
    else:
        # OUTPUT MODE: We punish Alpha (Disengagement).
        ALPHA_THRESH = 1.0  # Strict. Alpha = zoning out.
        THETA_THRESH = 1.5  # Strict.
    # ---------------------------------------------------------

    drift_score = 0
    markers = []

    # 2. SCORING
    z_alpha = z_scores.get('alpha', 0)
    z_theta = z_scores.get('theta', 0)
    z_beta = z_scores.get('beta', 0)
    z_delta = z_scores.get('delta', 0)
    z_lz = z_scores.get('lz', 0)

    # Alpha Check (Context Dependent)
    if z_alpha > ALPHA_THRESH:
        drift_score += 30
        markers.append('Alpha++')

    # Theta Check (Context Dependent)
    if z_theta > THETA_THRESH:
        drift_score += 25
        markers.append('Theta++')

    # Universal Checks (Always Bad)
    if z_delta > 1.0:  # Sleep pressure is never good for any task
        drift_score += 30
        markers.append('Delta+')

    if z_beta < -1.5:  # Extreme arousal drop
        drift_score += 20
        markers.append('Beta--')

    is_drifting = drift_score >= 40

    if drift_score >= 70:
        drift_label = 'SEVERE'
    elif drift_score >= 40:
        drift_label = 'MODERATE'
    else:
        drift_label = 'NONE'

    return {
        'is_drifting': is_drifting,
        'drift_strength': min(100, drift_score),
        'drift_label': drift_label,
        'drift_markers': markers,
    }

# ============================================================================
# BASELINE DETECTOR
# ============================================================================
class DailyBaselineManager:
    """
    Separate from ChunkedBaselineDetector
    Used ONLY for initial recommendations, not session monitoring
    """

    def __init__(self, user_id="default"):
        self.user_id = user_id
        self.baseline_dir = Path.home() / '.cognitive_monitor' / 'daily_baselines'
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_file = self.baseline_dir / f"{user_id}_baseline_v1.json"  # Changed name

        self.daily_baseline = None
        self.baseline_date = None
        self._load_baseline()


    def _load_baseline(self):
        """Load today's morning baseline if exists"""
        if not self.baseline_file.exists():
            return

        try:
            with open(self.baseline_file, 'r') as f:
                data = json.load(f)

            from datetime import datetime, date
            baseline_date = datetime.fromisoformat(data['date']).date()
            today = date.today()

            if baseline_date == today:
                self.daily_baseline = data['baseline']
                self.baseline_date = baseline_date
                print(f"✅ Loaded morning baseline from {data['time']}")
        except:
            pass

    def needs_baseline(self):
        """Check if we need new morning baseline"""
        from datetime import date
        return self.daily_baseline is None or self.baseline_date != date.today()

    def store_baseline(self, feature_list):
        """
        Store morning baseline from list of feature Series
        Args:
            feature_list: List of pd.Series (from 60-sec collection)
        """
        if len(feature_list) < 20:
            raise ValueError(f"Need 20+ epochs, got {len(feature_list)}")

        df = pd.concat(feature_list, axis=1).T
        baseline = {}

        for feature_name in VALIDATED_Z_SCORES:
            if feature_name == 'theta_beta_ratio':
                col = 'tbr_raw'
            elif feature_name == 'alpha_relative':
                col = 'alpha_relative'
            else:
                col = feature_name

            if col not in df.columns:
                continue

            values = df[col].dropna()
            if len(values) < 10:
                continue

            baseline[feature_name] = {
                'mean': float(values.mean()),
                'std': float(max(values.std(), 0.5))
            }

        from datetime import datetime
        now = datetime.now()

        data = {
            'date': now.date().isoformat(),
            'time': now.time().strftime('%H:%M:%S'),
            'baseline': baseline,
            'n_epochs': len(feature_list)
        }

        with open(self.baseline_file, 'w') as f:
            json.dump(data, f, indent=2)

        self.daily_baseline = baseline
        self.baseline_date = now.date()

        print(f"✅ Morning baseline stored ({len(baseline)} features)")
        return baseline

    def compute_z_scores(self, features):
        """Compute z-scores relative to morning baseline"""
        if self.daily_baseline is None:
            return None

        z_scores = {}

        for feature_name in VALIDATED_Z_SCORES:
            if feature_name not in self.daily_baseline:
                continue

            if feature_name == 'theta_beta_ratio':
                col = 'tbr_raw'
            elif feature_name == 'alpha_relative':
                col = 'alpha_relative'
            else:
                col = feature_name

            if col not in features.index:
                continue

            value = features[col]
            if pd.isna(value):
                continue

            bl = self.daily_baseline[feature_name]
            z = (value - bl['mean']) / bl['std']
            z_scores[feature_name] = float(z)

        return z_scores if z_scores else None
class ChunkedBaselineDetector:
    def __init__(self, chunk_size=100, min_trials=15):
        self.chunk_size = chunk_size
        self.min_trials = min_trials
        self.trial_count = 0
        self.buffer = []
        self.baseline = None
        self.baseline_ready = False

    def process_trial(self, trial_features, is_optimal: bool):
        self.trial_count += 1

        # First chunk: always collect; later: only collect when in optimal state
        is_first_chunk = (self.trial_count <= self.chunk_size)
        should_collect = is_first_chunk or is_optimal

        if should_collect:
            marker_values = {}
            for marker_name in VALIDATED_Z_SCORES:
                if marker_name == 'theta_beta_ratio':
                    value = trial_features.get('tbr_raw', 0)
                elif marker_name == 'alpha_relative':
                    value = trial_features.get('alpha_relative', 0)
                else:
                    value = trial_features.get(marker_name, 0)

                if pd.notna(value) and not np.isnan(value):
                    marker_values[marker_name] = float(value)

            if marker_values:
                self.buffer.append(marker_values)

        # âœ… Update baseline as soon as enough data is available,
        #    then at chunk boundaries for later refinement.
        if (not self.baseline_ready and len(self.buffer) >= self.min_trials) or \
           (self.trial_count % self.chunk_size == 0):
            self._update_baseline()
            if self.trial_count % self.chunk_size == 0:
                # For subsequent chunks, reset buffer at boundaries
                self.buffer = []

        # Compute z-scores (will return None if baseline not ready yet)
        return self.compute_z_scores(trial_features)


    def _update_baseline(self):
        if len(self.buffer) < self.min_trials:
            return
        buffer_df = pd.DataFrame(self.buffer)
        new_baseline = {}
        for marker_name in VALIDATED_Z_SCORES:
            if marker_name in buffer_df.columns:
                mean_val = buffer_df[marker_name].mean()
                std_val = buffer_df[marker_name].std()
                robust_std = max(std_val, 0.5) if marker_name in ['pe', 'wpe', 'mse'] else max(std_val, 0.3)
                new_baseline[marker_name] = {'mean': mean_val, 'std': robust_std}
        self.baseline = new_baseline
        self.baseline_ready = True

    def compute_z_scores(self, trial_features):
        if self.baseline is None:
            return None
        z_scores = {}
        for marker_name in VALIDATED_Z_SCORES:
            if marker_name not in self.baseline:
                continue
            if marker_name == 'theta_beta_ratio':
                value = trial_features.get('tbr_raw', 0)
            elif marker_name == 'alpha_relative':
                value = trial_features.get('alpha_relative', 0)
            else:
                value = trial_features.get(marker_name, 0)

            if pd.isna(value) or np.isnan(value):
                continue
            baseline = self.baseline[marker_name]

            if marker_name in ['theta', 'beta', 'alpha'] and not getattr(self, "quiet_mode", False):
                print(f"  {marker_name}: raw={value}, baseline_mean={baseline['mean']}, baseline_std={baseline['std']}")

            z = (value - baseline['mean']) / baseline['std']
            z_scores[marker_name] = z
        return z_scores if z_scores else None


# ============================================================================
# STATE CLASSIFICATION
# ============================================================================

def classify_state(z_scores: Dict[str, float]) -> str:
    if z_scores is None or not z_scores:
        return "Calibrating"

    z_theta = z_scores.get('theta', 0)
    z_beta = z_scores.get('beta', 0)
    z_tbr = z_scores.get('theta_beta_ratio', 0)
    z_alpha = z_scores.get('alpha', 0)
    z_gamma = z_scores.get('gamma', 0)
    z_delta = z_scores.get('delta', 0)
    z_pe = z_scores.get('pe', 0)
    z_lz = z_scores.get('lz', 0)
    z_pac = z_scores.get('pac', 0)

    if z_theta > OVERLOAD_THRESHOLDS['theta_extreme']:
        if z_alpha > 1.5 and z_beta < -1.5:
            return 'Overload'

    if z_alpha > FATIGUE_THRESHOLDS['alpha_elevation']:
        fatigue_markers = sum([
            z_delta > FATIGUE_THRESHOLDS['delta_elevation'],
            z_theta > FATIGUE_THRESHOLDS['theta_elevation'],
            z_beta < FATIGUE_THRESHOLDS['beta_decrease'],
            z_gamma < -0.3,
        ])
        if fatigue_markers >= 2:
            return 'Fatigue'

    if z_tbr > MW_THRESHOLDS['tbr_moderate']:
        mw_markers = sum([
            z_alpha < MW_THRESHOLDS['alpha_decrease'],
            z_pe < MW_THRESHOLDS['pe_moderate'],
            z_beta < -0.2,
            z_lz < -0.3,
            z_pac < -0.3,
        ])
        if mw_markers >= 2:
            return 'Mind-Wandering'

    if z_alpha < MW_THRESHOLDS['alpha_decrease'] and z_theta > 0.3:
        if z_beta < -0.2 or z_gamma < -0.2:
            return 'Mind-Wandering'

    if z_lz < -0.5 and z_pac < -0.5:
        if z_tbr > 0.15 or z_theta > 0.3:
            return 'Mind-Wandering'
    # NEW: Check for passive consumption (watching videos, light reading)
    if (z_alpha > 0.3 and z_theta > 0.2 and
            z_beta < -0.2 and z_gamma < -0.2 and
            z_lz < -0.3 and z_pac < -0.3):
        return 'Passive-Consumption'

    engagement_score = 0
    if z_beta > 0.3: engagement_score += 2.5
    if z_gamma > 0.2: engagement_score += 2.5
    if z_lz > 0.2: engagement_score += 2.0
    if z_pac > 0.2: engagement_score += 2.0
    if z_beta > 0.2 and z_gamma > 0.2:
        engagement_score += 2.0

    if engagement_score >= 6:
        return 'Optimal-Engaged'
    elif engagement_score >= 3:
        return 'Optimal-Engaged'

    return 'Optimal-Monitoring'

# ============================================================================
# TASK RECOMMENDATION ENGINE
# ============================================================================

# !/usr/bin/env python3
"""
FIXED TASK RECOMMENDATION ENGINE
Based on validated neuroscience principles:
- Arousal (Beta/Gamma) = Energy availability
- Frontal Theta = Executive control capacity
- Control (Alpha) = Inhibition/gating ability

Key insight: "Optimal-Engaged" can be passive OR generative
Discriminator: Frontal Midline Theta
"""


class TaskRecommendationEngine:
    """
    Neuroscience-grounded task recommendation
    Separates receptive vs. generative states
    """

    def __init__(self):
        pass

    def compute_sigma_state(self, z_scores):
        """
        THREE-AXIS MODEL (not two):
        1. Arousal (Energy) = Beta/Gamma/LZ
        2. Control (Inhibition) = Alpha - Theta_temporal
        3. Executive (Frontal) = Theta_frontal
        """
        if not z_scores:
            return 0.0, 0.0, 0.0

        # 1. AROUSAL (Metabolic energy available)
        z_beta = z_scores.get('beta', 0)
        z_gamma = z_scores.get('gamma', 0)
        z_lz = z_scores.get('lz', 0)
        arousal_sigma = (0.4 * z_beta) + (0.4 * z_gamma) + (0.2 * z_lz)

        # 2. CONTROL (Inhibitory gating)
        z_alpha = z_scores.get('alpha', 0)
        z_theta_temporal = z_scores.get('theta', 0)  # TP9/TP10
        control_sigma = (0.7 * z_alpha) - (0.3 * z_theta_temporal)

        # 3. EXECUTIVE (Frontal engagement) - THE KEY DISCRIMINATOR
        # This separates "watching Netflix" from "solving problems"
        z_theta_frontal = z_scores.get('theta_frontal', 0)  # Fp1/Fp2
        executive_sigma = z_theta_frontal

        if abs(arousal_sigma) > 5 or abs(executive_sigma) > 5:
            print(f"⚠️  WARNING: Extreme sigma detected!")
            print(f"   Arousal: {arousal_sigma:.2f}σ")
            print(f"   Executive: {executive_sigma:.2f}σ")
            print(f"   This indicates artifacts in the signal.")

            # Clip to reasonable range
            arousal_sigma = np.clip(arousal_sigma, -3, 3)
            executive_sigma = np.clip(executive_sigma, -3, 3)

        return arousal_sigma, control_sigma, executive_sigma

    def recommend_task(self, z_scores_list):
        """
        Decision tree based on three-axis model (Refined v7.4)
        """
        if not z_scores_list or len([z for z in z_scores_list if z]) < 5:
            return {
                'recommendation': '⚙️ INSUFFICIENT DATA',
                'tasks': ['Collecting baseline...'],
                'duration': 'Wait...',
                'confidence': 0.0,
            }

        # Average recent measurements
        valid_list = [z for z in z_scores_list if z]
        recent_list = valid_list[-10:] if len(valid_list) >= 10 else valid_list

        avg_z = {}
        if recent_list:
            for k in recent_list[0].keys():
                vals = [z[k] for z in recent_list if k in z]
                if vals: avg_z[k] = sum(vals) / len(vals)

        ar, ctrl, exec_fn = self.compute_sigma_state(avg_z)

        # ====================================================================
        # DECISION TREE (Neuroscience-Grounded v7.4)
        # ====================================================================

        # ZONE 0: METABOLIC CRASH (Priority Check)
        # If energy is tanked, executive function is irrelevant.
        # This fixes the "Learning Mode when Exhausted" bug (ar = -1.67).
        if (ar < -1.0):
            rec = "😴 FATIGUE (Metabolic Dip)"
            tasks = [
                "Nap (20 min)",
                "Meditation (NSDR)",
                "Walk Outside",
                "End Session"
            ]
            duration = "Rest 20 min"
            mode = "FATIGUE"

        # ZONE 1: GENERATIVE PEAK STATE
        # High arousal + High executive + Intact control
        # Lowered Arousal threshold (0.5 -> 0.2) to capture "Calm Focus"
        elif (ar > 0.2) and (exec_fn > 0.2) and (ctrl > -0.5):
            rec = "🎯 PEAK STATE (Generative Flow)"
            tasks = [
                "Deep Coding (new algorithms)",
                "Hard Problem Solving",
                "Technical Writing",
                "Complex Design Work"
            ]
            duration = "90 min (protect this window)"
            mode = "PEAK_GENERATIVE"

        # ZONE 2: RECEPTIVE PEAK STATE
        # High arousal but Low Executive (Input Mode)
        # Changed exec_fn < 0.0 to <= 0.2 to catch borderline cases
        elif (ar > 0.2) and (exec_fn <= 0.2) and (ctrl > -0.5):
            rec = "📚 ACTIVE LEARNING (Receptive)"
            tasks = [
                "Video Lectures",
                "Reading Technical Papers",
                "Code Review (others' code)",
                "Listening to Podcasts"
            ]
            duration = "60 min"
            mode = "RECEPTIVE_ENGAGED"

        # ZONE 3: EXECUTIVE DEPLETION (Can't initiate)
        # Raised threshold from -1.0 to -0.8 to be safer
        elif (exec_fn < -0.8):
            rec = "🎬 PASSIVE ONLY (Executive Depleted)"
            tasks = [
                "Netflix/YouTube",
                "Light Reading (fiction)",
                "Social Media Scrolling",
                "Music Listening"
            ]
            duration = "30 min, then break"
            mode = "PASSIVE_CONSUMPTION"

        # ZONE 4: OVERLOAD
        # High arousal but lost control (Anxiety/Noise)
        elif (ar > 2.0) or (ar > 1.0 and ctrl < -1.0):
            rec = "⚠️ OVERLOAD (High Noise)"
            tasks = [
                "Organization/Tidying",
                "Easy Emails",
                "Breathwork (4-7-8)",
                "Walk Outside"
            ]
            duration = "15 min break needed"
            mode = "OVERLOAD"

        # ZONE 6: BASELINE / STANDARD (The Middle Ground)
        else:
            # Moderate Arousal (-1.0 to 0.2)
            # If executive is positive, you can do routine work
            if exec_fn > -0.2:
                rec = "🔨 STANDARD WORK (Moderate Load)"
                tasks = [
                    "Refactoring Code",
                    "Documentation",
                    "Routine Tasks",
                    "Practice Problems"
                ]
                duration = "45-60 min"
                mode = "STANDARD_WORK"
            else:
                # Only here if not tired enough for Fatigue, but low Exec
                rec = "📖 LEARNING MODE (Low Load)"
                tasks = [
                    "Study (reading + notes)",
                    "Tutorial Following",
                    "Flashcard Review"
                ]
                duration = "45 min"
                mode = "RECEPTIVE_ENGAGED" # Map to Receptive internally

        # ====================================================================
        # COMPUTE CONFIDENCE
        # ====================================================================
        # Based on signal stability and clarity

        # Check variance in recent measurements
        ar_list = [self.compute_sigma_state(z)[0] for z in recent_list]
        exec_list = [self.compute_sigma_state(z)[2] for z in recent_list]

        ar_std = np.std(ar_list) if len(ar_list) > 1 else 1.0
        exec_std = np.std(exec_list) if len(exec_list) > 1 else 1.0

        # Low variance = high confidence
        stability_conf = 1.0 - min(1.0, (ar_std + exec_std) / 2.0)

        # Clear state = high confidence
        clarity_conf = min(1.0, (abs(ar) + abs(exec_fn)) / 2.0)

        confidence = (0.6 * stability_conf + 0.4 * clarity_conf)

        return {
            'recommendation': rec,
            'tasks': tasks,
            'duration': duration,
            'confidence': confidence,
            'arousal_sigma': round(ar, 2),
            'control_sigma': round(ctrl, 2),
            'executive_sigma': round(exec_fn, 2),
            'mode': mode,
        }

# ====================================================================
# VALIDATION PROTOCOL
# ====================================================================

def validate_recommendation(session_log, user_report):
    """
    Post-session validation to refine thresholds

    Args:
        session_log: CSV with trial-level data
        user_report: User's actual experience

    Returns:
        Match score and suggested threshold adjustments
    """

    # Load session data
    df = pd.read_csv(session_log)

    # Calculate average state
    avg_arousal = df['arousal_sigma'].mean()
    avg_executive = df['executive_sigma'].mean()
    avg_control = df['control_sigma'].mean()

    # User reported outcomes
    actual_mode = user_report['actual_task_type']
    felt_flow = user_report['felt_flow']  # 1-5 scale
    productivity = user_report['productivity']  # 1-5 scale
    fatigue = user_report['felt_fatigue']  # 1-5 scale

    # Check if recommendation matched reality
    predicted_mode = df['mode'].iloc[0]  # From first trial

    validation = {
        'predicted_mode': predicted_mode,
        'actual_mode': actual_mode,
        'match': predicted_mode == actual_mode,
        'avg_arousal': avg_arousal,
        'avg_executive': avg_executive,
        'avg_control': avg_control,
        'user_flow': felt_flow,
        'user_productivity': productivity,
        'user_fatigue': fatigue,
    }

    # Suggest threshold adjustments
    if not validation['match']:
        if actual_mode == 'PEAK_GENERATIVE' and predicted_mode != 'PEAK_GENERATIVE':
            validation['adjustment'] = f"Lower executive threshold (was {avg_executive:.2f}, felt peak)"
        elif actual_mode == 'PASSIVE_CONSUMPTION' and predicted_mode != 'PASSIVE_CONSUMPTION':
            validation['adjustment'] = f"Raise executive threshold (was {avg_executive:.2f}, felt passive)"

    return validation



# ============================================================================
# TEMPORAL SMOOTHING
# ============================================================================

class TemporalSmoothingEngine:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.state_history = deque(maxlen=window_size)
        self.drift_trajectory = deque(maxlen=20)

    def add_trial(self, state: str, drift_strength: int):
        self.state_history.append(state)
        self.drift_trajectory.append(drift_strength)

    def predict_drift_risk(self) -> int:
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
    """
    Track cumulative drift % over rolling window
    Based on lab data: 41.8% mean cumulative drift, intervention at 50%+
    """

    def __init__(self, window_seconds=120):
        """
        Args:
            window_seconds: Rolling window size (default 120 = 2 minutes)
        """
        self.window_seconds = window_seconds
        self.window_trials = int(window_seconds / 2)  # 2 sec per trial
        self.drift_history = deque(maxlen=self.window_trials)  # 0 or 1 for each trial

    def add_trial(self, is_drift: bool):
        """Add trial to history (drift or not drift)"""
        self.drift_history.append(1 if is_drift else 0)

    def get_cumulative_drift_pct(self) -> float:
        """Calculate % of window spent in drift state"""
        if len(self.drift_history) == 0:
            return 0.0
        return (sum(self.drift_history) / len(self.drift_history)) * 100.0

    def is_above_threshold(self, threshold_pct: float) -> bool:
        """Check if cumulative drift exceeds threshold"""
        return self.get_cumulative_drift_pct() > threshold_pct

# ============================================================================
# IG COMPUTER
# ============================================================================

class RealTimeIGComputer:
    def __init__(self):
        self.pca = None
        self.scaler = StandardScaler()
        self.pca_frozen = False

        # Baseline Manifold Parameters
        self.base_mean = None
        self.base_cov_inv = None
        self.base_cov_det = None
        self.base_cov = None

    def update_baseline(self, z_scores_list):
        if self.pca_frozen:
            return
        z_scores_clean = [z for z in z_scores_list if z]
        if len(z_scores_clean) < 50:
            return
        try:
            # Fit PCA on baseline window
            X = np.array([list(z.values()) for z in z_scores_clean[-300:]])
            X_scaled = self.scaler.fit_transform(X)
            self.pca = PCA(n_components=0.9)
            X_pca = self.pca.fit_transform(X_scaled)

            # Store Manifold Statistics (The "Safe Zone")
            self.base_mean = np.mean(X_pca, axis=0)
            self.base_cov = np.cov(X_pca.T) + np.eye(X_pca.shape[1]) * 1e-6
            self.base_cov_inv = np.linalg.pinv(self.base_cov)
            self.base_cov_det = np.linalg.det(self.base_cov)

            self.pca_frozen = True
            print(f"✅ PCA FROZEN: {self.pca.n_components_} components")
        except:
            pass

    def compute_distance(self, recent_z_scores):
        """Calculate how far current state is from baseline (Drift)"""
        if not self.pca_frozen or not recent_z_scores:
            return 0.0, 0.0
        try:
            X = np.array([list(z.values()) for z in recent_z_scores if z])
            if len(X) < 5: return 0.0, 0.0

            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)

            curr_mean = np.mean(X_pca, axis=0)
            curr_cov = np.cov(X_pca.T) + np.eye(X_pca.shape[1]) * 1e-6

            # 1. Mahalanobis Distance (Drift Magnitude)
            diff = curr_mean - self.base_mean
            mahal = np.sqrt(diff.T @ self.base_cov_inv @ diff)

            # 2. KL Divergence (State Scatter)
            k = len(curr_mean)
            term1 = np.trace(self.base_cov_inv @ curr_cov)
            term2 = diff.T @ self.base_cov_inv @ diff
            term3 = k
            term4 = np.log(self.base_cov_det / (np.linalg.det(curr_cov) + 1e-10))
            kl = 0.5 * (term1 + term2 - term3 + term4)

            return max(0.0, float(mahal)), max(0.0, float(kl))
        except:
            return 0.0, 0.0
# ============================================================================
# BASELINE PERSISTENCE
# ============================================================================

class BaselinePersistence:
    def __init__(self, user_id="default"):
        self.user_id = user_id
        self.profile_dir = Path.home() / '.cognitive_monitor' / 'baselines' / user_id
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    def save_baseline(self, baseline_detector, ig_computer, session_name=None):
        import datetime
        if baseline_detector.baseline is None:
            print("âŒ Baseline not ready")
            return None
        session_name = session_name or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        baseline_file = self.profile_dir / f"{session_name}_baseline.pkl"
        with open(baseline_file, 'wb') as f:
            pickle.dump({
                'baseline': baseline_detector.baseline,
                'pca': ig_computer.pca,
                'scaler': ig_computer.scaler,
            }, f)
        print(f"âœ… Baseline saved: {baseline_file.name}")
        return baseline_file

    def load_baseline(self, session_name="latest"):
        baseline_files = sorted(self.profile_dir.glob("*_baseline.pkl"))
        if not baseline_files:
            return None
        baseline_file = baseline_files[-1]
        with open(baseline_file, 'rb') as f:
            data = pickle.load(f)
        print(f"âœ… Loaded baseline: {baseline_file.name}")
        return data


class RollingVolatilityTracker:
    """
    Tracks relative focus drops (Audit Mode Logic).
    Flags if intensity drops 1.5 deviations below the last 60s average.
    """

    def __init__(self, window_trials=30, sensitivity=1.5):
        from collections import deque
        import numpy as np
        self.history = deque(maxlen=window_trials)  # 30 trials = 60 seconds
        self.sensitivity = sensitivity

    def update(self, current_intensity: float):
        self.history.append(current_intensity)

        # Need 30s of data (15 trials) to start flagging
        if len(self.history) < 15:
            return False, 50.0, 0.0

        mean_val = np.mean(self.history)
        std_val = np.std(self.history)
        threshold = mean_val - (self.sensitivity * std_val)

        is_drop = current_intensity < threshold
        return is_drop, mean_val, threshold
# ============================================================================
# COGNITIVE MONITOR (MAIN ENGINE) - COMPLETE v7.3
# ============================================================================

class CognitiveMonitorV7_3:
    def __init__(self, task_type="learning", user_id="default"):
        self.task_type = task_type

        # TWO SEPARATE BASELINES:
        self.baseline_detector = ChunkedBaselineDetector()  # KEEP - for session monitoring
        self.daily_baseline_mgr = DailyBaselineManager(user_id)  # ADD - for recommendations

        self.ig_computer = RealTimeIGComputer()
        self.signal_quality = SignalQualityTracker()
        self.windowed_classifier = WindowedStateClassifier()
        self.temporal_smoother = TemporalSmoothingEngine()
        self.task_recommender = TaskRecommendationEngine()
        self.cumulative_drift_tracker = CumulativeDriftTracker(window_seconds=120)
        self.rolling_volatility_tracker = RollingVolatilityTracker(window_trials=30, sensitivity=1.5)
        self.trial_count = 0
        self.epoch_count = 0
        self.rejected_count = 0
        self.z_scores_history = []
        self.state_history = []
        self.quiet_mode = False
        self.baseline_detector.quiet_mode = False
        self.ml_model = None
        self.ml_features = None
        self.ml_scaler = None
        self._load_ml_model()

    def _load_ml_model(self):
        """Load the pre-trained Gradient Boosting model"""
        try:
            # 👇 UPDATE THIS PATH to your actual file location 👇
            model_path = r"C:\Users\rapol\Downloads\lab_analysis_v6_0_grounded\phase2_complete_v6\error_model_v6.pkl"

            data = joblib.load(model_path)
            self.ml_model = data['model']
            self.ml_features = data['features']
            self.ml_scaler = data['scaler']
            print("✅ ML Error Model loaded successfully")
        except Exception as e:
            print(f"⚠️ ML Model not found (skipping error prediction): {e}")

    def predict_error_risk(self, z_scores):
        """Predict probability of error (0-100%) using 32-feature vector"""
        if not self.ml_model or not z_scores:
            return 0.0

        try:
            # Construct feature vector in exact order expected by model
            feature_vector = []
            for f in self.ml_features:
                # Model expects keys like 'theta', 'beta', but z_scores might have them
                # Adjust key matching if necessary (e.g. removing 'z_' prefix if your z_scores dict doesn't have it)
                key = f.replace('z_', '')
                feature_vector.append(z_scores.get(key, 0))

            # Scale and predict
            X = self.ml_scaler.transform([feature_vector])
            prob = self.ml_model.predict_proba(X)[0][1]  # Probability of Class 1 (Error)
            return round(prob * 100, 1)
        except:
            return 0.0

    def process_epoch(self, epoch_data, is_artifact=False):
        self.epoch_count += 1
        self.signal_quality.add_epoch(epoch_data, is_artifact)
        if is_artifact:
            self.rejected_count += 1

    def process_trial(self, raw_features: pd.Series) -> Dict:
        self.trial_count += 1
        import time
        timestamp = time.time()

        # Compute z-scores from baseline detector
        z_scores = self.baseline_detector.process_trial(raw_features, is_optimal=False)
        self.z_scores_history.append(z_scores)

        # Instant classification
        instant_state = classify_state(z_scores)
        signal_quality = self.signal_quality.get_mean_quality()

        # Windowed classifier
        self.windowed_classifier.add_instant_classification(instant_state, z_scores, signal_quality)
        windowed_state, confidence, avg_z_scores = self.windowed_classifier.get_windowed_state()
        self.state_history.append(windowed_state)

        if self.task_type in ['PEAK_GENERATIVE', 'peak_work']:
            # Only run the heavy ML model if the task demands it
            error_risk = self.predict_error_risk(z_scores)
        else:
            # Return 0.0 (Risk Undefined) for all other modes
            error_risk = 0.0

        # Baseline update when in optimal states
        is_optimal = windowed_state in OPTIMAL_STATES
        is_passive = windowed_state in PASSIVE_STATES

        if is_optimal and z_scores:
            self.baseline_detector.process_trial(raw_features, is_optimal=True)

        # Use windowed z if available
        use_z = avg_z_scores if avg_z_scores else z_scores

        # Intensity / engagement
        intensity = compute_intensity(use_z)
        # engagement = compute_engagement(use_z)

        # Scale intensity by state
        if windowed_state == 'Optimal-Engaged':
            intensity = min(100, int(intensity * 1.3))
        elif windowed_state in ['Mind-Wandering', 'Fatigue']:
            intensity = max(0, int(intensity * 0.7))
        elif windowed_state == 'Passive-Consumption':
            intensity = max(0, int(intensity * 0.5))

        rolling_drift_detected, rolling_baseline, rolling_threshold = self.rolling_volatility_tracker.update(intensity)
        # Drift estimation
        drift_info = detect_drift_enhanced(use_z, windowed_state, self.task_type)
        self.temporal_smoother.add_trial(windowed_state, drift_info['drift_strength'])
        drift_risk = self.temporal_smoother.predict_drift_risk()

        # ✅ ADD: Track cumulative drift
        is_drift_state = windowed_state in DRIFT_STATES
        self.cumulative_drift_tracker.add_trial(is_drift_state)
        cumulative_drift_pct = self.cumulative_drift_tracker.get_cumulative_drift_pct()
        ar_sigma, ctrl_sigma, exec_sigma = self.task_recommender.compute_sigma_state(use_z)
        # [IG / EFFICIENCY UPDATE]
        if self.baseline_detector.baseline_ready:
            self.ig_computer.update_baseline(self.z_scores_history)

        # Get recent window for stability check
        recent_z = list(self.windowed_classifier.instant_z_scores)[-15:]
        mahal, kl = self.ig_computer.compute_distance(recent_z)

        # 🧪 LAB FORMULA
        w_mahal = 0.5
        w_kl = 0.2
        w_int = 0.3

        score_stability = 100 / (1 + mahal)
        score_entropy = 100 / (1 + kl / 100)
        score_intensity = intensity

        efficiency = (w_mahal * score_stability) + (w_kl * score_entropy) + (w_int * score_intensity)
        output = {
            'timestamp': timestamp,
            'trial': self.trial_count,
            'instant_state': instant_state,
            'windowed_state': windowed_state,
            'confidence': round(confidence, 3),
            'intensity': intensity,
            # 'engagement': engagement,
            'signal_quality': round(signal_quality, 2),
            'drift_strength': drift_info['drift_strength'],
            'drift_label': drift_info['drift_label'],
            'drift_markers': ','.join(drift_info['drift_markers']),
            'drift_risk': drift_risk,
            'baseline_ready': self.baseline_detector.baseline_ready,
            'cumulative_drift_pct': round(cumulative_drift_pct, 1),
            'rolling_drift': rolling_drift_detected,
            'rolling_threshold': round(rolling_threshold, 1),
            'arousal_sigma': round(ar_sigma, 2),
            'control_sigma': round(ctrl_sigma, 2),
            'executive_sigma': round(exec_sigma, 2),  # ADD THIS LINE
            'error_risk': error_risk,
            'efficiency': round(efficiency, 1),
            'mahalanobis': round(mahal, 2),
        }

        if avg_z_scores:
            for key, val in avg_z_scores.items():
                output[f'z_{key}'] = val

        return output

