#!/usr/bin/env python3
"""
SESSION RUNNER v7.3 - COMPLETE ORCHESTRATOR
=====================================================================
✅ Real Muse EEG streaming + Adaptive artifact rejection
✅ Live CSV logging with 35+ features per trial
✅ Non-blocking pause/resume + quit controls
✅ Task recommendations every 5 trials (with arousal/readiness trends)
✅ Windowed state classification (30-sec aggregation)
✅ Real confidence scores + signal quality tracking
✅ Enhanced drift detection + intensity/engagement metrics
✅ Cross-session baseline persistence (instant start)

This file ONLY handles session management + real-time I/O.
All cognitive processing delegated to real_time_monitor_v7_3_complete.py

USAGE:
  python session_runner_v7_3_complete.py --task learning --duration 90
  python session_runner_v7_3_complete.py --task learning --duration 5 --no-lsl
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
import argparse
import threading
import queue
import datetime
import sys
from real_time_monitor_v7_3_complete import (
    CognitiveMonitorV7_3,
    connect_muse,
    collect_epoch,
    extract_features,
    is_epoch_clean,
    EntropyEngine,
    SAMPLE_RATE,
    EPOCH_DURATION,
    AMP_THRESHOLD_UV,
    KURTOSIS_THRESHOLD,
    USE_ADAPTIVE_ARTIFACT,
    BaselinePersistence,
)


# ============================================================================
# DESKTOP NOTIFICATION (CROSS-PLATFORM)
# ============================================================================

def send_desktop_notification(title, message, urgency='normal'):
    """
    Send desktop notification (cross-platform)
    Args:
        title: Notification title
        message: Notification body
        urgency: 'low', 'normal', or 'critical'
    """
    try:
        # Try plyer (cross-platform)
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name='Cognitive Monitor',
            timeout=10,  # seconds
        )
        print(f"✅ Desktop notification sent: {title}")
    except ImportError as e:
        # Fallback: print to console
        print(f"\n🔔 NOTIFICATION (plyer not installed): {title}")
        print(f" {message}\n")
    except Exception as e:
        # Catch ALL other errors and report them
        print(f"\n⚠️ Notification failed: {type(e).__name__}: {e}")
        print(f"🔔 FALLBACK NOTIFICATION: {title}")
        print(f" {message}\n")

# ============================================================================
# PAUSE MANAGER - NON-BLOCKING
# ============================================================================

class PauseManager:
    """Handles pause/resume without blocking main loop (background thread)"""

    def __init__(self):
        self.paused = False
        self.quit_flag = False
        self.input_queue = queue.Queue()
        self.input_thread = threading.Thread(target=self._listen_for_input, daemon=True)
        self.input_thread.start()

    def _listen_for_input(self):
        """Background thread listens for 'p' or 'q' input"""
        try:
            while True:
                user_input = input().strip().lower()
                if user_input == 'p':
                    self.input_queue.put('pause')
                elif user_input == 'q':
                    self.input_queue.put('quit')
        except (EOFError, KeyboardInterrupt):
            pass

    def check_input(self):
        """Check if there's a pause/quit command (non-blocking)"""
        try:
            command = self.input_queue.get_nowait()
            if command == 'pause':
                self.paused = True
            elif command == 'quit':
                self.quit_flag = True
        except queue.Empty:
            pass

    def handle_pause(self):
        """If paused, wait for user to press Enter to resume"""
        if self.paused:
            print("\n⏸️  PAUSED - Press Enter to resume...")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            self.paused = False
            print("▶️  RESUMED\n")

    def should_quit(self):
        """Check if user requested quit"""
        return self.quit_flag

# ============================================================================
# SESSION MANAGER - LOGGING + METADATA
# ============================================================================

class SessionManager:
    """Manages recording sessions, logging, and file I/O"""

    def __init__(self, task_type="learning", session_name=None, output_dir="./sessions"):
        self.task_type = task_type
        self.session_name = session_name or f"{task_type}_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_file = self.output_dir / f"{self.session_name}.csv"
        self.metadata_file = self.output_dir / f"{self.session_name}_metadata.json"

        self.trial_count = 0
        self.start_time = None
        self.end_time = None

        self.metadata = {
            'session_name': self.session_name,
            'task_type': task_type,
            'start_time': None,
            'end_time': None,
            'duration_minutes': 0,
            'total_trials': 0,
            'total_epochs_rejected': 0,
            'artifact_rejection_rate': 0,
            'state_distribution': {},
        }

    def start_session(self):
        """Start recording session"""
        self.start_time = datetime.datetime.now()
        self.metadata['start_time'] = self.start_time.isoformat()
        print("=" * 120)
        print(f"✅ SESSION STARTED: {self.session_name}")
        print("=" * 120)
        print(f"   Task type: {self.task_type}")
        print(f"   Start time: {self.start_time}")
        print(f"   Output directory: {self.output_dir}")
        print("=" * 120 + "\n")

    def end_session(self):
        """End recording session"""
        self.end_time = datetime.datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() / 60
        self.metadata['end_time'] = self.end_time.isoformat()
        self.metadata['duration_minutes'] = round(duration, 2)
        self.metadata['total_trials'] = self.trial_count

        print("\n" + "=" * 120)
        print(f"✅ SESSION ENDED: {self.session_name}")
        print("=" * 120)
        print(f"   Duration: {duration:.2f} minutes")
        print(f"   Total trials: {self.trial_count}")
        print(f"   Files saved:")
        print(f"   - {self.session_file}")
        print(f"   - {self.metadata_file}")
        print("=" * 120 + "\n")

    def log_trial(self, output_dict):
        """Log trial to CSV (ensure all z-score cols are present)"""
        from real_time_monitor_v7_3_complete import VALIDATED_Z_SCORES
        import numpy as np
        import pandas as pd

        # On first write, force all z_* columns into the header
        if not self.session_file.exists():
            for marker_name in VALIDATED_Z_SCORES:
                col_name = f"z_{marker_name}"
                if col_name not in output_dict:
                    output_dict[col_name] = np.nan

            df = pd.DataFrame([output_dict])
            df.to_csv(self.session_file, mode='w', index=False)
        else:
            df = pd.DataFrame([output_dict])
            df.to_csv(self.session_file, mode='a', header=False, index=False)

        self.trial_count += 1

    def save_metadata(self, monitor):
        """Save session metadata"""
        if monitor.epoch_count > 0:
            self.metadata['total_epochs_rejected'] = monitor.rejected_count
            self.metadata['artifact_rejection_rate'] = (
                (monitor.rejected_count / monitor.epoch_count) * 100 if monitor.epoch_count > 0 else 0
            )

        if monitor.state_history:
            state_counts = {}
            for state in monitor.state_history:
                state_counts[state] = state_counts.get(state, 0) + 1
            self.metadata['state_distribution'] = state_counts

        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def update_task_type(self, new_task_type):
        """Updates task type and renames session files BEFORE recording starts"""
        self.task_type = new_task_type

        # Regenerate session name with new task type
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_name = f"{new_task_type}_session_{timestamp}"

        # Update file paths
        self.session_file = self.output_dir / f"{self.session_name}.csv"
        self.metadata_file = self.output_dir / f"{self.session_name}_metadata.json"

        # Update metadata dict
        self.metadata['session_name'] = self.session_name
        self.metadata['task_type'] = new_task_type
        print(f"🔄 Task updated to: {new_task_type}")
        print(f"📄 New filename: {self.session_name}")
# ============================================================================
# REALTIME DISPLAY - LIVE CONSOLE METRICS (UPDATED)
# ============================================================================

class RealtimeDisplay:
    """Live console display with complete v7.3 metrics"""

    def __init__(self, update_interval=10):
        self.update_interval = update_interval
        self.trial_count = 0

    def update(self, output_dict, trial_num):
        """Update display with complete v7.3 trial results"""
        self.trial_count = trial_num
        # === INSERT THIS BLOCK ===
        rolling_drift = output_dict.get('rolling_drift', False)
        if rolling_drift:
            intensity = output_dict.get('intensity', 0)
            thresh = output_dict.get('rolling_threshold', 0)
            # This prints a visual ALERT regardless of update interval
            print(f"\n📉 AUDIT ALERT: Focus Crash! (Int: {intensity} < Thr: {thresh})")
        if trial_num % self.update_interval != 0:
            return

        instant = output_dict.get('instant_state', 'UNKNOWN')
        windowed = output_dict.get('windowed_state', 'UNKNOWN')
        conf = output_dict.get('confidence', 0)
        intensity = output_dict.get('intensity', 0)
        # engagement = output_dict.get('engagement', 0)
        signal_q = output_dict.get('signal_quality', 0)
        drift_label = output_dict.get('drift_label', 'NONE')
        cum_drift = output_dict.get('cumulative_drift_pct', 0)  # ✅ ADD THIS

        emoji = {
            'Optimal-Engaged': '🎯',
            'Optimal-Monitoring': '✅',
            'Mind-Wandering': '🌀',
            'Fatigue': '😴',
            'Overload': '🔥',
            'Calibrating': '⚙️',
        }

        ar_sigma = output_dict.get('arousal_sigma', 0.0)
        ctrl_sigma = output_dict.get('control_sigma', 0.0)

        print(
            f"[{trial_num:4d}] {emoji.get(instant, '❓')}{instant:20s} → "
            f"{emoji.get(windowed, '❓')}{windowed:20s} | "
            f"Ar:{ar_sigma:+.1f}σ | Ctl:{ctrl_sigma:+.1f}σ | "  # <--- CHANGED THIS
            f"SQ:{signal_q:.2f} | CumDrift:{cum_drift:5.1f}% | Drift:{drift_label:8s}"
        )
# ============================================================================
# SESSION RUNNER - MAIN ORCHESTRATOR
# ============================================================================

class SessionRunner:
    """Main session runner - integrates monitor, logging, display, pause control"""

    def __init__(
            self,
            task_type="learning",
            duration_minutes=90,
            session_name=None,
            test_interventions=False,  # ✅ ADD THIS
    ):
        self.task_type = task_type
        self.duration_minutes = duration_minutes
        self.test_interventions = test_interventions  # ✅ ADD THIS

        self.monitor = CognitiveMonitorV7_3(task_type=task_type)
        self.session_mgr = SessionManager(task_type=task_type, session_name=session_name)
        self.display = RealtimeDisplay(update_interval=10)
        self.pause_mgr = PauseManager()
        self.baseline_persistence = BaselinePersistence(user_id="default")

        # ✅ ADD: Recommendation tracking
        self.recommendation_shown = False
        self.last_recommendation_trial = 0

        # ✅ NEW: quiet-mode flag
        self.quiet_mode = False

        # ✅ Initial sync of quiet flag to engine
        self.monitor.quiet_mode = self.quiet_mode
        self.monitor.baseline_detector.quiet_mode = self.quiet_mode

        self.inlet = None
        self.epoch_count = 0
        self.rejected_count = 0
        # ✅ ADD: Intervention tracking
        self.last_intervention_time = None
        self.intervention_cooldown = 300  # 5 minutes between interventions (300 seconds)
        self.interventions_sent = 0
        self.interventions_ignored = 0
        self.stream_fail_count = 0
        self.initial_recommendation = None  # Store pre-session recommendation
        self.trials_output = []  # Store all trial outputs

        # Try to load previous baseline (instant calibration)
        # prev_baseline = self.baseline_persistence.load_baseline(session_name="latest")
        # if prev_baseline:
        #     print("✅ Loaded previous baseline - instant calibration!")
        #     self.monitor.baseline_detector.baseline = prev_baseline.get('baseline')
        #     self.monitor.baseline_detector.baseline_ready = True
        #     self.monitor.ig_computer.pca = prev_baseline.get('pca')
        #     self.monitor.ig_computer.scaler = prev_baseline.get('scaler')

        print("✅ SessionRunner initialized")
        print(f"   Task: {task_type}")
        print(f"   Duration: {duration_minutes} min")
        print(f"   Controls: Press 'p' to pause, 'q' to quit anytime\n")

    def setup_connection(self):
        """Setup Muse EEG connection via LSL"""
        try:
            self.inlet = connect_muse()
            print("✅ Connected to Muse via LSL\n")
        except Exception as e:
            print(f"❌ Failed to connect Muse: {e}")
            print("Cannot proceed without LSL stream\n")
            raise

    def collect_morning_baseline(self):
        """Collect 60-sec morning baseline (separate from session monitoring)"""
        print("\n" + "=" * 100)
        print("☀️ MORNING BASELINE - ONE-TIME SETUP")
        print("=" * 100)
        print("This captures your fresh brain state for today.")
        print()
        print("Instructions:")
        print("  • Sit comfortably, eyes open")
        print("  • Look at neutral point")
        print("  • Relax, don't think about tasks")
        print("  • Duration: 60 seconds")
        print()

        input("Press Enter when ready...")

        features_collected = []
        print("\n⏱️  Collecting (60 sec)...\n")

        for i in range(30):  # 30 epochs × 2 sec
            try:
                epoch_data = collect_epoch(self.inlet, duration=EPOCH_DURATION, sfreq=SAMPLE_RATE)
                if epoch_data is None:
                    continue

                # ADD THIS: Reject artifacts during baseline collection
                is_clean, reason = is_epoch_clean(epoch_data, use_adaptive=USE_ADAPTIVE_ARTIFACT)
                if not is_clean:
                    print(f"  Rejected: {reason}")  # Show user what's happening
                    continue

                features = extract_features(epoch_data, sfreq=SAMPLE_RATE, entropy_engine=EntropyEngine)
                if features is not None:
                    features_collected.append(features)

                if (i + 1) % 5 == 0:
                    print(f"  ✓ {len(features_collected)} clean epochs...")
            except:
                continue

        if len(features_collected) < 20:
            raise ValueError(f"Only {len(features_collected)}/20 clean epochs")

        baseline = self.monitor.daily_baseline_mgr.store_baseline(features_collected)

        # ✅ ADD THIS: Show 3-axis preview
        print(f"\n{'=' * 60}")
        print("📊 YOUR MORNING BASELINE (Reference State)")
        print(f"{'=' * 60}")

        # Compute initial 3-axis from morning baseline
        # (All should be ~0σ since this IS the baseline)
        sample_features = features_collected[-1]  # Use last epoch
        z_scores = self.monitor.daily_baseline_mgr.compute_z_scores(sample_features)

        if z_scores:
            ar, ctrl, ex = self.monitor.task_recommender.compute_sigma_state(z_scores)

            print(f"   Arousal:   {ar:+5.2f}σ")
            print(f"   Control:   {ctrl:+5.2f}σ")
            print(f"   Executive: {ex:+5.2f}σ")

            # ✅ ARTIFACT CHECK
            if abs(ex) > 5 or abs(ar) > 5 or abs(ctrl) > 5:
                print(f"\n⚠️  WARNING: Possible artifact detected!")
                print(f"   If any value > 5σ, consider retaking baseline.")
                response = input("\n   Retake baseline? (y/n): ").strip().lower()
                if response == 'y':
                    print("\n🔄 Retaking baseline...\n")
                    return self.collect_morning_baseline()  # Recursive retry

            print(f"\n   This is your reference state for today.")
            print(f"   Pre-session measurements compare against this.\n")
            print(f"{'=' * 60}\n")

        return baseline

    def measure_quick_state(self):
        """Collect 30-sec current state"""
        print("🧠 Measuring current state (60 sec)...\n")

        features_collected = []

        for i in range(30):  # 30 epochs × 2 sec
            try:
                epoch_data = collect_epoch(self.inlet, duration=EPOCH_DURATION, sfreq=SAMPLE_RATE)
                if epoch_data is None:
                    continue

                is_clean, _ = is_epoch_clean(epoch_data, use_adaptive=USE_ADAPTIVE_ARTIFACT)
                if not is_clean:
                    continue

                features = extract_features(epoch_data, sfreq=SAMPLE_RATE, entropy_engine=EntropyEngine)
                if features is not None:
                    features_collected.append(features)
            except:
                continue

        if len(features_collected) < 10:
            raise ValueError(f"Only {len(features_collected)}/10 clean epochs")

        print(f"✅ Measurement complete ({len(features_collected)} epochs)\n")
        return pd.concat(features_collected, axis=1).T.mean()

    def _print_recommendation_with_dynamic_options(self, rec):
        """
        Present recommendation and FORCE user choice (Blocking Input).
        No timeouts. No auto-select.
        """
        print(f"\n{'=' * 100}")
        print(f"📊 YOUR CURRENT BRAIN STATE")
        print(f"{'=' * 100}")

        # 1. DISPLAY NEUROSCIENCE METRICS
        ar = rec.get('arousal_sigma', 0.0)
        ct = rec.get('control_sigma', 0.0)
        ex = rec.get('executive_sigma', 0.0)

        print(f"   {rec['recommendation']}")
        print(f"   Arousal:   {ar:+5.2f}σ | Control:   {ct:+5.2f}σ | Executive: {ex:+5.2f}σ")

        print("\n   Best for:")
        for task in rec['tasks'][:4]:
            print(f"     • {task}")

        print(f"\n   Duration: {rec['duration']}")
        if rec.get('boost_suggestions'):
            print(f"   💡 Boosts: {', '.join(rec['boost_suggestions'][:2])}")
        print(f"   Confidence: {rec.get('confidence', 0):.0%}")
        print(f"{'=' * 100}\n")

        print("What will you ACTUALLY do right now?")

        # 2. DEFINE OPTIONS (The 5 Validated Modes)
        # Format: "Key": ("INTERNAL_MODE_NAME", "Display Label")
        options = {
            "1": ("PEAK_GENERATIVE", "🚀 PEAK WORK (Complex Creation/Coding)"),
            "2": ("RECEPTIVE_ENGAGED", "📚 ACTIVE LEARNING (Study/Review)"),
            "3": ("STANDARD_WORK", "🔨 STANDARD WORK (Routine/Admin)"),
            "4": ("PASSIVE_CONSUMPTION", "🎬 PASSIVE (Watch/Read)"),
            "5": ("FATIGUE", "😴 TAKE A BREAK")
        }

        # 3. IDENTIFY SYSTEM RECOMMENDATION
        # We rely on the 'mode' key from the engine, or infer it if missing
        rec_mode = rec.get('mode', 'STANDARD_WORK')

        # Print options with star next to recommendation
        for key, (mode_code, label) in options.items():
            prefix = "⭐ " if mode_code == rec_mode else "   "
            print(f"{prefix}{key}) {label}")

        print()

        # 4. BLOCKING INPUT LOOP (The Fix)
        selected_mode = None

        while True:
            # This input() will PAUSE everything until you type something
            choice = input(f"Enter 1-5 (default {rec_mode}): ").strip()

            # ENTER accepts the recommendation
            if choice == "":
                print(f"\n→ You accepted: {rec_mode}")
                selected_mode = rec_mode
                break

            # Valid choice
            if choice in options:
                selected_mode = options[choice][0]
                print(f"\n→ You chose OVERRIDE: {selected_mode}")
                break

            print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")

        # 5. HANDLE BREAK VS SESSION
        if selected_mode == "FATIGUE":
            print("\n💤 Taking a break - session will not start")
            print("   Run the program again when ready\n")
            import sys
            sys.exit(0)

        # 6. UPDATE SESSION MANAGER & RETURN
        self.session_mgr.update_task_type(selected_mode)

        # Log the specific target for later analysis
        self.session_mgr.metadata['target_state'] = selected_mode
        self.session_mgr.metadata['recommendation_text'] = rec['recommendation']

        return selected_mode

    def _get_choice_input(self, max_choice):
        """Helper to get valid choice input"""
        while True:
            try:
                choice = input(f"\nEnter 1-{max_choice}: ").strip()
                if choice in [str(i) for i in range(1, max_choice + 1)]:
                    return choice
                else:
                    print(f"⚠️  Invalid choice. Enter 1-{max_choice}")
            except (EOFError, KeyboardInterrupt):
                return "1"  # Default to recommended option

    def _check_and_intervene(self, output):
        """
        Check if intervention is needed based on cumulative drift

        Intervention strategy (from lab data analysis):
        - Cumulative drift >50% for 2 minutes = intervention trigger
        - Cooldown: 5 minutes between interventions
        - Notification: Desktop popup with suggested action
        """

        cumulative_drift = output.get('cumulative_drift_pct', 0)
        windowed_state = output.get('windowed_state', '')

        # Check cooldown
        current_time = time.time()
        if self.last_intervention_time is not None:
            time_since_last = current_time - self.last_intervention_time
            if time_since_last < self.intervention_cooldown:
                return  # Still in cooldown

        # Intervention threshold: 50% cumulative drift
        INTERVENTION_THRESHOLD = 50.0

        if cumulative_drift > INTERVENTION_THRESHOLD:
            self.interventions_sent += 1
            self.last_intervention_time = current_time

            # Generate context-specific message
            if windowed_state == 'Mind-Wandering':
                title = "🌀 Attention Drifting Detected"
                message = (
                    f"You've been drifting {cumulative_drift:.0f}% of the last 2 minutes.\n\n"
                    "Suggested actions:\n"
                    "  • Pause and take 3 notes on what you learned\n"
                    "  • Switch to a different subtask\n"
                    "  • Take a 5-minute break"
                )

            elif windowed_state == 'Fatigue':
                title = "😴 Fatigue Detected"
                message = (
                    f"You've been in fatigue state {cumulative_drift:.0f}% of the last 2 minutes.\n\n"
                    "Suggested actions:\n"
                    "  • Take a 15-minute break (walk, coffee)\n"
                    "  • Power nap (20 minutes)\n"
                    "  • End session (work quality is dropping)"
                )

            else:
                title = "⚠️ Sustained Low Focus"
                message = (
                    f"You've been drifting {cumulative_drift:.0f}% of the last 2 minutes.\n\n"
                    "Suggested actions:\n"
                    "  • Take a brief break (5-10 min)\n"
                    "  • Switch to easier task\n"
                    "  • Check if you're stuck on a problem"
                )

            # Send notification
            send_desktop_notification(title, message, urgency='normal')

            # Also print to console
            print(f"\n{'=' * 100}")
            print(f"{title}")
            print(f"{'=' * 100}")
            print(message)
            print(f"{'=' * 100}\n")

            # Log intervention
            self.session_mgr.metadata.setdefault('interventions', []).append({
                'trial': output.get('trial'),
                'time': datetime.datetime.now().isoformat(),
                'cumulative_drift_pct': cumulative_drift,
                'state': windowed_state,
                'message': message,
            })

    def run(self):
        self.setup_connection()

        # PRE-SESSION RECOMMENDATION
        if self.monitor.daily_baseline_mgr.needs_baseline():
            print("\n📅 New day - morning baseline required\n")
            try:
                self.collect_morning_baseline()
            except Exception as e:
                print(f"❌ Baseline failed: {e}")

        if not self.monitor.daily_baseline_mgr.needs_baseline():
            print("\n" + "=" * 100)
            print("🎯 PRE-SESSION READINESS CHECK")
            print("=" * 100)

            try:
                current_features = self.measure_quick_state()
                z_scores = self.monitor.daily_baseline_mgr.compute_z_scores(current_features)

                if z_scores:
                    rec = self.monitor.task_recommender.recommend_task([z_scores] * 20)

                    # ✅ STORE pre-session metrics in metadata
                    ar_pre, ctrl_pre, ex_pre = self.monitor.task_recommender.compute_sigma_state(z_scores)

                    self.session_mgr.metadata['pre_session'] = {
                        'arousal_sigma': round(ar_pre, 2),
                        'control_sigma': round(ctrl_pre, 2),
                        'executive_sigma': round(ex_pre, 2),
                        'recommendation': rec['recommendation'],
                        'recommended_mode': rec['mode'],
                        'confidence': round(rec['confidence'], 2)
                    }
                    # STORE recommendation
                    self.initial_recommendation = rec  # ADD THIS LINE

                    task_mode = self._print_recommendation_with_dynamic_options(rec)
                    self.session_mgr.metadata['pre_session_recommendation'] = rec
                    self.session_mgr.metadata['user_task_choice'] = task_mode
                    self.session_mgr.update_task_type(task_mode)
                    self.monitor.task_type = task_mode
                    self.session_mgr.start_session()
                    self.quiet_mode = True
                    self.monitor.quiet_mode = True
                    self.monitor.baseline_detector.quiet_mode = True
            except Exception as e:
                print(f"⚠️  Could not generate recommendation: {e}\n")

        # Enable quiet mode
        self.quiet_mode = True
        self.monitor.quiet_mode = True
        self.monitor.baseline_detector.quiet_mode = True

        print("=" * 100)
        print("🚀 SESSION RUNNING (Real Muse Data)...")
        if not self.monitor.baseline_detector.baseline_ready:
            print("Waiting for session baseline (3-4 min, ~100 trials)...")
        print("Controls: 'p' pause, 'q' quit")
        print("=" * 100 + "\n")

        start_time = time.time()
        target_duration = self.duration_minutes * 60

        try:
            while (time.time() - start_time) < target_duration:
                self.pause_mgr.check_input()
                if self.pause_mgr.should_quit():
                    print("❌ User requested quit")
                    break
                self.pause_mgr.handle_pause()

                loop_iteration_start = time.time()

                try:
                    epoch_data = collect_epoch(self.inlet, duration=EPOCH_DURATION, sfreq=SAMPLE_RATE)

                    if epoch_data is None:
                        self.stream_fail_count += 1
                        print(f"⚠️ Stream Missing... ({self.stream_fail_count})")

                        if self.stream_fail_count >= 3 and (self.stream_fail_count - 3) % 10 == 0:
                            print('\a')
                            print("\n" + "!" * 50)
                            print(f"❌ CRITICAL: STREAM LOST (Attempt {self.stream_fail_count})")
                            print("!" * 50 + "\n")
                            try:
                                send_desktop_notification(
                                    "❌ HEADSET DISCONNECTED",
                                    f"Stream down for {self.stream_fail_count * 2}s! Check Bluetooth.",
                                    urgency='critical'
                                )
                            except Exception as e:
                                print(f"Notification failed: {e}")
                        continue
                    else:
                        if self.stream_fail_count > 0:
                            print("✅ Stream Recovered!")
                        self.stream_fail_count = 0

                    self.epoch_count += 1

                    if self.epoch_count % 10 == 0:
                        elapsed = time.time() - loop_iteration_start
                        if elapsed < 1.5:
                            print(f"⚠️ Trial {self.epoch_count}: Too fast! ({elapsed:.2f}s)")

                    is_clean, reason = is_epoch_clean(
                        epoch_data,
                        amp_thresh=AMP_THRESHOLD_UV,
                        kurt_thresh=KURTOSIS_THRESHOLD,
                        use_adaptive=USE_ADAPTIVE_ARTIFACT,
                    )

                    self.monitor.process_epoch(epoch_data, is_artifact=not is_clean)

                    if not is_clean:
                        self.rejected_count += 1
                        loop_elapsed = time.time() - loop_iteration_start
                        if loop_elapsed < EPOCH_DURATION:
                            time.sleep(EPOCH_DURATION - loop_elapsed)
                        continue

                    features = extract_features(epoch_data, sfreq=SAMPLE_RATE, entropy_engine=EntropyEngine)
                    if features is None or features.isnull().all():
                        self.rejected_count += 1
                        loop_elapsed = time.time() - loop_iteration_start
                        if loop_elapsed < EPOCH_DURATION:
                            time.sleep(EPOCH_DURATION - loop_elapsed)
                        continue

                    output = self.monitor.process_trial(features)
                    trial_num = self.monitor.trial_count

                    if self.test_interventions and trial_num == 50:
                        print("\n🧪 TEST MODE: Injecting fake cumulative drift (75%)...\n")
                        output['cumulative_drift_pct'] = 75.0

                    self.session_mgr.log_trial(output)

                    # STORE trial output for later analysis
                    self.trials_output.append(output)  # ADD THIS LINE

                    self.display.update(output, trial_num)

                    if self.monitor.baseline_detector.baseline_ready and trial_num > 30:
                        self._check_and_intervene(output)

                    loop_elapsed = time.time() - loop_iteration_start
                    if loop_elapsed < EPOCH_DURATION:
                        time.sleep(EPOCH_DURATION - loop_elapsed)

                except KeyboardInterrupt:
                    print("\n❌ User interrupted session (Ctrl+C)")
                    break
                except Exception as e:
                    print(f"❌ Trial processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    loop_elapsed = time.time() - loop_iteration_start
                    if loop_elapsed < EPOCH_DURATION:
                        time.sleep(EPOCH_DURATION - loop_elapsed)
                    continue

        except KeyboardInterrupt:
            print("\n❌ Session interrupted by user")

        finally:
            self.session_mgr.end_session()
            self.session_mgr.save_metadata(self.monitor)

            # ✅ ADD: Session averages to metadata
            if len(self.trials_output) > 0:
                try:
                    avg_arousal = np.mean([t.get('arousal_sigma', 0) for t in self.trials_output])
                    avg_control = np.mean([t.get('control_sigma', 0) for t in self.trials_output])
                    avg_executive = np.mean([t.get('executive_sigma', 0) for t in self.trials_output])
                    avg_efficiency = np.mean([t.get('efficiency', 0) for t in self.trials_output])

                    self.session_mgr.metadata['session_averages'] = {
                        'arousal_sigma': round(avg_arousal, 2),
                        'control_sigma': round(avg_control, 2),
                        'executive_sigma': round(avg_executive, 2),
                        'efficiency': round(avg_efficiency, 1)
                    }

                    # ✅ Validation logging
                    from validation_logger import ValidationLogger
                    logger = ValidationLogger()
                    logger.prompt_post_session(
                        session_id=self.session_mgr.session_name,
                        recommendation=self.initial_recommendation['recommendation'],
                        avg_metrics={
                            'arousal': avg_arousal,
                            'control': avg_control,
                            'executive': avg_executive
                        }
                    )
                except Exception as e:
                    print(f"⚠️  Metadata update failed: {e}")
            # FIXED: Only do validation logging if we have data
            if len(self.trials_output) > 0 and self.initial_recommendation is not None:
                try:
                    avg_arousal = np.mean([t.get('arousal_sigma', 0) for t in self.trials_output])
                    avg_control = np.mean([t.get('control_sigma', 0) for t in self.trials_output])
                    avg_executive = np.mean([t.get('executive_sigma', 0) for t in self.trials_output])

                    from validation_logger import ValidationLogger
                    logger = ValidationLogger()
                    logger.prompt_post_session(
                        session_id=self.session_mgr.session_name,
                        recommendation=self.initial_recommendation['recommendation'],
                        avg_metrics={
                            'arousal': avg_arousal,
                            'control': avg_control,
                            'executive': avg_executive
                        }
                    )
                except Exception as e:
                    print(f"⚠️  Validation logging skipped: {e}")
                    # 👇 ADD THIS BLOCK FOR THIEL METRICS 👇
                    if len(self.trials_output) > 10:
                        try:
                            # Convert list of dicts to DataFrame for easy math
                            df_sess = pd.DataFrame(self.trials_output)

                            # 1. Calculate Core Metrics
                            avg_intensity = df_sess['intensity'].mean()
                            # stability_cost = df_sess['control_sigma'].std()
                            #
                            # # Prevent division by zero
                            # if stability_cost < 0.1: stability_cost = 0.1

                            # 2. The Thiel Metric: Neural Efficiency Index
                            # Formula: Output (Intensity) divided by Biological Cost (Instability)
                            # nei = avg_intensity / stability_cost

                            # print("\n" + "=" * 60)
                            # print("🔬 THIEL VALIDATION METRICS")
                            # print("=" * 60)
                            # print(f"🧠 NEURAL EFFICIENCY INDEX (NEI): {nei:.2f}")
                            # print(f"   (Intensity: {avg_intensity:.1f} / Stability Cost: {stability_cost:.2f})")

                            # 3. Context-Aware Success Criteria
                            # if self.monitor.task_type in ['PEAK_GENERATIVE', 'STANDARD_WORK', 'peak_work']:
                            #     print(f"📊 MODE: WORK ({self.monitor.task_type})")
                            #     print(f"   ✅ Success = High NEI (>50)")
                            #     print(f"   ❌ Failure = High Cost (Stability > 0.5)")
                            # else:
                            #     print(f"🧘 MODE: RECOVERY ({self.monitor.task_type})")
                            #     print(f"   ✅ Success = Low Cost (Stability < 0.2)")
                            #     print(f"   ℹ️ Note: NEI is less relevant here.")
                            #
                            # # Save to metadata for permanent record
                            # self.session_mgr.metadata['nei_score'] = round(nei, 2)
                            # self.session_mgr.metadata['stability_cost'] = round(stability_cost, 2)

                        except Exception as e:
                            print(f"⚠️ Could not calculate Thiel metrics: {e}")
                    # 👆 END NEW BLOCK 👆
            baseline_file = self.baseline_persistence.save_baseline(
                self.monitor.baseline_detector,
                self.monitor.ig_computer,
                session_name=self.session_mgr.session_name
            )

            if baseline_file:
                print(f"✅ Baseline saved for next session: {baseline_file.name}")
            else:
                print("⚠️  Baseline not ready - will need full calibration next session")

            self._print_final_stats()




    def _print_final_stats(self):
        """Print final session statistics"""
        total_trials = self.monitor.trial_count
        total_epochs = self.epoch_count
        rejection_rate = (self.rejected_count / total_epochs * 100) if total_epochs > 0 else 0

        print("\n" + "=" * 120)
        print("📊 FINAL STATISTICS")
        print("=" * 120)
        print(f"Total epochs collected: {total_epochs}")
        print(f"Epochs rejected: {self.rejected_count} ({rejection_rate:.1f}%)")
        print(f"Valid trials: {total_trials}")

        if total_trials > 0:
            state_counts = {}
            for state in self.monitor.state_history:
                state_counts[state] = state_counts.get(state, 0) + 1

            print(f"\nState distribution:")
            for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
                pct = (count / total_trials) * 100
                print(f"  {state:25s}: {count:4d} ({pct:5.1f}%)")
        # ✅ ADD: Intervention statistics
        if hasattr(self, 'interventions_sent') and self.interventions_sent > 0:
            print(f"\n📢 Interventions:")
            print(f"  Total sent: {self.interventions_sent}")
            if hasattr(self, 'interventions_ignored'):
                print(f"  User ignored: {self.interventions_ignored}")
            print(
                f"  Avg cumulative drift at intervention: {self.session_mgr.metadata.get('avg_intervention_drift', 0):.1f}%")

        print("=" * 120 + "\n")



# ============================================================================
# MAIN - CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cognitive Monitor v7.3 COMPLETE - Real-Time EEG Session Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python session_runner_v7_3_complete.py --task learning --duration 90
    Run 90-minute learning session with real Muse

  python session_runner_v7_3_complete.py --task work --duration 60 --session-name focus_block
    Run 60-minute work session with custom name

  python session_runner_v7_3_complete.py --task learning --duration 5 --no-lsl
    Quick 5-minute test with simulated data (no Muse needed)

Controls during session:
  Press 'p' to pause (non-blocking), Enter to resume
  Press 'q' to quit anytime
        """,
    )

    parser.add_argument(
        '--task',
        choices=['learning', 'work'],
        default='learning',
        help='Initial task type (learning or work)',
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=90,
        help='Session duration in minutes (default 90)',
    )
    parser.add_argument(
        '--session-name',
        type=str,
        default=None,
        help='Custom session name (default auto-generated)',
    )
    parser.add_argument(
        '--no-lsl',
        action='store_true',
        help='Use simulated data instead of Muse LSL stream',
    )
    parser.add_argument(
        '--test-interventions',
        action='store_true',
        help='Test mode: Force intervention at trial 50',
    )  # ✅ ADD THIS

    args = parser.parse_args()

    runner = SessionRunner(
        task_type=args.task,
        duration_minutes=args.duration,
        session_name=args.session_name,
        test_interventions=args.test_interventions,  # ✅ ADD THIS
    )

    runner.run()



if __name__ == '__main__':
    main()

