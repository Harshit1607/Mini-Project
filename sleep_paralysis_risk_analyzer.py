"""
SOMNUSGUARD - ENHANCED RISK ANALYSIS (V2)
Per-Subject Analysis Pipeline - Random Forest Compatible

This script works with the Random Forest classifier:
1.  Loads the trained RF model and the StandardScaler.
2.  Loops through each subject in the data directory.
3.  For each subject:
    a. Loads their full, continuous data.
    b. Extracts features and scales them.
    c. Generates predictions using the RF model.
    d. Calculates the model's accuracy for this subject.
    e. Runs the SleepParalysisRiskAnalyzer on the predictions.
4.  Collects all results and generates a final summary report.

Authors: Harshit Bareja, Ishika Manchanda, Teena Kaintura
Guide: Ms. Nupur Chugh
Institution: Bharati Vidyapeeth College of Engineering
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import butter, filtfilt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import Counter
import joblib
import time

warnings.filterwarnings('ignore')

# --- Configuration ---
EPOCH_DURATION = 30  # seconds (standard)
SAMPLING_RATE = 50   # Hz (approximate from dataset)
NUM_FEATURES = 25    # Must match the model!

# --- File Paths ---
DATA_PATH = './sleep-accel-data/'
OUTPUT_DIR = './outputs'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_sleep_model.joblib')
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler.joblib')

# --- Mappings (Must match training script) ---
LABEL_MAPPING = {
    -1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6
}
REVERSE_LABEL_MAPPING = {
    0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5
}
SLEEP_STAGE_MAPPING = {
    -1: 'Unknown', 0: 'Wake', 1: 'N1', 2: 'N2', 
    3: 'N3', 4: 'N4', 5: 'REM'
}

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def load_subject_data(subject_id, data_path):
    """
    Load acceleration and labeled sleep data for a single subject
    """
    accel_file = os.path.join(data_path, f'{subject_id}_acceleration.txt')
    accel_df = pd.read_csv(accel_file, sep=' ', names=['timestamp', 'x', 'y', 'z'])
    
    labels_file = os.path.join(data_path, f'{subject_id}_labeled_sleep.txt')
    labels_df = pd.read_csv(labels_file, sep=' ', names=['timestamp', 'stage'])
    
    return accel_df, labels_df

def calculate_magnitude(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def extract_epoch_features(epoch_data):
    """
    Extract statistical features from 30-second epoch
    """
    features = {}
    
    magnitude = calculate_magnitude(
        epoch_data['x'].values,
        epoch_data['y'].values, 
        epoch_data['z'].values
    )
    
    features['mean_mag'] = np.mean(magnitude)
    features['std_mag'] = np.std(magnitude)
    features['min_mag'] = np.min(magnitude)
    features['max_mag'] = np.max(magnitude)
    features['median_mag'] = np.median(magnitude)
    features['var_mag'] = np.var(magnitude)
    features['range_mag'] = np.max(magnitude) - np.min(magnitude)
    features['q25_mag'] = np.percentile(magnitude, 25)
    features['q75_mag'] = np.percentile(magnitude, 75)
    features['iqr_mag'] = features['q75_mag'] - features['q25_mag']
    features['skew_mag'] = stats.skew(magnitude)
    features['kurtosis_mag'] = stats.kurtosis(magnitude)
    features['energy_mag'] = np.sum(magnitude**2)
    
    magnitude_centered = magnitude - np.mean(magnitude)
    zero_crossings = np.sum(np.diff(np.sign(magnitude_centered)) != 0)
    features['zero_crossing_rate'] = zero_crossings / len(magnitude)
    
    for axis in ['x', 'y', 'z']:
        data = epoch_data[axis].values
        features[f'mean_{axis}'] = np.mean(data)
        features[f'std_{axis}'] = np.std(data)
        features[f'range_{axis}'] = np.max(data) - np.min(data)
    
    diff_mag = np.diff(magnitude)
    features['mean_diff'] = np.mean(np.abs(diff_mag))
    features['std_diff'] = np.std(diff_mag)
    
    return features

def create_epochs(accel_df, labels_df):
    """
    Segment continuous data into 30-second epochs and extract features
    """
    features_list = []
    labels_list = []
    
    start_time = max(accel_df['timestamp'].min(), labels_df['timestamp'].min())
    end_time = min(accel_df['timestamp'].max(), labels_df['timestamp'].max())
    
    current_time = start_time
    while current_time + EPOCH_DURATION <= end_time:
        epoch_accel = accel_df[
            (accel_df['timestamp'] >= current_time) & 
            (accel_df['timestamp'] < current_time + EPOCH_DURATION)
        ]
        
        epoch_label = labels_df[
            (labels_df['timestamp'] >= current_time) &
            (labels_df['timestamp'] < current_time + EPOCH_DURATION)
        ]
        
        if len(epoch_accel) > 10 and len(epoch_label) > 0:
            features = extract_epoch_features(epoch_accel)
            features_list.append(features)
            
            label = epoch_label['stage'].mode()[0]
            labels_list.append(label)
        
        current_time += EPOCH_DURATION
    
    return features_list, labels_list

# ========================================================================
# RISK ANALYZER CLASS
# ========================================================================

class SleepParalysisRiskAnalyzer:
    """
    Analyzes hypnogram (sleep stage sequence) to detect patterns 
    associated with sleep paralysis risk.
    """
    
    def __init__(self, epoch_duration=30):
        self.epoch_duration = epoch_duration
        self.risk_thresholds = {
            'rem_wake_transitions_high': 8,
            'rem_wake_transitions_moderate': 5,
            'rem_fragmentation_high': 12,
            'rem_fragmentation_moderate': 8,
            'total_awakenings_high': 20,
            'rem_percentage_low': 15,
            'rem_percentage_high': 30,
            'rem_latency_long': 120,
            'stage_transition_rate_high': 25
        }
        
    def analyze_hypnogram(self, sleep_stages):
        stages = np.array(sleep_stages)
        
        risk_report = {
            'total_epochs': len(stages),
            'duration_hours': len(stages) * self.epoch_duration / 3600,
            'stage_distribution': self._calculate_stage_distribution(stages),
            'rem_analysis': self._analyze_rem_patterns(stages),
            'transition_analysis': self._analyze_transitions(stages),
            'fragmentation_analysis': self._analyze_fragmentation(stages),
            'sleep_architecture': self._analyze_sleep_architecture(stages),
            'risk_factors': [],
            'risk_score': 0,
            'risk_level': 'Low'
        }
        
        risk_report = self._calculate_risk_score(risk_report)
        
        return risk_report
    
    def _calculate_stage_distribution(self, stages):
        stage_counts = Counter(stages)
        total = len(stages)
        
        distribution = {
            'wake_pct': (stage_counts.get(0, 0) / total) * 100,
            'n1_pct': (stage_counts.get(1, 0) / total) * 100,
            'n2_pct': (stage_counts.get(2, 0) / total) * 100,
            'n3_pct': (stage_counts.get(3, 0) / total) * 100,
            'n4_pct': (stage_counts.get(4, 0) / total) * 100,
            'rem_pct': (stage_counts.get(5, 0) / total) * 100,
            'unknown_pct': (stage_counts.get(-1, 0) / total) * 100
        }
        distribution['deep_sleep_pct'] = distribution['n3_pct'] + distribution['n4_pct']
        return distribution
    
    def _analyze_rem_patterns(self, stages):
        rem_analysis = {
            'rem_periods': [], 'rem_period_count': 0,
            'rem_latency_minutes': None, 'longest_rem_period': 0,
            'shortest_rem_period': float('inf'), 'avg_rem_period_length': 0,
            'rem_fragmentation_index': 0
        }
        
        in_rem = False
        rem_start = None
        
        for i, stage in enumerate(stages):
            if stage == 5 and not in_rem:
                in_rem = True
                rem_start = i
            elif stage != 5 and in_rem:
                in_rem = False
                rem_duration = i - rem_start
                rem_analysis['rem_periods'].append({
                    'start_epoch': rem_start, 'end_epoch': i,
                    'duration_minutes': (rem_duration * self.epoch_duration) / 60
                })
        
        if in_rem:
            rem_duration = len(stages) - rem_start
            rem_analysis['rem_periods'].append({
                'start_epoch': rem_start, 'end_epoch': len(stages),
                'duration_minutes': (rem_duration * self.epoch_duration) / 60
            })
            
        if rem_analysis['rem_periods']:
            rem_analysis['rem_period_count'] = len(rem_analysis['rem_periods'])
            first_rem = rem_analysis['rem_periods'][0]['start_epoch']
            rem_analysis['rem_latency_minutes'] = (first_rem * self.epoch_duration) / 60
            
            durations = [p['duration_minutes'] for p in rem_analysis['rem_periods']]
            rem_analysis['longest_rem_period'] = max(durations)
            rem_analysis['shortest_rem_period'] = min(durations)
            rem_analysis['avg_rem_period_length'] = np.mean(durations)
            rem_analysis['rem_fragmentation_index'] = len(rem_analysis['rem_periods'])
            
        return rem_analysis
        
    def _analyze_transitions(self, stages):
        transition_analysis = {
            'total_transitions': 0, 'rem_to_wake_count': 0,
            'wake_to_rem_count': 0, 'rem_to_any_count': 0,
            'transition_rate_per_hour': 0, 'transition_matrix': {},
            'critical_transitions': []
        }
        
        stage_names = {-1: 'Unknown', 0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'N4', 5: 'REM'}
        
        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]
            
            if current_stage != next_stage:
                transition_analysis['total_transitions'] += 1
                
                if current_stage == 5 and next_stage == 0:
                    transition_analysis['rem_to_wake_count'] += 1
                    transition_analysis['critical_transitions'].append({
                        'epoch': i, 'time_hours': (i * self.epoch_duration) / 3600,
                        'transition': 'REM to Wake'
                    })
                
                if current_stage == 0 and next_stage == 5:
                    transition_analysis['wake_to_rem_count'] += 1
                
                if current_stage == 5:
                    transition_analysis['rem_to_any_count'] += 1
                    
                trans_key = f"{stage_names.get(current_stage, 'Unknown')} to {stage_names.get(next_stage, 'Unknown')}"
                transition_analysis['transition_matrix'][trans_key] = \
                    transition_analysis['transition_matrix'].get(trans_key, 0) + 1
        
        duration_hours = len(stages) * self.epoch_duration / 3600
        transition_analysis['transition_rate_per_hour'] = \
            transition_analysis['total_transitions'] / duration_hours if duration_hours > 0 else 0
            
        return transition_analysis
        
    def _analyze_fragmentation(self, stages):
        fragmentation = {
            'awakening_count': 0, 'awakening_index': 0,
            'sleep_efficiency': 0, 'wake_after_sleep_onset_minutes': 0
        }
        
        sleep_onset_idx = None
        for i, stage in enumerate(stages):
            if stage != 0:
                sleep_onset_idx = i
                break
        
        if sleep_onset_idx is not None:
            in_wake = False
            for i in range(sleep_onset_idx, len(stages)):
                if stages[i] == 0 and not in_wake:
                    fragmentation['awakening_count'] += 1
                    in_wake = True
                elif stages[i] != 0:
                    in_wake = False
            
            wake_epochs = sum(1 for s in stages[sleep_onset_idx:] if s == 0)
            fragmentation['wake_after_sleep_onset_minutes'] = \
                (wake_epochs * self.epoch_duration) / 60
                
            time_in_bed = len(stages) * self.epoch_duration / 60
            time_asleep = time_in_bed - fragmentation['wake_after_sleep_onset_minutes']
            fragmentation['sleep_efficiency'] = (time_asleep / time_in_bed * 100) if time_in_bed > 0 else 0
            
            duration_hours = len(stages) * self.epoch_duration / 3600
            fragmentation['awakening_index'] = \
                fragmentation['awakening_count'] / duration_hours if duration_hours > 0 else 0
        
        return fragmentation
        
    def _analyze_sleep_architecture(self, stages):
        architecture = {
            'sleep_cycles_detected': 0, 'notes': []
        }
        return architecture
        
    def _calculate_risk_score(self, risk_report):
        risk_factors = []
        risk_score = 0
        
        # Factor 1: REM-to-Wake Transitions (MOST CRITICAL)
        rem_wake_trans = risk_report['transition_analysis']['rem_to_wake_count']
        if rem_wake_trans >= self.risk_thresholds['rem_wake_transitions_high']:
            risk_factors.append({
                'factor': 'High REM-to-Wake Transitions', 'severity': 'HIGH',
                'value': rem_wake_trans, 'points': 35
            })
            risk_score += 35
        elif rem_wake_trans >= self.risk_thresholds['rem_wake_transitions_moderate']:
            risk_factors.append({
                'factor': 'Moderate REM-to-Wake Transitions', 'severity': 'MODERATE',
                'value': rem_wake_trans, 'points': 20
            })
            risk_score += 20
            
        # Factor 2: REM Fragmentation
        rem_frag = risk_report['rem_analysis']['rem_period_count']
        if rem_frag >= self.risk_thresholds['rem_fragmentation_high']:
            risk_factors.append({
                'factor': 'High REM Fragmentation', 'severity': 'HIGH',
                'value': rem_frag, 'points': 25
            })
            risk_score += 25
        elif rem_frag >= self.risk_thresholds['rem_fragmentation_moderate']:
            risk_factors.append({
                'factor': 'Moderate REM Fragmentation', 'severity': 'MODERATE',
                'value': rem_frag, 'points': 15
            })
            risk_score += 15
            
        # Factor 3: Overall Sleep Fragmentation
        awakenings = risk_report['fragmentation_analysis']['awakening_count']
        if awakenings >= self.risk_thresholds['total_awakenings_high']:
            risk_factors.append({
                'factor': 'High Sleep Fragmentation', 'severity': 'MODERATE',
                'value': awakenings, 'points': 15
            })
            risk_score += 15
            
        # Factor 4: Abnormal REM Percentage
        rem_pct = risk_report['stage_distribution']['rem_pct']
        if rem_pct < self.risk_thresholds['rem_percentage_low']:
            risk_factors.append({
                'factor': 'Low REM Percentage', 'severity': 'MODERATE',
                'value': f"{rem_pct:.1f}%", 'points': 10
            })
            risk_score += 10
        elif rem_pct > self.risk_thresholds['rem_percentage_high']:
            risk_factors.append({
                'factor': 'High REM Percentage', 'severity': 'MODERATE',
                'value': f"{rem_pct:.1f}%", 'points': 10
            })
            risk_score += 10
            
        # Factor 5: High Transition Rate
        trans_rate = risk_report['transition_analysis']['transition_rate_per_hour']
        if trans_rate >= self.risk_thresholds['stage_transition_rate_high']:
            risk_factors.append({
                'factor': 'High Stage Transition Rate', 'severity': 'MODERATE',
                'value': f"{trans_rate:.1f}/hr", 'points': 10
            })
            risk_score += 10

        # Risk level classification
        if risk_score >= 70:
            risk_level = 'VERY HIGH'
        elif risk_score >= 50:
            risk_level = 'HIGH'
        elif risk_score >= 25:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
            
        risk_report['risk_factors'] = risk_factors
        risk_report['risk_score'] = risk_score
        risk_report['risk_level'] = risk_level
        
        return risk_report

# ========================================================================
# MAIN ANALYSIS FUNCTION
# ========================================================================

def main():
    """
    Main execution - loops through all subjects, predicts,
    analyzes, and builds a final report.
    """
    print("="*80)
    print("SOMNUSGUARD - ENHANCED RISK ANALYSIS REPORT")
    print("Running Per-Subject Analysis with Random Forest...")
    print("="*80)

    # --- 1. Load Model and Scaler ---
    print(f"\nLoading Random Forest model from: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"Could not load model: {e}")
        print("Please make sure you have run the classifier script first.")
        print(f"Expected file: {os.path.abspath(MODEL_PATH)}")
        return

    print(f"Loading scaler from: {SCALER_PATH}")
    try:
        scaler = joblib.load(SCALER_PATH)
        print("✓ Scaler loaded successfully")
    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"Could not load scaler: {e}")
        print("Please make sure you have run the classifier script first.")
        print(f"Expected file: {os.path.abspath(SCALER_PATH)}")
        return
        
    print("\n✓ Model and scaler loaded successfully.\n")

    # --- 2. Instantiate Analyzer ---
    analyzer = SleepParalysisRiskAnalyzer()

    # --- 3. Find Subjects ---
    try:
        subject_files = [f for f in os.listdir(DATA_PATH) if f.endswith('_acceleration.txt')]
        subject_ids = [f.replace('_acceleration.txt', '') for f in subject_files]
        if not subject_ids:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"\n--- FATAL ERROR ---")
        print(f"No data files found in: {os.path.abspath(DATA_PATH)}")
        print("Please ensure your data is in the correct folder.")
        return
        
    print(f"Found {len(subject_ids)} subjects. Starting analysis...\n")

    # --- 4. Loop, Predict, and Analyze ---
    all_reports = []
    
    for i, subject_id in enumerate(subject_ids, 1):
        print(f"Processing Subject {i}/{len(subject_ids)}: {subject_id} ...")
        try:
            # Load
            accel_df, labels_df = load_subject_data(subject_id, DATA_PATH)
            
            # Extract Features and True Labels
            features, labels = create_epochs(accel_df, labels_df)
            if len(features) == 0:
                print("  -> Skipping: No valid epochs extracted.")
                continue
                
            y_true = np.array(labels)
            X_df = pd.DataFrame(features)
            
            # Verify feature count
            if X_df.shape[1] != NUM_FEATURES:
                print(f"  -> Warning: Expected {NUM_FEATURES} features, got {X_df.shape[1]}")
            
            # Scale
            X_scaled = scaler.transform(X_df)
            
            # Predict with Random Forest (direct prediction, no sequences!)
            y_pred_mapped = model.predict(X_scaled)
            
            # Map back to original labels
            y_pred_hypnogram = np.vectorize(REVERSE_LABEL_MAPPING.get)(y_pred_mapped)
            
            # Analyze
            risk_report = analyzer.analyze_hypnogram(y_pred_hypnogram)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred_hypnogram)
            
            # Store results
            risk_report['subject_id'] = subject_id
            risk_report['accuracy'] = accuracy * 100
            all_reports.append(risk_report)
            
            print(f"  -> Accuracy: {accuracy*100:.1f}% | Risk: {risk_report['risk_level']}")

        except Exception as e:
            print(f"  -> ERROR processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    if not all_reports:
        print("No subjects were successfully analyzed.")
        return

    # --- 5. Generate Final Report ---
    
    # Flatten the reports for the DataFrame
    flat_reports = []
    for r in all_reports:
        try:
            flat_row = {
                'subject_id': r['subject_id'],
                'accuracy': r['accuracy'],
                'risk_level': r['risk_level'],
                'risk_score': r['risk_score'],
                'rem_to_wake': r['transition_analysis']['rem_to_wake_count'],
                'rem_fragmentation': r['rem_analysis']['rem_period_count'],
                'rem_pct': r['stage_distribution']['rem_pct'],
                'sleep_efficiency': r['fragmentation_analysis']['sleep_efficiency']
            }
            flat_reports.append(flat_row)
        except KeyError as e:
            print(f"Warning: Could not parse report for subject {r.get('subject_id', 'unknown')}. Missing key: {e}")

    results_df = pd.DataFrame(flat_reports)
    results_df = results_df.sort_values('risk_score', ascending=False)
    
    # Save detailed CSV
    csv_path = os.path.join(OUTPUT_DIR, 'somnusguard_full_report.csv')
    results_df.to_csv(csv_path, index=False, float_format='%.1f')
    print(f"\nDetailed CSV report saved to: {os.path.abspath(csv_path)}")
    
    # Prepare text report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("SOMNUSGUARD - ENHANCED RISK ANALYSIS REPORT")
    report_lines.append("Complete Sleep Paralysis Risk Assessment (Random Forest)")
    report_lines.append("="*80)
    report_lines.append(f"\nTotal Subjects Analyzed: {len(results_df)}")
    report_lines.append(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary Stats
    report_lines.append("--- EXECUTIVE SUMMARY ---")
    risk_dist = results_df['risk_level'].value_counts(normalize=True) * 100
    report_lines.append("Risk Level Distribution:")
    for level in ['VERY HIGH', 'HIGH', 'MODERATE', 'LOW']:
        if level in risk_dist.index:
            report_lines.append(f"  {level:<10}: {risk_dist[level]:5.1f}%")
        
    report_lines.append(f"\nMean Risk Score: {results_df['risk_score'].mean():.1f}")
    report_lines.append(f"Median Risk Score: {results_df['risk_score'].median():.1f}")
    
    report_lines.append(f"\nMean Model Accuracy: {results_df['accuracy'].mean():.1f}%")
    report_lines.append(f"Median Model Accuracy: {results_df['accuracy'].median():.1f}%")
    
    # High Risk Subjects
    high_risk_df = results_df[results_df['risk_score'] >= 50]
    report_lines.append("\n\n--- HIGH RISK SUBJECTS ---")
    report_lines.append(f"Found {len(high_risk_df)} subjects at elevated risk:\n")
    
    for _, row in high_risk_df.iterrows():
        report_lines.append(f"Subject: {row['subject_id']}")
        report_lines.append(f"  Risk Level: {row['risk_level']}")
        report_lines.append(f"  Risk Score: {row['risk_score']:.0f}")
        report_lines.append(f"  REM->Wake Transitions: {row['rem_to_wake']:.0f}")
        report_lines.append(f"  REM Fragmentation: {row['rem_fragmentation']:.0f} periods")
        report_lines.append(f"  Sleep Efficiency: {row['sleep_efficiency']:.1f}%")
        report_lines.append(f"  Model Accuracy: {row['accuracy']:.1f}%")
        report_lines.append("")

    # Full Report
    report_lines.append("\n--- COMPLETE SUBJECT-BY-SUBJECT ANALYSIS ---")
    report_lines.append(results_df.to_string(
        float_format="%.1f",
        columns=['subject_id', 'accuracy', 'risk_level', 'risk_score', 'rem_to_wake', 'rem_fragmentation', 'rem_pct', 'sleep_efficiency'],
        index=False
    ))
    
    report_lines.append("\n\n" + "="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    # Print to console
    final_report_text = "\n".join(report_lines)
    print(final_report_text)
    
    # Save text report
    txt_path = os.path.join(OUTPUT_DIR, 'somnusguard_full_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(final_report_text)
    print(f"\nFormatted text report saved to: {os.path.abspath(txt_path)}")


if __name__ == "__main__":
    main()