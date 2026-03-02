"""
SOMNUSGUARD - UNIFIED MULTI-MODEL ANALYZER
Analyzes all subjects using all 3 trained models and generates comprehensive reports

This script:
1. Loads all 3 models (RF, LSTM, XGBoost)
2. Processes each subject through each model
3. Generates individual model reports
4. Creates comparison reports
5. Exports everything as JSON for frontend

Authors: Harshit Bareja, Ishika Manchanda, Teena Kaintura
Guide: Ms. Nupur Chugh
Institution: Bharati Vidyapeeth College of Engineering
"""

import numpy as np
import pandas as pd
import os
import json
from scipy import stats
from sklearn.metrics import accuracy_score
from collections import Counter
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# CONFIGURATION
# ========================================================================

EPOCH_DURATION = 30
DATA_PATH = '../sleep-accel-data/'
OUTPUT_DIR = '../client/src/data'
INPUT_DIR = './outputs'

# Model configurations with prefixes
MODELS = {
    'rf': {
        'name': 'Random Forest',
        'model_path': os.path.join(INPUT_DIR, 'rf_sleep_model.joblib'),
        'scaler_path': os.path.join(INPUT_DIR, 'rf_scaler.joblib'),
        'color': '#3498db',
        'prefix': 'rf'
    },
    'lstm': {
        'name': 'LSTM',
        'model_path': os.path.join(INPUT_DIR, 'lstm_sleep_model.h5'),
        'scaler_path': os.path.join(INPUT_DIR, 'lstm_scaler.joblib'),
        'color': '#e74c3c',
        'prefix': 'lstm'
    },
    'xgboost': {
        'name': 'XGBoost',
        'model_path': os.path.join(INPUT_DIR, 'xgb_sleep_model.joblib'),
        'scaler_path': os.path.join(INPUT_DIR, 'xgb_scaler.joblib'),
        'color': '#2ecc71',
        'prefix': 'xgb'
    }
}

LABEL_MAPPING = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
REVERSE_LABEL_MAPPING = {0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
SLEEP_STAGE_MAPPING = {
    -1: 'Unknown', 0: 'Wake', 1: 'N1', 2: 'N2', 
    3: 'N3', 4: 'N4', 5: 'REM'
}

# ========================================================================
# FEATURE EXTRACTION (Same for all models)
# ========================================================================

def load_subject_data(subject_id, data_path):
    """Load acceleration and labeled sleep data"""
    accel_file = os.path.join(data_path, f'{subject_id}_acceleration.txt')
    accel_df = pd.read_csv(accel_file, sep=' ', names=['timestamp', 'x', 'y', 'z'])
    
    labels_file = os.path.join(data_path, f'{subject_id}_labeled_sleep.txt')
    labels_df = pd.read_csv(labels_file, sep=' ', names=['timestamp', 'stage'])
    
    return accel_df, labels_df

def calculate_magnitude(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def extract_epoch_features(epoch_data):
    """Extract 25 features from 30-second epoch"""
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
    """Create 30-second epochs"""
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
# RISK ANALYZER
# ========================================================================

class SleepParalysisRiskAnalyzer:
    """Analyzes sleep patterns for risk factors"""
    
    def __init__(self):
        self.risk_thresholds = {
            'rem_wake_transitions_high': 8,
            'rem_wake_transitions_moderate': 5,
            'rem_fragmentation_high': 12,
            'rem_fragmentation_moderate': 8,
        }
    
    def analyze_hypnogram(self, sleep_stages):
        """Analyze sleep stage sequence for risk factors"""
        stages = np.array(sleep_stages)
        
        # Stage distribution
        stage_counts = Counter(stages)
        total = len(stages)
        rem_pct = (stage_counts.get(5, 0) / total) * 100
        
        # REM-to-Wake transitions
        rem_to_wake = sum(1 for i in range(len(stages)-1) 
                         if stages[i] == 5 and stages[i+1] == 0)
        
        # REM fragmentation
        rem_periods = 0
        in_rem = False
        for stage in stages:
            if stage == 5 and not in_rem:
                rem_periods += 1
                in_rem = True
            elif stage != 5:
                in_rem = False
        
        # Sleep efficiency
        sleep_onset_idx = next((i for i, s in enumerate(stages) if s != 0), None)
        sleep_efficiency = 100.0
        if sleep_onset_idx is not None:
            wake_epochs = sum(1 for s in stages[sleep_onset_idx:] if s == 0)
            time_in_bed = len(stages)
            time_asleep = time_in_bed - wake_epochs
            sleep_efficiency = (time_asleep / time_in_bed * 100) if time_in_bed > 0 else 0
        
        # Calculate risk score
        risk_score = 0
        if rem_to_wake >= 8:
            risk_score += 35
        elif rem_to_wake >= 5:
            risk_score += 20
        
        if rem_periods >= 12:
            risk_score += 25
        elif rem_periods >= 8:
            risk_score += 15
        
        # Risk level
        if risk_score >= 70:
            risk_level = 'VERY HIGH'
        elif risk_score >= 50:
            risk_level = 'HIGH'
        elif risk_score >= 25:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'rem_to_wake': rem_to_wake,
            'rem_fragmentation': rem_periods,
            'rem_pct': rem_pct,
            'sleep_efficiency': sleep_efficiency
        }

# ========================================================================
# MAIN ANALYSIS FUNCTION
# ========================================================================

def analyze_subject_with_model(subject_id, model_key, model, scaler):
    """Analyze one subject with one model"""
    try:
        # Load data
        accel_df, labels_df = load_subject_data(subject_id, DATA_PATH)
        
        # Extract features
        features, labels = create_epochs(accel_df, labels_df)
        if len(features) == 0:
            return None
        
        y_true = np.array(labels)
        X_df = pd.DataFrame(features)
        
        # Scale
        X_scaled = scaler.transform(X_df)
        
        # Predict
        if model_key == 'lstm':
            # LSTM needs sequences - use simple windowing
            sequence_length = min(10, len(X_scaled))
            if len(X_scaled) < sequence_length:
                return None
            
            # Create sequences
            X_seq = []
            for i in range(len(X_scaled) - sequence_length + 1):
                X_seq.append(X_scaled[i:i+sequence_length])
            X_seq = np.array(X_seq)
            
            # Predict
            y_pred_probs = model.predict(X_seq, verbose=0)
            y_pred_mapped = np.argmax(y_pred_probs, axis=1)
            
            # Align with true labels
            y_true = y_true[sequence_length-1:]
        else:
            # RF and XGBoost - direct prediction
            y_pred_mapped = model.predict(X_scaled)
        
        # Map back to original labels
        y_pred = np.vectorize(REVERSE_LABEL_MAPPING.get)(y_pred_mapped)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred) * 100
        
        # Analyze risk
        analyzer = SleepParalysisRiskAnalyzer()
        risk_report = analyzer.analyze_hypnogram(y_pred)
        
        return {
            'subject_id': subject_id,
            'model': model_key,
            'accuracy': accuracy,
            **risk_report
        }
        
    except Exception as e:
        print(f"  ✗ Error analyzing {subject_id} with {model_key}: {e}")
        return None

def main():
    """Main execution - analyze all subjects with all models"""
    
    print("="*80)
    print("SOMNUSGUARD - UNIFIED MULTI-MODEL ANALYSIS")
    print("="*80)
    print()
    
    # Load all models
    loaded_models = {}
    for model_key, config in MODELS.items():
        print(f"Loading {config['name']}...")
        
        if not os.path.exists(config['model_path']):
            print(f"  ⚠ Model not found: {config['model_path']}")
            print(f"  Skipping {config['name']}")
            continue
        
        try:
            # Load model
            if model_key == 'lstm':
                from tensorflow import keras
                model = keras.models.load_model(config['model_path'])
            else:
                model = joblib.load(config['model_path'])
            
            # Load scaler
            scaler = joblib.load(config['scaler_path'])
            
            loaded_models[model_key] = {
                'model': model,
                'scaler': scaler,
                'config': config
            }
            print(f"  ✓ {config['name']} loaded successfully")
            
        except Exception as e:
            print(f"  ✗ Error loading {config['name']}: {e}")
            continue
    
    if not loaded_models:
        print("\n✗ No models loaded. Please train models first.")
        return
    
    print(f"\n✓ Loaded {len(loaded_models)} models: {', '.join(loaded_models.keys())}")
    print()
    
    # Get subject list
    subject_files = [f for f in os.listdir(DATA_PATH) if f.endswith('_acceleration.txt')]
    subject_ids = [f.replace('_acceleration.txt', '') for f in subject_files]
    
    if not subject_ids:
        print(f"✗ No subjects found in {DATA_PATH}")
        return
    
    print(f"Found {len(subject_ids)} subjects")
    print("="*80)
    print()
    
    # Analyze all subjects with all models
    all_results = []
    
    for i, subject_id in enumerate(subject_ids, 1):
        print(f"Processing Subject {i}/{len(subject_ids)}: {subject_id}")
        
        for model_key, model_data in loaded_models.items():
            result = analyze_subject_with_model(
                subject_id,
                model_key,
                model_data['model'],
                model_data['scaler']
            )
            
            if result:
                all_results.append(result)
                print(f"  ✓ {model_data['config']['name']}: {result['accuracy']:.1f}% | {result['risk_level']}")
        
        print()
    
    if not all_results:
        print("✗ No results generated")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save individual model reports
    print("="*80)
    print("SAVING REPORTS")
    print("="*80)
    print()
    
    for model_key in loaded_models.keys():
        model_results = results_df[results_df['model'] == model_key]
        
        if len(model_results) > 0:
            # CSV report
            csv_path = os.path.join(OUTPUT_DIR, f'{model_key}_report.csv')
            model_results.to_csv(csv_path, index=False, float_format='%.2f')
            print(f"✓ {MODELS[model_key]['name']}: {csv_path}")
    
    # Save combined report
    combined_path = os.path.join(OUTPUT_DIR, 'all_models_combined.csv')
    results_df.to_csv(combined_path, index=False, float_format='%.2f')
    print(f"✓ Combined report: {combined_path}")
    
    # Generate comparison summary
    comparison_data = []
    for model_key in loaded_models.keys():
        model_results = results_df[results_df['model'] == model_key]
        
        if len(model_results) > 0:
            comparison_data.append({
                'model_key': model_key,
                'model_name': MODELS[model_key]['name'],
                'color': MODELS[model_key]['color'],
                'subjects_analyzed': len(model_results),
                'mean_accuracy': model_results['accuracy'].mean(),
                'std_accuracy': model_results['accuracy'].std(),
                'high_risk_count': len(model_results[model_results['risk_score'] >= 50]),
                'mean_risk_score': model_results['risk_score'].mean()
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(OUTPUT_DIR, 'model_comparison_summary.csv')
    comparison_df.to_csv(comparison_path, index=False, float_format='%.2f')
    print(f"✓ Model comparison: {comparison_path}")
    
    # Generate JSON files for frontend
    print()
    print("Generating JSON files for frontend...")
    
    # 1. Individual model reports as JSON
    for model_key in loaded_models.keys():
        model_results = results_df[results_df['model'] == model_key]
        if len(model_results) > 0:
            json_data = model_results.to_dict('records')
            json_path = os.path.join(OUTPUT_DIR, f'{model_key}_report.json')
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"✓ {MODELS[model_key]['name']} JSON: {json_path}")
    
    # 2. Combined report as JSON
    json_path = os.path.join(OUTPUT_DIR, 'all_models_combined.json')
    with open(json_path, 'w') as f:
        json.dump(results_df.to_dict('records'), f, indent=2)
    print(f"✓ Combined JSON: {json_path}")
    
    # 3. Comparison summary as JSON
    json_path = os.path.join(OUTPUT_DIR, 'model_comparison_summary.json')
    with open(json_path, 'w') as f:
        json.dump(comparison_df.to_dict('records'), f, indent=2)
    print(f"✓ Comparison JSON: {json_path}")
    
    # 4. Statistics summary as JSON
    stats_data = {
        'total_subjects': len(subject_ids),
        'models_analyzed': list(loaded_models.keys()),
        'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'overall_stats': {
            'mean_accuracy': float(results_df['accuracy'].mean()),
            'mean_risk_score': float(results_df['risk_score'].mean()),
            'high_risk_count': int(len(results_df[results_df['risk_score'] >= 50])),
            'risk_distribution': results_df['risk_level'].value_counts().to_dict()
        },
        'by_model': comparison_df.to_dict('records')
    }
    
    json_path = os.path.join(OUTPUT_DIR, 'statistics.json')
    with open(json_path, 'w') as f:
        json.dump(stats_data, f, indent=2)
    print(f"✓ Statistics JSON: {json_path}")
    
    # Print summary
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("Model Comparison:")
    print(comparison_df.to_string(index=False))
    print()
    print("Files generated:")
    print(f"  📊 CSV Reports: {OUTPUT_DIR}/*_report.csv")
    print(f"  📄 JSON Files: {OUTPUT_DIR}/*.json")
    print()
    print("✅ All files ready for frontend!")
    print("="*80)

if __name__ == "__main__":
    main()