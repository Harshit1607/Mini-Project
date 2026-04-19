"""
SOMNUSGUARD - UNIFIED MULTI-MODEL ANALYZER
Analyzes preprocessed data using trained RF and XGBoost models.

This script:
1. Loads feature_manifest.json and validates feature schema
2. Loads trained RF and XGBoost models
3. Processes custom preprocessed CSV files
4. Calculates risk scores using SleepParalysisRiskAnalyzer
5. Outputs comprehensive risk reports
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path to import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import EPOCH_SECONDS, STAGE_LABELS, RISK_WEIGHTS

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'outputs')
CUSTOM_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data_preprocessing', 'processed_custom')
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'client', 'src', 'data')

class SleepParalysisRiskAnalyzer:
    """Analyzes sleep patterns for risk factors based on constants.py weights"""
    
    def __init__(self):
        self.weights = RISK_WEIGHTS
        
    def analyze_hypnogram(self, sleep_stages):
        stages = np.array(sleep_stages)
        total_epochs = len(stages)
        
        if total_epochs == 0:
            return {"error": "Empty hypnogram"}
            
        stage_counts = Counter(stages)
        
        rem_pct = (stage_counts.get(4, 0) / total_epochs) * 100
        
        # REM-to-Wake transitions (REM is 4, Wake is 0 based on STAGE_LABELS)
        rem_to_wake = sum(1 for i in range(len(stages)-1) 
                         if stages[i] == 4 and stages[i+1] == 0)
                         
        # Stage transitions
        transitions = sum(1 for i in range(len(stages)-1) if stages[i] != stages[i+1])
        duration_hours = total_epochs * EPOCH_SECONDS / 3600
        transition_rate = transitions / duration_hours if duration_hours > 0 else 0
        
        # REM fragmentation (number of distinct REM periods)
        rem_periods = 0
        in_rem = False
        rem_start_idx = -1
        
        for i, stage in enumerate(stages):
            if stage == 4 and not in_rem:
                rem_periods += 1
                in_rem = True
                if rem_start_idx == -1:
                    rem_start_idx = i
            elif stage != 4:
                in_rem = False
                
        rem_latency_mins = (rem_start_idx * EPOCH_SECONDS) / 60 if rem_start_idx != -1 else float('inf')
        
        # Sleep efficiency
        sleep_onset_idx = next((i for i, s in enumerate(stages) if s != 0), None)
        sleep_efficiency = 100.0
        awakenings = 0
        if sleep_onset_idx is not None:
            wake_epochs = sum(1 for s in stages[sleep_onset_idx:] if s == 0)
            time_in_bed = total_epochs
            time_asleep = time_in_bed - wake_epochs
            sleep_efficiency = (time_asleep / time_in_bed * 100) if time_in_bed > 0 else 0
            
            in_wake = False
            for s in stages[sleep_onset_idx:]:
                if s == 0 and not in_wake:
                    awakenings += 1
                    in_wake = True
                elif s != 0:
                    in_wake = False
        
        # Calculate risk score
        risk_score = 0
        flags = []
        
        if rem_to_wake >= self.weights['rem_wake_transitions_high']:
            risk_score += 35
            flags.append("High REM-to-Wake Transitions")
        elif rem_to_wake >= self.weights['rem_wake_transitions_moderate']:
            risk_score += 20
            flags.append("Moderate REM-to-Wake Transitions")
            
        if rem_periods >= self.weights['rem_fragmentation_high']:
            risk_score += 25
            flags.append("High REM Fragmentation")
        elif rem_periods >= self.weights['rem_fragmentation_moderate']:
            risk_score += 15
            flags.append("Moderate REM Fragmentation")
            
        if awakenings >= self.weights['total_awakenings_high']:
            risk_score += 15
            flags.append("High Sleep Fragmentation")
            
        if rem_pct < self.weights['rem_percentage_low']:
            risk_score += 10
            flags.append("Low REM Percentage")
        elif rem_pct > self.weights['rem_percentage_high']:
            risk_score += 10
            flags.append("High REM Percentage")
            
        if transition_rate >= self.weights['stage_transition_rate_high']:
            risk_score += 10
            flags.append("High Stage Transition Rate")
            
        # Risk level categorization
        if risk_score >= 70:
            risk_level = 'VERY HIGH'
        elif risk_score >= 50:
            risk_level = 'HIGH'
        elif risk_score >= 25:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
            
        # Per stage percentages
        percentages = {STAGE_LABELS.get(k, f"Stage_{k}"): (v / total_epochs * 100) for k, v in stage_counts.items()}
            
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'flags': flags,
            'metrics': {
                'rem_to_wake': rem_to_wake,
                'rem_fragmentation': rem_periods,
                'rem_pct': rem_pct,
                'sleep_efficiency': sleep_efficiency,
                'transition_rate': transition_rate
            },
            'stage_percentages': percentages
        }


class SomnusAnalyzer:
    def __init__(self):
        self.manifest = None
        self.models = {}
        self.risk_analyzer = SleepParalysisRiskAnalyzer()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.load_resources()
        
    def load_resources(self):
        print("Loading resources...")
        
        # 1. Load manifest
        manifest_path = os.path.join(MODELS_DIR, 'feature_manifest.json')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Feature manifest not found at {manifest_path}. Train models first.")
            
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        print(f"Loaded manifest with {self.manifest['num_features']} expected features.")
        
        # 2. Load RF
        rf_path = os.path.join(MODELS_DIR, 'rf_sleep_model.joblib')
        rf_scaler_path = os.path.join(MODELS_DIR, 'rf_scaler.joblib')
        if os.path.exists(rf_path) and os.path.exists(rf_scaler_path):
            self.models['RF'] = {
                'model': joblib.load(rf_path),
                'scaler': joblib.load(rf_scaler_path)
            }
            print("Loaded Random Forest model.")
            
        # 3. Load XGBoost
        xgb_path = os.path.join(MODELS_DIR, 'xgb_sleep_model.joblib')
        xgb_scaler_path = os.path.join(MODELS_DIR, 'xgb_scaler.joblib')
        xgb_encoder_path = os.path.join(MODELS_DIR, 'xgb_label_encoder.joblib')
        if os.path.exists(xgb_path) and os.path.exists(xgb_scaler_path) and os.path.exists(xgb_encoder_path):
            self.models['XGBoost'] = {
                'model': joblib.load(xgb_path),
                'scaler': joblib.load(xgb_scaler_path),
                'encoder': joblib.load(xgb_encoder_path)
            }
            print("Loaded XGBoost model.")
            
        if not self.models:
            raise ValueError("No trained models found! Please run training scripts.")

    def analyze_data(self, df_or_path):
        """Analyze custom preprocessed data (accepts DataFrame or path to CSV)"""
        
        if isinstance(df_or_path, str):
            df = pd.read_csv(df_or_path)
            source_name = os.path.basename(df_or_path)
        else:
            df = df_or_path
            source_name = "DataFrame"
            
        # Validate columns against manifest
        expected_features = self.manifest['features']
        missing_cols = [col for col in expected_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input data is missing required columns: {missing_cols}")
            
        X = df[expected_features]
        print(f"\nAnalyzing {source_name} ({len(X)} epochs)...")
        
        results = {}
        for model_name, components in self.models.items():
            print(f"  -> Running {model_name} prediction...")
            
            # Scale
            X_scaled = components['scaler'].transform(X)
            
            # Predict
            if model_name == 'XGBoost':
                preds_encoded = components['model'].predict(X_scaled)
                preds = components['encoder'].inverse_transform(preds_encoded)
            else:
                preds = components['model'].predict(X_scaled)
                
            # Run Risk Analyzer
            risk_report = self.risk_analyzer.analyze_hypnogram(preds)
            results[model_name] = risk_report
            
            print(f"     Risk Score: {risk_report['risk_score']} | Category: {risk_report['risk_level']}")
            if risk_report['flags']:
                print(f"     Flags: {', '.join(risk_report['flags'])}")
                
            # --- Generate Hypnogram ---
            try:
                import matplotlib.pyplot as plt
                
                # Create a mapping that orders the stages nicely for a hypnogram
                # Usually: Wake (top), REM, N1, N2, N3 (bottom)
                stage_to_y = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0} # 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
                y_labels = ['N3', 'N2', 'N1', 'REM', 'Wake']
                
                y_vals = [stage_to_y.get(stage, 0) for stage in preds]
                x_vals = np.arange(len(preds)) * (EPOCH_SECONDS / 60.0) # Time in minutes
                
                plt.figure(figsize=(12, 4))
                # Plot the stepped hypnogram line
                plt.step(x_vals, y_vals, where='post', color='#2c3e50', linewidth=2)
                
                # Fill the REM stages with a highlight color
                for i in range(len(preds)):
                    if preds[i] == 4: # REM
                        start_time = i * (EPOCH_SECONDS / 60.0)
                        end_time = (i + 1) * (EPOCH_SECONDS / 60.0)
                        plt.axvspan(start_time, end_time, color='red', alpha=0.3)
                        
                plt.yticks([0, 1, 2, 3, 4], y_labels)
                plt.xlabel("Time (Minutes)")
                plt.ylabel("Sleep Stage")
                plt.title(f"Hypnogram - {source_name} ({model_name})")
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save the hypnogram
                if isinstance(df_or_path, str):
                    save_dir = os.path.dirname(df_or_path)
                else:
                    save_dir = OUTPUT_DIR
                
                hypnogram_path = os.path.join(save_dir, f"{model_name}_hypnogram.png")
                plt.savefig(hypnogram_path, dpi=300)
                plt.close()
                print(f"     -> Hypnogram saved to {hypnogram_path}")
            except Exception as e:
                print(f"     -> Failed to generate hypnogram: {e}")
                
        return results

def process_custom_directory():
    print("="*60)
    print("SOMNUSGUARD INFERENCE BATCH RUNNER")
    print("="*60)
    
    try:
        analyzer = SomnusAnalyzer()
    except Exception as e:
        print(f"Initialization Error: {e}")
        return
        
    if not os.path.exists(CUSTOM_DATA_DIR):
        print(f"Custom data directory not found: {CUSTOM_DATA_DIR}")
        return
        
    processed_count = 0
    for person in os.listdir(CUSTOM_DATA_DIR):
        person_dir = os.path.join(CUSTOM_DATA_DIR, person)
        if not os.path.isdir(person_dir):
            continue
            
        for day in os.listdir(person_dir):
            day_dir = os.path.join(person_dir, day)
            if not os.path.isdir(day_dir):
                continue
                
            csv_path = os.path.join(day_dir, "Preprocessed_Window_Features.csv")
            if os.path.exists(csv_path):
                print(f"\n--- Subject: {person} | Day: {day} ---")
                try:
                    analyzer.analyze_data(csv_path)
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")
                    
    print(f"\nFinished processing {processed_count} files.")
    print("="*60)

if __name__ == "__main__":
    process_custom_directory()