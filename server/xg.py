"""
EARLY DETECTION OF SLEEP DISORDERS AND PARASOMNIAS
Sleep Stage Classification using XGBoost

This script trains an XGBoost model using the preprocessed PhysioNet dataset.
It ensures feature schema consistency by importing from constants.py and saving
a feature manifest for the inference pipeline.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path to import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import FEATURE_COLUMNS, SAMPLE_RATE_HZ, STAGE_LABELS, EPOCH_SECONDS

# Configuration
RANDOM_STATE = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         'data_preprocessing', 'processed_physionet', 'physionet_features.csv')

def main():
    print("="*60)
    print("TRAINING XGBOOST MODEL (Consistent Pipeline)")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Preprocessed data not found at {DATA_PATH}")
        print("Please run data_preprocessing/preprocess_physionet.py first.")
        return
        
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Validate columns
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Training data is missing required feature columns: {missing_cols}")
        
    X = df[FEATURE_COLUMNS]
    y = df['stage']
    
    print(f"Data loaded successfully. Found {len(X)} epochs.")
    print("Feature validation passed. Exact column match confirmed.")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} epochs")
    print(f"Testing set:  {len(X_test)} epochs")
    
    # Encode labels (XGBoost requires sequential classes starting from 0)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print("\nLabel mapping:")
    for orig, enc in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        stage_name = STAGE_LABELS.get(int(orig), f'Stage_{orig}')
        print(f"  {orig} ({stage_name}) -> {enc}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train_scaled, y_train_encoded)
    
    # Evaluate
    print("\nEvaluating model...")
    preds_encoded = model.predict(X_test_scaled)
    preds = label_encoder.inverse_transform(preds_encoded)
    
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # Classification Report
    unique_stages = sorted(y_test.unique())
    target_names = [STAGE_LABELS.get(int(i), f'Stage_{i}') for i in unique_stages]
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=target_names))
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('XGBoost - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(OUTPUT_DIR, 'xg_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save Model, Scaler, and Label Encoder
    model_path = os.path.join(OUTPUT_DIR, 'xgb_sleep_model.joblib')
    scaler_path = os.path.join(OUTPUT_DIR, 'xgb_scaler.joblib')
    encoder_path = os.path.join(OUTPUT_DIR, 'xgb_label_encoder.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Label encoder saved to {encoder_path}")
    
    # Save Feature Manifest
    # (Only writing manifest if it doesn't exist to avoid conflict, or just overwriting is fine)
    manifest = {
        "features": FEATURE_COLUMNS,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "num_features": len(FEATURE_COLUMNS)
    }
    manifest_path = os.path.join(OUTPUT_DIR, 'feature_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    print(f"Feature manifest saved to {manifest_path}")
    
    # --- NEW: Generate Hypnogram for a Sample Subject ---
    print("\nGenerating sample hypnogram for Subject 1066528...")
    # subject_id might be a string in some contexts, or int. preprocess_physionet saves as string?
    # Actually checking df types is safer.
    sample_subject = df[df['subject_id'].astype(str) == '1066528']
    if sample_subject.empty:
        first_id = df['subject_id'].iloc[0]
        sample_subject = df[df['subject_id'] == first_id]
        print(f"Subject 1066528 not found, using {first_id} instead.")
        
    if not sample_subject.empty:
        sample_subject = sample_subject.sort_values('window')
        X_sample = sample_subject[FEATURE_COLUMNS]
        y_true = sample_subject['stage'].values
        
        X_sample_scaled = scaler.transform(X_sample)
        preds_encoded = model.predict(X_sample_scaled)
        y_pred = label_encoder.inverse_transform(preds_encoded)
        
        # Plotting
        plt.figure(figsize=(15, 6))
        x_axis = np.arange(len(y_pred)) * (EPOCH_SECONDS / 60.0)
        
        # Mapping for hypnogram levels
        stage_map = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0} 
        y_true_plot = [stage_map.get(int(s), 0) for s in y_true]
        y_pred_plot = [stage_map.get(int(s), 0) for s in y_pred]
        
        plt.step(x_axis, y_true_plot, label='True Stage', alpha=0.5, color='gray')
        plt.step(x_axis, y_pred_plot, label='Predicted Stage', color='green', linewidth=1.5)
        
        plt.yticks([0, 1, 2, 3, 4], ['N3', 'N2', 'N1', 'REM', 'Wake'])
        plt.xlabel("Time (Minutes)")
        plt.ylabel("Sleep Stage")
        plt.title(f"XGBoost Hypnogram - Subject {sample_subject['subject_id'].iloc[0]}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        hyp_path = os.path.join(OUTPUT_DIR, 'xg_physionet_sample_hypnogram.png')
        plt.savefig(hyp_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sample hypnogram saved to {hyp_path}")

    print("\n" + "="*60)
    print("XGBOOST TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()