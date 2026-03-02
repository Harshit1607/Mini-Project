"""
EARLY DETECTION OF SLEEP DISORDERS AND PARASOMNIAS
Sleep Stage Classification using XGBoost

Authors: Harshit Bareja, Ishika Manchanda, Teena Kaintura
Guide: Ms. Nupur Chugh
Institution: Bharati Vidyapeeth College of Engineering

This implementation uses XGBoost to classify sleep stages from accelerometer data
Dataset: PhysioNet Sleep-Accel Database (https://physionet.org/content/sleep-accel/1.0.0/)

UPDATED VERSION - Saves predictions for Step 2 analysis
FIXED VERSION - Handles -1 class with label encoding
"""

import numpy as np
import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# Configuration
EPOCH_DURATION = 30  # seconds
SAMPLING_RATE = 50   # Hz
RANDOM_STATE = 42
OUTPUT_DIR = './outputs'

class SleepStageClassifierXGB:
    """
    Complete pipeline for sleep stage classification using XGBoost
    """

    def __init__(self, data_path='../sleep-accel-data/', output_dir='./outputs'):
        """
        Initialize the classifier
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        os.makedirs(self.output_dir, exist_ok=True)

        self.sleep_stage_mapping = {
            -1: 'Unknown',
            0: 'Wake',
            1: 'N1',
            2: 'N2',
            3: 'N3',
            4: 'N4',
            5: 'REM'
        }

    def load_subject_data(self, subject_id):
        accel_file = os.path.join(self.data_path, f'{subject_id}_acceleration.txt')
        labels_file = os.path.join(self.data_path, f'{subject_id}_labeled_sleep.txt')
        accel_df = pd.read_csv(accel_file, sep=' ', names=['timestamp', 'x', 'y', 'z'])
        labels_df = pd.read_csv(labels_file, sep=' ', names=['timestamp', 'stage'])
        return accel_df, labels_df

    def butter_lowpass_filter(self, data, cutoff=3, fs=50, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def calculate_magnitude(self, x, y, z):
        return np.sqrt(x**2 + y**2 + z**2)

    def extract_epoch_features(self, epoch_data):
        features = {}
        magnitude = self.calculate_magnitude(
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

    def create_epochs(self, accel_df, labels_df):
        features_list, labels_list = [], []
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
                features = self.extract_epoch_features(epoch_accel)
                features_list.append(features)
                label = epoch_label['stage'].mode()[0]
                labels_list.append(label)
            current_time += EPOCH_DURATION
        return features_list, labels_list

    def load_all_subjects(self):
        print("Loading and processing all subjects...")
        all_features, all_labels = [], []
        subject_files = [f for f in os.listdir(self.data_path)
                         if f.endswith('_acceleration.txt')]
        subject_ids = [f.replace('_acceleration.txt', '') for f in subject_files]

        print(f"Found {len(subject_ids)} subjects")
        for i, subject_id in enumerate(subject_ids, 1):
            try:
                print(f"Processing subject {i}/{len(subject_ids)}: {subject_id}")
                accel_df, labels_df = self.load_subject_data(subject_id)
                features, labels = self.create_epochs(accel_df, labels_df)
                all_features.extend(features)
                all_labels.extend(labels)
                print(f"  -> Extracted {len(features)} epochs")
            except Exception as e:
                print(f"  -> Error processing subject {subject_id}: {e}")
                continue

        X = pd.DataFrame(all_features)
        y = pd.Series(all_labels, name='stage')

        print(f"\nTotal epochs: {len(X)}")
        print(f"Feature dimensions: {X.shape}")
        print(f"\nSleep stage distribution:")
        print(y.value_counts().sort_index())
        return X, y

    def train_model(self, X_train, y_train):
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)

        # Encode labels to handle -1 and ensure sequential classes
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        print(f"\nOriginal classes: {sorted(y_train.unique())}")
        print(f"Encoded classes: {sorted(np.unique(y_train_encoded))}")
        print(f"Label mapping:")
        for orig, enc in zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))):
            stage_name = self.sleep_stage_mapping.get(int(orig), f'Stage_{orig}')
            print(f"  {orig} ({stage_name}) -> {enc}")

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        print("\nTraining in progress...")
        self.model.fit(X_train_scaled, y_train_encoded)
        print("\nModel training completed!")

        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train_encoded, cv=5,
                                    scoring='accuracy', n_jobs=-1)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    def evaluate_model(self, X_test, y_test):
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Encode test labels
        y_test_encoded = self.label_encoder.transform(y_test)
        
        X_test_scaled = self.scaler.transform(X_test)
        predictions_encoded = self.model.predict(X_test_scaled)
        
        # Decode predictions back to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nTest Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        unique_stages = sorted(y_test.unique())
        target_names = [self.sleep_stage_mapping.get(int(i), f'Stage_{i}') for i in unique_stages]
        print("\nDetailed Classification Report:")
        print("-" * 60)
        print(classification_report(y_test, predictions, target_names=target_names, digits=4))
        cm = confusion_matrix(y_test, predictions)
        print("\nConfusion Matrix:\n", cm)
        return predictions

    def plot_results(self, X_test, y_test, predictions):
        print("\nGenerating visualizations...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        cm = confusion_matrix(y_test, predictions)
        unique_stages = sorted(y_test.unique())
        stage_names = [self.sleep_stage_mapping.get(int(i), f'Stage_{i}') for i in unique_stages]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=stage_names, yticklabels=stage_names, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        axes[0, 1].barh(range(len(feature_importance)), feature_importance['importance'])
        axes[0, 1].set_yticks(range(len(feature_importance)))
        axes[0, 1].set_yticklabels(feature_importance['feature'])
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')

        stage_dist = pd.DataFrame({
            'True': y_test.value_counts().sort_index(),
            'Predicted': pd.Series(predictions).value_counts().sort_index()
        })
        stage_dist.index = [self.sleep_stage_mapping.get(int(i), f'Stage_{i}') for i in stage_dist.index]
        stage_dist.plot(kind='bar', ax=axes[1, 0], color=['#3498db', '#e74c3c'])
        axes[1, 0].set_title('Sleep Stage Distribution', fontsize=14, fontweight='bold')

        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, labels=unique_stages, zero_division=0)
        metrics_df = pd.DataFrame({
            'Precision': precision, 'Recall': recall, 'F1-Score': f1
        }, index=stage_names)
        metrics_df.plot(kind='bar', ax=axes[1, 1],
                        color=['#2ecc71', '#f39c12', '#9b59b6'])
        axes[1, 1].set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'xgb_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {os.path.abspath(output_path)}")

    def generate_hypnogram_sample(self, predictions, sample_size=500):
        print("\nGenerating sample hypnogram...")
        sample_pred = predictions[:min(sample_size, len(predictions))]
        time_hours = np.arange(len(sample_pred)) * (EPOCH_DURATION / 3600)
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(time_hours, sample_pred, linewidth=2, color='#2c3e50')
        ax.fill_between(time_hours, sample_pred, alpha=0.3, color='#3498db')
        ax.set_yticks(sorted(np.unique(sample_pred)))
        ax.set_yticklabels([self.sleep_stage_mapping.get(int(i), f'Stage_{i}')
                            for i in sorted(np.unique(sample_pred))])
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Sleep Stage', fontsize=12)
        ax.set_title('Sample Hypnogram - XGBoost', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'xgb_hypnogram.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Hypnogram saved to: {os.path.abspath(output_path)}")


def main():
    print("="*60)
    print("SLEEP STAGE CLASSIFICATION USING XGBOOST")
    print("SomnusGuard - Step 1: Sleep Stage Classification (XGBoost)")
    print("="*60)

    classifier = SleepStageClassifierXGB(
        data_path='../sleep-accel-data/',
        output_dir=OUTPUT_DIR
    )
    print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")

    print("\n" + "="*60)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*60)
    X, y = classifier.load_all_subjects()

    print("\n" + "="*60)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")

    classifier.train_model(X_train, y_train)
    predictions = classifier.evaluate_model(X_test, y_test)

    print("\n" + "="*60)
    print("STEP 5: VISUALIZATION")
    print("="*60)
    classifier.plot_results(X_test, y_test, predictions)
    classifier.generate_hypnogram_sample(predictions)

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': classifier.model.feature_importances_
    }).sort_values('Importance', ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, 'xgb_feature_importance.csv')
    feature_importance.to_csv(csv_path, index=False)
    print(f"\nFeature importance saved to: {os.path.abspath(csv_path)}")

    predictions_file = os.path.join(OUTPUT_DIR, 'xgb_predictions.npy')
    np.save(predictions_file, predictions)
    print(f"✓ Predictions saved to: {os.path.abspath(predictions_file)}")

    predictions_csv = os.path.join(OUTPUT_DIR, 'xgb_predictions.csv')
    pd.DataFrame({
        'epoch': range(len(predictions)),
        'sleep_stage': predictions,
        'stage_name': [classifier.sleep_stage_mapping.get(int(s), f'Stage_{s}') for s in predictions]
    }).to_csv(predictions_csv, index=False)
    print(f"✓ Human-readable predictions saved to: {os.path.abspath(predictions_csv)}")

    # Save model, scaler, and label encoder for the unified analyzer
    print("\n" + "="*60)
    print("SAVING MODEL, SCALER, AND LABEL ENCODER FOR ANALYZER")
    print("="*60)
    
    model_path = os.path.join(OUTPUT_DIR, 'xgb_sleep_model.joblib')
    joblib.dump(classifier.model, model_path)
    print(f"✓ Model saved to: {os.path.abspath(model_path)}")
    
    scaler_path = os.path.join(OUTPUT_DIR, 'xgb_scaler.joblib')
    joblib.dump(classifier.scaler, scaler_path)
    print(f"✓ Scaler saved to: {os.path.abspath(scaler_path)}")
    
    encoder_path = os.path.join(OUTPUT_DIR, 'xgb_label_encoder.joblib')
    joblib.dump(classifier.label_encoder, encoder_path)
    print(f"✓ Label encoder saved to: {os.path.abspath(encoder_path)}")

    print("\n" + "="*60)
    print("STEP 1 COMPLETED SUCCESSFULLY! - XGBOOST VERSION")
    print("="*60)
    print(f"All output files saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()