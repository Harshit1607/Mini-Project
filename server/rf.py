"""
EARLY DETECTION OF SLEEP DISORDERS AND PARASOMNIAS
Sleep Stage Classification using Random Forest

Authors: Harshit Bareja, Ishika Manchanda, Teena Kaintura
Guide: Ms. Nupur Chugh
Institution: Bharati Vidyapeeth College of Engineering

This implementation uses Random Forest to classify sleep stages from accelerometer data
Dataset: PhysioNet Sleep-Accel Database (https://physionet.org/content/sleep-accel/1.0.0/)

UPDATED VERSION - Compatible with Enhanced Risk Analyzer
"""

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import butter, filtfilt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
EPOCH_DURATION = 30  # seconds (standard for sleep stage classification)
SAMPLING_RATE = 50  # Hz (approximate from dataset)
RANDOM_STATE = 42
OUTPUT_DIR = './outputs'

# --- Label Mappings (Must match analyzer) ---
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

class SleepStageClassifier:
    """
    Complete pipeline for sleep stage classification using Random Forest
    Compatible with Enhanced Risk Analyzer
    """
    
    def __init__(self, data_path='../sleep-accel-data/', output_dir='./outputs'):
        """
        Initialize the classifier
        
        Args:
            data_path: Path to the downloaded PhysioNet sleep-accel dataset
            output_dir: Directory to save output files
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.model = None
        self.scaler = StandardScaler()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.sleep_stage_mapping = SLEEP_STAGE_MAPPING
        
    def load_subject_data(self, subject_id):
        """
        Load acceleration and labeled sleep data for a single subject
        
        Args:
            subject_id: Subject identifier (e.g., '1360686')
            
        Returns:
            accel_df: DataFrame with acceleration data
            labels_df: DataFrame with sleep stage labels
        """
        # Load acceleration data
        accel_file = os.path.join(self.data_path, f'{subject_id}_acceleration.txt')
        accel_df = pd.read_csv(accel_file, sep=' ', 
                               names=['timestamp', 'x', 'y', 'z'])
        
        # Load labeled sleep stages
        labels_file = os.path.join(self.data_path, f'{subject_id}_labeled_sleep.txt')
        labels_df = pd.read_csv(labels_file, sep=' ',
                               names=['timestamp', 'stage'])
        
        return accel_df, labels_df
    
    def butter_lowpass_filter(self, data, cutoff=3, fs=50, order=4):
        """
        Apply low-pass Butterworth filter to reduce noise
        
        Args:
            data: Input signal
            cutoff: Cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order
            
        Returns:
            Filtered signal
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    def calculate_magnitude(self, x, y, z):
        """
        Calculate magnitude of 3D acceleration vector
        
        Args:
            x, y, z: Acceleration components
            
        Returns:
            Magnitude of acceleration
        """
        return np.sqrt(x**2 + y**2 + z**2)
    
    def extract_epoch_features(self, epoch_data):
        """
        Extract time-domain and statistical features from 30-second epoch
        MUST MATCH THE ANALYZER'S FEATURE EXTRACTION!
        
        Features extracted:
        - Mean, Std, Min, Max, Median
        - Variance, Range, IQR
        - Skewness, Kurtosis
        - Zero-crossing rate
        - Percentiles (25th, 75th)
        - Energy (sum of squared values)
        
        Args:
            epoch_data: Acceleration data for one epoch
            
        Returns:
            Dictionary of features (25 features total)
        """
        features = {}
        
        # Calculate magnitude
        magnitude = self.calculate_magnitude(
            epoch_data['x'].values,
            epoch_data['y'].values, 
            epoch_data['z'].values
        )
        
        # Basic statistics
        features['mean_mag'] = np.mean(magnitude)
        features['std_mag'] = np.std(magnitude)
        features['min_mag'] = np.min(magnitude)
        features['max_mag'] = np.max(magnitude)
        features['median_mag'] = np.median(magnitude)
        features['var_mag'] = np.var(magnitude)
        features['range_mag'] = np.max(magnitude) - np.min(magnitude)
        
        # Percentiles
        features['q25_mag'] = np.percentile(magnitude, 25)
        features['q75_mag'] = np.percentile(magnitude, 75)
        features['iqr_mag'] = features['q75_mag'] - features['q25_mag']
        
        # Shape statistics
        features['skew_mag'] = stats.skew(magnitude)
        features['kurtosis_mag'] = stats.kurtosis(magnitude)
        
        # Energy
        features['energy_mag'] = np.sum(magnitude**2)
        
        # Zero crossing rate (movement changes)
        magnitude_centered = magnitude - np.mean(magnitude)
        zero_crossings = np.sum(np.diff(np.sign(magnitude_centered)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(magnitude)
        
        # Per-axis features (capturing directional movement)
        for axis in ['x', 'y', 'z']:
            data = epoch_data[axis].values
            features[f'mean_{axis}'] = np.mean(data)
            features[f'std_{axis}'] = np.std(data)
            features[f'range_{axis}'] = np.max(data) - np.min(data)
        
        # Movement intensity (high-frequency component)
        diff_mag = np.diff(magnitude)
        features['mean_diff'] = np.mean(np.abs(diff_mag))
        features['std_diff'] = np.std(diff_mag)
        
        return features
    
    def create_epochs(self, accel_df, labels_df):
        """
        Segment continuous data into 30-second epochs and extract features
        
        Args:
            accel_df: Acceleration data
            labels_df: Sleep stage labels
            
        Returns:
            features_list: List of feature dictionaries
            labels_list: List of corresponding sleep stage labels
        """
        features_list = []
        labels_list = []
        
        # Determine time range
        start_time = max(accel_df['timestamp'].min(), labels_df['timestamp'].min())
        end_time = min(accel_df['timestamp'].max(), labels_df['timestamp'].max())
        
        # Create 30-second epochs
        current_time = start_time
        while current_time + EPOCH_DURATION <= end_time:
            # Get acceleration data for this epoch
            epoch_accel = accel_df[
                (accel_df['timestamp'] >= current_time) & 
                (accel_df['timestamp'] < current_time + EPOCH_DURATION)
            ]
            
            # Get corresponding label
            epoch_label = labels_df[
                (labels_df['timestamp'] >= current_time) &
                (labels_df['timestamp'] < current_time + EPOCH_DURATION)
            ]
            
            # Only process if we have both data and label
            if len(epoch_accel) > 10 and len(epoch_label) > 0:
                # Extract features
                features = self.extract_epoch_features(epoch_accel)
                features_list.append(features)
                
                # Use most common label in epoch
                label = epoch_label['stage'].mode()[0]
                labels_list.append(label)
            
            current_time += EPOCH_DURATION
        
        return features_list, labels_list
    
    def load_all_subjects(self):
        """
        Load and process data from all subjects in the dataset
        
        Returns:
            X: Feature matrix
            y: Label vector
        """
        print("Loading and processing all subjects...")
        
        all_features = []
        all_labels = []
        
        # Get list of subject files
        subject_files = [f for f in os.listdir(self.data_path) 
                        if f.endswith('_acceleration.txt')]
        subject_ids = [f.replace('_acceleration.txt', '') for f in subject_files]
        
        print(f"Found {len(subject_ids)} subjects")
        
        for i, subject_id in enumerate(subject_ids, 1):
            try:
                print(f"Processing subject {i}/{len(subject_ids)}: {subject_id}")
                
                # Load data
                accel_df, labels_df = self.load_subject_data(subject_id)
                
                # Create epochs and extract features
                features, labels = self.create_epochs(accel_df, labels_df)
                
                all_features.extend(features)
                all_labels.extend(labels)
                
                print(f"  -> Extracted {len(features)} epochs")
                
            except Exception as e:
                print(f"  -> Error processing subject {subject_id}: {e}")
                continue
        
        # Convert to DataFrames
        X = pd.DataFrame(all_features)
        y = pd.Series(all_labels, name='stage')
        
        print(f"\nTotal epochs: {len(X)}")
        print(f"Feature dimensions: {X.shape}")
        print(f"\nSleep stage distribution:")
        print(y.value_counts().sort_index())
        
        return X, y
    
    def train_model(self, X_train, y_train):
        """
        Train Random Forest classifier with mapped labels
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        # Map labels to 0-6 range for consistency with analyzer
        y_train_mapped = y_train.map(LABEL_MAPPING)
        
        # Initialize Random Forest with optimized hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        print("\nTraining in progress...")
        self.model.fit(X_train_scaled, y_train_mapped)
        
        print("\nModel training completed!")
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train_mapped, 
                                    cv=5, scoring='accuracy', n_jobs=-1)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            predictions: Model predictions (in original label space)
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Map test labels
        y_test_mapped = y_test.map(LABEL_MAPPING)
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions (in mapped space)
        predictions_mapped = self.model.predict(X_test_scaled)
        
        # Map back to original labels
        predictions = np.vectorize(REVERSE_LABEL_MAPPING.get)(predictions_mapped)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nTest Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print("-" * 60)
        unique_stages = sorted(y_test.unique())
        target_names = [self.sleep_stage_mapping.get(int(i), f'Stage_{i}') for i in unique_stages]
        print(classification_report(y_test, predictions, 
                                   target_names=target_names,
                                   digits=4))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        return predictions
    
    def plot_results(self, X_test, y_test, predictions):
        """
        Visualize model performance
        
        Args:
            X_test: Test features
            y_test: True labels
            predictions: Predicted labels
        """
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        unique_stages = sorted(y_test.unique())
        stage_names = [self.sleep_stage_mapping.get(int(i), f'Stage_{i}') for i in unique_stages]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=stage_names, yticklabels=stage_names,
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label', fontsize=12)
        axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
        
        # 2. Feature Importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        axes[0, 1].barh(range(len(feature_importance)), feature_importance['importance'])
        axes[0, 1].set_yticks(range(len(feature_importance)))
        axes[0, 1].set_yticklabels(feature_importance['feature'])
        axes[0, 1].set_xlabel('Importance', fontsize=12)
        axes[0, 1].set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
        axes[0, 1].invert_yaxis()
        
        # 3. Sleep Stage Distribution
        stage_dist = pd.DataFrame({
            'True': y_test.value_counts().sort_index(),
            'Predicted': pd.Series(predictions).value_counts().sort_index()
        })
        stage_dist.index = [self.sleep_stage_mapping.get(int(i), f'Stage_{i}') for i in stage_dist.index]
        stage_dist.plot(kind='bar', ax=axes[1, 0], color=['#3498db', '#e74c3c'])
        axes[1, 0].set_title('Sleep Stage Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Sleep Stage', fontsize=12)
        axes[1, 0].set_ylabel('Number of Epochs', fontsize=12)
        axes[1, 0].legend(['True', 'Predicted'])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Per-class Accuracy
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, 
                                                                     labels=unique_stages,
                                                                     zero_division=0)
        
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=[self.sleep_stage_mapping.get(int(i), f'Stage_{i}') for i in unique_stages])
        
        metrics_df.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#f39c12', '#9b59b6'])
        axes[1, 1].set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Sleep Stage', fontsize=12)
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].legend(loc='lower right')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'sleep_classification_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {os.path.abspath(output_path)}")
        
        return fig
    
    def generate_hypnogram_sample(self, predictions, sample_size=500):
        """
        Generate a sample hypnogram visualization
        
        Args:
            predictions: Predicted sleep stages
            sample_size: Number of epochs to display
        """
        print("\nGenerating sample hypnogram...")
        
        # Take a sample of predictions
        sample_pred = predictions[:min(sample_size, len(predictions))]
        
        # Create time axis (in hours)
        time_hours = np.arange(len(sample_pred)) * (EPOCH_DURATION / 3600)
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot hypnogram
        ax.plot(time_hours, sample_pred, linewidth=2, color='#2c3e50')
        ax.fill_between(time_hours, sample_pred, alpha=0.3, color='#3498db')
        
        # Formatting
        unique_stages_in_sample = sorted(np.unique(sample_pred))
        ax.set_yticks(unique_stages_in_sample)
        ax.set_yticklabels([self.sleep_stage_mapping.get(int(i), f'Stage_{i}') 
                           for i in unique_stages_in_sample])
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sleep Stage', fontsize=12, fontweight='bold')
        ax.set_title('Sample Hypnogram - Sleep Stage Progression', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time_hours[-1]])
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'sample_hypnogram.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Hypnogram saved to: {os.path.abspath(output_path)}")
        
        return fig
    
    def save_model_and_scaler(self):
        """
        Save the trained model and scaler for use with the risk analyzer
        """
        print("\n" + "="*60)
        print("SAVING MODEL AND SCALER")
        print("="*60)
        
        # Save Random Forest model
        model_path = os.path.join(OUTPUT_DIR, 'rf_sleep_model.joblib')

        joblib.dump(self.model, model_path)
        print(f"✓ Model saved to: {os.path.abspath(model_path)}")
        
        # Save scaler
        scaler_path = os.path.join(OUTPUT_DIR, 'rf_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved to: {os.path.abspath(scaler_path)}")
        
        return model_path, scaler_path


def main():
    """
    Main execution function
    """
    print("="*60)
    print("SLEEP STAGE CLASSIFICATION USING RANDOM FOREST")
    print("SomnusGuard - Compatible with Enhanced Risk Analyzer")
    print("="*60)
    
    # Initialize classifier
    classifier = SleepStageClassifier(
        data_path='../sleep-accel-data/',
        output_dir=OUTPUT_DIR
    )
    
    print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")
    
    # STEP 1: Load and process all subject data
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*60)
    
    X, y = classifier.load_all_subjects()
    
    # Verify we have 25 features (required by analyzer)
    print(f"\n✓ Feature count: {X.shape[1]} (Required: 25)")
    if X.shape[1] != 25:
        print("⚠ WARNING: Feature count mismatch! Analyzer expects 25 features.")
    
    # STEP 2: Split data
    print("\n" + "="*60)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set size: {len(X_train)} epochs")
    print(f"Test set size: {len(X_test)} epochs")
    
    # STEP 3: Train model
    classifier.train_model(X_train, y_train)
    
    # STEP 4: Evaluate model
    predictions = classifier.evaluate_model(X_test, y_test)
    
    # STEP 5: Visualize results
    print("\n" + "="*60)
    print("STEP 5: VISUALIZATION")
    print("="*60)
    
    classifier.plot_results(X_test, y_test, predictions)
    classifier.generate_hypnogram_sample(predictions)
    
    # STEP 6: Feature importance
    print("\n" + "="*60)
    print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': classifier.model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    csv_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
    feature_importance.to_csv(csv_path, index=False)
    print(f"\nFeature importance saved to: {os.path.abspath(csv_path)}")
    
    # STEP 7: Save model and scaler for analyzer
    classifier.save_model_and_scaler()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nAll output files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("  1. sleep_classification_results.png")
    print("  2. sample_hypnogram.png")
    print("  3. feature_importance.csv")
    print("  4. rf_sleep_model.joblib ← Model for analyzer")
    print("  5. scaler.joblib ← Scaler for analyzer")
    print("\n" + "="*60)
    print("⚠ IMPORTANT: Update the analyzer to load RF model")
    print("="*60)
    print("\nThe analyzer needs to be modified to use:")
    print("  model = joblib.load('rf_sleep_model.joblib')")
    print("  predictions = model.predict(X_scaled)  # No sequences needed")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()