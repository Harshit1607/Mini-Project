"""
EARLY DETECTION OF SLEEP DISORDERS AND PARASOMNIAS
Sleep Stage Classification using a Hybrid CNN-LSTM Network (V2.1 FIXED)

Authors: Harshit Bareja, Ishika Manchanda, Teena Kaintura
Guide: Ms. Nupur Chugh
Institution: Bharati Vidyapeeth College of Engineering

This is an improved implementation (V2.1) FIXED FOR ANALYZER COMPATIBILITY

Key Improvements:
1.  *Hybrid CNN-LSTM Architecture:*
2.  *Class Weighting:*
3.  *FIXED: Proper __init__ method (was _init_)*
4.  *FIXED: Correct magnitude calculation*
5.  *ANALYZER COMPATIBLE: Saves with correct filenames and metadata*

Requires: tensorflow, scikit-learn, joblib
(pip install tensorflow scikit-learn joblib)
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import butter, filtfilt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight 
import joblib 
import json

# --- TensorFlow/Keras Imports ---
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, TimeDistributed,
    Conv1D, MaxPooling1D, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')

# --- Configuration ---
EPOCH_DURATION = 30  # seconds (standard)
SAMPLING_RATE = 50   # Hz (approximate from dataset)
RANDOM_STATE = 42
OUTPUT_DIR = './outputs'  # Cross-platform compatible output directory

# --- LSTM Configuration ---
SEQUENCE_LENGTH = 10  # Use 10 epochs (5 minutes) - COMPATIBLE WITH ANALYZER
LSTM_EPOCHS = 30      # Train for a bit longer
LSTM_BATCH_SIZE = 64  # Batch size for LSTM training


class SleepStageClassifierCNNLSTM:
    """
    Complete pipeline for sleep stage classification using a CNN-LSTM.
    """
    
    def __init__(self, data_path='../sleep-accel-data/', output_dir='./outputs'):
        """
        Initialize the classifier
        FIXED: Was _init_ now __init__
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.model = None
        self.scaler = StandardScaler()
        self.history = None # To store training history
        self.sequence_length = SEQUENCE_LENGTH  # Store for metadata
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # --- Label Mapping for Keras ---
        self.label_mapping = {
            -1: 0,
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        self.num_classes = len(self.label_mapping)

        self.sleep_stage_mapping = {
            -1: 'Unknown',
            0: 'Wake',
            1: 'N1',
            2: 'N2', 
            3: 'N3',
            4: 'N4',
            5: 'REM'
        }
        print(f"CNN-LSTM Classifier initialized. Using {self.num_classes} classes.")

    # --- Data Loading and Feature Extraction ---
    
    def load_subject_data(self, subject_id):
        accel_file = os.path.join(self.data_path, f'{subject_id}_acceleration.txt')
        accel_df = pd.read_csv(accel_file, sep=' ', names=['timestamp', 'x', 'y', 'z'])
        
        labels_file = os.path.join(self.data_path, f'{subject_id}_labeled_sleep.txt')
        labels_df = pd.read_csv(labels_file, sep=' ', names=['timestamp', 'stage'])
        
        return accel_df, labels_df
    
    def calculate_magnitude(self, x, y, z):
        """
        FIXED: Was x*2 + y2 + z*2, now x**2 + y**2 + z**2
        """
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
                features = self.extract_epoch_features(epoch_accel)
                features_list.append(features)
                
                label = epoch_label['stage'].mode()[0]
                labels_list.append(label)
            
            current_time += EPOCH_DURATION
        
        return features_list, labels_list

    # --- Data Preparation for CNN-LSTM ---

    def load_and_process_all_subjects(self):
        print("Loading and processing all subjects...")
        
        subject_data_list = []
        all_features_dfs = []
        
        subject_files = [f for f in os.listdir(self.data_path) if f.endswith('_acceleration.txt')]
        subject_ids = [f.replace('_acceleration.txt', '') for f in subject_files]
        
        print(f"Found {len(subject_ids)} subjects")
        
        for i, subject_id in enumerate(subject_ids, 1):
            try:
                print(f"Processing subject {i}/{len(subject_ids)}: {subject_id}")
                
                accel_df, labels_df = self.load_subject_data(subject_id)
                features, labels = self.create_epochs(accel_df, labels_df)
                
                if len(features) > SEQUENCE_LENGTH:
                    X_subj = pd.DataFrame(features)
                    y_subj = pd.Series(labels, name='stage')
                    
                    subject_data_list.append((X_subj, y_subj))
                    all_features_dfs.append(X_subj)
                    
                    print(f"  -> Extracted {len(features)} epochs")
                else:
                    print(f"  -> Skipping subject: Not enough epochs ({len(features)})")
                    
            except Exception as e:
                print(f"  -> Error processing subject {subject_id}: {e}")
                continue
        
        # Handle potential empty list
        if not all_features_dfs:
            print("No subjects processed successfully. Exiting.")
            return None, None
            
        X_all_df = pd.concat(all_features_dfs, ignore_index=True)
        
        print(f"\nTotal epochs from all subjects: {len(X_all_df)}")
        return subject_data_list, X_all_df
        
    def create_lstm_sequences(self, subject_data_list, sequence_length):
        print(f"\nCreating sequences of length {sequence_length}...")
        
        X_sequences = []
        y_sequences = []
        
        for X_subj, y_subj in subject_data_list:
            X_subj_scaled = self.scaler.transform(X_subj)
            
            for i in range(len(X_subj_scaled) - sequence_length + 1):
                X_sequences.append(X_subj_scaled[i : i + sequence_length])
                y_sequences.append(y_subj.iloc[i + sequence_length - 1])
        
        X_out = np.array(X_sequences)
        y_out = np.array(y_sequences)
        
        print(f"Total sequences created: {len(X_out)}")
        print(f"X_sequences shape: {X_out.shape}")
        print(f"y_sequences shape: {y_out.shape}")
        
        return X_out, y_out

    # --- CNN-LSTM Model Building and Training ---

    def build_and_train_cnn_lstm(self, X_train, y_train_mapped, X_val, y_val_mapped, class_weights):
        print("\n" + "="*60)
        print("BUILDING AND TRAINING *CNN-LSTM* MODEL")
        print("="*60)
        
        n_sequences, n_timesteps, n_features = X_train.shape
        
        input_layer = Input(shape=(n_timesteps, n_features))
        
        x = tf.keras.layers.Reshape((n_timesteps, n_features, 1))(input_layer)

        cnn = TimeDistributed(Conv1D(
            filters=64, 
            kernel_size=3, 
            activation='relu', 
            padding='same'
        ))(x)
        cnn = TimeDistributed(MaxPooling1D(pool_size=2, padding='same'))(cnn)
        cnn = TimeDistributed(Conv1D(
            filters=64, 
            kernel_size=3, 
            activation='relu', 
            padding='same'
        ))(cnn)
        cnn = TimeDistributed(MaxPooling1D(pool_size=2, padding='same'))(cnn)
        
        cnn = TimeDistributed(Flatten())(cnn)
        
        lstm = LSTM(100, return_sequences=False)(cnn)
        lstm = Dropout(0.4)(lstm)
        
        dense = Dense(64, activation='relu')(lstm)
        dense = Dropout(0.2)(dense)
        
        output_layer = Dense(self.num_classes, activation='softmax')(dense)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model.summary()
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5,
            restore_best_weights=True
        )
        
        # FIXED: Use lstm_sleep_model.h5 for analyzer compatibility
        model_checkpoint_path = os.path.join(self.output_dir, 'lstm_sleep_model.h5')
        model_checkpoint = ModelCheckpoint(
            filepath=model_checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        print("\nTraining CNN-LSTM in progress...")
        print(f"Using class weights: {class_weights}")
        
        self.history = self.model.fit(
            X_train,
            y_train_mapped,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_data=(X_val, y_val_mapped),
            callbacks=[early_stopping, model_checkpoint],
            class_weight=class_weights,
            verbose=1
        )
        
        print("\nModel training completed!")
        print(f"Best model saved to {model_checkpoint_path}")
        
        self.model.load_weights(model_checkpoint_path)

    # --- Evaluation and Plotting ---

    def evaluate_model(self, X_test, y_test_original):
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        y_pred_probs = self.model.predict(X_test)
        y_pred_mapped = np.argmax(y_pred_probs, axis=1)
        
        predictions_original_format = np.vectorize(self.reverse_label_mapping.get)(y_pred_mapped)
        
        accuracy = accuracy_score(y_test_original, predictions_original_format)
        print(f"\nTest Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nDetailed Classification Report:")
        print("-" * 60)
        
        unique_stages = sorted(np.unique(y_test_original))
        target_names = [self.sleep_stage_mapping.get(int(i), f'Stage_{i}') for i in unique_stages]
        
        print(classification_report(
            y_test_original, 
            predictions_original_format, 
            target_names=target_names,
            labels=unique_stages,
            digits=4,
            zero_division=0
        ))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test_original, predictions_original_format, labels=unique_stages)
        print(cm)
        
        return predictions_original_format

    def save_confusion_matrix(self, y_test, predictions):
        print("\nSaving confusion matrix...")
        plt.figure(figsize=(10, 8))
        
        unique_stages = sorted(np.unique(y_test))
        stage_names = [self.sleep_stage_mapping.get(int(i), f'Stage_{i}') for i in unique_stages]
        cm = confusion_matrix(y_test, predictions, labels=unique_stages)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=stage_names, yticklabels=stage_names)
        plt.title('CNN-LSTM - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        output_path = os.path.join(self.output_dir, 'lstm_confusion_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {os.path.abspath(output_path)}")
        
    def generate_hypnogram_sample(self, predictions, sample_size=500):
        print("\nGenerating sample hypnogram...")
        sample_pred = predictions[:min(sample_size, len(predictions))]
        time_hours = np.arange(len(sample_pred)) * (EPOCH_DURATION / 3600)
        
        plt.figure(figsize=(15, 6))
        plt.plot(time_hours, sample_pred, linewidth=2, color='#2c3e50')
        plt.fill_between(time_hours, sample_pred, alpha=0.3, color='#3498db')
        
        unique_stages_in_sample = sorted(np.unique(sample_pred))
        plt.yticks(unique_stages_in_sample, [self.sleep_stage_mapping.get(int(i), f'Stage_{i}') 
                            for i in unique_stages_in_sample])
        plt.xlabel('Time (hours)', fontsize=12, fontweight='bold')
        plt.ylabel('Sleep Stage', fontsize=12, fontweight='bold')
        plt.title('CNN-LSTM - Sample Hypnogram', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, time_hours[-1]])
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'lstm_hypnogram.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Hypnogram saved to: {os.path.abspath(output_path)}")


def main():
    print("="*60)
    print("SLEEP STAGE CLASSIFICATION (V2 - CNN-LSTM FIXED)")
    print("SomnusGuard - Step 1: Sleep Stage Classification")
    print("="*60)
    
    classifier = SleepStageClassifierCNNLSTM(
        data_path='../sleep-accel-data/',
        output_dir=OUTPUT_DIR
    )
    
    print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")
    
    # STEP 1
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*60)
    
    subject_data_list, X_all_df = classifier.load_and_process_all_subjects()
    
    if X_all_df is None:
        print("Halting execution due to data loading errors.")
        return
    
    # STEP 2
    print("\n" + "="*60)
    print("STEP 2: FITTING SCALER & CREATING SEQUENCES")
    print("="*60)
    
    print(f"Fitting StandardScaler on {len(X_all_df)} total epochs...")
    classifier.scaler.fit(X_all_df)
    print("Scaler fitted.")
    
    # FIXED: Save scaler with correct filename for analyzer
    scaler_path = os.path.join(OUTPUT_DIR, 'lstm_scaler.joblib')
    joblib.dump(classifier.scaler, scaler_path)
    print(f"✓ Scaler saved to: {os.path.abspath(scaler_path)}")
    
    X_seq, y_seq = classifier.create_lstm_sequences(
        subject_data_list, 
        sequence_length=SEQUENCE_LENGTH
    )
    
    # STEP 3
    print("\n" + "="*60)
    print("STEP 3: LABEL MAPPING AND TRAIN-TEST SPLIT")
    print("="*60)
    
    y_seq_mapped = np.vectorize(classifier.label_mapping.get)(y_seq)
    
    X_train, X_test, y_train_mapped, y_test_mapped, y_train_orig, y_test_orig = train_test_split(
        X_seq, 
        y_seq_mapped, 
        y_seq,
        test_size=0.2, 
        random_state=RANDOM_STATE, 
        stratify=y_seq_mapped
    )
    
    print(f"Training sequences: {len(X_train)}")
    print(f"Testing sequences: {len(X_test)}")
    
    # STEP 4
    print("\n" + "="*60)
    print("STEP 4: CALCULATING CLASS WEIGHTS")
    print("="*60)
    
    unique_classes = np.unique(y_train_mapped)
    print(f"Calculating weights for classes: {unique_classes}")
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train_mapped
    )
    
    class_weights_dict = dict(zip(unique_classes, class_weights))
    print("\nClass weights calculated:")
    for stage_mapped, weight in class_weights_dict.items():
        stage_orig = classifier.reverse_label_mapping[stage_mapped]
        stage_name = classifier.sleep_stage_mapping[stage_orig]
        print(f"  {stage_name} (Class {stage_mapped}): {weight:.2f}")

    # STEP 5
    classifier.build_and_train_cnn_lstm(
        X_train, y_train_mapped, 
        X_test, y_test_mapped,
        class_weights_dict
    )
    
    # STEP 6
    predictions = classifier.evaluate_model(X_test, y_test_orig)
    
    # STEP 7
    print("\n" + "="*60)
    print("STEP 7: VISUALIZATION")
    print("="*60)
    classifier.save_confusion_matrix(y_test_orig, predictions)
    classifier.generate_hypnogram_sample(predictions)
    
    # STEP 8
    print("\n" + "="*60)
    print("STEP 8: SAVING MODEL METADATA")
    print("="*60)
    
    # NEW: Save model metadata for analyzer
    metadata = {
        'sequence_length': SEQUENCE_LENGTH,
        'num_features': X_train.shape[2],
        'num_classes': classifier.num_classes,
        'label_mapping': classifier.label_mapping,
        'reverse_label_mapping': classifier.reverse_label_mapping,
        'model_type': 'CNN-LSTM'
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, 'lstm_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Model metadata saved to: {os.path.abspath(metadata_path)}")
    
    print("\n" + "="*60)
    print("✅ CNN-LSTM TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nFiles saved:")
    print(f"  📦 Model: {os.path.join(OUTPUT_DIR, 'lstm_sleep_model.h5')}")
    print(f"  📊 Scaler: {os.path.join(OUTPUT_DIR, 'lstm_scaler.joblib')}")
    print(f"  📄 Metadata: {os.path.join(OUTPUT_DIR, 'lstm_metadata.json')}")
    print(f"  🖼 Confusion Matrix: {os.path.join(OUTPUT_DIR, 'lstm_confusion_matrix.png')}")
    print(f"  🖼 Hypnogram: {os.path.join(OUTPUT_DIR, 'lstm_hypnogram.png')}")
    print("\nNext Steps:")
    print("  1. Ensure RF and XGBoost models are trained")
    print("  2. Run unified_analyzer.py to analyze all subjects with all models")
    print("="*60)

if __name__ == "__main__":
    main()