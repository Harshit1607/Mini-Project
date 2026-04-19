import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import warnings

# Add parent directory to path to import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SAMPLE_RATE_HZ, EPOCH_SECONDS, FEATURE_COLUMNS, STAGE_LABELS

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "sleep-accel-data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_physionet")

def extract_features(group):
    """
    Extract exact features matching FEATURE_COLUMNS.
    For PhysioNet, audio is missing, so we fill with 0s.
    """
    x = group['x'].values
    y = group['y'].values
    z = group['z'].values
    
    # Calculate dynamic magnitude
    mag = np.sqrt(x**2 + y**2 + z**2)
    
    f = {}
    
    # Magnitude stats
    f['mean_mag'] = np.mean(mag)
    f['std_mag'] = np.std(mag)
    f['min_mag'] = np.min(mag)
    f['max_mag'] = np.max(mag)
    f['median_mag'] = np.median(mag)
    f['var_mag'] = np.var(mag)
    f['range_mag'] = np.max(mag) - np.min(mag)
    f['q25_mag'] = np.percentile(mag, 25) if len(mag) > 0 else 0
    f['q75_mag'] = np.percentile(mag, 75) if len(mag) > 0 else 0
    f['iqr_mag'] = f['q75_mag'] - f['q25_mag']
    f['skew_mag'] = stats.skew(mag) if len(mag) > 0 else 0
    f['kurtosis_mag'] = stats.kurtosis(mag) if len(mag) > 0 else 0
    f['energy_mag'] = np.sum(mag**2)
    
    # Zero crossing
    mag_centered = mag - np.mean(mag)
    f['zero_crossing_rate'] = np.sum(np.diff(np.sign(mag_centered)) != 0) / len(mag) if len(mag) > 0 else 0
    
    # Per-axis
    f['mean_x'], f['std_x'], f['range_x'] = np.mean(x), np.std(x), np.max(x) - np.min(x)
    f['mean_y'], f['std_y'], f['range_y'] = np.mean(y), np.std(y), np.max(y) - np.min(y)
    f['mean_z'], f['std_z'], f['range_z'] = np.mean(z), np.std(z), np.max(z) - np.min(z)
    
    # Movement dynamics
    diff_mag = np.diff(mag)
    f['mean_diff'] = np.mean(np.abs(diff_mag)) if len(diff_mag) > 0 else 0
    f['std_diff'] = np.std(diff_mag) if len(diff_mag) > 0 else 0
    
    # Missing Audio - handle case where audio is absent
    f['audio_mean'] = 0.0
    f['audio_max'] = 0.0
    f['audio_std'] = 0.0
    
    # Return as series ordered by FEATURE_COLUMNS
    return pd.Series({col: f[col] for col in FEATURE_COLUMNS})

def process_subject(subject_id):
    print(f"Processing PhysioNet Subject: {subject_id}")
    
    accel_file = os.path.join(DATA_DIR, f'{subject_id}_acceleration.txt')
    labels_file = os.path.join(DATA_DIR, f'{subject_id}_labeled_sleep.txt')
    
    if not os.path.exists(accel_file) or not os.path.exists(labels_file):
        print(f"Missing data for {subject_id}")
        return None
        
    acc = pd.read_csv(accel_file, sep=' ', names=['timestamp', 'x', 'y', 'z'])
    labels = pd.read_csv(labels_file, sep=' ', names=['timestamp', 'stage'])
    
    # Map labels (PhysioNet labels: 0=Wake, 1=N1, 2=N2, 3=N3, 4=N4, 5=REM)
    # Mapping to 5-class schema: 0: Wake, 1: N1, 2: N2, 3: N3/N4, 4: REM
    LABEL_MAP = {-1: -1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}
    labels['stage'] = labels['stage'].map(LABEL_MAP)
    
    # Resample to common time grid based on SAMPLE_RATE_HZ
    start_time = max(acc["timestamp"].min(), labels["timestamp"].min())
    end_time = min(acc["timestamp"].max(), labels["timestamp"].max())
    
    time_step = 1.0 / SAMPLE_RATE_HZ
    common_time = np.arange(start_time, end_time, time_step)
    
    x_interp = np.interp(common_time, acc["timestamp"], acc["x"])
    y_interp = np.interp(common_time, acc["timestamp"], acc["y"])
    z_interp = np.interp(common_time, acc["timestamp"], acc["z"])
    
    merged = pd.DataFrame({
        "timestamp": common_time,
        "x": x_interp,
        "y": y_interp,
        "z": z_interp
    })
    
    # Segment into epochs
    merged["window"] = ((merged["timestamp"] - start_time) // EPOCH_SECONDS).astype(int)
    
    features = merged.groupby("window").apply(extract_features).reset_index()
    
    # Assign labels to windows
    labels["window"] = ((labels["timestamp"] - start_time) // EPOCH_SECONDS).astype(int)
    window_labels = labels.groupby("window")["stage"].apply(lambda x: x.mode()[0] if not x.empty else -1).reset_index()
    
    # Merge features and labels
    final_df = pd.merge(features, window_labels, on="window", how="inner")
    
    # Filter out unknown stages
    final_df = final_df[final_df['stage'] != -1]
    
    if len(final_df) == 0:
        return None
        
    # Ensure exact column order for features
    return final_df[['window', 'stage'] + FEATURE_COLUMNS]

def process_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found.")
        return
        
    subject_files = [f for f in os.listdir(DATA_DIR) if f.endswith('_acceleration.txt')]
    subject_ids = [f.replace('_acceleration.txt', '') for f in subject_files]
    
    all_data = []
    
    for subject_id in subject_ids:
        df = process_subject(subject_id)
        if df is not None:
            df['subject_id'] = subject_id
            all_data.append(df)
            
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        out_path = os.path.join(OUTPUT_DIR, "physionet_features.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Saved preprocessed PhysioNet data to {out_path} with {len(final_df)} epochs.")
    else:
        print("No valid data processed.")

if __name__ == "__main__":
    process_all()
