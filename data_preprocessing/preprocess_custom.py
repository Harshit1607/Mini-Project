import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import warnings

# Add parent directory to path to import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SAMPLE_RATE_HZ, EPOCH_SECONDS, FEATURE_COLUMNS

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_custom")

def extract_features(group):
    """
    Extract exact features matching FEATURE_COLUMNS.
    """
    x = group['x'].values
    y = group['y'].values
    z = group['z'].values
    
    # Calculate dynamic magnitude
    mag = np.sqrt(x**2 + y**2 + z**2)
    
    has_audio = 'audio_energy' in group.columns and not group['audio_energy'].isna().all()
    audio = group['audio_energy'].values if has_audio else np.zeros_like(mag)
    
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
    
    # Audio features
    if has_audio and len(audio) > 0:
        f['audio_mean'] = np.mean(audio)
        f['audio_max'] = np.max(audio)
        f['audio_std'] = np.std(audio)
    else:
        f['audio_mean'] = 0.0
        f['audio_max'] = 0.0
        f['audio_std'] = 0.0
        
    return pd.Series({col: f[col] for col in FEATURE_COLUMNS})

def process_day(person_folder, day_folder):
    day_path = os.path.join(DATA_DIR, person_folder, day_folder)
    accel_path = os.path.join(day_path, "Accelerometer.csv")
    mic_path = os.path.join(day_path, "Microphone.csv")

    if not os.path.exists(accel_path):
        print(f"Missing Accelerometer.csv in {person_folder}/{day_folder}")
        return

    has_mic = os.path.exists(mic_path) and os.path.getsize(mic_path) > 0
    if not has_mic:
        print(f"Warning: Missing or empty Microphone.csv in {person_folder}/{day_folder}. Audio features will be 0.")

    print(f"Processing: {person_folder} - {day_folder}")

    acc = pd.read_csv(accel_path)
    acc = acc.sort_values("seconds_elapsed").reset_index(drop=True)
    
    # --- ENHANCED NORMALIZATION ---
    # 1. Centering: Remove DC offset/bias from linear acceleration
    acc["x"] = acc["x"] - acc["x"].mean()
    acc["y"] = acc["y"] - acc["y"].mean()
    acc["z"] = acc["z"] - acc["z"].mean()

    # 2. Denoising: Apply smoothing to reduce high-frequency sensor jitter
    # (PhysioNet data is typically much cleaner than raw phone sensors)
    window_size = 5 # 5 samples @ 50Hz = 0.1s smoothing
    acc["x"] = acc["x"].rolling(window=window_size, center=True).mean().fillna(acc["x"])
    acc["y"] = acc["y"].rolling(window=window_size, center=True).mean().fillna(acc["y"])
    acc["z"] = acc["z"].rolling(window=window_size, center=True).mean().fillna(acc["z"])

    # 3. Scaling: De-sensitize phone-on-bed movement to match wrist-worn baseline
    # PhysioNet std_mag in sleep is ~0.0015, custom was ~0.0075. Scaling by ~0.2x.
    scaling_factor = 0.25
    acc["x"] *= scaling_factor
    acc["y"] *= scaling_factor
    acc["z"] *= scaling_factor

    # 4. Simulated Gravity: Add 1.0g to Z axis to align with raw PhysioNet magnitude profile
    acc["z"] = acc["z"] + 1.0
    
    start_time = acc["seconds_elapsed"].min()
    end_time = acc["seconds_elapsed"].max()

    if has_mic:
        mic = pd.read_csv(mic_path)
        mic = mic.sort_values("seconds_elapsed").reset_index(drop=True)
        # Handle dBFS: usually negative. -dBFS makes it positive energy.
        mic["audio_energy"] = -mic["dBFS"]
        
        # NORMALIZATION: Min-Max scale the audio energy per night
        # Using a small epsilon to avoid division by zero and handle quiet environments
        audio_min = mic["audio_energy"].min()
        audio_max = mic["audio_energy"].max()
        if audio_max > audio_min + 0.1:
            mic["audio_energy"] = (mic["audio_energy"] - audio_min) / (audio_max - audio_min)
        else:
            mic["audio_energy"] = 0.0
            
        start_time = max(start_time, mic["seconds_elapsed"].min())
        end_time = min(end_time, mic["seconds_elapsed"].max())

    time_step = 1.0 / SAMPLE_RATE_HZ
    common_time = np.arange(start_time, end_time, time_step)

    x_interp = np.interp(common_time, acc["seconds_elapsed"], acc["x"])
    y_interp = np.interp(common_time, acc["seconds_elapsed"], acc["y"])
    z_interp = np.interp(common_time, acc["seconds_elapsed"], acc["z"])
    
    merged_data = {
        "seconds_elapsed": common_time,
        "x": x_interp,
        "y": y_interp,
        "z": z_interp
    }

    if has_mic:
        merged_data["audio_energy"] = np.interp(common_time, mic["seconds_elapsed"], mic["audio_energy"])

    merged = pd.DataFrame(merged_data)
    merged["window"] = ((merged["seconds_elapsed"] - start_time) // EPOCH_SECONDS).astype(int)

    print(f"Extracting features from {len(merged.groupby('window'))} windows...")
    features = merged.groupby("window").apply(extract_features).reset_index()

    person_output = os.path.join(OUTPUT_DIR, person_folder, day_folder)
    os.makedirs(person_output, exist_ok=True)

    features.to_csv(os.path.join(person_output, "Preprocessed_Window_Features.csv"), index=False)
    print(f"Saved exact {len(FEATURE_COLUMNS)} feature columns for {person_folder}/{day_folder}")

def process_all():
    if not os.path.exists(DATA_DIR):
        print(f"data folder {DATA_DIR} not found")
        return

    for person in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person)
        if not os.path.isdir(person_path):
            continue
        for day in os.listdir(person_path):
            day_path = os.path.join(person_path, day)
            if os.path.isdir(day_path):
                process_day(person, day)

if __name__ == "__main__":
    process_all()
