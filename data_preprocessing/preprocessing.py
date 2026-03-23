import os
import pandas as pd
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")


# =====================================================
# PROCESS ONE PERSON ONE DAY
# =====================================================

def process_day(person_folder, day_folder):

    day_path = os.path.join(DATA_DIR, person_folder, day_folder)

    accel_path = os.path.join(day_path, "Accelerometer.csv")
    mic_path = os.path.join(day_path, "Microphone.csv")

    if not os.path.exists(accel_path):
        print(f"Missing Accelerometer.csv in {person_folder}/{day_folder}")
        return

    if not os.path.exists(mic_path):
        print(f"Missing Microphone.csv in {person_folder}/{day_folder}")
        return

    # Check for empty files
    if os.path.getsize(accel_path) == 0 or os.path.getsize(mic_path) == 0:
        print(f"Empty Accelerometer/Microphone csv in {person_folder}/{day_folder}")
        return

    print(f"Processing: {person_folder} - {day_folder}")

    # =====================================================
    # 1. LOAD DATA
    # =====================================================

    acc = pd.read_csv(accel_path)
    mic = pd.read_csv(mic_path)

    acc = acc.sort_values("seconds_elapsed").reset_index(drop=True)
    mic = mic.sort_values("seconds_elapsed").reset_index(drop=True)

    # =====================================================
    # 2. ACCELEROMETER RAW MAGNITUDE
    # =====================================================

    acc["magnitude"] = np.sqrt(acc["x"]**2 + acc["y"]**2 + acc["z"]**2)

    # =====================================================
    # 3. MICROPHONE ENERGY
    # =====================================================

    # Use the same audio energy metric as before
    mic["audio_energy"] = -mic["dBFS"]

    # =====================================================
    # 4. RESAMPLE TO COMMON TIME GRID (10Hz)
    # =====================================================

    start_time = max(acc["seconds_elapsed"].min(), mic["seconds_elapsed"].min())
    end_time = min(acc["seconds_elapsed"].max(), mic["seconds_elapsed"].max())
    common_time = np.arange(start_time, end_time, 0.1)

    # We need to interpolate X, Y, Z, and Audio Energy for synchronicity
    x_interp = np.interp(common_time, acc["seconds_elapsed"], acc["x"])
    y_interp = np.interp(common_time, acc["seconds_elapsed"], acc["y"])
    z_interp = np.interp(common_time, acc["seconds_elapsed"], acc["z"])
    mag_interp = np.interp(common_time, acc["seconds_elapsed"], acc["magnitude"])
    mic_interp = np.interp(common_time, mic["seconds_elapsed"], mic["audio_energy"])

    merged = pd.DataFrame({
        "seconds_elapsed": common_time,
        "x": x_interp,
        "y": y_interp,
        "z": z_interp,
        "magnitude": mag_interp,
        "audio_energy": mic_interp
    })

    # =====================================================
    # 5. SEGMENT INTO 30-SECOND WINDOWS (Sleep Medicine Standard)
    # =====================================================

    window_size = 30
    merged["window"] = (merged["seconds_elapsed"] // window_size).astype(int)

    # =====================================================
    # 6. EXTRACT COMPATIBLE FEATURES (25 Accel + 3 Audio)
    # =====================================================

    def extract_features(group):
        mag = group['magnitude'].values
        x = group['x'].values
        y = group['y'].values
        z = group['z'].values
        audio = group['audio_energy'].values
        
        # 25 Accelerometer features used by RF/XGB/LSTM models
        f = {}
        
        # Magnitude stats (13 features)
        f['mean_mag'] = np.mean(mag)
        f['std_mag'] = np.std(mag)
        f['min_mag'] = np.min(mag)
        f['max_mag'] = np.max(mag)
        f['median_mag'] = np.median(mag)
        f['var_mag'] = np.var(mag)
        f['range_mag'] = np.max(mag) - np.min(mag)
        f['q25_mag'] = np.percentile(mag, 25)
        f['q75_mag'] = np.percentile(mag, 75)
        f['iqr_mag'] = f['q75_mag'] - f['q25_mag']
        f['skew_mag'] = stats.skew(mag)
        f['kurtosis_mag'] = stats.kurtosis(mag)
        f['energy_mag'] = np.sum(mag**2)
        
        # Zero Crossing Rate (1 feature)
        mag_centered = mag - np.mean(mag)
        f['zero_crossing_rate'] = np.sum(np.diff(np.sign(mag_centered)) != 0) / len(mag)
        
        # Axis stats (9 features)
        f['mean_x'], f['std_x'], f['range_x'] = np.mean(x), np.std(x), np.max(x) - np.min(x)
        f['mean_y'], f['std_y'], f['range_y'] = np.mean(y), np.std(y), np.max(y) - np.min(y)
        f['mean_z'], f['std_z'], f['range_z'] = np.mean(z), np.std(z), np.max(z) - np.min(z)
        
        # Movement Intensity (2 features)
        diff_mag = np.diff(mag)
        f['mean_diff'] = np.mean(np.abs(diff_mag))
        f['std_diff'] = np.std(diff_mag)
        
        # Additional Audio features (from your original logic)
        f['audio_mean'] = np.mean(audio)
        f['audio_max'] = np.max(audio)
        f['audio_std'] = np.std(audio)
        
        return pd.Series(f)

    print(f"Extracting 28 features from {len(merged.groupby('window'))} windows...")
    features = merged.groupby("window").apply(extract_features).reset_index()

    # =====================================================
    # 7. SAVE OUTPUT (PERSON → DAY)
    # =====================================================

    person_output = os.path.join(OUTPUT_DIR, person_folder, day_folder)
    os.makedirs(person_output, exist_ok=True)

    merged.to_csv(
        os.path.join(person_output, "Preprocessed_TimeSeries.csv"),
        index=False
    )

    features.to_csv(
        os.path.join(person_output, "Preprocessed_Window_Features.csv"),
        index=False
    )

    print(f"Saved output for {person_folder}/{day_folder}")


# =====================================================
# MAIN LOOP
# =====================================================

def process_all():
    if not os.path.exists(DATA_DIR):
        raise ValueError("data folder not found")

    for person in os.listdir(DATA_DIR):

        person_path = os.path.join(DATA_DIR, person)

        if not os.path.isdir(person_path):
            continue

        # Loop over days
        for day in os.listdir(person_path):

            day_path = os.path.join(person_path, day)

            if os.path.isdir(day_path):
                process_day(person, day)


if __name__ == "__main__":
    process_all()