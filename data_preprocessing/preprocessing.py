import os
import pandas as pd
import numpy as np

# =====================================================
# PATH SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

# =====================================================
# PROCESS ONE PERSON
# =====================================================

def process_person(person_folder):

    person_path = os.path.join(DATA_DIR, person_folder)

    accel_path = os.path.join(person_path, "Accelerometer.csv")
    mic_path = os.path.join(person_path, "Microphone.csv")

    if not os.path.exists(accel_path):
        print(f"Missing Accelerometer.csv in {person_folder}")
        return

    if not os.path.exists(mic_path):
        print(f"Missing Microphone.csv in {person_folder}")
        return

    print(f"Processing: {person_folder}")

    # =====================================================
    # 1. LOAD DATA
    # =====================================================

    acc = pd.read_csv(accel_path)
    mic = pd.read_csv(mic_path)

    acc = acc.sort_values("seconds_elapsed").reset_index(drop=True)
    mic = mic.sort_values("seconds_elapsed").reset_index(drop=True)

    # =====================================================
    # 2. ACCELEROMETER PREPROCESSING
    # =====================================================

    acc["magnitude"] = np.sqrt(
        acc["x"]**2 +
        acc["y"]**2 +
        acc["z"]**2
    )

    acc["baseline"] = acc["magnitude"].rolling(
        window=200,
        center=True,
        min_periods=1
    ).median()

    acc["movement_energy"] = acc["magnitude"] - acc["baseline"]

    acc["movement_energy_norm"] = (
        acc["movement_energy"] - acc["movement_energy"].mean()
    ) / acc["movement_energy"].std()

    # =====================================================
    # 3. MICROPHONE PREPROCESSING
    # =====================================================

    mic["audio_energy"] = -mic["dBFS"]

    mic["audio_energy_norm"] = (
        mic["audio_energy"] - mic["audio_energy"].mean()
    ) / mic["audio_energy"].std()

    # =====================================================
    # 4. RESAMPLE TO COMMON TIME GRID
    # =====================================================

    start_time = max(acc["seconds_elapsed"].min(),
                     mic["seconds_elapsed"].min())

    end_time = min(acc["seconds_elapsed"].max(),
                   mic["seconds_elapsed"].max())

    common_time = np.arange(start_time, end_time, 0.1)

    acc_interp = np.interp(
        common_time,
        acc["seconds_elapsed"],
        acc["movement_energy_norm"]
    )

    mic_interp = np.interp(
        common_time,
        mic["seconds_elapsed"],
        mic["audio_energy_norm"]
    )

    merged = pd.DataFrame({
        "seconds_elapsed": common_time,
        "movement_energy_norm": acc_interp,
        "audio_energy_norm": mic_interp
    })

    # =====================================================
    # 5. SEGMENT INTO 5-SECOND WINDOWS
    # =====================================================

    window_size = 5
    merged["window"] = (
        merged["seconds_elapsed"] // window_size
    ).astype(int)

    # =====================================================
    # 6. EXTRACT WINDOW FEATURES
    # =====================================================

    features = merged.groupby("window").agg(
        movement_mean=("movement_energy_norm", "mean"),
        movement_max=("movement_energy_norm", "max"),
        movement_std=("movement_energy_norm", "std"),
        audio_mean=("audio_energy_norm", "mean"),
        audio_max=("audio_energy_norm", "max"),
        audio_std=("audio_energy_norm", "std")
    ).reset_index()

    # =====================================================
    # 7. SAVE OUTPUT
    # =====================================================

    person_output = os.path.join(OUTPUT_DIR, person_folder)
    os.makedirs(person_output, exist_ok=True)

    merged.to_csv(
        os.path.join(person_output, "Preprocessed_TimeSeries.csv"),
        index=False
    )

    features.to_csv(
        os.path.join(person_output, "Preprocessed_Window_Features.csv"),
        index=False
    )

    print(f"Saved output for {person_folder}")


# =====================================================
# MAIN LOOP
# =====================================================

def process_all():
    if not os.path.exists(DATA_DIR):
        raise ValueError("data folder not found")

    for person in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person)

        if os.path.isdir(person_path):
            process_person(person)


if __name__ == "__main__":
    process_all()