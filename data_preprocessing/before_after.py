import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")


def plot_day(person, day):
    print(f"Plotting: {person} - {day}")

    # ---------------------------
    # FILE PATHS
    # ---------------------------
    raw_acc_path = os.path.join(DATA_DIR, person, day, "Accelerometer.csv")
    raw_mic_path = os.path.join(DATA_DIR, person, day, "Microphone.csv")

    processed_path = os.path.join(
        PROCESSED_DIR,
        person,
        day,
        "Preprocessed_TimeSeries.csv"
    )

    if not (os.path.exists(raw_acc_path) and os.path.exists(raw_mic_path) and os.path.exists(processed_path)):
        print("Missing files, skipping...")
        return

    # ---------------------------
    # LOAD DATA
    # ---------------------------
    acc = pd.read_csv(raw_acc_path)
    mic = pd.read_csv(raw_mic_path)
    processed = pd.read_csv(processed_path)

    acc = acc.sort_values("seconds_elapsed")
    mic = mic.sort_values("seconds_elapsed")

    # ---------------------------
    # RAW ACCELEROMETER (magnitude)
    # ---------------------------
    acc["magnitude"] = np.sqrt(
        acc["x"]**2 + acc["y"]**2 + acc["z"]**2
    )

    # ADD THIS (same as your preprocessing)
    acc["baseline"] = acc["magnitude"].rolling(
        window=200,
        center=True,
        min_periods=1
    ).median()

    acc["movement_energy"] = acc["magnitude"] - acc["baseline"]

    # ---------------------------
    # PLOT 1: ACCELEROMETER BEFORE vs AFTER (FIXED)
    # ---------------------------
    plt.figure()

    plt.plot(acc["seconds_elapsed"], acc["magnitude"], label="Raw Magnitude", alpha=0.5)
    plt.plot(acc["seconds_elapsed"], acc["baseline"], label="Baseline", linestyle="--")
    plt.plot(acc["seconds_elapsed"], acc["movement_energy"], label="Movement Signal")

    plt.title(f"Accelerometer: Before vs After ({person}-{day})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Signal")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{person}_{day}_accelerometer.png")
    plt.close()

    # ---------------------------
    # PLOT 2: MICROPHONE BEFORE vs AFTER
    # ---------------------------
    mic["audio_energy"] = -mic["dBFS"]

    plt.figure()

    plt.plot(mic["seconds_elapsed"], mic["audio_energy"], label="Raw Audio Energy", alpha=0.6)
    plt.plot(processed["seconds_elapsed"], processed["audio_energy_norm"], label="Processed (Normalized)", alpha=0.8)

    plt.title(f"Microphone: Before vs After ({person}-{day})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Signal")
    plt.legend()

    plt.savefig(f"{person}_{day}_microphone.png")
    plt.close()

    # ---------------------------
    # PLOT 3: COMBINED SIGNAL (IMPORTANT)
    # ---------------------------
    plt.figure()

    plt.plot(processed["seconds_elapsed"], processed["movement_energy_norm"], label="Movement")
    plt.plot(processed["seconds_elapsed"], processed["audio_energy_norm"], label="Audio")

    plt.title(f"Combined Signals ({person}-{day})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Signal")
    plt.legend()

    plt.savefig(f"{person}_{day}_combined.png")
    plt.close()

    print(f"Saved plots for {person}-{day}")


def plot_all():
    for person in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person)

        if not os.path.isdir(person_path):
            continue

        for day in os.listdir(person_path):
            day_path = os.path.join(person_path, day)

            if os.path.isdir(day_path):
                plot_day(person, day)


if __name__ == "__main__":
    plot_all()