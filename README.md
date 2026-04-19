# SomnusGuard: Sleep Stage Classification & Parasomnia Detection
## Early Detection of Sleep Disorders and Parasomnias

**Authors:** Harshit Bareja, Ishika Manchanda, Teena Kaintura  
**Guide:** Ms. Nupur Chugh  
**Institution:** Bharati Vidyapeeth College of Engineering

---

## 🚀 Unified Architecture Overview
SomnusGuard uses a standardized pipeline to bridge the gap between clinical gold-standard data (PhysioNet) and home-collected sensor data. By enforcing a **28-feature unified schema** and implementing **Dynamic Gravity & Audio Normalization**, we ensure that models trained on clinical data perform accurately on real-world smartphone data.

---

## 🛠️ Step-by-Step Workflow

### Phase 1: Pretrain Your Models
Before analyzing your own data, you must generate the clean training dataset and train the clinical-grade models.

#### 1. Process the PhysioNet Training Data
Navigate to the `data_preprocessing` folder and run the PhysioNet script. This converts raw clinical text files into a clean, 28-feature CSV format that perfectly matches our system schema.
```bash
cd data_preprocessing
python preprocess_physionet.py
```
*   **Output:** `data_preprocessing/processed_physionet/physionet_features.csv`

#### 2. Train the Random Forest Model
Navigate to the `server` folder. The training script will read the Step 1 data, validate the column schema, and save the model artifacts.
```bash
cd ../server
python rf.py
```
*   **Outputs:** `rf_sleep_model.joblib`, `rf_scaler.joblib`, `feature_manifest.json` (saved to `server/outputs/`)

#### 3. Train the XGBoost Model
Run the XGBoost training script for high-sensitivity sleep stage detection (especially effective for REM).
```bash
python xg.py
```
*   **Outputs:** `xgb_sleep_model.joblib`, `xgb_scaler.joblib`, `xgb_label_encoder.joblib` (saved to `server/outputs/`)

---

### Phase 2: Process & Analyze Your Custom Data
Once models are trained, you can run inference on your own smartphone-collected data.

#### 4. Add Your Own Data
Place your collected `Accelerometer.csv` and `Microphone.csv` into the folder structure:  
`data_preprocessing/data/<PersonName>/<DayName>/`

#### 5. Preprocess Your Custom Data
Run the custom preprocessor. This automatically syncs audio and accelerometer data, applies **centering**, **denoising**, and **variance scaling** to match the clinical baseline.
```bash
cd ../data_preprocessing
python preprocess_custom.py
```
*   **Output:** `Preprocessed_Window_Features.csv` saved inside `data_preprocessing/processed_custom/<PersonName>/<DayName>/`

#### 6. Run the Final Inference Engine
Go back to the `server` folder and run the unified analyzer. It detects your processed data, runs both RF and XGBoost predictions, and generates a **Risk Report**.
```bash
cd ../server
python analyser.py
```
*   **Risk Metrics:** Risk Scores, Risk Levels (Low to Very High), and Flags (e.g., "High REM Fragmentation").
*   **Visualizations:** Step-based Hypnograms for both models.

---

## 🧪 Unified Feature Schema (28 Features)
Every epoch (30s) is reduced to a consistent 28-feature vector used by all models:

*   **Accelerometer (25):** 
    *   Magnitude: Mean, Std, Max, Min, Range, Median, Var, IQR, 25th/75th Pct, Skew, Kurtosis, Energy, Zero-Crossing.
    *   Axes (X, Y, Z): Mean, Std, Range for each.
    *   Dynamics: Mean/Std of Magnitude Differences.
*   **Audio (3):** 
    *   Mean, Max, and Standard Deviation of normalized audio energy (dBFS-derived).

---

## 📁 Project Structure
```
├── constants.py                # Single source of truth (Sample Rate, Weights, Features)
├── data_preprocessing/
│   ├── preprocess_physionet.py  # Clinical data pipeline
│   ├── preprocess_custom.py     # Home-data pipeline (with Normalization)
│   ├── data/                   # Source data directory
│   └── processed_custom/        # Inference-ready CSVs
├── server/
│   ├── rf.py                   # Random Forest Trainer
│   ├── xg.py                   # XGBoost Trainer
│   ├── analyser.py             # Unified Inference & Risk Engine
│   └── outputs/                # Trained models and manifests (Git Ignored)
└── requirements.txt            # Dependencies (scikit-learn, xgboost, etc.)
```

---

## ⚖️ Configuration
If you need to change a rule (e.g., sample rate, window size, or how heavily "REM-to-Wake transitions" are penalized), you only need to edit **`constants.py`**. The entire pipeline dynamically adapts.

---

## 🎓 Academic Context
This implementation is for educational purposes as part of the Mini Project for B.Tech Computer Science & Engineering at Bharati Vidyapeeth College of Engineering.

**Guide:** Ms. Nupur Chugh, Professor, BVCOE.

*Last Updated: April 2026*