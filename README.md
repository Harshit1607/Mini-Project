# Sleep Stage Classification using Random Forest
## Early Detection of Sleep Disorders and Parasomnias - SomnusGuard

**Authors:** Harshit Bareja, Ishika Manchanda, Teena Kaintura  
**Guide:** Ms. Nupur Chugh  
**Institution:** Bharati Vidyapeeth College of Engineering

---

## Quick Start Guide (Ready in 5 minutes!)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
```bash
python download_data.py
```

### Step 3: Run Training & Evaluation
```bash
python sleep_paralysis_rf_classifier.py
```

That's it! The model will train and generate results automatically.

---

## What This Does

This implementation provides a complete **Random Forest-based sleep stage classification system** using accelerometer data from the PhysioNet Sleep-Accel dataset.

### Key Features:

1. **Automated Data Processing Pipeline**
   - Loads multi-subject accelerometer data
   - Segments into 30-second epochs (standard for sleep studies)
   - Extracts 25+ time-domain and statistical features

2. **Random Forest Classifier**
   - 200 decision trees with optimized hyperparameters
   - Classifies 5 sleep stages: Wake, N1, N2, N3, REM
   - Cross-validation for robust performance estimation

3. **Comprehensive Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrices
   - Feature importance analysis
   - Hypnogram visualization

4. **Professional Visualizations**
   - Performance metrics plots
   - Feature importance rankings
   - Sleep stage distribution
   - Sample hypnogram

---

## Output Files

After running, you'll get:

1. **`sleep_classification_results.png`** - 4-panel visualization with:
   - Confusion matrix
   - Top 15 feature importances
   - Sleep stage distribution comparison
   - Per-class performance metrics

2. **`sample_hypnogram.png`** - Visual representation of sleep stage progression

3. **`feature_importance.csv`** - Complete ranking of all features

---

## Technical Details

### Features Extracted (per 30-second epoch):

**Magnitude-based features:**
- Mean, Standard Deviation, Variance
- Min, Max, Range, Median
- 25th & 75th Percentiles, IQR
- Skewness, Kurtosis
- Energy (sum of squares)
- Zero-crossing rate

**Per-axis features (X, Y, Z):**
- Mean, Standard Deviation, Range

**Movement dynamics:**
- Mean & Standard Deviation of magnitude differences

**Total: 25 features per epoch**

### Model Architecture:

```python
RandomForestClassifier(
    n_estimators=200,        # 200 decision trees
    max_depth=20,            # Maximum tree depth
    min_samples_split=10,    # Min samples to split
    min_samples_leaf=4,      # Min samples in leaf
    max_features='sqrt',     # Features per split
    random_state=42
)
```

### Dataset Information:

- **Source:** PhysioNet Sleep-Accel Database v1.0.0
- **Subjects:** 31 individuals
- **Data:** Wrist-worn accelerometer + PSG-labeled sleep stages
- **Labels:** Wake (0), N1 (1), N2 (2), N3 (3), REM (5)
- **Citation:** Walch et al., "Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device", SLEEP (2019)

---

## Expected Performance

Based on similar studies with this dataset, you should achieve:

- **Overall Accuracy:** 75-85%
- **Wake Detection:** High precision (~85-90%)
- **NREM/REM Classification:** Moderate to good (~70-80%)
- **N1 Stage:** Lower accuracy (challenging even for experts)

### Performance Factors:
- N1 is the most difficult stage to classify (even experts struggle)
- Wake and REM typically have highest accuracies
- N2 and N3 (deep sleep) show good discrimination
- Results vary by individual sleep patterns

---

## Integration with SomnusGuard

This trained model serves as the **foundation** for the SomnusGuard system:

1. **Sleep Stage Classification** → Generates hypnogram
2. **REM Instability Analysis** → Detects sleep paralysis risk patterns
3. **Night Terror Detection** → Identifies NREM-based terror events

### Next Steps for Complete System:

1. **Sleep Paralysis Risk Model:**
   ```python
   # Count REM-to-Wake transitions
   rem_wake_transitions = count_transitions(hypnogram, REM, WAKE)
   if rem_wake_transitions > threshold:
       flag_sleep_paralysis_risk()
   ```

2. **Night Terror Detection:**
   ```python
   # Detect simultaneous audio + movement spikes in NREM
   if audio_spike AND movement_spike AND stage == NREM_DEEP:
       flag_potential_night_terror()
   ```

---

## Code Structure

```
├── sleep_paralysis_rf_classifier.py  # Main implementation
│   ├── SleepStageClassifier          # Complete pipeline class
│   │   ├── load_subject_data()       # Data loading
│   │   ├── extract_epoch_features()  # Feature engineering
│   │   ├── create_epochs()           # Data segmentation
│   │   ├── train_model()             # RF training
│   │   ├── evaluate_model()          # Performance metrics
│   │   └── plot_results()            # Visualization
│   └── main()                        # Orchestrates pipeline
│
├── download_data.py                  # Dataset downloader
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Troubleshooting

### Issue: "Data directory not found"
**Solution:** Run `python download_data.py` first

### Issue: "Module not found"
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Issue: Low accuracy on first run
**Solution:** Normal! Random Forest uses randomization. Try running 2-3 times.

### Issue: Memory error
**Solution:** Reduce subjects or batch processing (modify code to process fewer subjects)

---

## Presentation Tips

When showing this to your teacher:

1. **Start with the problem:** "Traditional PSG is expensive and inaccessible"

2. **Show the solution:** "We use smartphone accelerometer + ML"

3. **Demonstrate results:** 
   - Run the code live (takes 5-10 minutes)
   - Show the confusion matrix
   - Explain the hypnogram

4. **Highlight innovation:**
   - Multi-modal approach (accel + audio)
   - Privacy-first (on-device processing)
   - Parasomnia-specific detection

5. **Future work:**
   - Add more algorithms (SVM, Neural Networks)
   - Real-time mobile implementation
   - Clinical validation study

---

## References

1. Walch et al. (2019). "Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device." *Sleep*, zsz180.

2. American Academy of Sleep Medicine. (2014). *International Classification of Sleep Disorders*, 3rd ed.

3. PhysioNet Sleep-Accel Database: https://physionet.org/content/sleep-accel/1.0.0/

---

## License

This implementation is for educational purposes as part of the Mini Project for B.Tech Computer Science & Engineering at Bharati Vidyapeeth College of Engineering.

Dataset: Open Data Commons Attribution License v1.0

---

## Contact

For questions or issues:
- Harshit Bareja (00611502723)
- Ishika Manchanda (02611502723)
- Teena Kaintura (05711502723)

**Guide:** Ms. Nupur Chugh, Professor, Bharati Vidyapeeth College of Engineering

---

*Last Updated: September 2025*