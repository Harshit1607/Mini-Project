"""
QUICK DEMO - Sleep Stage Classification
Run this if you need immediate results without downloading the full dataset
This creates synthetic data that mimics the structure of real sleep data
pip install numpy pandas scikit-learn scipy matplotlib seaborn
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_synthetic_sleep_data(n_epochs=5000):
    """
    Generate synthetic accelerometer features that mimic real sleep patterns
    
    Sleep stage characteristics:
    - Wake: High movement variance, high mean
    - N1: Moderate movement, transitional
    - N2: Lower movement, some spindles
    - N3: Very low movement (deep sleep)
    - REM: Low movement but occasional twitches
    """
    
    print("Generating synthetic sleep data...")
    
    features_list = []
    labels_list = []
    
    # Define sleep stage proportions (realistic distribution)
    stage_proportions = {
        0: 0.15,  # Wake - 15%
        1: 0.10,  # N1 - 10%
        2: 0.45,  # N2 - 45%
        3: 0.15,  # N3 - 15%
        5: 0.15   # REM - 15%
    }
    
    for stage, proportion in stage_proportions.items():
        n_stage_epochs = int(n_epochs * proportion)
        
        for _ in range(n_stage_epochs):
            # Generate features based on sleep stage characteristics
            if stage == 0:  # Wake
                features = {
                    'mean_mag': np.random.uniform(1.2, 2.0),
                    'std_mag': np.random.uniform(0.3, 0.6),
                    'var_mag': np.random.uniform(0.09, 0.36),
                    'range_mag': np.random.uniform(1.0, 2.5),
                    'max_mag': np.random.uniform(2.0, 3.5),
                    'min_mag': np.random.uniform(0.5, 1.0),
                    'median_mag': np.random.uniform(1.1, 1.8),
                    'q25_mag': np.random.uniform(0.8, 1.3),
                    'q75_mag': np.random.uniform(1.4, 2.2),
                    'iqr_mag': np.random.uniform(0.4, 0.9),
                    'skew_mag': np.random.uniform(-0.5, 0.5),
                    'kurtosis_mag': np.random.uniform(-1, 1),
                    'energy_mag': np.random.uniform(80, 150),
                    'zero_crossing_rate': np.random.uniform(0.3, 0.6),
                    'mean_x': np.random.uniform(-0.5, 0.5),
                    'std_x': np.random.uniform(0.2, 0.5),
                    'range_x': np.random.uniform(0.8, 1.5),
                    'mean_y': np.random.uniform(-0.5, 0.5),
                    'std_y': np.random.uniform(0.2, 0.5),
                    'range_y': np.random.uniform(0.8, 1.5),
                    'mean_z': np.random.uniform(0.8, 1.2),
                    'std_z': np.random.uniform(0.2, 0.5),
                    'range_z': np.random.uniform(0.8, 1.5),
                    'mean_diff': np.random.uniform(0.15, 0.35),
                    'std_diff': np.random.uniform(0.15, 0.35)
                }
                
            elif stage == 1:  # N1 (Light sleep transition)
                features = {
                    'mean_mag': np.random.uniform(1.0, 1.4),
                    'std_mag': np.random.uniform(0.15, 0.35),
                    'var_mag': np.random.uniform(0.02, 0.12),
                    'range_mag': np.random.uniform(0.5, 1.2),
                    'max_mag': np.random.uniform(1.5, 2.2),
                    'min_mag': np.random.uniform(0.6, 1.0),
                    'median_mag': np.random.uniform(0.95, 1.3),
                    'q25_mag': np.random.uniform(0.85, 1.1),
                    'q75_mag': np.random.uniform(1.1, 1.5),
                    'iqr_mag': np.random.uniform(0.2, 0.5),
                    'skew_mag': np.random.uniform(-0.3, 0.3),
                    'kurtosis_mag': np.random.uniform(-0.5, 0.5),
                    'energy_mag': np.random.uniform(50, 90),
                    'zero_crossing_rate': np.random.uniform(0.2, 0.4),
                    'mean_x': np.random.uniform(-0.3, 0.3),
                    'std_x': np.random.uniform(0.1, 0.3),
                    'range_x': np.random.uniform(0.4, 0.9),
                    'mean_y': np.random.uniform(-0.3, 0.3),
                    'std_y': np.random.uniform(0.1, 0.3),
                    'range_y': np.random.uniform(0.4, 0.9),
                    'mean_z': np.random.uniform(0.85, 1.15),
                    'std_z': np.random.uniform(0.1, 0.3),
                    'range_z': np.random.uniform(0.4, 0.9),
                    'mean_diff': np.random.uniform(0.08, 0.2),
                    'std_diff': np.random.uniform(0.08, 0.2)
                }
                
            elif stage == 2:  # N2 (Light sleep)
                features = {
                    'mean_mag': np.random.uniform(0.95, 1.15),
                    'std_mag': np.random.uniform(0.08, 0.2),
                    'var_mag': np.random.uniform(0.006, 0.04),
                    'range_mag': np.random.uniform(0.3, 0.8),
                    'max_mag': np.random.uniform(1.2, 1.8),
                    'min_mag': np.random.uniform(0.7, 1.0),
                    'median_mag': np.random.uniform(0.92, 1.12),
                    'q25_mag': np.random.uniform(0.88, 1.05),
                    'q75_mag': np.random.uniform(1.0, 1.2),
                    'iqr_mag': np.random.uniform(0.1, 0.3),
                    'skew_mag': np.random.uniform(-0.2, 0.2),
                    'kurtosis_mag': np.random.uniform(-0.3, 0.3),
                    'energy_mag': np.random.uniform(40, 70),
                    'zero_crossing_rate': np.random.uniform(0.15, 0.3),
                    'mean_x': np.random.uniform(-0.2, 0.2),
                    'std_x': np.random.uniform(0.05, 0.2),
                    'range_x': np.random.uniform(0.2, 0.6),
                    'mean_y': np.random.uniform(-0.2, 0.2),
                    'std_y': np.random.uniform(0.05, 0.2),
                    'range_y': np.random.uniform(0.2, 0.6),
                    'mean_z': np.random.uniform(0.9, 1.1),
                    'std_z': np.random.uniform(0.05, 0.2),
                    'range_z': np.random.uniform(0.2, 0.6),
                    'mean_diff': np.random.uniform(0.04, 0.12),
                    'std_diff': np.random.uniform(0.04, 0.12)
                }
                
            elif stage == 3:  # N3 (Deep sleep)
                features = {
                    'mean_mag': np.random.uniform(0.92, 1.05),
                    'std_mag': np.random.uniform(0.02, 0.08),
                    'var_mag': np.random.uniform(0.0004, 0.006),
                    'range_mag': np.random.uniform(0.1, 0.4),
                    'max_mag': np.random.uniform(1.05, 1.3),
                    'min_mag': np.random.uniform(0.85, 0.98),
                    'median_mag': np.random.uniform(0.93, 1.03),
                    'q25_mag': np.random.uniform(0.91, 1.0),
                    'q75_mag': np.random.uniform(0.98, 1.08),
                    'iqr_mag': np.random.uniform(0.05, 0.15),
                    'skew_mag': np.random.uniform(-0.1, 0.1),
                    'kurtosis_mag': np.random.uniform(-0.2, 0.2),
                    'energy_mag': np.random.uniform(25, 50),
                    'zero_crossing_rate': np.random.uniform(0.05, 0.15),
                    'mean_x': np.random.uniform(-0.1, 0.1),
                    'std_x': np.random.uniform(0.02, 0.1),
                    'range_x': np.random.uniform(0.1, 0.3),
                    'mean_y': np.random.uniform(-0.1, 0.1),
                    'std_y': np.random.uniform(0.02, 0.1),
                    'range_y': np.random.uniform(0.1, 0.3),
                    'mean_z': np.random.uniform(0.95, 1.05),
                    'std_z': np.random.uniform(0.02, 0.1),
                    'range_z': np.random.uniform(0.1, 0.3),
                    'mean_diff': np.random.uniform(0.01, 0.05),
                    'std_diff': np.random.uniform(0.01, 0.05)
                }
                
            else:  # stage == 5 (REM)
                features = {
                    'mean_mag': np.random.uniform(0.98, 1.12),
                    'std_mag': np.random.uniform(0.1, 0.25),
                    'var_mag': np.random.uniform(0.01, 0.06),
                    'range_mag': np.random.uniform(0.4, 0.9),
                    'max_mag': np.random.uniform(1.3, 1.9),
                    'min_mag': np.random.uniform(0.75, 0.95),
                    'median_mag': np.random.uniform(0.96, 1.1),
                    'q25_mag': np.random.uniform(0.9, 1.05),
                    'q75_mag': np.random.uniform(1.05, 1.18),
                    'iqr_mag': np.random.uniform(0.12, 0.25),
                    'skew_mag': np.random.uniform(0, 0.4),
                    'kurtosis_mag': np.random.uniform(-0.2, 0.5),
                    'energy_mag': np.random.uniform(45, 75),
                    'zero_crossing_rate': np.random.uniform(0.18, 0.35),
                    'mean_x': np.random.uniform(-0.15, 0.15),
                    'std_x': np.random.uniform(0.08, 0.25),
                    'range_x': np.random.uniform(0.3, 0.7),
                    'mean_y': np.random.uniform(-0.15, 0.15),
                    'std_y': np.random.uniform(0.08, 0.25),
                    'range_y': np.random.uniform(0.3, 0.7),
                    'mean_z': np.random.uniform(0.92, 1.08),
                    'std_z': np.random.uniform(0.08, 0.25),
                    'range_z': np.random.uniform(0.3, 0.7),
                    'mean_diff': np.random.uniform(0.06, 0.15),
                    'std_diff': np.random.uniform(0.06, 0.15)
                }
            
            features_list.append(features)
            labels_list.append(stage)
    
    # Shuffle the data
    combined = list(zip(features_list, labels_list))
    np.random.shuffle(combined)
    features_list, labels_list = zip(*combined)
    
    X = pd.DataFrame(features_list)
    y = pd.Series(labels_list, name='stage')
    
    return X, y


def train_and_evaluate():
    """Train and evaluate Random Forest on synthetic data"""
    
    print("="*60)
    print("QUICK DEMO - SLEEP STAGE CLASSIFICATION")
    print("Using Synthetic Data")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_sleep_data(n_epochs=5000)
    
    print(f"\nDataset created:")
    print(f"  Total epochs: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"\nSleep stage distribution:")
    stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 5: 'REM'}
    for stage, count in y.value_counts().sort_index().items():
        print(f"  {stage_names[stage]}: {count} ({count/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print("-" * 60)
    target_names = [stage_names[i] for i in sorted(y_test.unique())]
    print(classification_report(y_test, predictions, target_names=target_names, digits=4))
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=target_names, yticklabels=target_names,
               ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    axes[0, 1].barh(range(len(feature_importance)), feature_importance['importance'])
    axes[0, 1].set_yticks(range(len(feature_importance)))
    axes[0, 1].set_yticklabels(feature_importance['feature'])
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_title('Top 15 Features', fontsize=14, fontweight='bold')
    axes[0, 1].invert_yaxis()
    
    # Stage Distribution
    stage_dist = pd.DataFrame({
        'True': y_test.value_counts().sort_index(),
        'Predicted': pd.Series(predictions).value_counts().sort_index()
    })
    stage_dist.index = [stage_names[i] for i in stage_dist.index]
    stage_dist.plot(kind='bar', ax=axes[1, 0], color=['#3498db', '#e74c3c'])
    axes[1, 0].set_title('Sleep Stage Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Sleep Stage')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend(['True', 'Predicted'])
    
    # Sample Hypnogram
    sample_pred = predictions[:min(500, len(predictions))]
    time_hours = np.arange(len(sample_pred)) * (30 / 3600)
    
    axes[1, 1].plot(time_hours, sample_pred, linewidth=2, color='#2c3e50')
    axes[1, 1].fill_between(time_hours, sample_pred, alpha=0.3, color='#3498db')
    axes[1, 1].set_yticks([0, 1, 2, 3, 5])
    axes[1, 1].set_yticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Sleep Stage')
    axes[1, 1].set_title('Sample Hypnogram', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/demo_results.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)
    print("\nResults saved to: demo_results.png")
    print("\nThis was synthetic data. For real results, run:")
    print("  1. python download_data.py")
    print("  2. python sleep_paralysis_rf_classifier.py")


if __name__ == "__main__":
    train_and_evaluate()