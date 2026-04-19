SAMPLE_RATE_HZ = 50
EPOCH_SECONDS = 30
SAMPLES_PER_EPOCH = SAMPLE_RATE_HZ * EPOCH_SECONDS

FEATURE_COLUMNS = [
    'mean_mag', 'std_mag', 'min_mag', 'max_mag', 'median_mag', 'var_mag', 'range_mag', 
    'q25_mag', 'q75_mag', 'iqr_mag', 'skew_mag', 'kurtosis_mag', 'energy_mag', 
    'zero_crossing_rate', 'mean_x', 'std_x', 'range_x', 'mean_y', 'std_y', 'range_y', 
    'mean_z', 'std_z', 'range_z', 'mean_diff', 'std_diff', 'audio_mean', 'audio_max', 'audio_std'
]

STAGE_LABELS = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM'
}

RISK_WEIGHTS = {
    'rem_wake_transitions_high': 8,
    'rem_wake_transitions_moderate': 5,
    'rem_fragmentation_high': 12,
    'rem_fragmentation_moderate': 8,
    'total_awakenings_high': 20,
    'rem_percentage_low': 15,
    'rem_percentage_high': 30,
    'rem_latency_long': 120,
    'stage_transition_rate_high': 25
}
