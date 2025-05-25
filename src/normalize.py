"""
Normalization module for Titanic dataset preprocessing.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_features(df):
    """
    Normalize numerical features using StandardScaler.
    
    Args:
        df (pd.DataFrame): Dataset to normalize
        
    Returns:
        tuple: (normalized_dataset, scaler_object)
    """
    print("\n" + "=" * 50)
    print("NORMALIZING FEATURES")
    print("=" * 50)
    
    df_normalized = df.copy()
    scaler = StandardScaler()
    
    # Features to normalize
    features_to_scale = []
    
    if 'Age' in df_normalized.columns:
        features_to_scale.append('Age')
    
    # Use log-transformed Fare if available, otherwise original Fare
    if 'Fare_log' in df_normalized.columns:
        features_to_scale.append('Fare_log')
    elif 'Fare' in df_normalized.columns:
        features_to_scale.append('Fare')
    
    if features_to_scale:
        df_normalized[features_to_scale] = scaler.fit_transform(df_normalized[features_to_scale])
        print(f"✓ Normalized features: {features_to_scale}")
        
        for feature in features_to_scale:
            mean_val = df_normalized[feature].mean()
            std_val = df_normalized[feature].std()
            print(f"  {feature}: mean={mean_val:.3f}, std={std_val:.3f}")
    else:
        print("⚠️ No numerical features found for normalization")
    
    return df_normalized, scaler
