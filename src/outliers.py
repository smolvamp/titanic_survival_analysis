"""
Outlier detection and handling module for Titanic dataset preprocessing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def handle_outliers(df):
    """
    Identify and handle outliers in numerical columns.
    
    Args:
        df (pd.DataFrame): Dataset to process
        
    Returns:
        pd.DataFrame: Dataset with outliers handled
    """
    print("\n" + "=" * 50)
    print("HANDLING OUTLIERS")
    print("=" * 50)
    
    df_outliers = df.copy()
    
    # Create boxplots for Fare and Age
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fare analysis and transformation
    if 'Fare' in df_outliers.columns:
        axes[0].boxplot(df_outliers['Fare'].dropna())
        axes[0].set_title('Fare Distribution (Before)')
        axes[0].set_ylabel('Fare')
        
        fare_skewness = df_outliers['Fare'].skew()
        print(f"📊 Fare skewness: {fare_skewness:.3f}")
        
        if fare_skewness > 1:  # Highly skewed
            df_outliers['Fare_log'] = np.log1p(df_outliers['Fare'])
            print(f"✓ Fare: Applied log transformation (new skewness: {df_outliers['Fare_log'].skew():.3f})")
    
    # Age analysis
    if 'Age' in df_outliers.columns:
        axes[1].boxplot(df_outliers['Age'].dropna())
        axes[1].set_title('Age Distribution')
        axes[1].set_ylabel('Age')
        
        age_skewness = df_outliers['Age'].skew()
        print(f"📊 Age skewness: {age_skewness:.3f}")
    
    plt.tight_layout()
    plt.savefig('outlier_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Boxplots saved as 'outlier_boxplots.png'")
    
    return df_outliers
