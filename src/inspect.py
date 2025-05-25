"""
Data inspection module for Titanic dataset preprocessing.
"""

import pandas as pd


def inspect_dataset(df):
    """
    Perform initial inspection of the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to inspect
    """
    print("\n" + "=" * 50)
    print("DATASET INSPECTION")
    print("=" * 50)
    
    print("\n📊 Dataset Head:")
    print(df.head())
    
    print("\n📋 Dataset Info:")
    print(df.info())
    
    print("\n🔍 Missing Values:")
    missing_vals = df.isnull().sum()
    missing_percent = (missing_vals / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_vals,
        'Percentage': missing_percent
    }).sort_values('Missing Count', ascending=False)
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print("\n📈 Descriptive Statistics:")
    print(df.describe())
    
    print("\n🏷️ Unique Values:")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique()} unique values")
