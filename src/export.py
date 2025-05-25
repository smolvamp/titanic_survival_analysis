"""
Export module for Titanic dataset preprocessing.
"""

import pandas as pd


def export_cleaned_dataset(df, filename='titanic_cleaned.csv'):
    """
    Export the cleaned and processed dataset.
    
    Args:
        df (pd.DataFrame): Cleaned dataset to export
        filename (str): Output filename
    """
    print("\n" + "=" * 50)
    print("EXPORTING CLEANED DATASET")
    print("=" * 50)
    
    try:
        df.to_csv(filename, index=False)
        print(f"✓ Dataset exported successfully!")
        print(f"  Filename: {filename}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        print(f"\n📊 CLEANED DATASET PREVIEW:")
        print(df.head())
        
        print(f"\n📋 FINAL DATA TYPES:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
            
    except Exception as e:
        print(f"✗ Error exporting dataset: {e}")
