"""
Data loading module for Titanic dataset preprocessing.
"""

import pandas as pd


def load_titanic_data(filepath='data/train.csv'):
    """
    Load the Titanic dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset or None if error
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Dataset loaded successfully from {filepath}")
        print(f"  Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None
