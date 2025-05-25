"""
Data cleaning module for Titanic dataset preprocessing.
"""

import pandas as pd


def clean_missing_values(df):
    """
    Clean missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to clean
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("\n" + "=" * 50)
    print("CLEANING MISSING VALUES")
    print("=" * 50)
    
    df_clean = df.copy()
    
    # Impute Age with median
    if 'Age' in df_clean.columns and df_clean['Age'].isnull().any():
        age_median = df_clean['Age'].median()
        missing_age = df['Age'].isnull().sum()
        df_clean['Age'].fillna(age_median, inplace=True)
        print(f"✓ Age: Filled {missing_age} missing values with median ({age_median:.1f})")
    
    # Impute Embarked with mode
    if 'Embarked' in df_clean.columns and df_clean['Embarked'].isnull().any():
        embarked_mode = df_clean['Embarked'].mode()[0] if not df_clean['Embarked'].mode().empty else 'S'
        missing_embarked = df['Embarked'].isnull().sum()
        df_clean['Embarked'].fillna(embarked_mode, inplace=True)
        print(f"✓ Embarked: Filled {missing_embarked} missing values with mode ('{embarked_mode}')")
    
    # Handle Cabin - convert to binary HasCabin feature
    if 'Cabin' in df_clean.columns:
        df_clean['HasCabin'] = df_clean['Cabin'].notna().astype(int)
        df_clean.drop('Cabin', axis=1, inplace=True)
        print("✓ Cabin: Converted to binary 'HasCabin' feature")
    
    # Fill Fare with median if missing
    if 'Fare' in df_clean.columns and df_clean['Fare'].isnull().any():
        fare_median = df_clean['Fare'].median()
        missing_fare = df['Fare'].isnull().sum()
        df_clean['Fare'].fillna(fare_median, inplace=True)
        print(f"✓ Fare: Filled {missing_fare} missing values with median ({fare_median:.2f})")
    
    remaining_missing = df_clean.isnull().sum().sum()
    print(f"\n📊 Total missing values after cleaning: {remaining_missing}")
    
    return df_clean


def drop_irrelevant_columns(df):
    """
    Drop columns that are not useful for analysis.
    
    Args:
        df (pd.DataFrame): Dataset to process
        
    Returns:
        pd.DataFrame: Dataset with irrelevant columns removed
    """
    print("\n" + "=" * 50)
    print("DROPPING IRRELEVANT COLUMNS")
    print("=" * 50)
    
    df_processed = df.copy()
    columns_to_drop = []
    
    # Drop PassengerId and Ticket
    for col in ['PassengerId', 'Ticket']:
        if col in df_processed.columns:
            columns_to_drop.append(col)
    
    if columns_to_drop:
        df_processed.drop(columns_to_drop, axis=1, inplace=True)
        print(f"✓ Dropped columns: {columns_to_drop}")
    
    print(f"📊 Remaining columns: {list(df_processed.columns)}")
    return df_processed


def check_duplicates(df):
    """
    Check for and remove duplicate rows.
    
    Args:
        df (pd.DataFrame): Dataset to check
        
    Returns:
        pd.DataFrame: Dataset with duplicates removed
    """
    print("\n" + "=" * 50)
    print("CHECKING DUPLICATES")
    print("=" * 50)
    
    duplicate_count = df.duplicated().sum()
    print(f"📊 Duplicate rows found: {duplicate_count}")
    
    if duplicate_count > 0:
        df_no_dupes = df.drop_duplicates()
        print(f"✓ Removed {duplicate_count} duplicate rows")
        return df_no_dupes
    else:
        print("✓ No duplicate rows found")
        return df
