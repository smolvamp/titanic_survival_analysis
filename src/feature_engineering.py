"""
Feature engineering module for Titanic dataset preprocessing.
"""

import pandas as pd


def create_new_features(df):
    """
    Create new features from existing ones.
    
    Args:
        df (pd.DataFrame): Dataset to enhance
        
    Returns:
        pd.DataFrame: Dataset with new features
    """
    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING")
    print("=" * 50)
    
    df_featured = df.copy()
    
    # Extract Title from Name
    if 'Name' in df_featured.columns:
        df_featured['Title'] = df_featured['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Group rare titles
        title_counts = df_featured['Title'].value_counts()
        rare_titles = title_counts[title_counts < 10].index
        df_featured['Title'] = df_featured['Title'].replace(rare_titles, 'Rare')
        
        print("✓ Title: Extracted from Name column")
        print(f"  Distribution: {df_featured['Title'].value_counts().to_dict()}")
        
        # Drop Name column after title extraction
        df_featured.drop('Name', axis=1, inplace=True)
        print("✓ Name: Dropped after title extraction")
    
    # Create FamilySize
    if 'SibSp' in df_featured.columns and 'Parch' in df_featured.columns:
        df_featured['FamilySize'] = df_featured['SibSp'] + df_featured['Parch'] + 1
        print(f"✓ FamilySize: Created (range: {df_featured['FamilySize'].min()}-{df_featured['FamilySize'].max()})")
    
    # Create IsAlone
    if 'FamilySize' in df_featured.columns:
        df_featured['IsAlone'] = (df_featured['FamilySize'] == 1).astype(int)
        alone_count = df_featured['IsAlone'].sum()
        print(f"✓ IsAlone: Created ({alone_count} passengers traveling alone)")
    
    return df_featured
