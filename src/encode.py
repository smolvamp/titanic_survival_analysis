"""
Encoding module for Titanic dataset preprocessing.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categorical_variables(df):
    """
    Encode categorical variables for analysis.
    
    Args:
        df (pd.DataFrame): Dataset to encode
        
    Returns:
        tuple: (encoded_dataset, label_encoders_dict)
    """
    print("\n" + "=" * 50)
    print("ENCODING CATEGORICAL VARIABLES")
    print("=" * 50)
    
    df_encoded = df.copy()
    label_encoders = {}
    
    # Binary encoding for Sex
    if 'Sex' in df_encoded.columns:
        df_encoded['Sex'] = df_encoded['Sex'].map({'male': 1, 'female': 0})
        print("✓ Sex: Encoded (male=1, female=0)")
    
    # Label encoding for other categorical variables
    categorical_cols = ['Embarked', 'Title']
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"✓ {col}: Label encoded - {mapping}")
    
    # Ensure Pclass is integer
    if 'Pclass' in df_encoded.columns:
        df_encoded['Pclass'] = df_encoded['Pclass'].astype(int)
        print("✓ Pclass: Ensured integer format")
    
    return df_encoded, label_encoders
