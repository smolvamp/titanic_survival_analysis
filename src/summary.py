"""
Summary statistics module for Titanic dataset preprocessing.
"""

import pandas as pd


def generate_summary_statistics(df, original_df_path='data/train.csv'):
    """
    Generate comprehensive summary statistics.
    
    Args:
        df (pd.DataFrame): Processed dataset
        original_df_path (str): Path to original dataset for better stats
    """
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    
    try:
        original_df = pd.read_csv(original_df_path)
    except:
        original_df = df  # Fallback to processed data
    
    # Basic dataset statistics
    print("📊 DATASET OVERVIEW:")
    print(f"  Total records: {len(df)}")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Features: {list(df.columns)}")
    
    # Age statistics
    if 'Age' in original_df.columns:
        print(f"\n📊 AGE STATISTICS:")
        print(f"  Mean age: {original_df['Age'].mean():.1f} years")
        print(f"  Median age: {original_df['Age'].median():.1f} years")
        print(f"  Age range: {original_df['Age'].min():.1f} - {original_df['Age'].max():.1f} years")
    
    # Fare statistics
    if 'Fare' in original_df.columns:
        print(f"\n💰 FARE STATISTICS:")
        print(f"  Mean fare: ${original_df['Fare'].mean():.2f}")
        print(f"  Median fare: ${original_df['Fare'].median():.2f}")
        print(f"  Fare range: ${original_df['Fare'].min():.2f} - ${original_df['Fare'].max():.2f}")
    
    # Class distribution
    if 'Pclass' in original_df.columns:
        print(f"\n🎫 PASSENGER CLASS DISTRIBUTION:")
        class_dist = original_df['Pclass'].value_counts().sort_index()
        for class_num, count in class_dist.items():
            percentage = (count / len(original_df)) * 100
            print(f"  Class {class_num}: {count} passengers ({percentage:.1f}%)")
    
    # Gender distribution
    if 'Sex' in original_df.columns:
        print(f"\n👥 GENDER DISTRIBUTION:")
        gender_dist = original_df['Sex'].value_counts()
        for gender, count in gender_dist.items():
            percentage = (count / len(original_df)) * 100
            print(f"  {gender.title()}: {count} passengers ({percentage:.1f}%)")
    
    # Embarkation statistics
    if 'Embarked' in original_df.columns:
        print(f"\n⚓ EMBARKATION PORT DISTRIBUTION:")
        embarked_dist = original_df['Embarked'].value_counts()
        port_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
        for port, count in embarked_dist.items():
            percentage = (count / len(original_df)) * 100
            port_name = port_names.get(port, port)
            print(f"  {port_name} ({port}): {count} passengers ({percentage:.1f}%)")
    
    # Family statistics
    if 'FamilySize' in df.columns:
        print(f"\n👨‍👩‍👧‍👦 FAMILY STATISTICS:")
        print(f"  Average family size: {df['FamilySize'].mean():.1f}")
        if 'IsAlone' in df.columns:
            alone_pct = (df['IsAlone'].sum() / len(df)) * 100
            print(f"  Passengers traveling alone: {df['IsAlone'].sum()} ({alone_pct:.1f}%)")
        
        family_dist = df['FamilySize'].value_counts().sort_index()
        print(f"  Family size distribution:")
        for size, count in family_dist.items():
            percentage = (count / len(df)) * 100
            print(f"    Size {size}: {count} passengers ({percentage:.1f}%)")
