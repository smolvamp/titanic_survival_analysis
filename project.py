"""
Titanic Dataset Preprocessing and Analysis Script
================================================

This script performs comprehensive preprocessing and analysis of the Titanic dataset,
including data cleaning, feature engineering, visualization, and statistical analysis.

Author: Data Analysis Assistant
Date: May 22, 2025
"""

# ============================================================================
# 1. IMPORT NECESSARY LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("TITANIC DATASET PREPROCESSING AND ANALYSIS")
print("=" * 60)

# ============================================================================
# 2. LOAD THE DATASET
# ============================================================================

def load_dataset(filepath):
    """Load the Titanic dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Dataset loaded successfully!")
        print(f"  Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None

# Load the dataset
df = load_dataset('test.csv')
if df is None:
    exit(1)

# ============================================================================
# 3. INITIAL INSPECTION
# ============================================================================

def initial_inspection(df):
    """Perform initial inspection of the dataset."""
    print("\n" + "=" * 60)
    print("INITIAL DATASET INSPECTION")
    print("=" * 60)
    
    # Display basic information
    print("\n📊 DATASET HEAD:")
    print(df.head())
    
    print("\n📋 DATASET INFO:")
    print(df.info())
    
    print("\n🔍 MISSING VALUES:")
    missing_vals = df.isnull().sum()
    missing_percent = (missing_vals / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_vals,
        'Percentage': missing_percent
    }).sort_values('Missing Count', ascending=False)
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print("\n📈 DESCRIPTIVE STATISTICS:")
    print(df.describe())
    
    print("\n🏷️ UNIQUE VALUES PER COLUMN:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

initial_inspection(df)

# ============================================================================
# 4. CLEAN MISSING VALUES
# ============================================================================

def clean_missing_values(df):
    """Clean missing values in the dataset."""
    print("\n" + "=" * 60)
    print("CLEANING MISSING VALUES")
    print("=" * 60)
    
    df_clean = df.copy()
    
    # Impute Age with median
    if 'Age' in df_clean.columns:
        age_median = df_clean['Age'].median()
        df_clean['Age'].fillna(age_median, inplace=True)
        print(f"✓ Age: Filled {df['Age'].isnull().sum()} missing values with median ({age_median:.1f})")
    
    # Impute Embarked with mode
    if 'Embarked' in df_clean.columns:
        embarked_mode = df_clean['Embarked'].mode()[0] if not df_clean['Embarked'].mode().empty else 'S'
        df_clean['Embarked'].fillna(embarked_mode, inplace=True)
        print(f"✓ Embarked: Filled {df['Embarked'].isnull().sum()} missing values with mode ('{embarked_mode}')")
    
    # Handle Cabin - convert to binary HasCabin feature
    if 'Cabin' in df_clean.columns:
        df_clean['HasCabin'] = df_clean['Cabin'].notna().astype(int)
        df_clean.drop('Cabin', axis=1, inplace=True)
        print(f"✓ Cabin: Converted to binary 'HasCabin' feature")
    
    # Fill any remaining missing values in Fare with median
    if 'Fare' in df_clean.columns and df_clean['Fare'].isnull().any():
        fare_median = df_clean['Fare'].median()
        df_clean['Fare'].fillna(fare_median, inplace=True)
        print(f"✓ Fare: Filled {df['Fare'].isnull().sum()} missing values with median ({fare_median:.2f})")
    
    print(f"\n📊 Missing values after cleaning:")
    remaining_missing = df_clean.isnull().sum().sum()
    print(f"Total missing values: {remaining_missing}")
    
    return df_clean

df_clean = clean_missing_values(df)

# ============================================================================
# 5. DROP IRRELEVANT COLUMNS
# ============================================================================

def drop_irrelevant_columns(df):
    """Drop columns that are not useful for analysis."""
    print("\n" + "=" * 60)
    print("DROPPING IRRELEVANT COLUMNS")
    print("=" * 60)
    
    df_processed = df.copy()
    columns_to_drop = []
    
    # Drop PassengerId and Ticket as they don't provide meaningful information
    for col in ['PassengerId', 'Ticket']:
        if col in df_processed.columns:
            columns_to_drop.append(col)
    
    if columns_to_drop:
        df_processed.drop(columns_to_drop, axis=1, inplace=True)
        print(f"✓ Dropped columns: {columns_to_drop}")
    
    print(f"📊 Remaining columns: {list(df_processed.columns)}")
    return df_processed

df_processed = drop_irrelevant_columns(df_clean)

# ============================================================================
# 6. FEATURE ENGINEERING
# ============================================================================

def feature_engineering(df):
    """Create new features from existing ones."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    df_featured = df.copy()
    
    # Extract Title from Name
    if 'Name' in df_featured.columns:
        df_featured['Title'] = df_featured['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Group rare titles
        title_counts = df_featured['Title'].value_counts()
        rare_titles = title_counts[title_counts < 10].index
        df_featured['Title'] = df_featured['Title'].replace(rare_titles, 'Rare')
        
        print(f"✓ Title: Extracted from Name column")
        print(f"  Title distribution: {df_featured['Title'].value_counts().to_dict()}")
        
        # Now we can drop the Name column
        df_featured.drop('Name', axis=1, inplace=True)
        print(f"✓ Name: Dropped after title extraction")
    
    # Create FamilySize
    if 'SibSp' in df_featured.columns and 'Parch' in df_featured.columns:
        df_featured['FamilySize'] = df_featured['SibSp'] + df_featured['Parch'] + 1
        print(f"✓ FamilySize: Created (SibSp + Parch + 1)")
        print(f"  FamilySize range: {df_featured['FamilySize'].min()} to {df_featured['FamilySize'].max()}")
    
    # Create IsAlone
    if 'FamilySize' in df_featured.columns:
        df_featured['IsAlone'] = (df_featured['FamilySize'] == 1).astype(int)
        alone_count = df_featured['IsAlone'].sum()
        print(f"✓ IsAlone: Created ({alone_count} passengers traveling alone)")
    
    return df_featured

df_featured = feature_engineering(df_processed)

# ============================================================================
# 7. ENCODE CATEGORICAL VARIABLES
# ============================================================================

def encode_categorical_variables(df):
    """Encode categorical variables for analysis."""
    print("\n" + "=" * 60)
    print("ENCODING CATEGORICAL VARIABLES")
    print("=" * 60)
    
    df_encoded = df.copy()
    
    # Binary encoding for Sex
    if 'Sex' in df_encoded.columns:
        df_encoded['Sex'] = df_encoded['Sex'].map({'male': 1, 'female': 0})
        print("✓ Sex: Encoded (male=1, female=0)")
    
    # Label encoding for other categorical variables
    categorical_cols = ['Embarked', 'Title']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            print(f"✓ {col}: Label encoded")
            print(f"  Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Pclass is already numeric, but let's ensure it's properly formatted
    if 'Pclass' in df_encoded.columns:
        df_encoded['Pclass'] = df_encoded['Pclass'].astype(int)
        print("✓ Pclass: Ensured integer format")
    
    return df_encoded, label_encoders

df_encoded, encoders = encode_categorical_variables(df_featured)

# ============================================================================
# 8. CHECK FOR DUPLICATES
# ============================================================================

def check_duplicates(df):
    """Check for and remove duplicate rows."""
    print("\n" + "=" * 60)
    print("CHECKING FOR DUPLICATES")
    print("=" * 60)
    
    initial_shape = df.shape
    duplicate_count = df.duplicated().sum()
    
    print(f"📊 Duplicate rows found: {duplicate_count}")
    
    if duplicate_count > 0:
        df_no_dupes = df.drop_duplicates()
        print(f"✓ Removed {duplicate_count} duplicate rows")
        print(f"  Shape before: {initial_shape}")
        print(f"  Shape after: {df_no_dupes.shape}")
        return df_no_dupes
    else:
        print("✓ No duplicate rows found")
        return df

df_no_dupes = check_duplicates(df_encoded)

# ============================================================================
# 9. HANDLE OUTLIERS
# ============================================================================

def handle_outliers(df):
    """Identify and handle outliers in numerical columns."""
    print("\n" + "=" * 60)
    print("HANDLING OUTLIERS")
    print("=" * 60)
    
    df_outliers = df.copy()
    
    # Create boxplots for Fare and Age
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fare boxplot
    if 'Fare' in df_outliers.columns:
        axes[0].boxplot(df_outliers['Fare'].dropna())
        axes[0].set_title('Fare Distribution (Before Transformation)')
        axes[0].set_ylabel('Fare')
        
        # Check if log transformation is needed
        fare_skewness = df_outliers['Fare'].skew()
        print(f"📊 Fare skewness: {fare_skewness:.3f}")
        
        if fare_skewness > 1:  # Highly skewed
            # Add small constant to handle zero values
            df_outliers['Fare_log'] = np.log1p(df_outliers['Fare'])
            print("✓ Fare: Applied log transformation due to high skewness")
            print(f"  New skewness: {df_outliers['Fare_log'].skew():.3f}")
    
    # Age boxplot
    if 'Age' in df_outliers.columns:
        axes[1].boxplot(df_outliers['Age'].dropna())
        axes[1].set_title('Age Distribution')
        axes[1].set_ylabel('Age')
        
        age_skewness = df_outliers['Age'].skew()
        print(f"📊 Age skewness: {age_skewness:.3f}")
    
    plt.tight_layout()
    plt.savefig('outlier_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Boxplots saved as 'outlier_boxplots.png'")
    
    return df_outliers

df_outliers = handle_outliers(df_no_dupes)

# ============================================================================
# 10. NORMALIZE/STANDARDIZE NUMERICAL FEATURES
# ============================================================================

def normalize_features(df):
    """Normalize numerical features using StandardScaler."""
    print("\n" + "=" * 60)
    print("NORMALIZING NUMERICAL FEATURES")
    print("=" * 60)
    
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
    
    return df_normalized, scaler

df_final, scaler = normalize_features(df_outliers)

# ============================================================================
# 11. VISUALIZATIONS
# ============================================================================

def create_visualizations(df):
    """Create comprehensive visualizations of the dataset."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Set up the plotting environment
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Correlation heatmap saved as 'correlation_heatmap.png'")
    
    # 2. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Age distribution
    if 'Age' in df.columns:
        axes[0, 0].hist(df['Age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Frequency')
    
    # Fare distribution
    fare_col = 'Fare_log' if 'Fare_log' in df.columns else 'Fare'
    if fare_col in df.columns:
        axes[0, 1].hist(df[fare_col], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        title = 'Fare Distribution (Log-transformed)' if fare_col == 'Fare_log' else 'Fare Distribution'
        axes[0, 1].set_title(title)
        axes[0, 1].set_xlabel(fare_col)
        axes[0, 1].set_ylabel('Frequency')
    
    # Sex distribution
    if 'Sex' in df.columns:
        sex_counts = df['Sex'].value_counts()
        sex_labels = ['Female' if x == 0 else 'Male' for x in sex_counts.index]
        axes[1, 0].bar(sex_labels, sex_counts.values, color=['pink', 'lightblue'])
        axes[1, 0].set_title('Gender Distribution')
        axes[1, 0].set_ylabel('Count')
    
    # Pclass distribution
    if 'Pclass' in df.columns:
        pclass_counts = df['Pclass'].value_counts().sort_index()
        axes[1, 1].bar(pclass_counts.index, pclass_counts.values, color='orange', alpha=0.7)
        axes[1, 1].set_title('Passenger Class Distribution')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('distribution_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Distribution plots saved as 'distribution_plots.png'")
    
    # 3. Count plots for categorical variables
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sex vs other variables (using original data for better interpretation)
    original_df = pd.read_csv('test.csv')  # Reload for better labels
    
    if 'Sex' in original_df.columns and 'Pclass' in original_df.columns:
        sns.countplot(data=original_df, x='Sex', hue='Pclass', ax=axes[0, 0])
        axes[0, 0].set_title('Gender by Passenger Class')
        axes[0, 0].legend(title='Class')
    
    if 'Sex' in original_df.columns and 'Embarked' in original_df.columns:
        sns.countplot(data=original_df, x='Sex', hue='Embarked', ax=axes[0, 1])
        axes[0, 1].set_title('Gender by Embarkation Port')
        axes[0, 1].legend(title='Embarked')
    
    if 'Pclass' in original_df.columns and 'Embarked' in original_df.columns:
        sns.countplot(data=original_df, x='Pclass', hue='Embarked', ax=axes[1, 0])
        axes[1, 0].set_title('Passenger Class by Embarkation Port')
        axes[1, 0].legend(title='Embarked')
    
    # Family Size distribution
    if 'FamilySize' in df.columns:
        family_counts = df['FamilySize'].value_counts().sort_index()
        axes[1, 1].bar(family_counts.index, family_counts.values, color='purple', alpha=0.7)
        axes[1, 1].set_title('Family Size Distribution')
        axes[1, 1].set_xlabel('Family Size')
        axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('categorical_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Categorical plots saved as 'categorical_plots.png'")
    
    print("✓ All visualizations completed successfully!")

create_visualizations(df_final)

# ============================================================================
# 12. SUMMARY STATISTICS
# ============================================================================

def generate_summary_statistics(df, original_df):
    """Generate comprehensive summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
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
        print(f"  Passengers traveling alone: {df['IsAlone'].sum()} ({(df['IsAlone'].sum()/len(df)*100):.1f}%)")
        
        family_dist = df['FamilySize'].value_counts().sort_index()
        print(f"  Family size distribution:")
        for size, count in family_dist.items():
            percentage = (count / len(df)) * 100
            print(f"    Size {size}: {count} passengers ({percentage:.1f}%)")

# Load original data for better summary statistics
original_df = pd.read_csv('test.csv')
generate_summary_statistics(df_final, original_df)

# ============================================================================
# 13. EXPORT CLEANED DATASET
# ============================================================================

def export_cleaned_dataset(df, filename='titanic_cleaned.csv'):
    """Export the cleaned and processed dataset."""
    print("\n" + "=" * 60)
    print("EXPORTING CLEANED DATASET")
    print("=" * 60)
    
    try:
        df.to_csv(filename, index=False)
        print(f"✓ Cleaned dataset exported successfully!")
        print(f"  Filename: {filename}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Display first few rows of the cleaned dataset
        print(f"\n📊 CLEANED DATASET PREVIEW:")
        print(df.head())
        
        # Show data types
        print(f"\n📋 FINAL DATA TYPES:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
            
    except Exception as e:
        print(f"✗ Error exporting dataset: {e}")

export_cleaned_dataset(df_final)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY! 🎉")
print("=" * 60)

print("\n📋 PROCESSING SUMMARY:")
print("✓ Dataset loaded and inspected")
print("✓ Missing values handled")
print("✓ Irrelevant columns removed")
print("✓ New features engineered")
print("✓ Categorical variables encoded")
print("✓ Duplicates checked and removed")
print("✓ Outliers identified and handled")
print("✓ Numerical features normalized")
print("✓ Comprehensive visualizations created")
print("✓ Summary statistics generated")
print("✓ Cleaned dataset exported")

print("\n📁 FILES CREATED:")
print("• titanic_cleaned.csv - Cleaned and processed dataset")
print("• correlation_heatmap.png - Feature correlation visualization")
print("• distribution_plots.png - Feature distribution plots")
print("• categorical_plots.png - Categorical variable visualizations")
print("• outlier_boxplots.png - Outlier detection plots")

print(f"\n🎯 FINAL DATASET SHAPE: {df_final.shape}")
print(f"📊 FEATURES READY FOR ANALYSIS: {len(df_final.columns)}")

print("\n" + "=" * 60)
print("The dataset is now ready for machine learning modeling!")
print("=" * 60)