"""
Visualization module for Titanic dataset preprocessing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_visualizations(df, original_df_path='data/train.csv'):
    """
    Create comprehensive visualizations of the dataset.
    
    Args:
        df (pd.DataFrame): Processed dataset
        original_df_path (str): Path to original dataset for better labels
    """
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, square=True, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Correlation heatmap saved")
    
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
    plt.close()
    print("✓ Distribution plots saved")
    
    # 3. Categorical plots using original data for better interpretation
    try:
        original_df = pd.read_csv(original_df_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gender by class
        if 'Sex' in original_df.columns and 'Pclass' in original_df.columns:
            sns.countplot(data=original_df, x='Sex', hue='Pclass', ax=axes[0, 0])
            axes[0, 0].set_title('Gender by Passenger Class')
            axes[0, 0].legend(title='Class')
        
        # Gender by embarked
        if 'Sex' in original_df.columns and 'Embarked' in original_df.columns:
            sns.countplot(data=original_df, x='Sex', hue='Embarked', ax=axes[0, 1])
            axes[0, 1].set_title('Gender by Embarkation Port')
            axes[0, 1].legend(title='Embarked')
        
        # Class by embarked
        if 'Pclass' in original_df.columns and 'Embarked' in original_df.columns:
            sns.countplot(data=original_df, x='Pclass', hue='Embarked', ax=axes[1, 0])
            axes[1, 0].set_title('Passenger Class by Embarkation Port')
            axes[1, 0].legend(title='Embarked')
        
        # Family size distribution
        if 'FamilySize' in df.columns:
            family_counts = df['FamilySize'].value_counts().sort_index()
            axes[1, 1].bar(family_counts.index, family_counts.values, color='purple', alpha=0.7)
            axes[1, 1].set_title('Family Size Distribution')
            axes[1, 1].set_xlabel('Family Size')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('categorical_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Categorical plots saved")
        
    except Exception as e:
        print(f"⚠️ Could not create categorical plots: {e}")
    
    print("✓ All visualizations completed")
