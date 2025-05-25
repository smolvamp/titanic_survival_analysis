"""
Main orchestration script for Titanic dataset preprocessing.

This script coordinates all preprocessing steps in the correct order.
"""

from src.load_data import load_titanic_data
from src.inspect import inspect_dataset
from src.clean import clean_missing_values, drop_irrelevant_columns, check_duplicates
from src.feature_engineering import create_new_features
from src.encode import encode_categorical_variables
from src.outliers import handle_outliers
from src.normalize import normalize_features
from src.visualize import create_visualizations
from src.summary import generate_summary_statistics
from src.export import export_cleaned_dataset


def main():
    """Execute the complete Titanic dataset preprocessing pipeline."""
    print("=" * 60)
    print("TITANIC DATASET PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load the dataset
    df = load_titanic_data('data/train.csv')
    if df is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    # Step 2: Initial inspection
    inspect_dataset(df)
    
    # Step 3: Clean missing values
    df = clean_missing_values(df)
    
    # Step 4: Drop irrelevant columns
    df = drop_irrelevant_columns(df)
    
    # Step 5: Check for duplicates
    df = check_duplicates(df)
    
    # Step 6: Feature engineering
    df = create_new_features(df)
    
    # Step 7: Encode categorical variables
    df, encoders = encode_categorical_variables(df)
    
    # Step 8: Handle outliers
    df = handle_outliers(df)
    
    # Step 9: Normalize features
    df, scaler = normalize_features(df)
    
    # Step 10: Create visualizations
    create_visualizations(df, 'data/train.csv')
    
    # Step 11: Generate summary statistics
    generate_summary_statistics(df, 'data/train.csv')
    
    # Step 12: Export cleaned dataset
    export_cleaned_dataset(df, 'titanic_cleaned.csv')
    
    # Final summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)
    
    print("\n📋 PROCESSING SUMMARY:")
    print("✓ Dataset loaded and inspected")
    print("✓ Missing values handled")
    print("✓ Irrelevant columns removed")
    print("✓ Duplicates checked")
    print("✓ New features engineered")
    print("✓ Categorical variables encoded")
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
    
    print(f"\n🎯 FINAL DATASET SHAPE: {df.shape}")
    print(f"📊 FEATURES READY FOR ANALYSIS: {len(df.columns)}")
    
    print("\n" + "=" * 60)
    print("The dataset is now ready for machine learning modeling!")
    print("=" * 60)


if __name__ == "__main__":
    main()
