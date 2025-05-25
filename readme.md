# Titanic Dataset Preprocessing Pipeline

A comprehensive, modular preprocessing pipeline for the Titanic dataset that performs data cleaning, feature engineering, visualization, and analysis.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   - Place your Titanic dataset as `data/train.csv`
   - Ensure the `data/` directory exists

3. **Run the pipeline:**
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
titanic-project/
├── data/
│   └── train.csv                # Input Titanic dataset
├── src/
│   ├── load_data.py            # Data loading functionality
│   ├── inspect.py              # Dataset inspection and overview
│   ├── clean.py                # Missing value handling and cleaning
│   ├── feature_engineering.py  # Feature creation (Title, FamilySize, IsAlone)
│   ├── encode.py               # Categorical variable encoding
│   ├── outliers.py             # Outlier detection and handling
│   ├── normalize.py            # Feature normalization/standardization
│   ├── visualize.py            # Comprehensive visualizations
│   ├── summary.py              # Statistical summaries and insights
│   └── export.py               # Dataset export functionality
├── main.py                     # Main orchestration script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔄 Processing Pipeline

The pipeline executes the following steps in order:

1. **Data Loading** - Load the Titanic dataset from CSV
2. **Initial Inspection** - Display basic dataset information and statistics
3. **Data Cleaning** - Handle missing values, drop irrelevant columns
4. **Duplicate Check** - Identify and remove duplicate records
5. **Feature Engineering** - Create new features (Title, FamilySize, IsAlone)
6. **Encoding** - Convert categorical variables to numerical format
7. **Outlier Handling** - Detect outliers and apply transformations
8. **Normalization** - Standardize numerical features
9. **Visualization** - Generate comprehensive plots and charts
10. **Summary Statistics** - Produce detailed statistical analysis
11. **Export** - Save the cleaned dataset for further analysis

## 📊 Output Files

After running the pipeline, the following files will be created:

- `titanic_cleaned.csv` - Final preprocessed dataset
- `correlation_heatmap.png` - Feature correlation visualization
  ![correlation_heatmap](https://github.com/user-attachments/assets/c3decea2-fd60-4940-b740-0a5771c8971e)

- `distribution_plots.png` - Feature distribution analysis
  ![distribution_plots](https://github.com/user-attachments/assets/3b9d4ed0-5ac3-4860-addc-e3abd6ce81fc)

- `categorical_plots.png` - Categorical variable relationships
  ![categorical_plots](https://github.com/user-attachments/assets/24a4a091-9b8a-4ee3-870f-3e96c10120e4)

- `outlier_boxplots.png` - Outlier detection visualizations
![outlier_boxplots](https://github.com/user-attachments/assets/6a63418e-95f7-4894-a033-a35e2ba1144d)

## 🛠️ Features

### Data Cleaning
- **Missing Value Imputation**: Age (median), Embarked (mode), Fare (median)
- **Feature Creation**: Convert Cabin to binary HasCabin feature
- **Column Removal**: Drop PassengerId and Ticket columns

### Feature Engineering
- **Title Extraction**: Extract titles from passenger names
- **Family Size**: Calculate total family size (SibSp + Parch + 1)
- **Is Alone**: Binary indicator for solo travelers

### Advanced Processing
- **Categorical Encoding**: Label encoding for categorical variables
- **Outlier Handling**: Log transformation for highly skewed features
- **Normalization**: StandardScaler for numerical features
- **Duplicate Detection**: Identify and remove duplicate records

### Comprehensive Analysis
- **Statistical Summaries**: Detailed statistics for all features
- **Correlation Analysis**: Feature relationship heatmaps
- **Distribution Plots**: Visualize feature distributions
- **Categorical Analysis**: Cross-tabulation and count plots

## 🎯 Usage Examples

### Run with custom dataset path:
```python
from src.load_data import load_titanic_data

# Load from custom path
df = load_titanic_data('path/to/your/dataset.csv')
```

### Use individual modules:
```python
from src.clean import clean_missing_values
from src.feature_engineering import create_new_features

# Process step by step
df_clean = clean_missing_values(df)
df_featured = create_new_features(df_clean)
```

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements or additional features.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
