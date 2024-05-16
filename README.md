# Binary Classification Model Training and Evaluation

This script trains and evaluates machine learning models for binary classification tasks. It supports multiple classification models, including Random Forest, KNN, SVM, Logistic Regression, and XGBoost.

## Features

- Train and evaluate multiple models for binary classification.
- Plot ROC curves and calculate AUC scores for model comparison.
- Calculate and plot feature importance using XGBoost models.
- Generate SHAP summary plots for model interpretability.

## Requirements

To run this script, you will need the following packages:

- pandas
- scikit-learn
- xgboost
- matplotlib
- joblib
- shap
- numpy

You can install these packages using the following command:

```bash
pip install pandas scikit-learn xgboost matplotlib joblib shap numpy
```

## Usage

To use this script, you need to provide a CSV data file and specify the feature columns. The data file should contain a column named "Label" for the target labels and other columns for the features.

Example command:

```bash
python binary_classification.py data.csv 1-5
```

In this example, `data.csv` is the path to the CSV data file, and `1-5` specifies that columns 1 to 5 (inclusive) are the feature columns.

## Input

- `data_file`: Path to the CSV data file.
- `feature_columns`: Feature column range (e.g., `1,3,6` or `1-5` or `3`).

## Output

- Models trained for each class pair are saved in the `models` directory.
- ROC curves are saved in the `plots` directory.
- Feature importance plots are saved in the `plots` directory.
- SHAP summary plots are saved in the `plots` directory.
- A CSV file named `model_performance_results.csv` containing the performance metrics of the trained models.

## Structure

The main components of the script are:

- `create_binary_labels()`: Creates binary labels for a given class pair.
- `split_data()`: Splits the data into training and testing sets.
- `plot_auc_combined()`: Plots the ROC curves for all models.
- `train_and_evaluate_all()`: Trains and evaluates all models for a given class pair.
- `get_feature_importance()`: Retrieves the feature importance from an XGBoost model.
- `plot_shap_values()`: Plots the SHAP values for an XGBoost model.
- `main()`: The main function that orchestrates the training and evaluation process.

## Note

- The script assumes that the target column in the data file is named "Label".