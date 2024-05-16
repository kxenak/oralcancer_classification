import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np

def create_binary_labels(df, class1, class2):
    """
    Creates binary labels for a given class pair.

    Parameters:
    - df: DataFrame containing the data.
    - class1: The first class in the binary classification.
    - class2: The second class in the binary classification.

    Returns:
    - A DataFrame with binary labels for the specified class pair.
    """
    df_binary = df[df["Label"].isin([class1, class2])]
    df_binary["Label"] = df_binary["Label"].map({class1: 0, class2: 1})
    return df_binary

def split_data(df):
    """
    Splits the data into training and testing sets.

    Parameters:
    - df: DataFrame containing the data.

    Returns:
    - X_train: Features for the training set.
    - X_test: Features for the testing set.
    - y_train: Labels for the training set.
    - y_test: Labels for the testing set.
    """
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def plot_auc_combined(y_test, auc_data, pair_name, plots_dir):
    """
    Plots the ROC curves for all models and saves the plot to a file.

    Parameters:
    - y_test: The ground truth labels for the testing set.
    - auc_data: A dictionary containing the model names and their corresponding predicted probabilities.
    - pair_name: The name of the class pair being evaluated.
    - plots_dir: The directory where the plot will be saved.
    """
    best_model = max(auc_data, key=lambda model: roc_auc_score(y_test, auc_data[model]))
    for model_name, y_pred_proba in auc_data.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        if model_name == best_model:
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})", linewidth=2.5, linestyle="-", color="grey")
        else:
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})", linestyle="--")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    pair = pair_name.split("_")
    plt.title(f"{pair[0]} vs {pair[1]}")
    plt.legend(loc="lower right")
    plt.savefig(f"{plots_dir}/{pair_name}_roc_curve.png")
    plt.clf()

def train_and_evaluate_all(X_train, y_train, X_test, y_test, pair_name, results_df, models, models_dir, plots_dir):
    """
    Trains and evaluates all models for a given class pair.

    Parameters:
    - X_train: Features for the training set.
    - y_train: Labels for the training set.
    - X_test: Features for the testing set.
    - y_test: Labels for the testing set.
    - pair_name: The name of the class pair being evaluated.
    - results_df: DataFrame to store the model performance results.
    - models: Dictionary of models to be trained.
    - models_dir: Directory where trained models will be saved.
    - plots_dir: Directory where plots will be saved.

    Returns:
    - Updated results_df with the performance metrics of the trained models.
    """
    auc_data = {}
    for model_name, model in models.items():
        print(f"Training {model_name} for {pair_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)
        print(f"Accuracy for {model_name} on {pair_name}: {accuracy}\n")
        print(f"Classification Report for {model_name} on {pair_name}:\n{report}\n")
        temp_df = pd.DataFrame({"Class Pair": [pair_name], "Model": [model_name], 
                                "Accuracy": [accuracy], "AUC Score": [auc_score]})
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
        joblib.dump(model, f"{models_dir}/{model_name}_{pair_name}.joblib")
        auc_data[model_name] = y_pred_proba
    plot_auc_combined(y_test, auc_data, pair_name, plots_dir)
    return results_df

def get_feature_importance(model):
    """
    Retrieves the feature importance from an XGBoost model.

    Parameters:
    - model: The XGBoost model.

    Returns:
    - A Series containing the feature importance.
    """
    importance = model.get_booster().get_score(importance_type="weight")
    return pd.Series(importance)

def plot_shap_values(model, X_train, class_pair, plots_dir):
    """
    Plots the SHAP values for an XGBoost model.

    Parameters:
    - model: The XGBoost model.
    - X_train: Features for the training set.
    - class_pair: The name of the class pair being evaluated.
    - plots_dir: The directory where the plot will be saved.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_train)

    print(f"Mean SHAP value for {class_pair}: {np.mean(np.abs(shap_values))}")

    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f"SHAP Summary Plot for {class_pair}")
    plt.savefig(f"{plots_dir}/{class_pair}_shap_summary.png")
    plt.clf()

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate machine learning models for binary classification.")
    parser.add_argument("data_file", help="Path to the CSV data file")
    parser.add_argument("feature_columns", help="Feature column range (e.g., 1, 3, 6 or 1-5 or 3)")
    args = parser.parse_args()

    # Create directories for saving models and plots
    models_dir = "models"
    plots_dir = "plots"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Load the CSV data file into a pandas DataFrame
    df = pd.read_csv(args.data_file)

    # Parse the feature column range
    if "," in args.feature_columns:
        feature_columns = [int(col) - 1 for col in args.feature_columns.split(",")]
    elif "-" in args.feature_columns:
        start, end = args.feature_columns.split("-")
        feature_columns = list(range(int(start) - 1, int(end)))
    else:
        feature_columns = [int(args.feature_columns) - 1]

    # Select the specified columns as features
    feature_columns = [df.columns[i] for i in feature_columns]
    df = df[["Label"] + feature_columns]

    # Create a DataFrame to store the model performance results
    results_df = pd.DataFrame(columns=["Class Pair", "Model", "Accuracy", "AUC Score"])

    # Create binary labels for different class pairs
    df_normal_oscc = create_binary_labels(df, "NORMAL", "OSCC")
    df_oscc_osmf = create_binary_labels(df, "OSCC", "OSMF")
    df_normal_osmf = create_binary_labels(df, "NORMAL", "OSMF")

    # Split the data for each class pair
    X_train_no, X_test_no, y_train_no, y_test_no = split_data(df_normal_oscc)
    X_train_oo, X_test_oo, y_train_oo, y_test_oo = split_data(df_oscc_osmf)
    X_train_nm, X_test_nm, y_train_nm, y_test_nm = split_data(df_normal_osmf)

    # Define the models to be trained
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    # Train and evaluate models for each class pair
    print("Training for NORMAL vs OSCC")
    results_df = train_and_evaluate_all(X_train_no, y_train_no, X_test_no, y_test_no, "NORMAL_OSCC", results_df, models, models_dir, plots_dir)

    print("Training for OSCC vs OSMF")
    results_df = train_and_evaluate_all(X_train_oo, y_train_oo, X_test_oo, y_test_oo, "OSCC_OSMF", results_df, models, models_dir, plots_dir)

    print("Training for NORMAL vs OSMF")
    results_df = train_and_evaluate_all(X_train_nm, y_train_nm, X_test_nm, y_test_nm, "NORMAL_OSMF", results_df, models, models_dir, plots_dir)

    # Save the model performance results to a CSV file
    results_df.to_csv("model_performance_results.csv", index=False)

    # Load the trained XGBoost models
    xgb_normal_oscc = joblib.load(f"{models_dir}/XGBoost_NORMAL_OSCC.joblib")
    xgb_oscc_osmf = joblib.load(f"{models_dir}/XGBoost_OSCC_OSMF.joblib")
    xgb_normal_osmf = joblib.load(f"{models_dir}/XGBoost_NORMAL_OSMF.joblib")

    # Get feature importance for each class pair
    df_normal_oscc = get_feature_importance(xgb_normal_oscc)
    df_oscc_osmf = get_feature_importance(xgb_oscc_osmf)
    df_normal_osmf = get_feature_importance(xgb_normal_osmf)

    # Combine feature importance into a single DataFrame
    combined_df = pd.DataFrame({"NORMAL_OSCC": df_normal_oscc, 
                                "OSCC_OSMF": df_oscc_osmf, 
                                "NORMAL_OSMF": df_normal_osmf})

    # Fill missing values with 0 and calculate total importance
    combined_df_filled = combined_df.fillna(0)
    combined_df_filled["Total Importance"] = combined_df_filled.sum(axis=1)
    combined_df_sorted = combined_df_filled.sort_values(by="Total Importance", ascending=False)
    combined_df_sorted = combined_df_sorted.drop(columns=["Total Importance"])

    # Plot the feature importance
    ax = combined_df_sorted.plot(kind="bar", figsize=(8, 4), width=0.8)
    plt.title("XGBoost Feature Importance Across Class Pairs")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=0)
    plt.legend(title="Class Pair")
    plt.savefig(f"{plots_dir}/feature_importance.png")
    plt.clf()

    # Plot SHAP values for each class pair
    plot_shap_values(xgb_normal_oscc, X_train_no, "NORMAL vs OSCC", plots_dir)
    plot_shap_values(xgb_oscc_osmf, X_train_oo, "OSCC vs OSMF", plots_dir)
    plot_shap_values(xgb_normal_osmf, X_train_nm, "NORMAL vs OSMF", plots_dir)

if __name__ == "__main__":
    main()