import os
import zipfile
import glob
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

print("=" * 60)
print("PREDICTIVE MAINTENANCE PROJECT")
print("=" * 60)

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

zip_files = glob.glob("*.zip")

if zip_files:
    zip_file = zip_files[0]

    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall("data")
        print("\nZIP file extracted successfully!")
    except Exception as e:
        print("\nZIP extraction error:", e)

csv_files = glob.glob("data/**/*.csv", recursive=True)

if not csv_files:
    print("\nCSV file not found.")
    exit()

csv_file = csv_files[0]

print("CSV file found:", csv_file)

df = pd.read_csv(csv_file)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nOriginal Column Names:")
print(df.columns.tolist())

rename_columns = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf"
}

df.rename(columns=rename_columns, inplace=True)

print("\nStandardized Column Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nMachine Failure Values:")
print(df["machine_failure"].value_counts())

print("\nMachine Type Distribution:")
print(df["type"].value_counts())

print("\nFailure Type Distribution:")

failure_columns = ["twf", "hdf", "pwf", "osf", "rnf"]

for col in failure_columns:
    print("\n" + col.upper())
    print(df[col].value_counts())

failure_sum = df[failure_columns].sum(axis=1)

print("\nMachine failure = 0 but failure type exists:",
      ((df["machine_failure"] == 0) & (failure_sum > 0)).sum())

print("Machine failure = 1 but no failure type exists:",
      ((df["machine_failure"] == 1) & (failure_sum == 0)).sum())

print("\nFailure Flag Combinations:")
print(df[failure_columns].value_counts())

failure_percentage = df["machine_failure"].value_counts(normalize=True) * 100

print("\nMachine Failure Percentage:")
print(failure_percentage)

df["failure_flag_sum"] = failure_sum

df["temp_diff"] = (
    df["process_temperature"] -
    df["air_temperature"]
)

df["power_proxy"] = (
    df["torque"] *
    df["rotational_speed"]
)

print("\nFeature Engineering Completed!")

print("\nNew Features:")
print(df[["temp_diff", "power_proxy"]].head())

print("\nNew Feature Statistics:")
print(df[["temp_diff", "power_proxy"]].describe())

print("\nCreating EDA visualizations...")

plt.figure(figsize=(8, 5))

sns.barplot(
    x=df["machine_failure"].value_counts().index,
    y=df["machine_failure"].value_counts().values
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Number of Machines")
plt.tight_layout()
plt.savefig("plots/01_machine_failure_distribution.png")
plt.close()

print("Graph 1 saved.")

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="machine_failure",
    y="torque",
    data=df
)

plt.title("Torque vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Torque [Nm]")
plt.tight_layout()
plt.savefig("plots/02_torque_vs_failure.png")
plt.close()

print("Graph 2 saved.")

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="machine_failure",
    y="tool_wear",
    data=df
)

plt.title("Tool Wear vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Tool Wear [min]")
plt.tight_layout()
plt.savefig("plots/03_tool_wear_vs_failure.png")
plt.close()

print("Graph 3 saved.")

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="machine_failure",
    y="temp_diff",
    data=df
)

plt.title("Temperature Difference vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Temperature Difference [K]")
plt.tight_layout()
plt.savefig("plots/04_temperature_difference.png")
plt.close()

print("Graph 4 saved.")

failure_rates = []

for col in failure_columns:
    rate = df.groupby(col)["machine_failure"].mean().get(1, 0)
    failure_rates.append(rate)

plt.figure(figsize=(8, 5))

plt.bar(
    [x.upper() for x in failure_columns],
    failure_rates
)

plt.title("Failure Rate by Failure Type")
plt.xlabel("Failure Type")
plt.ylabel("Machine Failure Rate")
plt.tight_layout()
plt.savefig("plots/05_failure_type_rates.png")
plt.close()

print("Graph 5 saved.")

plt.figure(figsize=(10, 7))

correlation_columns = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "temp_diff",
    "power_proxy",
    "machine_failure"
]

correlation_matrix = df[correlation_columns].corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Sensor Features Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/06_correlation_heatmap.png")
plt.close()

print("Graph 6 saved.")

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df.sample(min(3000, len(df)), random_state=42),
    x="rotational_speed",
    y="torque",
    hue="machine_failure"
)

plt.title("Rotational Speed vs Torque")
plt.xlabel("Rotational Speed [rpm]")
plt.ylabel("Torque [Nm]")
plt.tight_layout()
plt.savefig("plots/07_speed_torque.png")
plt.close()

print("Graph 7 saved.")

print("\nEDA completed.")

print("\nRemoving identifier columns...")

model_df = df.drop(
    columns=["udi", "product_id"]
)

model_df = pd.get_dummies(
    model_df,
    columns=["type"],
    drop_first=True,
    dtype=int
)

print("\nColumns after encoding:")
print(model_df.columns.tolist())

feature_columns = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "temp_diff",
    "power_proxy",
    "type_L",
    "type_M"
]

X = model_df[feature_columns]

y = model_df["machine_failure"]

print("\nBinary Classification Features:")
print(feature_columns)

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nTarget Distribution:")
print(y.value_counts())

print("\nTarget Percentage:")
print(y.value_counts(normalize=True) * 100)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=feature_columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=feature_columns,
    index=X_test.index
)

print("\nFeature Scaling Completed.")

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

print("Logistic Regression trained successfully.")

print("\nTraining Random Forest...")

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train,
    y_train
)

print("Random Forest trained successfully.")

print("\nTraining XGBoost...")

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("XGBoost Scale Pos Weight:", scale_pos_weight)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train,
    y_train
)

print("XGBoost trained successfully.")

def evaluate_binary_model(
    name,
    model,
    X_data,
    y_data,
    scaled=False
):

    if scaled:
        predictions = model.predict(X_data)
        probabilities = model.predict_proba(X_data)[:, 1]
    else:
        predictions = model.predict(X_data)
        probabilities = model.predict_proba(X_data)[:, 1]

    accuracy = accuracy_score(
        y_data,
        predictions
    )

    precision = precision_score(
        y_data,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_data,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_data,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_data,
        probabilities
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc
    }

binary_results = []

binary_results.append(
    evaluate_binary_model(
        "Logistic Regression",
        logistic_model,
        X_test_scaled,
        y_test,
        scaled=True
    )
)

binary_results.append(
    evaluate_binary_model(
        "Random Forest",
        random_forest_model,
        X_test,
        y_test
    )
)

binary_results.append(
    evaluate_binary_model(
        "XGBoost",
        xgb_model,
        X_test,
        y_test
    )
)

binary_results_df = pd.DataFrame(binary_results)

print("\nBinary Model Comparison:")
print(binary_results_df)

binary_results_df.to_csv(
    "results/binary_model_comparison.csv",
    index=False
)

best_binary_name = binary_results_df.sort_values(
    "F1",
    ascending=False
).iloc[0]["Model"]

print("\nBest Binary Model based on F1:", best_binary_name)

print("\nTuning XGBoost using GridSearchCV...")

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8, 1.0]
}

grid_xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=grid_xgb,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(
    X_train,
    y_train
)

tuned_xgb = grid_search.best_estimator_

print("\nBest XGBoost Parameters:")
print(grid_search.best_params_)

tuned_predictions = tuned_xgb.predict(X_test)

tuned_probabilities = tuned_xgb.predict_proba(
    X_test
)[:, 1]

tuned_accuracy = accuracy_score(
    y_test,
    tuned_predictions
)

tuned_precision = precision_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_recall = recall_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_f1 = f1_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_auc = roc_auc_score(
    y_test,
    tuned_probabilities
)

print("\nTuned XGBoost Results:")
print("Accuracy :", round(tuned_accuracy, 4))
print("Precision:", round(tuned_precision, 4))
print("Recall   :", round(tuned_recall, 4))
print("F1 Score :", round(tuned_f1, 4))
print("ROC-AUC  :", round(tuned_auc, 4))

binary_results_df = pd.concat(
    [
        binary_results_df,
        pd.DataFrame(
            [{
                "Model": "Tuned XGBoost",
                "Accuracy": tuned_accuracy,
                "Precision": tuned_precision,
                "Recall": tuned_recall,
                "F1": tuned_f1,
                "ROC-AUC": tuned_auc
            }]
        )
    ],
    ignore_index=True
)

binary_results_df.to_csv(
    "results/binary_model_comparison_with_tuning.csv",
    index=False
)

print("\nFinal Binary Model Comparison:")
print(binary_results_df)

print("\nBinary Classification Report:")
print(
    classification_report(
        y_test,
        tuned_predictions,
        target_names=["No Failure", "Failure"],
        zero_division=0
    )
)

binary_cm = confusion_matrix(
    y_test,
    tuned_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    binary_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Failure", "Failure"],
    yticklabels=["No Failure", "Failure"]
)

plt.title("Binary Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/08_binary_confusion_matrix.png")
plt.close()

print("Binary confusion matrix saved.")

print("\nCreating Multi-Class Failure Target...")

def get_failure_type(row):

    if row["machine_failure"] == 0:
        return "No Failure"

    if row["twf"] == 1:
        return "TWF"

    if row["hdf"] == 1:
        return "HDF"

    if row["pwf"] == 1:
        return "PWF"

    if row["osf"] == 1:
        return "OSF"

    if row["rnf"] == 1:
        return "RNF"

    return "Unknown Failure"

df["failure_type"] = df.apply(
    get_failure_type,
    axis=1
)

print("\nMulti-Class Target Distribution:")
print(df["failure_type"].value_counts())

valid_classes = [
    "No Failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Unknown Failure"
]

class_mapping = {
    "No Failure": 0,
    "TWF": 1,
    "HDF": 2,
    "PWF": 3,
    "OSF": 4,
    "RNF": 5,
    "Unknown Failure": 6
}

df["failure_type_encoded"] = df["failure_type"].map(
    class_mapping
)

multi_X = model_df[feature_columns]

multi_y = df["failure_type_encoded"]

multi_X_train, multi_X_test, multi_y_train, multi_y_test = train_test_split(
    multi_X,
    multi_y,
    test_size=0.20,
    random_state=42,
    stratify=multi_y
)

print("\nMulti-Class Training Samples:", len(multi_X_train))
print("Multi-Class Testing Samples:", len(multi_X_test))

multi_class_model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Multi-Class Random Forest...")

multi_class_model.fit(
    multi_X_train,
    multi_y_train
)

print("Multi-Class Random Forest trained successfully.")

multi_predictions = multi_class_model.predict(
    multi_X_test
)

multi_accuracy = accuracy_score(
    multi_y_test,
    multi_predictions
)

multi_precision = precision_score(
    multi_y_test,
    multi_predictions,
    average="macro",
    zero_division=0
)

multi_recall = recall_score(
    multi_y_test,
    multi_predictions,
    average="macro",
    zero_division=0
)

multi_f1 = f1_score(
    multi_y_test,
    multi_predictions,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 60)
print("MULTI-CLASS MODEL RESULTS")
print("=" * 60)

print("Accuracy :", round(multi_accuracy, 4))
print("Macro Precision:", round(multi_precision, 4))
print("Macro Recall   :", round(multi_recall, 4))
print("Macro F1       :", round(multi_f1, 4))

print("\nMulti-Class Classification Report:")

present_labels = sorted(
    np.unique(
        np.concatenate(
            [multi_y_test, multi_predictions]
        )
    )
)

present_names = [
    list(class_mapping.keys())[list(class_mapping.values()).index(label)]
    for label in present_labels
]

print(
    classification_report(
        multi_y_test,
        multi_predictions,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    )
)

multi_results_df = pd.DataFrame(
    [{
        "Model": "Random Forest Multi-Class",
        "Accuracy": multi_accuracy,
        "Macro Precision": multi_precision,
        "Macro Recall": multi_recall,
        "Macro F1": multi_f1
    }]
)

print("\nMulti-Class Model Comparison:")
print(multi_results_df)

multi_results_df.to_csv(
    "results/multiclass_model_comparison.csv",
    index=False
)

multi_cm = confusion_matrix(
    multi_y_test,
    multi_predictions,
    labels=present_labels
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    multi_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=present_names,
    yticklabels=present_names
)

plt.title("Multi-Class Failure Type Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/09_multiclass_confusion_matrix.png")
plt.close()

print("Multi-class confusion matrix saved.")

print("\nFeature Importance - Binary XGBoost")

binary_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": tuned_xgb.feature_importances_
})

binary_importance = binary_importance.sort_values(
    "Importance",
    ascending=False
)

print(binary_importance)

binary_importance.to_csv(
    "results/binary_feature_importance.csv",
    index=False
)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=binary_importance,
    x="Importance",
    y="Feature"
)

plt.title("Binary Model Feature Importance")
plt.tight_layout()
plt.savefig("plots/10_binary_feature_importance.png")
plt.close()

print("\nFeature Importance - Multi-Class Random Forest")

multi_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": multi_class_model.feature_importances_
})

multi_importance = multi_importance.sort_values(
    "Importance",
    ascending=False
)

print(multi_importance)

multi_importance.to_csv(
    "results/multiclass_feature_importance.csv",
    index=False
)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=multi_importance,
    x="Importance",
    y="Feature"
)

plt.title("Multi-Class Model Feature Importance")
plt.tight_layout()
plt.savefig("plots/11_multiclass_feature_importance.png")
plt.close()

print("\nSaving cleaned dataset...")

cleaned_columns = [
    "type",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "machine_failure",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
    "temp_diff",
    "power_proxy",
    "failure_type"
]

df[cleaned_columns].to_csv(
    "results/cleaned_predictive_maintenance_dataset.csv",
    index=False
)

print("Cleaned dataset saved.")

print("\nSaving trained models...")

import joblib

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    tuned_xgb,
    "models/binary_xgboost_model.pkl"
)

joblib.dump(
    multi_class_model,
    "models/multiclass_random_forest_model.pkl"
)

joblib.dump(
    feature_columns,
    "models/feature_columns.pkl"
)

joblib.dump(
    class_mapping,
    "models/class_mapping.pkl"
)

print("Models saved successfully.")

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print("\nDataset:")
print("10,000 machine records")

print("\nMachine Failure:")
print("Normal:", (df["machine_failure"] == 0).sum())
print("Failure:", (df["machine_failure"] == 1).sum())

print("\nBest Binary Model:")
print("Tuned XGBoost")

print("Binary F1 Score:", round(tuned_f1, 4))
print("Binary ROC-AUC:", round(tuned_auc, 4))

print("\nMulti-Class Model:")
print("Random Forest")

print("Multi-Class Macro F1:", round(multi_f1, 4))

print("\nTop Binary Features:")

for _, row in binary_importance.head(5).iterrows():
    print(
        row["Feature"],
        "->",
        round(row["Importance"], 4)
    )

print("\nTop Multi-Class Features:")

for _, row in multi_importance.head(5).iterrows():
    print(
        row["Feature"],
        "->",
        round(row["Importance"], 4)
    )

print("\nOutput folders:")
print("plots/")
print("models/")
print("results/")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print()
print("Binary Model Evaluation")

binary_predictions =xgb_model.predict(X_test_scaled)

binary_probabilities = xgb_model.predict_proba(
    X_test_scaled
)[:, 1]

accuracy = accuracy_score(
    y_test,
    binary_predictions
)

precision = precision_score(
    y_test,
    binary_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    binary_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    binary_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    binary_probabilities
)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

print()
print("Binary Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        binary_predictions
    )
)

print()
print("Binary Classification Report:")

print(
    classification_report(
        y_test,
        binary_predictions,
        zero_division=0
    )
)

print()
print("Evaluation completed successfully.")