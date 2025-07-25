# trustpulse.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
def load_and_preprocess(df):
    """
    Takes in raw churn dataset and prepares X_train, X_test, y_train, y_test.
    """
    df = df.copy()

    # Drop irrelevant columns
    df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

    # Encode categorical features
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    df = pd.get_dummies(df, drop_first=True)

    # Define features/target
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

# -----------------------------
def train_trust_model(X_train, X_test, y_train, y_test, threshold=0.35):
    """
    Trains Random Forest and returns predictions with trust scores.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_thresh = (y_proba >= threshold).astype(int)

    # Evaluation metrics
    report = classification_report(y_test, y_pred_thresh, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred_thresh)
    conf_matrix = confusion_matrix(y_test, y_pred_thresh)

    # Generate output DataFrame
    results_df = X_test.copy()
    results_df["Churn_Prob"] = y_proba
    results_df["Trust_Score"] = pd.cut(
        y_proba,
        bins=[-0.01, 0.3, 0.6, 1.0],
        labels=["High", "Medium", "Low"]
    )

    return model, results_df, report, accuracy, conf_matrix

# -----------------------------
def summarize_results(results_df, model):
    """
    Prints summary of churn risks and plots.
    """
    print("📊 Trust Score Distribution:")
    print(results_df["Trust_Score"].value_counts())

    print("\n⚠️ Top 5 High-Risk Customers:")
    print(results_df.sort_values("Churn_Prob", ascending=False).head(5)[["Churn_Prob", "Trust_Score"]])

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(results_df["Churn_Prob"], bins=20, color="orange", edgecolor="black")
    plt.axvline(results_df["Churn_Prob"].mean(), color="red", linestyle="--", label="Mean")
    plt.title("Churn Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Feature importance
    importances = model.feature_importances_
    features = model.feature_names_in_
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(6, 5))
    plt.barh(features[sorted_idx], importances[sorted_idx])
    plt.xlabel("Importance")
    plt.title("Top Influential Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
