"""
Crop Recommendation System - Model Training
============================================
Trains a Random Forest classifier on the Crop Recommendation dataset.
Saves the trained model and label encoder for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import pickle
import os
import warnings
warnings.filterwarnings("ignore")


# ── 1. Load Dataset ──────────────────────────────────────────────────────────
def load_data(path="data/Crop_recommendation.csv"):
    """Load the crop recommendation dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset\n"
            "Place the CSV file inside the 'data/' folder."
        )
    df = pd.read_csv(path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── 2. Exploratory Data Analysis ─────────────────────────────────────────────
def run_eda(df):
    """Print basic EDA and save plots."""
    print("\n── Dataset Info ──")
    print(df.info())
    print("\n── First 5 Rows ──")
    print(df.head())
    print("\n── Summary Statistics ──")
    print(df.describe())
    print("\n── Crop Distribution ──")
    print(df["label"].value_counts())
    print(f"\n── Missing Values ──\n{df.isnull().sum()}")

    os.makedirs("plots", exist_ok=True)

    # Correlation heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(df.drop("label", axis=1).corr(), annot=True, fmt=".2f", cmap="YlGn")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png", dpi=150)
    plt.close()

    # Crop distribution bar chart
    plt.figure(figsize=(14, 5))
    df["label"].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Number of Samples per Crop")
    plt.xlabel("Crop")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/crop_distribution.png", dpi=150)
    plt.close()

    print("\n✅ EDA plots saved in 'plots/' directory.")


# ── 3. Preprocessing ──────────────────────────────────────────────────────────
def preprocess(df):
    """Encode labels and split into train/test sets."""
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])

    FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[FEATURES]
    y = df["label_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✅ Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, le


# ── 4. Train Model ────────────────────────────────────────────────────────────
def train(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("\n✅ Model training complete.")
    return model


# ── 5. Evaluate Model ─────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, le):
    """Print metrics and save confusion matrix."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n── Model Accuracy: {acc * 100:.2f}% ──")
    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(16, 13))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=150)
    plt.close()
    print("✅ Confusion matrix saved.")

    # Feature importance
    FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=FEATURES, palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150)
    plt.close()
    print("✅ Feature importance plot saved.")

    return acc


# ── 6. Save Model ─────────────────────────────────────────────────────────────
def save_artifacts(model, le):
    """Pickle the model and label encoder."""
    os.makedirs("models", exist_ok=True)
    with open("models/crop_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("\n✅ Model and encoder saved in 'models/' directory.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    run_eda(df)
    X_train, X_test, y_train, y_test, le = preprocess(df)
    model = train(X_train, y_train)
    evaluate(model, X_test, y_test, le)
    save_artifacts(model, le)
    print("\n🎉 Pipeline complete! Run 'streamlit run app/app.py' to launch the app.")
