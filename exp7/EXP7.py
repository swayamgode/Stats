# Experiment: Supervised Classification for Diabetes Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("diabetes.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# --------------------------------------------------
# 2. Separate Features and Target
# --------------------------------------------------

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)

# --------------------------------------------------
# 3. Handle Invalid Values
# --------------------------------------------------

# In the Pima dataset, zero values are invalid for
# some medical attributes.

invalid_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace zero values with NaN
for col in invalid_columns:
    if col in X.columns:
        X[col] = X[col].replace(0, np.nan)

print("\nMissing values before imputation:")
print(X.isnull().sum())

# Fill missing values with median
for col in invalid_columns:
    if col in X.columns:
        X[col] = X[col].fillna(X[col].median())

print("\nMissing values after imputation:")
print(X.isnull().sum())

# --------------------------------------------------
# 4. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 5. Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 6. Create Classification Models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    )
}

results = []

# --------------------------------------------------
# 7-10. Train, Predict and Evaluate
# --------------------------------------------------

for name, model in models.items():

    # KNN and Logistic Regression require scaling
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    TN, FP, FN, TP = cm.ravel()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        auc
    ])

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nTrue Negative (TN):", TN)
    print("False Positive (FP):", FP)
    print("False Negative (FN):", FN)
    print("True Positive (TP):", TP)

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Diabetic", "Diabetic"],
        yticklabels=["Non-Diabetic", "Diabetic"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(name + " - Confusion Matrix")
    plt.show()

# --------------------------------------------------
# 11. Compare Models
# --------------------------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]
)

print("\n\nMODEL COMPARISON")
print("=" * 70)
print(results_df.round(4))

# --------------------------------------------------
# Find Best Model
# --------------------------------------------------

best_model = results_df.loc[
    results_df["F1-Score"].idxmax()
]

print("\nBest Model based on F1-Score:")
print(best_model)