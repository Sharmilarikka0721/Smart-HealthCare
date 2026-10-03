import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Remove ID
df = df.drop("id", axis=1)

# Convert BMI
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

X = df.drop("stroke", axis=1)
y = df["stroke"]

# -----------------------------
# Identify columns
# -----------------------------

numeric_features = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi"
]

categorical_features = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

# -----------------------------
# Preprocessing
# -----------------------------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# -----------------------------
# Train/test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Model 1 - Logistic Regression
# -----------------------------

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

# -----------------------------
# Model 2 - Random Forest
# -----------------------------

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

random_forest_model.fit(X_train, y_train)

rf_pred = random_forest_model.predict(X_test)
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]

# -----------------------------
# Evaluation function
# -----------------------------

def evaluate_model(name, y_test, prediction, probability):

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(
        y_test, prediction, zero_division=0
    )
    recall = recall_score(
        y_test, prediction, zero_division=0
    )
    f1 = f1_score(
        y_test, prediction, zero_division=0
    )
    auc = roc_auc_score(
        y_test, probability
    )

    print("\n----------------------------")
    print(name)
    print("----------------------------")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        prediction,
        zero_division=0
    ))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    }


# -----------------------------
# Compare models
# -----------------------------

result1 = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred,
    logistic_prob
)

result2 = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)

results = pd.DataFrame([
    result1,
    result2
])

print("\nMODEL COMPARISON")
print(results)

# -----------------------------
# Save models
# -----------------------------

import os

os.makedirs("models", exist_ok=True)

joblib.dump(
    logistic_model,
    "models/logistic_model.pkl"
)

joblib.dump(
    random_forest_model,
    "models/random_forest_model.pkl"
)

print("\nModels saved successfully.")