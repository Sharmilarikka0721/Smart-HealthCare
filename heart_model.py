import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# Load dataset
df = pd.read_csv("cardio_train.csv", sep=";")

print("Original shape:", df.shape)


# ---------------------------------------
# REMOVE ID
# ---------------------------------------

df = df.drop("id", axis=1)


# ---------------------------------------
# AGE CONVERSION
# ---------------------------------------

df["age"] = df["age"] / 365.25


# ---------------------------------------
# REMOVE CLEARLY INVALID VALUES
# ---------------------------------------

df = df[
    (df["height"] >= 100) &
    (df["height"] <= 220) &
    (df["weight"] >= 30) &
    (df["weight"] <= 200) &
    (df["ap_hi"] >= 70) &
    (df["ap_hi"] <= 250) &
    (df["ap_lo"] >= 40) &
    (df["ap_lo"] <= 160)
]


# ---------------------------------------
# FEATURES AND TARGET
# ---------------------------------------

X = df.drop("cardio", axis=1)

y = df["cardio"]


# ---------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000
    ))
])


logistic_model.fit(
    X_train,
    y_train
)


# ---------------------------------------
# RANDOM FOREST
# ---------------------------------------

random_forest_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])


random_forest_model.fit(
    X_train,
    y_train
)


# ---------------------------------------
# PREDICTIONS
# ---------------------------------------

logistic_pred = logistic_model.predict(X_test)

logistic_prob = logistic_model.predict_proba(
    X_test
)[:, 1]


rf_pred = random_forest_model.predict(X_test)

rf_prob = random_forest_model.predict_proba(
    X_test
)[:, 1]


# ---------------------------------------
# EVALUATION
# ---------------------------------------

def evaluate(name, y_test, prediction, probability):

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction
    )

    recall = recall_score(
        y_test,
        prediction
    )

    f1 = f1_score(
        y_test,
        prediction
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    print("\n-------------------------")
    print(name)
    print("-------------------------")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))


evaluate(
    "Logistic Regression",
    y_test,
    logistic_pred,
    logistic_prob
)


evaluate(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)


# ---------------------------------------
# SAVE MODELS
# ---------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    logistic_model,
    "models/heart_logistic_model.pkl"
)


joblib.dump(
    random_forest_model,
    "models/heart_random_forest_model.pkl"
)


print("\nHeart models saved successfully!")