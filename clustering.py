import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

features = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi"
]

X = df[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

print(df[["age", "avg_glucose_level", "bmi", "cluster"]].head())

print("\nCluster counts:")
print(df["cluster"].value_counts())

# Visualization

plt.scatter(
    df["age"],
    df["avg_glucose_level"],
    c=df["cluster"]
)

plt.xlabel("Age")
plt.ylabel("Average Glucose Level")
plt.title("Patient Clusters")

plt.show()