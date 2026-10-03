import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

df = df.drop("id", axis=1)

df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())

print("\nStroke count:")
print(df["stroke"].value_counts())

# 1. Stroke distribution
sns.countplot(x="stroke", data=df)
plt.title("Stroke Distribution")
plt.xlabel("Stroke")
plt.ylabel("Number of Patients")
plt.show()

# 2. Age distribution
plt.hist(df["age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()

# 3. Glucose distribution
plt.hist(df["avg_glucose_level"], bins=20)
plt.title("Average Glucose Level")
plt.xlabel("Glucose Level")
plt.ylabel("Number of Patients")
plt.show()

# 4. BMI distribution
plt.hist(df["bmi"], bins=20)
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Number of Patients")
plt.show()

# 5. Stroke vs age
sns.boxplot(x="stroke", y="age", data=df)
plt.title("Age vs Stroke")
plt.show()

# 6. Stroke vs glucose
sns.boxplot(x="stroke", y="avg_glucose_level", data=df)
plt.title("Glucose Level vs Stroke")
plt.show()

# 7. Stroke vs hypertension
sns.countplot(x="hypertension", hue="stroke", data=df)
plt.title("Hypertension and Stroke")
plt.show()

# 8. Stroke vs heart disease
sns.countplot(x="heart_disease", hue="stroke", data=df)
plt.title("Heart Disease and Stroke")
plt.show()

# 9. Smoking status
sns.countplot(x="smoking_status", hue="stroke", data=df)
plt.title("Smoking Status and Stroke")
plt.xticks(rotation=20)
plt.show()

# 10. Correlation
numeric = df.select_dtypes(include="number")

sns.heatmap(numeric.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()