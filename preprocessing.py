import pandas as pd

def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")

    # Remove unnecessary ID
    df = df.drop("id", axis=1)

    # Convert BMI to numeric
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # Fill missing BMI with median
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    return df


if __name__ == "__main__":
    df = load_data()

    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nShape:")
    print(df.shape)