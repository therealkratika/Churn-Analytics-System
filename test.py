from preprocess import preprocess


def test_preprocessing():
    try:
        # Run preprocessing
        df = preprocess("data/churn_data.csv")

        # Basic checks
        print("Data loaded and processed successfully ✅")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())

        # Check if target exists
        if "Churn" in df.columns:
            print("Target column present ✅")
        else:
            print("Target column missing ❌")

        # Check for null values
        if df.isnull().sum().sum() == 0:
            print("No missing values ✅")
        else:
            print("Missing values still present ❌")

        print("\nSample data:")
        print(df.head())

    except Exception as e:
        print("Error during preprocessing ❌")
        print(str(e))


if __name__ == "__main__":
    test_preprocessing()