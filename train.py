import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocess import preprocess


def train_model(data_path):
    df = preprocess(data_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)

    # Save model
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # ✅ Save feature columns (VERY IMPORTANT)
    with open("model/columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    print("Model and columns saved!")


if __name__ == "__main__":
    train_model("data/churn_data.csv")