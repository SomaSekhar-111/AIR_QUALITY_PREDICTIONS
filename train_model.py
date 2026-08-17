"""
Air Quality Prediction Project
------------------------------
A simple machine-learning project that predicts AQI from
weather, pollution, activity, and event-description data.

Run:
    python train_model.py
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "air_quality_india.csv"
MODEL_FILE = PROJECT_DIR / "air_quality_model.joblib"
RESULT_FILE = PROJECT_DIR / "prediction_results.csv"


def find_column(df, possible_names):
    """Return the first matching column name from a list of choices."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def get_aqi_category(aqi):
    """Give a simple description for a predicted AQI value."""
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Severe"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    print(f"Loaded {len(df):,} rows.")
    print("Cleaning the data...")

    # Remove exact duplicate rows.
    df = df.drop_duplicates().copy()

    # Find the target column.
    target = find_column(df, ["AQI", "aqi", "AQI_Value", "AQI value"])
    if target is None:
        raise ValueError("Could not find an AQI column in the dataset.")

    # Replace missing numeric values with their median.
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Replace missing text/category values with a simple placeholder.
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].fillna("Unknown")

    # Try to find a text column containing pollution-event descriptions.
    text_column = find_column(
        df,
        [
            "Pollution_Event_Description",
            "Pollution Event Description",
            "Event_Description",
            "Description",
            "Pollution_Event",
        ],
    )

    # Keep only useful columns for the model.
    ignored = {target}
    if "Date" in df.columns:
        ignored.add("Date")

    feature_columns = [c for c in df.columns if c not in ignored]
    X = df[feature_columns]
    y = df[target]

    # Separate numeric, categorical and text features.
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    if text_column in categorical_features:
        categorical_features.remove(text_column)

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numbers",
                StandardScaler(),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        )

    if text_column:
        transformers.append(
            (
                "event_text",
                TfidfVectorizer(max_features=300, ngram_range=(1, 2)),
                text_column,
            )
        )

    if not transformers:
        raise ValueError("No usable features were found in the dataset.")

    preprocessor = ColumnTransformer(transformers=transformers)

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("features", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("Training the model...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nModel results")
    print("-" * 30)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    results = X_test.copy()
    results["Actual_AQI"] = y_test.values
    results["Predicted_AQI"] = predictions.round(2)
    results["AQI_Category"] = [
        get_aqi_category(value) for value in predictions
    ]

    results.to_csv(RESULT_FILE, index=False)
    joblib.dump(pipeline, MODEL_FILE)

    print(f"\nSaved predictions to: {RESULT_FILE.name}")
    print(f"Saved trained model to: {MODEL_FILE.name}")
    print("\nDone! The model is ready to use.")


if __name__ == "__main__":
    main()
