"""Reproducible regression baseline on the labeled training file."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    data = pd.read_csv(Path(__file__).resolve().parent / "data" / "train.csv")
    x = data.drop(columns=["total_sales_price"])
    y = data["total_sales_price"]
    cat = list(x.select_dtypes(include="object").columns)
    num = [col for col in x if col not in cat]
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat),
    ])
    model = TransformedTargetRegressor(
        regressor=make_pipeline(preprocessing, Ridge(alpha=10.0)),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.20, random_state=42
    )
    model.fit(x_train, y_train)
    predictions = np.maximum(0.0, model.predict(x_valid))
    scores = {
        "n_train": len(x_train), "n_validation": len(x_valid),
        "mae": round(float(mean_absolute_error(y_valid, predictions)), 2),
        "mape": round(float(mean_absolute_percentage_error(y_valid, predictions)), 4),
        "r2": round(float(r2_score(y_valid, predictions)), 4),
    }
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
