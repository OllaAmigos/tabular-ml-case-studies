"""Reproducible classification baseline on the labeled training file."""
from pathlib import Path
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    data = pd.read_csv(Path(__file__).resolve().parent / "data" / "train.csv")
    x = data.drop(columns=["quality"])
    y = (data["quality"] >= 7).astype(int)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid)
    probs = model.predict_proba(x_valid)[:, 1]
    scores = {
        "n_train": len(x_train), "n_validation": len(x_valid),
        "positive_rate_validation": round(float(y_valid.mean()), 4),
        "precision": round(float(precision_score(y_valid, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_valid, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_valid, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_valid, probs)), 4),
    }
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
