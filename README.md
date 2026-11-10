# Tabular ML — Two Reproducible Case Studies

**Classification** · **Regression** · Data preprocessing · Holdout validation

This repository organizes two earlier ML assignments as case studies, separates exploratory notebooks from executable baseline scripts, and documents the distinction between a **labeled train split** and an **unlabeled test file**.

| Case study | Question | Target | Evaluation on labeled holdout |
| --- | --- | --- | --- |
| [Wine quality classification](wine-quality-classification/) | Can we identify wines with a quality score of **7 or above**? | Binary good-quality flag | Precision, recall, F1, ROC AUC |
| [Diamond price regression](diamond-price-regression/) | Can structured features predict a diamond's selling price? | `total_sales_price` | MAE, MAPE, R² |

## Run the reproducible baselines

```bash
python -m pip install -r requirements.txt
python wine-quality-classification/evaluate.py
python diamond-price-regression/evaluate.py
```

Each script creates an explicit random 80/20 holdout split **from its labeled training CSV**, with a fixed seed (`42`). The separately provided `test.csv` files contain no labels and cannot be used for validation.

Original exploratory notebooks are preserved in `notebooks/`, with historical execution outputs cleared and paths standardized. They are for reading and experimentation, not for claiming a production workflow.

## Baseline results (single holdout split)

| Case | Metric | Result |
| --- | --- | ---: |
| Wine quality classification | ROC AUC | **0.817** |
| Wine quality classification | Precision / Recall | 0.679 / 0.597 |
| Diamond price regression | MAPE | **10.51%** |
| Diamond price regression | R² | **0.9628** |

All numbers were computed by the included `evaluate.py` scripts on 20% held-out labeled records (`random_state=42`) using a new reproducible baseline. They are not historical notebook experiment scores.

## Notes on limitations

- Results are benchmark scores on a single holdout split, **not** external validation or production results.
- The educational data provenance, original competition constraints and potential distribution shift should be evaluated before real-world use.
- Feature engineering, error analysis, repeated cross-validation and probability calibration would be natural next steps.
