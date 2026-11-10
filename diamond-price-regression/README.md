# Diamond price regression

## Question

How accurately can a simple regression model predict `total_sales_price` from a diamond's size, grading and dimensions?

## Data

- `data/train.csv` — 67,598 labeled samples
- `data/test.csv` — 22,453 unlabeled samples (not used for the reported evaluation)
- Numeric dimensions and categorical descriptors such as color, clarity, cut, symmetry and polish

## Baseline

`evaluate.py` performs a random 80/20 split on labeled training data. Numeric features are imputed and scaled; categorical features are one-hot encoded. A ridge regression model is trained on `log1p(price)` and transformed back for metric calculation.

```bash
python diamond-price-regression/evaluate.py
```

## Validation results

**Fixed 80/20 holdout** (`random_state=42`, 13,520 validation rows):

| Metric | Score |
| --- | ---: |
| Mean Absolute Error | **306.46** (price units) |
| Mean Absolute Percentage Error | **10.51%** |
| R² | **0.9628** |

These are scores from the newly packaged ridge-regression baseline, not from the historical notebook experiments.

## Caveats

Prices can vary for unobserved market factors. MAPE can be unstable for very low target values; inspect the target distribution before deployment. Original model-search notebooks are under `notebooks/` and are not the validated baseline.
