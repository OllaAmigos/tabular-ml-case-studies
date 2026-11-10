# Wine quality classification

## Question

Can measurements of wine acidity, sugar, sulfates and alcohol identify samples with an expert quality score **≥ 7**?

## Data

- `data/train.csv` — 1,725 labeled samples (includes `quality`)
- `data/test.csv` — 576 unlabeled samples (not used for the reported evaluation)
- 11 continuous predictors

## Baseline

`evaluate.py` builds a pipeline of `StandardScaler` and `LogisticRegression`, then evaluates on a fixed stratified 20% holdout split.

```bash
python wine-quality-classification/evaluate.py
```

The project includes original, exploratory model-search notebooks under `notebooks/`, covering K-nearest neighbors and logistic regression. The packaged baseline is **independent of the original experimental notebook results**.

## Validation results

**Fixed 80/20 stratified holdout** (`random_state=42`, 345 validation rows):

| Metric | Score |
| --- | ---: |
| Precision | **0.679** |
| Recall | **0.597** |
| F1 | **0.635** |
| ROC AUC (using probabilities) | **0.817** |

These are metrics from the newly packaged logistic-regression baseline, not from the exploratory model-search notebooks.

## Caveats

A high-precision model may miss many genuinely high-quality wines. Precision, recall, and operational selection volume should therefore be considered jointly, not optimized by tuning a threshold on the final holdout.
