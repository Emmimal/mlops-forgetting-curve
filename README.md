# mlops-forgetting-curve
Diagnose whether your ML model drifts smoothly or collapses in shocks — and pick the right retraining strategy. R² = −0.31 on 555k fraud transactions.

# mlops-forgetting-curve

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-Kaggle%20Fraud%20Detection-20BEFF.svg)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
[![TDS Article](https://img.shields.io/badge/article-Towards%20Data%20Science-black.svg)](https://towardsdatascience.com)

Diagnose whether your production ML model decays gradually or collapses in sudden shocks — and choose the right retraining strategy accordingly.

---

## The Problem

Every major MLOps platform assumes model performance follows a smooth exponential decay. Fitting that curve to 555,000 real fraud transactions returned **R² = −0.31** — worse than predicting the mean. The model wasn't decaying. It was being shocked.

This repository contains the diagnostic tool, the fraud simulation, and all four publication charts from the TDS article.

---

## Key Concepts

| Regime | R² | Pattern | Right response |
|---|---|---|---|
| **Smooth** | ≥ 0.4 | Gradual, predictable drift | Calendar-based retraining |
| **Episodic** | < 0.4 | Sudden shocks and recoveries | Event-driven shock detection |

The `ModelForgettingTracker` fits an exponential curve to your weekly metrics, computes R², and tells you which regime you're in before you commit to an operations strategy.

---

## Quickstart

```bash
git clone https://github.com/Emmimal/mlops-forgetting-curve
cd mlops-forgetting-curve
pip install -r requirements.txt
python fraud_forgetting_demo.py
```

**Against the Kaggle fraud dataset:**

```bash
# Download fraudTrain.csv + fraudTest.csv from Kaggle and place them here
python fraud_forgetting_demo.py
```

**Against your own performance log:**

```python
from model_forgetting_curve import load_from_dataframe

tracker = load_from_dataframe(
    df,
    metric_col="weekly_recall",
    metric_name="Recall",
    baseline_method="top3",   # "mean" | "max" | "top3"
    retrain_threshold=0.07,
)

report = tracker.report()
print(f"Regime : {report.forgetting_regime}")   # "smooth" | "episodic"
print(f"R²     : {report.fit_r_squared:.3f}")

figs = tracker.plot(save_dir="./charts", dark_mode=True)
```

---

## Output

Running `fraud_forgetting_demo.py` produces:

```
FORGETTING CURVE REPORT
============================================================
Baseline Recall               : 0.8807  [top3]
Current  Recall               : 0.8507
Retention ratio               : 96.6%
Forgetting regime             : EPISODIC
Curve fit R-squared           : -0.3091
Worst single-week drop        : Week 7  (−0.1875)
============================================================
```

And four charts saved to `./fraud_charts/`:

| Figure | What it shows |
|---|---|
| `fig1_forgetting_curve.png` | Raw recall vs fitted exponential + R² box |
| `fig2_retention_ratio.png` | Weekly retention against threshold, breach markers |
| `fig3_rolling_decay.png` | Rolling λ bar chart — early warning radar |
| `fig4_retrain_countdown.png` | Semi-circular gauge + key stats block |

---

## Shock Detection (Episodic Regime)

When R² < 0.4, replace your calendar schedule with these three mechanisms:

```python
import pandas as pd
import numpy as np

recall_series = pd.Series(weekly_recalls)
fraud_counts  = pd.Series(weekly_fraud_counts)

# 1 — single-week shock detector
rolling_mean = recall_series.rolling(window=4).mean()
shock_flags  = recall_series < (rolling_mean * 0.92)

# 2 — volume-weighted recall (more stable than raw recall)
weighted_recall = np.average(recall_series, weights=fraud_counts)

# 3 — two-consecutive-week trigger (reduces false retrain alerts)
breach          = recall_series < (recall_series.mean() * 0.93)
retrain_trigger = breach & breach.shift(1).fillna(False)
```

Calibrate the `0.92` shock threshold and `0.93` retrain threshold against your domain's cost asymmetry — the ratio of missed-fraud cost to false-alarm cost.

---

## Repository Structure

```
mlops-forgetting-curve/
├── model_forgetting_curve.py   # Core tracker + ForgettingReport + plots
├── fraud_forgetting_demo.py    # End-to-end demo on Kaggle fraud dataset
├── requirements.txt
├── examples/
│   └── load_from_dataframe.py  # Minimal example for existing perf logs
└── assets/
    └── fig1_forgetting_curve.png
    └── fig2_retention_ratio.png
    └── fig3_rolling_decay.png
    └── fig4_retrain_countdown.png
```

---

## Tracker API

### `ModelForgettingTracker`

```python
tracker = ModelForgettingTracker(
    metric_name       = "Recall",
    higher_is_better  = True,
    retrain_threshold = 0.07,      # alert at 7% drop from baseline
    baseline_window   = 6,         # weeks used to establish baseline
    baseline_method   = "top3",    # mean | max | top3
)
tracker.log(0.9375)                # call once per evaluation window
report = tracker.report()          # ForgettingReport dataclass
figs   = tracker.plot()            # list of 4 matplotlib figures
```

### `ForgettingReport` fields

| Field | Type | Description |
|---|---|---|
| `forgetting_regime` | `str` | `"smooth"` or `"episodic"` |
| `fit_r_squared` | `float` | R² of the exponential fit |
| `baseline_metric` | `float` | Computed baseline recall |
| `retention_ratio` | `float` | Current / baseline |
| `decay_rate` | `float` | λ in R(t) = R₀·exp(−λt) |
| `half_life_days` | `float\|None` | ln(2)/λ — meaningful only in smooth regime |
| `retrain_recommended` | `bool` | True if retention ≤ 1 − threshold |
| `predicted_days_to_threshold` | `float\|None` | Days until retrain alert |
| `worst_drop_week` | `int\|None` | 1-based week of worst single-week drop |
| `worst_drop_magnitude` | `float\|None` | Absolute recall drop into that week |

---

## Dataset

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — Kartik Shenoy, Kaggle, 2020.
Synthetic data generated with [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation). Distributed under the Database Contents Licence (DbCL) v1.0.

> Dataset files are not included in this repository. Download `fraudTrain.csv` and `fraudTest.csv` from Kaggle and place them in the project root before running the demo.

---

## Requirements

```
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
scipy>=1.7
matplotlib>=3.4
lightgbm>=3.3
```

```bash
pip install -r requirements.txt
```

LightGBM is optional. The demo falls back to `LogisticRegression` if it is not installed, with a warning.

---

## Citation

If you use this diagnostic approach or the `ModelForgettingTracker` in your work, please cite the accompanying article:

```bibtex
@article{alexander2026forgetting,
  title   = {Why MLOps Retraining Schedules Fail — Models Don't Forget, They Get Shocked},
  author  = {Alexander, Emmimal P},
  journal = {Towards Data Science},
  year    = {2026},
  url     = {https://towardsdatascience.com}
}
```

---

## License

MIT © Emmimal P Alexander
