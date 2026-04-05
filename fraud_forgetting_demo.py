"""
fraud_forgetting_demo.py
========================
End-to-end demo of ModelForgettingTracker on the Kaggle Fraud Detection dataset:
    https://www.kaggle.com/datasets/kartik2112/fraud-detection

Dataset files expected in the same directory:
    fraudTrain.csv   fraudTest.csv

Pipeline
--------
  1. Feature engineering on real timestamps + demographics + merchant info
  2. Train LightGBM (falls back to LogisticRegression if not installed)
  3. Simulate production: evaluate on weekly rolling windows (fraudTest.csv)
  4. Feed F1-Score per window into ModelForgettingTracker
     - Windows with fewer than MIN_FRAUD_IN_WIN fraud cases are skipped
     - Windows with anomalously low fraud RATE are also skipped
       (catches holiday / data-quality outliers that skew the curve)
  5. Generate 4 separate publication-ready charts + print full report

Key fixes vs v1
---------------
  * MIN_FRAUD_IN_WIN raised to 30  → skips statistically noisy windows
  * MIN_FRAUD_RATE  guard added    → skips holiday/data-quality outliers
  * baseline_window raised to 6   → more stable baseline estimate
  * ForgettingReport now carries forgetting_regime ("smooth" vs "episodic")
    and worst_drop_week for richer chart annotations
  * Report printer updated to show regime + worst-drop info

Key fixes vs v2
---------------
  * Primary metric switched to Recall  → more appropriate for fraud
    (missing a real fraud costs far more than a false alarm)
  * baseline_method="top3"  → baseline = mean of top-3 early weeks,
    ignoring warm-up noise while tracking near-peak performance
  * Retrain logic in report() is now regime-aware (see model_forgetting_curve.py)

Install
-------
    pip install pandas numpy scikit-learn matplotlib scipy lightgbm

Run
---
    python fraud_forgetting_demo.py
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    _LGB = True
except ImportError:
    _LGB = False
    print("[INFO] lightgbm not installed - using LogisticRegression fallback.")
    print("       pip install lightgbm  for better results.\n")

import matplotlib.pyplot as plt
from model_forgetting_curve import ModelForgettingTracker

# =============================================================================
# CONFIG  - tweak here
# =============================================================================
TRAIN_PATH        = Path("fraudTrain.csv")
TEST_PATH         = Path("fraudTest.csv")

WINDOW_DAYS       = 7       # one production batch = 7 days of transactions

# --- Quality guards on each weekly window ------------------------------------
# Skip windows with fewer than this many confirmed fraud labels.
# 30 is the minimum for a statistically reliable F1 estimate.
MIN_FRAUD_IN_WIN  = 30

# Skip windows where fraud rate drops below this fraction of total transactions.
# Catches holiday / data-collection outliers (e.g. Dec-20 week in this dataset)
# without hard-coding dates.
MIN_FRAUD_RATE    = 0.0003   # 0.03 % of transactions must be fraud

RETRAIN_THRESHOLD = 0.07     # alert when F1 drops >= 7 % from baseline
METRIC            = "Recall"        # Recall preferred for fraud: cost of miss >> cost of FP
BASELINE_METHOD   = "top3"         # "mean" | "max" | "top3"
SAVE_DIR          = "./fraud_charts"
DARK_MODE         = True


# =============================================================================
# Step 1 - Load & validate
# =============================================================================

def load_data():
    for p in [TRAIN_PATH, TEST_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"\n[ERROR] '{p}' not found.\n"
                "Download from: https://www.kaggle.com/datasets/kartik2112/fraud-detection\n"
                "Place fraudTrain.csv and fraudTest.csv in the same folder as this script."
            )
    print("[1/5] Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    for df in [train, test]:
        df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)
    print(f"      Train: {len(train):,} rows  |  fraud={train['is_fraud'].sum():,} "
          f"({train['is_fraud'].mean()*100:.2f}%)")
    print(f"      Test : {len(test):,} rows  |  fraud={test['is_fraud'].sum():,} "
          f"({test['is_fraud'].mean()*100:.2f}%)")
    return train, test


# =============================================================================
# Step 2 - Feature engineering
# =============================================================================

CATEGORICAL_COLS = ["merchant", "category", "gender", "state", "job"]
DROP_COLS        = ["trans_num", "first", "last", "street", "city",
                    "dob", "unix_time", "is_fraud",
                    "trans_date_trans_time", "trans_dt", "dob_dt"]

_label_encoders: dict = {}


def _haversine(lat1, lon1, lat2, lon2):
    R  = 6371.0
    d1 = np.radians(lat2 - lat1)
    d2 = np.radians(lon2 - lon1)
    a  = (np.sin(d1/2)**2
          + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
          * np.sin(d2/2)**2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def engineer_features(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["trans_dt"]  = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]      = df["trans_dt"].dt.hour
    df["dayofweek"] = df["trans_dt"].dt.dayofweek
    df["month"]     = df["trans_dt"].dt.month
    df["is_night"]  = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_weekend"]= df["dayofweek"].isin([5, 6]).astype(int)
    df["distance_km"] = _haversine(
        df["lat"].values, df["long"].values,
        df["merch_lat"].values, df["merch_long"].values)
    df["dob_dt"] = pd.to_datetime(df["dob"])
    df["age"]    = (df["trans_dt"] - df["dob_dt"]).dt.days / 365.25

    global _label_encoders
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            _label_encoders[col] = le
        else:
            le = _label_encoders.get(col)
            if le:
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

    drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=drop, inplace=True)
    df.fillna(0, inplace=True)
    return df


def get_feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in ("is_fraud", "trans_date_trans_time")]


# =============================================================================
# Step 3 - Train
# =============================================================================

def train_model(X_train, y_train):
    if _LGB:
        print("[3/5] Training LightGBM...")
        scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = lgb.LGBMClassifier(
            n_estimators     = 400,
            learning_rate    = 0.05,
            num_leaves       = 63,
            scale_pos_weight = scale,
            random_state     = 42,
            n_jobs           = -1,
            verbose          = -1,
        )
        model.fit(X_train, y_train)
    else:
        print("[3/5] Training LogisticRegression (fallback)...")
        # Sub-sample to keep training time reasonable
        if len(X_train) > 200_000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), 200_000, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            print("      [sub-sampled to 200 k rows for speed]")
        model = LogisticRegression(
            max_iter     = 1000,
            class_weight = "balanced",
            random_state = 42,
            n_jobs       = -1,
            solver       = "lbfgs",   # much faster than saga for this size
        )
        model.fit(X_train, y_train)

    train_f1 = f1_score(y_train, model.predict(X_train), zero_division=0)
    print(f"      Train F1 = {train_f1:.4f}")
    return model


# =============================================================================
# Step 4 - Rolling production simulation
# =============================================================================

def simulate_production(model, test_raw, feature_cols, tracker):
    print("[4/5] Simulating production evaluation (weekly windows)...")
    test_raw = test_raw.copy()
    test_raw["trans_dt"] = pd.to_datetime(test_raw["trans_date_trans_time"])
    test_raw = test_raw.sort_values("trans_dt").reset_index(drop=True)

    start_dt   = test_raw["trans_dt"].min()
    end_dt     = test_raw["trans_dt"].max()
    total_days = (end_dt - start_dt).days
    print(f"      Test period: {start_dt.date()} -> {end_dt.date()} "
          f"({total_days} days)")
    print(f"      Quality filters: MIN_FRAUD_IN_WIN={MIN_FRAUD_IN_WIN}  "
          f"MIN_FRAUD_RATE={MIN_FRAUD_RATE:.4f}")

    window_start = start_dt
    win_idx  = 0
    skipped  = 0
    skip_log = []

    while window_start < end_dt:
        window_end = window_start + pd.Timedelta(days=WINDOW_DAYS)
        mask  = (test_raw["trans_dt"] >= window_start) & \
                (test_raw["trans_dt"] <  window_end)
        chunk = test_raw[mask].copy()

        if len(chunk) < 10:
            window_start = window_end
            continue

        y_true      = chunk["is_fraud"].values
        fraud_count = int(y_true.sum())
        fraud_rate  = fraud_count / len(chunk)

        # Quality gate 1: too few fraud labels → F1 is unreliable
        if fraud_count < MIN_FRAUD_IN_WIN:
            skip_log.append(
                f"      [SKIP] {window_start.date()}  fraud={fraud_count}  "
                f"< MIN_FRAUD_IN_WIN={MIN_FRAUD_IN_WIN}"
            )
            skipped += 1
            window_start = window_end
            continue

        # Quality gate 2: fraud rate collapsed → likely data / holiday artefact
        if fraud_rate < MIN_FRAUD_RATE:
            skip_log.append(
                f"      [SKIP] {window_start.date()}  fraud_rate={fraud_rate:.5f}  "
                f"< MIN_FRAUD_RATE={MIN_FRAUD_RATE:.4f}  "
                f"(possible data artefact / holiday effect)"
            )
            skipped += 1
            window_start = window_end
            continue

        chunk_feat = engineer_features(chunk, fit=False)
        for col in feature_cols:
            if col not in chunk_feat.columns:
                chunk_feat[col] = 0
        chunk_feat = chunk_feat[feature_cols]

        y_pred = model.predict(chunk_feat.values)

        f1  = f1_score(y_true, y_pred, zero_division=0)
        pr  = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        obs_ts = window_start.to_pydatetime()
        if obs_ts.tzinfo is None:
            obs_ts = obs_ts.replace(tzinfo=timezone.utc)

        # Primary metric is Recall: a missed fraud costs >> a false alarm.
        # F1 and precision are still logged as metadata for reference.
        tracker.log(
            metric_value = rec,
            timestamp    = obs_ts,
            n_samples    = len(chunk),
            fraud_cases  = fraud_count,
            fraud_rate   = round(fraud_rate, 6),
            precision    = round(pr, 4),
            recall       = round(rec, 4),
            window_index = win_idx,
            window_start = str(window_start.date()),
        )

        print(f"      Week {win_idx+1:>3}  [{window_start.date()}]  "
              f"n={len(chunk):>6,}  fraud={fraud_count:>4}  "
              f"rate={fraud_rate:.4f}  "
              f"R={rec:.4f}  F1={f1:.4f}  P={pr:.3f}")

        win_idx      += 1
        window_start  = window_end

    print(f"\n      Logged: {win_idx} windows  |  Skipped: {skipped}")
    if skip_log:
        print("      Skipped window details:")
        for msg in skip_log:
            print(msg)


# =============================================================================
# Step 5 - Report & visualise
# =============================================================================

def print_report(report) -> None:
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  FORGETTING CURVE REPORT")
    print(SEP)
    rows = [
        ("Baseline Recall",          f"{report.baseline_metric:.4f}  [{BASELINE_METHOD}]"),
        ("Current  Recall",          f"{report.current_metric:.4f}"),
        ("Retention ratio",          f"{report.retention_ratio:.1%}"),
        ("Decay rate  lambda",       f"{report.decay_rate:.6f}"),
        ("Half-life",                f"{report.half_life_days:.1f} days"
                                     if report.half_life_days else "--"),
        ("Forgetting speed",         report.forgetting_speed.upper()),
        ("Forgetting regime",        report.forgetting_regime.upper()),
        ("Curve fit R-squared",      f"{report.fit_r_squared:.4f}"
                                     if report.fit_r_squared is not None else "--"),
        ("Snapshots logged",         str(report.snapshots_used)),
        ("Retrain recommended NOW",  str(report.retrain_recommended)),
        ("Days until retrain alert", f"{report.predicted_days_to_threshold:.1f}"
                                     if report.predicted_days_to_threshold is not None
                                     else "--"),
        ("Recommended retrain date", report.recommended_retrain_date.strftime("%Y-%m-%d")
                                     if report.recommended_retrain_date else "--"),
        ("Worst single-week drop",   f"Week {report.worst_drop_week}  "
                                     f"(−{report.worst_drop_magnitude:.4f})"
                                     if report.worst_drop_week else "--"),
    ]
    for lbl, val in rows:
        print(f"  {lbl:<30}: {val}")
    print(SEP)

    # Human-readable interpretation
    print("\n  INTERPRETATION")
    print(f"  {'─'*56}")
    if report.forgetting_regime == "smooth":
        print("  Regime  : SMOOTH — performance decays gradually.")
        print("            The exponential model fits well (high R²).")
        print("            Classic concept drift: retrain on a schedule.")
    elif report.forgetting_regime == "episodic":
        print("  Regime  : EPISODIC — performance drops in sudden shocks.")
        print("            The exponential model fits poorly (low R²).")
        print("            Likely cause: holiday effects, fraud-pattern")
        print("            regime changes, or new merchant categories.")
        print("            Action: use anomaly detection + event-triggered")
        print("            retraining rather than fixed schedules.")
    else:
        print("  Regime  : UNKNOWN — not enough data to classify.")

    if report.retrain_recommended:
        print(f"\n  ⚠  RETRAIN NOW — retention has fallen to "
              f"{report.retention_ratio:.1%}")
    elif report.predicted_days_to_threshold is not None:
        print(f"\n  ✓  Healthy — estimated {report.predicted_days_to_threshold:.0f} days"
              f" until threshold breach.")
    print(SEP + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    # 1. Load
    train_raw, test_raw = load_data()

    # 2. Features
    print("[2/5] Engineering features...")
    train_feat   = engineer_features(train_raw, fit=True)
    feature_cols = get_feature_cols(train_feat)
    X_train      = train_feat[feature_cols].values
    y_train      = train_raw["is_fraud"].values
    print(f"      Features ({len(feature_cols)}): "
          f"{feature_cols[:6]}{'...' if len(feature_cols)>6 else ''}")

    # 3. Train
    model = train_model(X_train, y_train)

    # 4. Simulate + track
    tracker = ModelForgettingTracker(
        metric_name       = METRIC,
        higher_is_better  = True,
        retrain_threshold = RETRAIN_THRESHOLD,
        baseline_window   = 6,              # first 6 clean weeks establish baseline
        baseline_method   = BASELINE_METHOD,# "top3" = mean of top-3 early weeks
    )
    simulate_production(model, test_raw, feature_cols, tracker)

    if len(tracker._snapshots) < tracker.baseline_window:
        print(f"\n[WARN] Only {len(tracker._snapshots)} windows logged after "
              f"quality filtering. Need >= {tracker.baseline_window}.\n"
              "Try reducing MIN_FRAUD_IN_WIN or MIN_FRAUD_RATE.")
        return

    # 5. Report
    report = tracker.report()
    print_report(report)

    # 6. Visualise — 4 separate figures
    print(f"[5/5] Generating charts -> {SAVE_DIR}/")
    figs = tracker.plot(dark_mode=DARK_MODE, save_dir=SAVE_DIR)
    print(f"      {len(figs)} figures saved.")

    # -------------------------------------------------------------------------
    # LIVE TRACKING DEMO (uncomment to run against a real model endpoint)
    # -------------------------------------------------------------------------
    # from sklearn.metrics import f1_score as _f1
    # import itertools
    # _test_iter = itertools.cycle(range(len(test_raw)))
    # def live_metric_fn():
    #     idx   = next(_test_iter)
    #     chunk = test_raw.iloc[max(0,idx-500):idx+500]
    #     cf    = engineer_features(chunk, fit=False)
    #     for col in feature_cols:
    #         if col not in cf.columns: cf[col] = 0
    #     cf = cf[feature_cols]
    #     yt = chunk["is_fraud"].values
    #     yp = model.predict(cf.values)
    #     return _f1(yt, yp, zero_division=0)
    #
    # live_tracker = ModelForgettingTracker(
    #     metric_name="F1-Score", retrain_threshold=0.07, baseline_window=6)
    # live_tracker.live_track(metric_fn=live_metric_fn, interval_seconds=3)

    plt.show()


if __name__ == "__main__":
    main()
