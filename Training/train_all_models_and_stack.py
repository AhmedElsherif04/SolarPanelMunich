"""
Train, Compare, Stack & Save — All Models
==========================================
This script:
  1. Trains 5 base models (RandomForest, XGBoost, LightGBM, HistGBR, CatBoost)
  2. Saves each base model individually
  3. Builds a stacking ensemble (meta-learner on top of base predictions)
  4. Saves the stacking meta-learner + metadata
  5. Prints a full comparison table

All models are trained on the 2-stage pipeline:
  Stage 1: LGBMClassifier (has_solar? yes/no)
  Stage 2: Regression models (predict panel_area_log for solar tiles)
"""

import pandas as pd
import numpy as np
import json
import time
import joblib
import warnings
from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict, KFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "CleanupDataSet" / "final_model_ev_updated.csv"
SAVE_DIR = Path(__file__).resolve().parent / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate(y_true_log, y_pred_log, label=""):
    """Return a dict of log-scale and real-scale metrics."""
    y_true_real = np.expm1(y_true_log)
    y_pred_real = np.clip(np.expm1(y_pred_log), 0, None)
    return {
        "label": label,
        "r2_log": r2_score(y_true_log, y_pred_log),
        "rmse_log": rmse(y_true_log, y_pred_log),
        "r2_real": r2_score(y_true_real, y_pred_real),
        "mae_real": mean_absolute_error(y_true_real, y_pred_real),
        "rmse_real": rmse(y_true_real, y_pred_real),
    }


# ─────────────────────────────────────────────────────────────
# 1. Load & Prepare Data
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df["panel_area_log"] = np.log1p(df["panel_area_m2"])
df["has_solar"] = (df["panel_area_m2"] > 0).astype(int)

FEATURE_COLS = [
    "year", "total_rooftops", "Unemployment_Rate", "Average_Age",
    "Elderly_Population", "Young_Population", "Total_Population",
    "employed", "pv_price", "panel_area_lag1", "ev_points_164m",
    "tile_encoded", "tile_centroid_lat", "tile_centroid_lon",
]

# Clean up
df_model = df.dropna(subset=FEATURE_COLS + ["panel_area_log"]).copy()

# Remove extreme outliers (top 1% of positive)
upper_cutoff = df_model[df_model["panel_area_log"] > 0]["panel_area_log"].quantile(0.99)
df_model = df_model[df_model["panel_area_log"] <= upper_cutoff]

# Time-based split
df_model = df_model.sort_values("year")
test_year = df_model["year"].max()
val_year = df_model[df_model["year"] < test_year]["year"].max()

train_df = df_model[df_model["year"] < val_year]
val_df = df_model[df_model["year"] == val_year]
test_df = df_model[df_model["year"] == test_year]

# Stage 2 trains on positive-only tiles
train_pos = train_df[train_df["has_solar"] == 1]
val_pos = val_df[val_df["has_solar"] == 1]

X_train = train_pos[FEATURE_COLS]
y_train = train_pos["panel_area_log"]
X_val = val_pos[FEATURE_COLS]
y_val = val_pos["panel_area_log"]
X_test = test_df[FEATURE_COLS]
y_test = test_df["panel_area_log"]

print(f"Dataset: {len(df_model)} rows  |  Train: {len(train_pos)} (solar>0)")
print(f"Val: {len(val_pos)} (solar>0)  |  Test: {len(test_df)} (all)")
print(f"Feature columns: {len(FEATURE_COLS)}")
print(f"Years → Train: <{val_year}, Val: {val_year}, Test: {test_year}")

# ─────────────────────────────────────────────────────────────
# 2. Stage 1 — LGBMClassifier (same as before)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 1: LGBMClassifier (adoption probability)")
print("=" * 70)

clf = lgb.LGBMClassifier(
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=32,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)
clf.fit(
    train_df[FEATURE_COLS], train_df["has_solar"],
    eval_set=[(val_df[FEATURE_COLS], val_df["has_solar"])],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=100)],
)

from sklearn.metrics import roc_auc_score
p_solar_test = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(test_df["has_solar"], p_solar_test)
print(f"  ✓ AUC on test: {auc:.4f}  |  Best iteration: {clf.best_iteration_}")

# Save Stage 1
clf_path = SAVE_DIR / "stage1_classifier.joblib"
joblib.dump(clf, clf_path)
print(f"  ✓ Saved → {clf_path}")

# ─────────────────────────────────────────────────────────────
# 3. Stage 2 — Train 5 Base Models
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 2: Training 5 Base Regression Models")
print("=" * 70)

models_config = {
    "RandomForest": RandomForestRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=-1,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
    "HistGBR": HistGradientBoostingRegressor(
        max_iter=1000,
        max_depth=None,
        max_leaf_nodes=63,
        learning_rate=0.05,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=42,
    ),
    "CatBoost": cb.CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        subsample=0.8,
        random_seed=42,
        verbose=0,
    ),
}

trained_models = {}
results = []

for name, model in models_config.items():
    print(f"\n{'─'*60}")
    print(f"Training: {name}")
    print(f"{'─'*60}")

    t0 = time.time()

    # Fit with early stopping where supported
    if name == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best_iter = getattr(model, "best_iteration", None)
    elif name == "LightGBM":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        best_iter = getattr(model, "best_iteration_", None)
    elif name == "CatBoost":
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
        best_iter = model.get_best_iteration()
    else:
        model.fit(X_train, y_train)
        best_iter = None

    elapsed = time.time() - t0

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate (log-scale metrics on all test rows)
    train_metrics = evaluate(y_train, y_pred_train, f"{name}_train")
    test_metrics = evaluate(y_test, y_pred_test, f"{name}_test")

    result = {
        "model": name,
        "train_r2_log": train_metrics["r2_log"],
        "test_r2_log": test_metrics["r2_log"],
        "test_rmse_log": test_metrics["rmse_log"],
        "test_r2_real": test_metrics["r2_real"],
        "test_mae_real": test_metrics["mae_real"],
        "test_rmse_real": test_metrics["rmse_real"],
        "time_sec": elapsed,
        "best_iter": best_iter,
    }
    results.append(result)
    trained_models[name] = model

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top3 = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{n}({v:.3f})" for n, v in top3)
    elif hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()
        top3 = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{n}({v:.1f})" for n, v in top3)
    else:
        top3_str = "N/A"

    iter_str = f"  (best_iter={best_iter})" if best_iter else ""
    print(f"  Train R²(log): {train_metrics['r2_log']:.4f}")
    print(f"  Test  R²(log): {test_metrics['r2_log']:.4f}  RMSE(log): {test_metrics['rmse_log']:.4f}")
    print(f"  Test  R²(real): {test_metrics['r2_real']:.4f}  MAE(real): {test_metrics['mae_real']:.2f} m²  RMSE(real): {test_metrics['rmse_real']:.2f} m²")
    print(f"  Top-3 features: {top3_str}")
    print(f"  Training time: {elapsed:.1f}s{iter_str}")

    # Save individual model
    model_path = SAVE_DIR / f"stage2_{name.lower()}.joblib"
    joblib.dump(model, model_path)
    print(f"  ✓ Saved → {model_path}")

# ─────────────────────────────────────────────────────────────
# 4. Stacking Ensemble
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STACKING ENSEMBLE")
print("=" * 70)

# Generate out-of-fold (OOF) predictions for the meta-learner
# This prevents the meta-learner from just memorising train predictions
print("\n  Generating out-of-fold predictions for stacking...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# We need OOF predictions on the training set for the meta-learner,
# and regular predictions on the test set.
oof_preds = np.zeros((len(X_train), len(trained_models)))
test_preds = np.zeros((len(X_test), len(trained_models)))

model_names = list(trained_models.keys())

for i, name in enumerate(model_names):
    print(f"  [{i+1}/{len(model_names)}] Generating OOF for {name}...")

    # OOF predictions via cross_val_predict
    model_cfg = models_config[name]

    # Clone and retrain per fold for honest OOF
    fold_test_preds = np.zeros((len(X_test),))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr_fold = X_train.iloc[train_idx]
        y_tr_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]

        # Clone model
        if name == "RandomForest":
            m = RandomForestRegressor(**{k: v for k, v in model_cfg.get_params().items()})
        elif name == "XGBoost":
            m = xgb.XGBRegressor(**{k: v for k, v in model_cfg.get_params().items()})
        elif name == "LightGBM":
            m = lgb.LGBMRegressor(**{k: v for k, v in model_cfg.get_params().items()})
        elif name == "HistGBR":
            m = HistGradientBoostingRegressor(**{k: v for k, v in model_cfg.get_params().items()})
        elif name == "CatBoost":
            m = cb.CatBoostRegressor(
                iterations=1000, depth=6, learning_rate=0.05,
                l2_leaf_reg=3.0, subsample=0.8, random_seed=42, verbose=0,
            )

        # Fit
        if name == "CatBoost":
            m.fit(X_tr_fold, y_tr_fold, verbose=0)
        else:
            m.fit(X_tr_fold, y_tr_fold)

        # OOF predictions for this fold
        oof_preds[val_idx, i] = m.predict(X_val_fold)

        # Test predictions (average across folds)
        fold_test_preds += m.predict(X_test) / kf.n_splits

    test_preds[:, i] = fold_test_preds

print(f"\n  OOF matrix shape: {oof_preds.shape}  |  Test matrix shape: {test_preds.shape}")

# Train meta-learner (Ridge regression — simple and effective)
print("\n  Training meta-learner (Ridge regression)...")
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(oof_preds, y_train)

# Meta-learner weights (how much it trusts each base model)
print("\n  Meta-learner weights:")
for name, coef in zip(model_names, meta_learner.coef_):
    bar = "█" * int(abs(coef) * 20)
    print(f"    {name:15s}: {coef:+.4f}  {bar}")
print(f"    {'Intercept':15s}: {meta_learner.intercept_:+.4f}")

# Stacking predictions on test
y_pred_stack = meta_learner.predict(test_preds)
stack_metrics = evaluate(y_test, y_pred_stack, "Stacking")

stack_result = {
    "model": "Stacking (5-model)",
    "train_r2_log": r2_score(y_train, meta_learner.predict(oof_preds)),
    "test_r2_log": stack_metrics["r2_log"],
    "test_rmse_log": stack_metrics["rmse_log"],
    "test_r2_real": stack_metrics["r2_real"],
    "test_mae_real": stack_metrics["mae_real"],
    "test_rmse_real": stack_metrics["rmse_real"],
    "time_sec": 0,
    "best_iter": None,
}
results.append(stack_result)

print(f"\n  Stacking Test R²(log):  {stack_metrics['r2_log']:.4f}")
print(f"  Stacking Test RMSE(log): {stack_metrics['rmse_log']:.4f}")
print(f"  Stacking Test R²(real):  {stack_metrics['r2_real']:.4f}")
print(f"  Stacking Test MAE(real): {stack_metrics['mae_real']:.2f} m²")
print(f"  Stacking Test RMSE(real): {stack_metrics['rmse_real']:.2f} m²")

# Save stacking ensemble
stack_path = SAVE_DIR / "stage2_stacking.joblib"
joblib.dump({
    "meta_learner": meta_learner,
    "base_model_names": model_names,
    "feature_cols": FEATURE_COLS,
}, stack_path)
print(f"\n  ✓ Stacking meta-learner saved → {stack_path}")

# ─────────────────────────────────────────────────────────────
# 5. Final Comparison Table
# ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 100)
print("FINAL COMPARISON — ALL MODELS")
print("=" * 100)
print(f"{'Model':<20} {'Train R²(log)':>13} {'Test R²(log)':>13} {'Test RMSE(log)':>15} "
      f"{'Test R²(real)':>13} {'Test MAE(m²)':>13} {'Test RMSE(m²)':>14} {'Time(s)':>8}")
print("-" * 100)

for r in sorted(results, key=lambda x: x["test_mae_real"]):
    print(
        f"{r['model']:<20} {r['train_r2_log']:>13.4f} {r['test_r2_log']:>13.4f} "
        f"{r['test_rmse_log']:>15.4f} {r['test_r2_real']:>13.4f} "
        f"{r['test_mae_real']:>13.2f} {r['test_rmse_real']:>14.2f} {r['time_sec']:>8.1f}"
    )

# Winner
best = min(results, key=lambda x: x["test_mae_real"])
print(f"\n🏆 BEST MODEL (lowest MAE): {best['model']}  →  MAE = {best['test_mae_real']:.2f} m²")

# ─────────────────────────────────────────────────────────────
# 6. Save metadata
# ─────────────────────────────────────────────────────────────
metadata = {
    "feature_cols": FEATURE_COLS,
    "stage1": {
        "model": "LGBMClassifier",
        "path": str(clf_path),
        "auc": float(auc),
    },
    "stage2_models": {},
    "stacking": {
        "path": str(stack_path),
        "base_models": model_names,
        "meta_learner": "Ridge(alpha=1.0)",
        "weights": {n: float(c) for n, c in zip(model_names, meta_learner.coef_)},
        "intercept": float(meta_learner.intercept_),
    },
    "results": [],
}

for r in results:
    entry = {k: v for k, v in r.items() if k != "best_iter"}
    entry["best_iter"] = int(r["best_iter"]) if r["best_iter"] is not None else None
    metadata["results"].append(entry)

for name in model_names:
    metadata["stage2_models"][name] = str(SAVE_DIR / f"stage2_{name.lower()}.joblib")

meta_path = SAVE_DIR / "model_metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Metadata saved → {meta_path}")
print(f"\n✅ All models saved to: {SAVE_DIR}/")
print("   Files:")
for p in sorted(SAVE_DIR.glob("*")):
    size_mb = p.stat().st_size / 1024 / 1024
    print(f"   • {p.name:40s} ({size_mb:.1f} MB)")
