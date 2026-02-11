"""Run PCA and SHAP diagnostics for the trained models."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # ensure plots can be saved headlessly

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from analysis_utils import compute_shap, run_pca

try:
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    xgb = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "CleanupDataSet" / "final_model.csv"
TRAIN_DIR = BASE_DIR / "Training"
OUTPUT_DIR = TRAIN_DIR / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "year",
    "total_rooftops",
    "Unemployment_Rate",
    "Average_Age",
    "Elderly_Population",
    "Young_Population",
    "Total_Population",
    "tile_encoded",
    "employed",
    "pv_price",
    "panel_area_lag1",
]

MODEL_CONFIGS = [
    {
        "name": "stage1_lightgbm",
        "type": "lightgbm",
        "path": TRAIN_DIR / "lgb_model_1Stage_lag.txt",
    },
    {
        "name": "stage2_random_forest",
        "type": "random_forest",
        "path": TRAIN_DIR / "random_forest_model_lag.joblib",
    },
    {
        "name": "stage2_xgboost",
        "type": "xgboost",
        "path": TRAIN_DIR / "stage2_xgb.json",
    },
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=FEATURE_COLS)
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=FEATURE_COLS)
    return df


def load_model(config: Dict[str, Path]):
    model_type = config["type"]
    model_path = config["path"]
    if not model_path.exists():
        print(f"⚠️  Skipping {config['name']} (missing file: {model_path.name})")
        return None

    if model_type == "lightgbm":
        return lgb.Booster(model_file=str(model_path))
    if model_type == "random_forest":
        return joblib.load(model_path)
    if model_type == "xgboost":
        if xgboost_not_available():
            print("⚠️  Skipping XGBoost diagnostics (xgboost not installed)")
            return None
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        return model

    raise ValueError(f"Unknown model type: {model_type}")


def xgboost_not_available() -> bool:
    return xgb is None


def run_pca_diagnostics(df: pd.DataFrame):
    print("\n==== Running PCA diagnostics ====")
    X = df[FEATURE_COLS]
    pca_plot_path = OUTPUT_DIR / "pca_variance.png"
    run_pca(X, plot=True, show=False, save_path=str(pca_plot_path))


def run_shap_diagnostics(model_name: str, model, df: pd.DataFrame):
    print(f"\n==== SHAP for {model_name} ====")
    X = df[FEATURE_COLS]
    bar_path = OUTPUT_DIR / f"{model_name}_shap_bar.png"
    beeswarm_path = OUTPUT_DIR / f"{model_name}_shap_beeswarm.png"

    compute_shap(
        model,
        X,
        plot_type="bar",
        show=False,
        save_path=str(bar_path),
    )
    compute_shap(
        model,
        X,
        plot_type="dot",
        show=False,
        save_path=str(beeswarm_path),
    )


def main(selected_models: Optional[str] = None):
    df = load_data()
    run_pca_diagnostics(df)

    targets = selected_models.split(",") if selected_models else [cfg["name"] for cfg in MODEL_CONFIGS]

    for cfg in MODEL_CONFIGS:
        if cfg["name"] not in targets:
            continue
        model = load_model(cfg)
        if model is None:
            continue
        try:
            run_shap_diagnostics(cfg["name"], model, df)
        except Exception as exc:
            print(f"⚠️  Failed SHAP for {cfg['name']}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCA and SHAP for trained models")
    parser.add_argument(
        "--models",
        help="Comma-separated model names (default: all)",
    )
    args = parser.parse_args()
    main(args.models)
