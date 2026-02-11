"""Reusable PCA and SHAP helpers for training notebooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


@dataclass
class PCAResult:
    explained_variance_ratio_: np.ndarray
    cumulative_variance_: np.ndarray
    components_needed: int


def run_pca(
    X: pd.DataFrame | np.ndarray,
    variance_threshold: float = 0.95,
    plot: bool = True,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Standardize features and run PCA diagnostics."""
    if isinstance(X, pd.DataFrame):
        values = X.values
        columns = X.columns
    else:
        values = np.asarray(X)
        columns = [f"feature_{i}" for i in range(values.shape[1])]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(values)

    pca = PCA()
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    components_needed = int(np.argmax(cumulative >= variance_threshold) + 1)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].bar(range(1, len(explained) + 1), explained)
        axes[0].set_title("Variance Explained per Component")
        axes[0].set_xlabel("Component")
        axes[0].set_ylabel("Variance Ratio")

        axes[1].plot(range(1, len(cumulative) + 1), cumulative, marker="o")
        axes[1].axhline(variance_threshold, color="red", linestyle="--", label=f"{variance_threshold:.0%}")
        axes[1].set_title("Cumulative Variance")
        axes[1].set_xlabel("Component")
        axes[1].set_ylabel("Cumulative Ratio")
        axes[1].legend()

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"PCA: {components_needed} components reach {variance_threshold:.0%} variance")
    return PCAResult(explained, cumulative, components_needed)


def compute_shap(
    model,
    X: pd.DataFrame | np.ndarray,
    sample_size: int = 500,
    random_state: int = 42,
    plot_type: str = "bar",
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Compute SHAP summary plot for a model, sampling rows if needed."""
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(np.asarray(X))

    if sample_size and len(X_df) > sample_size:
        X_sample = X_df.sample(sample_size, random_state=random_state)
    else:
        X_sample = X_df

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type=plot_type,
        show=show,
    )
    if save_path is not None:
        fig = plt.gcf()
        fig.savefig(save_path, bbox_inches="tight")
        if not show:
            plt.close(fig)
    return shap_values
