"""Runtime-safe custom classes for the NHANES two-stage cascade model.

These classes mirror the custom notebook classes used when the joblib package was
created. They are required because notebook-trained sklearn/joblib objects often
reference custom classes from `__main__`.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

N_JOBS = 1
LABEL_ORDER = ["low", "moderate", "high"]


class OBUResampler(BaseEstimator):
    """OBU-like Stage-2 resampler used in the final notebook.

    Stage-2 labels are binary: 0=moderate, 1=high.
    This class is only used during training, but it must exist at load time
    because it is stored inside the fitted imblearn pipeline.
    """

    def __init__(self, k_neighbors=7, majority_remove_frac=0.25, target_ratio=1.0, random_state=42):
        self.k_neighbors = k_neighbors
        self.majority_remove_frac = majority_remove_frac
        self.target_ratio = target_ratio
        self.random_state = random_state

    def fit_resample(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            return X, y
        minority = classes[np.argmin(counts)]
        majority = classes[np.argmax(counts)]
        idx_min = np.where(y == minority)[0]
        idx_maj = np.where(y == majority)[0]
        if len(idx_min) < 3 or len(idx_maj) < 3:
            return X, y

        k = min(self.k_neighbors + 1, len(y))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        neigh = nn.kneighbors(X[idx_maj], return_distance=False)[:, 1:]
        minority_frac = (y[neigh] == minority).mean(axis=1)

        n_remove = int(len(idx_maj) * self.majority_remove_frac)
        keep_maj = idx_maj.copy()
        if n_remove > 0:
            remove = idx_maj[np.argsort(-minority_frac)[:n_remove]]
            keep_maj = np.setdiff1d(idx_maj, remove, assume_unique=False)

        target_min_n = int(len(keep_maj) * self.target_ratio)
        if target_min_n <= len(idx_min):
            idx_min_new = idx_min
        else:
            extra = rng.choice(idx_min, size=target_min_n - len(idx_min), replace=True)
            idx_min_new = np.concatenate([idx_min, extra])

        final_idx = np.concatenate([keep_maj, idx_min_new])
        rng.shuffle(final_idx)
        return X[final_idx], y[final_idx]


class TreeMedianSelector(BaseEstimator, TransformerMixin):
    """Tree-importance selector using median importance threshold."""

    def __init__(self, random_state=42, n_estimators=150, max_depth=5):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X, y):
        self.model_ = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=N_JOBS,
        )
        self.model_.fit(X, y)
        imp = np.asarray(self.model_.feature_importances_)
        threshold = np.median(imp)
        mask = imp >= threshold
        if mask.sum() == 0:
            mask[np.argmax(imp)] = True
        self.support_mask_ = mask
        self.importances_ = imp
        return self

    def transform(self, X):
        check_is_fitted(self, "support_mask_")
        return X[:, self.support_mask_]

    def get_support(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_


class TwoStageCascade(BaseEstimator, ClassifierMixin):
    """Final two-stage model: Stage-1 low/at-risk gate, Stage-2 moderate/high."""

    def __init__(self, stage1_pipeline=None, stage2_pipeline=None, stage1_threshold=0.44,
                 stage2_threshold=0.50, label_order=None, name="cascade"):
        self.stage1_pipeline = stage1_pipeline
        self.stage2_pipeline = stage2_pipeline
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold
        self.label_order = label_order or LABEL_ORDER
        self.name = name

    def _positive_proba(self, model, X, positive_class=1):
        p = model.predict_proba(X)
        try:
            cls = list(model.named_steps["model"].classes_)
        except Exception:
            cls = list(model.classes_)
        return p[:, cls.index(positive_class)] if positive_class in cls else p[:, -1]

    def predict_detail(self, X):
        p_atrisk = self._positive_proba(self.stage1_pipeline_, X, 1)
        gate = p_atrisk >= self.stage1_threshold
        pred = np.array(["low"] * len(X), dtype=object)
        p_high_cond_all = self._positive_proba(self.stage2_pipeline_, X, 1)

        if gate.sum() > 0:
            p_high_gate = p_high_cond_all[gate]
            pred[gate] = np.where(p_high_gate >= self.stage2_threshold, "high", "moderate")

        p_low = 1.0 - p_atrisk
        p_high = p_atrisk * p_high_cond_all
        p_moderate = p_atrisk * (1.0 - p_high_cond_all)
        proba = np.vstack([p_low, p_moderate, p_high]).T
        row_sum = proba.sum(axis=1, keepdims=True)
        proba = np.divide(proba, row_sum, where=row_sum != 0)

        return pd.DataFrame({
            "predicted": pred,
            "p_low": proba[:, 0],
            "p_moderate": proba[:, 1],
            "p_high": proba[:, 2],
            "p_atrisk_stage1": p_atrisk,
            "stage1_gate_sent_to_stage2": gate,
            "p_high_given_atrisk_stage2": p_high_cond_all,
        })

    def predict(self, X):
        return self.predict_detail(X)["predicted"].values

    def predict_proba(self, X):
        detail = self.predict_detail(X)
        return detail[["p_low", "p_moderate", "p_high"]].values


class Stage1EnsembleGate(BaseEstimator, ClassifierMixin):
    """Compatibility shim. Final selected model does not use this class."""

    def __init__(self, stage1_features=None, X_reference=None, base_configs=None,
                 mode="mean_proba", base_vote_threshold=0.475, random_state=42):
        self.stage1_features = stage1_features
        self.X_reference = X_reference
        self.base_configs = base_configs
        self.mode = mode
        self.base_vote_threshold = base_vote_threshold
        self.random_state = random_state


def install_pickle_aliases() -> None:
    """Expose custom classes under __main__ for notebook-created pickle files."""
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    for cls in [OBUResampler, TreeMedianSelector, TwoStageCascade, Stage1EnsembleGate]:
        setattr(main_mod, cls.__name__, cls)
