"""
Classifier builders for the model comparison, and a wrapper so every model
speaks class names.

    build_rf / build_xgb / build_lgbm / build_catboost(params, seed)
    build_ensemble(params_by_model, seed, calibrated=True)
        soft voting over XGBoost + LightGBM + RF; with calibrated=True each member
        is wrapped in CalibratedClassifierCV (isotonic, 5-fold) before voting
    PARAM_SPACES      random-search spaces (every model gets the same 30-trial budget)
    LabeledModel      estimator + LabelEncoder: predict() returns e.g. "LORA", not 2
    save_model / load_model   models/<name>.joblib (+ models/<name>.json metadata)

All estimators are single-threaded (n_jobs=1): a detection is one row, where
thread start-up costs more than it saves; tuning parallelises over CV fits.
"""

import json
import os

import joblib
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

MODELS_DIR = "models"

PARAM_SPACES = {
    "rf": {
        "n_estimators": randint(100, 501),
        "max_depth": [None, 8, 12, 16, 24],
        "min_samples_leaf": randint(1, 6),
        "max_features": ["sqrt", "log2", 0.5, None],
    },
    "xgb": {
        "n_estimators": randint(100, 501),
        "max_depth": randint(3, 9),
        "learning_rate": loguniform(0.02, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 8),
        "reg_lambda": loguniform(0.1, 10),
    },
    "lgbm": {
        "n_estimators": randint(100, 501),
        "num_leaves": randint(8, 64),
        "learning_rate": loguniform(0.02, 0.3),
        "subsample": uniform(0.6, 0.4),
        "subsample_freq": [1],
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_samples": randint(5, 41),
        "reg_lambda": loguniform(0.1, 10),
    },
    "cat": {
        "iterations": randint(150, 501),
        "depth": randint(4, 9),
        "learning_rate": loguniform(0.03, 0.3),
        "l2_leaf_reg": loguniform(1, 10),
    },
}


def build_rf(params=None, seed=0):
    return RandomForestClassifier(n_jobs=1, random_state=seed, **(params or {}))


def build_xgb(params=None, seed=0):
    return XGBClassifier(objective="multi:softprob", tree_method="hist", eval_metric="mlogloss",
                         n_jobs=1, random_state=seed, **(params or {}))


def build_lgbm(params=None, seed=0):
    return LGBMClassifier(n_jobs=1, random_state=seed, verbose=-1, **(params or {}))


def build_catboost(params=None, seed=0):
    return CatBoostClassifier(thread_count=1, random_seed=seed, verbose=False,
                              allow_writing_files=False, **(params or {}))


BUILDERS = {"rf": build_rf, "xgb": build_xgb, "lgbm": build_lgbm, "cat": build_catboost}


def build_ensemble(params_by_model, seed=0, calibrated=True):
    """Soft-voting XGBoost + LightGBM + RF, using each member's tuned params."""
    members = []
    for name in ("xgb", "lgbm", "rf"):
        est = BUILDERS[name](params_by_model.get(name), seed)
        if calibrated:
            est = CalibratedClassifierCV(est, method="isotonic", cv=5)
        members.append((name, est))
    return VotingClassifier(members, voting="soft", n_jobs=1)


class LabeledModel:
    """A fitted estimator trained on encoded labels, plus the encoder to name its outputs."""

    def __init__(self, estimator, encoder):
        self.estimator = estimator
        self.encoder = encoder

    @property
    def classes_(self):
        return self.encoder.classes_

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.encoder.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        return self.estimator.predict_proba(np.asarray(X, dtype=float))


def save_model(name, model, meta, models_dir=MODELS_DIR):
    """Write models/<name>.joblib and models/<name>.json; returns the paths."""
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{name}.joblib")
    meta_path = os.path.join(models_dir, f"{name}.json")
    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return model_path, meta_path


def load_model(name, models_dir=MODELS_DIR):
    """(LabeledModel, metadata dict) for models/<name>.joblib; FileNotFoundError if missing."""
    model = joblib.load(os.path.join(models_dir, f"{name}.joblib"))
    with open(os.path.join(models_dir, f"{name}.json")) as f:
        meta = json.load(f)
    return model, meta
