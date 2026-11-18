"""Model zoo for the candidate classifier.

Every model scores one (observation, candidate) row with P(candidate is the
reference arc). All learned preprocessing (StandardScaler) sits inside a
Pipeline / estimator, so it is fitted on the training fold only.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


class EpochMLP(BaseEstimator, ClassifierMixin):
    """sklearn MLPClassifier trained epoch by epoch with partial_fit so that the
    *log loss* on the training data and on held-out validation trips can be
    recorded after every epoch. Early stopping on validation log loss; the
    weights of the best epoch are restored."""

    def __init__(self, hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3, batch_size=1024,
                 max_epochs=30, patience=5, val_frac=0.15, random_state=SEED):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        y = np.asarray(y)
        if groups is None:
            groups = np.arange(len(y)) // 1000
        gss = GroupShuffleSplit(n_splits=1, test_size=self.val_frac, random_state=self.random_state)
        tr, va = next(gss.split(X, y, groups))
        self.scaler_ = StandardScaler().fit(X[tr])
        Xtr, Xva = self.scaler_.transform(X[tr]), self.scaler_.transform(X[va])
        ytr, yva = y[tr], y[va]
        rng = np.random.RandomState(self.random_state)
        mon = rng.choice(len(ytr), size=min(len(ytr), 200_000), replace=False)  # train-loss monitor sample
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                  learning_rate_init=self.learning_rate_init, batch_size=self.batch_size,
                                  solver="adam", shuffle=True, random_state=self.random_state)
        hist, best, best_state, bad = [], np.inf, None, 0
        for ep in range(1, self.max_epochs + 1):
            t0 = time.time()
            self.mlp_.partial_fit(Xtr, ytr, classes=np.array([0, 1]))
            tl = log_loss(ytr[mon], self.mlp_.predict_proba(Xtr[mon])[:, 1], labels=[0, 1])
            vl = log_loss(yva, self.mlp_.predict_proba(Xva)[:, 1], labels=[0, 1])
            hist.append({"epoch": ep, "train_log_loss": tl, "val_log_loss": vl, "epoch_s": time.time() - t0})
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = (copy.deepcopy(self.mlp_.coefs_), copy.deepcopy(self.mlp_.intercepts_), ep)
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.mlp_.coefs_, self.mlp_.intercepts_, self.best_epoch_ = best_state
        self.history_ = pd.DataFrame(hist)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(self.scaler_.transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# name -> (family, factory(params) -> estimator, default params, fit-row cap, needs groups)
def _lgbm(p):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**p)


def _xgb(p):
    from xgboost import XGBClassifier
    return XGBClassifier(**p)


ZOO = {
    "LogisticRegression": ("Linear / probabilistic",
                           lambda p: Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**p))]),
                           {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}, None, False),
    "GaussianNB": ("Probabilistic (generative)", lambda p: GaussianNB(**p), {"var_smoothing": 1e-9}, None, False),
    "KNN": ("Instance-based",
            lambda p: Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(**p))]),
            {"n_neighbors": 25, "weights": "distance", "n_jobs": 4}, 150_000, False),
    "RandomForest": ("Tree ensemble (bagging)", lambda p: RandomForestClassifier(**p),
                     {"n_estimators": 200, "min_samples_leaf": 5, "max_features": "sqrt", "n_jobs": 4,
                      "random_state": SEED}, None, False),
    "ExtraTrees": ("Tree ensemble (randomised)", lambda p: ExtraTreesClassifier(**p),
                   {"n_estimators": 300, "min_samples_leaf": 5, "max_features": "sqrt", "n_jobs": 4,
                    "random_state": SEED}, None, False),
    "HistGradientBoosting": ("Boosting", lambda p: HistGradientBoostingClassifier(**p),
                             {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                              "l2_regularization": 0.0, "early_stopping": False, "random_state": SEED}, None, False),
    "LightGBM": ("Boosting", _lgbm,
                 {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "subsample": 0.8,
                  "subsample_freq": 1, "colsample_bytree": 0.8, "min_child_samples": 20, "n_jobs": 4,
                  "random_state": SEED, "verbose": -1}, None, False),
    "XGBoost": ("Boosting", _xgb,
                {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
                 "colsample_bytree": 0.8, "tree_method": "hist", "n_jobs": 4, "random_state": SEED,
                 "eval_metric": "logloss"}, None, False),
    "MLP": ("Neural network", lambda p: EpochMLP(**p),
            {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3, "batch_size": 1024,
             "max_epochs": 30, "patience": 5, "val_frac": 0.15, "random_state": SEED}, None, True),
}


def make_model(name, params=None):
    family, factory, defaults, cap, needs_groups = ZOO[name]
    p = dict(defaults)
    if params:
        p.update(params)
    return factory(p), p


def fit_model(name, model, X, y, groups, seed=SEED):
    """Fit with the model-specific row cap (KNN) and group-aware validation (MLP)."""
    family, _, _, cap, needs_groups = ZOO[name]
    if cap is not None and len(y) > cap:
        idx = np.sort(np.random.RandomState(seed).choice(len(y), cap, replace=False))
        X, y, groups = X[idx], y[idx], groups[idx]
    if needs_groups:
        model.fit(X, y, groups=groups)
    else:
        model.fit(X, y)
    return model


def predict_pos(model, X, batch=200_000):
    out = np.empty(len(X), dtype=np.float64)
    for s in range(0, len(X), batch):
        out[s:s + batch] = model.predict_proba(X[s:s + batch])[:, 1]
    return out
