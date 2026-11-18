"""Step 4 - Bayesian hyperparameter optimisation of the chosen model (Optuna TPE).

Partition: TRAIN only (78 trips). Objective = mean observation-level Top-1
accuracy over the same 5-fold GroupKFold-by-trip split used in step 3 (fit on
every 3rd observation of the training trips, score all observations of the
held-out trips). Student-test is used only AFTER the search, to compare the
tuned and untuned models; it never feeds back into the search.

Sampler: TPE (seed 42). Pruner: MedianPruner on the running fold mean, so
clearly bad configurations stop after 2 folds (pruned trials are kept in
trials.csv with their partial value).

Outputs -> outputs/hpo/: trials.csv, best_params.json, search_space.json,
           tuned_vs_untuned.csv, optimisation_history.png, param_importances.png,
           tuned_vs_untuned.png ; data/processed/preds/tuned_oof.npy ;
           artifacts/model.joblib (final tuned model)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from pathsnap.evaluate import confusion
from pathsnap.experiment import (ROOT, SEED, TRAIN_STRIDE, continuity_for, load_partition, score, stride_mask,
                                 trip_folds, xy)
from pathsnap.features import FEATURES, prepare_trip
from pathsnap.io import list_trips, load_trip
from pathsnap.models import ZOO, fit_model, make_model, predict_pos

OUT = os.path.join(ROOT, "outputs", "hpo")
ART = os.path.join(ROOT, "artifacts")
PRED = os.path.join(ROOT, "data", "processed", "preds")
for d in (OUT, ART, PRED):
    os.makedirs(d, exist_ok=True)

SPACES = {
    "LightGBM": {
        "n_estimators": ("int_log", 100, 1500), "learning_rate": ("float_log", 0.01, 0.3),
        "num_leaves": ("int_log", 15, 255), "min_child_samples": ("int_log", 5, 300),
        "subsample": ("float", 0.5, 1.0), "colsample_bytree": ("float", 0.4, 1.0),
        "reg_lambda": ("float_log", 1e-3, 10.0), "reg_alpha": ("float_log", 1e-3, 10.0)},
    "XGBoost": {
        "n_estimators": ("int_log", 100, 1500), "learning_rate": ("float_log", 0.01, 0.3),
        "max_depth": ("int", 3, 12), "min_child_weight": ("float_log", 1.0, 50.0),
        "subsample": ("float", 0.5, 1.0), "colsample_bytree": ("float", 0.4, 1.0),
        "reg_lambda": ("float_log", 1e-3, 10.0), "gamma": ("float", 0.0, 5.0)},
    "HistGradientBoosting": {
        "learning_rate": ("float_log", 0.01, 0.3), "max_iter": ("int_log", 100, 1000),
        "max_leaf_nodes": ("int_log", 15, 255), "min_samples_leaf": ("int_log", 5, 300),
        "l2_regularization": ("float_log", 1e-3, 10.0)},
    "RandomForest": {
        "n_estimators": ("int", 100, 400), "min_samples_leaf": ("int_log", 1, 50),
        "max_features": ("float", 0.1, 0.8), "max_depth": ("int", 8, 40)},
    "ExtraTrees": {
        "n_estimators": ("int", 100, 500), "min_samples_leaf": ("int_log", 1, 50),
        "max_features": ("float", 0.1, 0.9), "max_depth": ("int", 8, 40)},
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None, help="defaults to outputs/models/model_selection.json 'chosen'")
ap.add_argument("--trials", type=int, default=40)
args = ap.parse_args()
name = args.model or json.load(open(os.path.join(ROOT, "outputs", "models", "model_selection.json")))["chosen"]
space = SPACES[name]
json.dump({"model": name, "space": space, "objective": "maximise mean 5-fold GroupKFold CV Top-1 (train, 78 trips)",
           "sampler": "TPESampler(seed=42, n_startup_trials=10)", "pruner": "MedianPruner(n_startup_trials=8, n_warmup_steps=1)",
           "n_trials": args.trials, "train_stride": TRAIN_STRIDE},
          open(os.path.join(OUT, "search_space.json"), "w"), indent=2)

D, PT = load_partition("train")
S, PS = load_partition("student_test")
X, y, g = xy(D)
Xs, ys, _ = xy(S)
sm = stride_mask(D)
folds = trip_folds(D.trip_id)


def suggest(trial):
    p = {}
    for k, (kind, lo, hi) in space.items():
        if kind == "int":
            p[k] = trial.suggest_int(k, lo, hi)
        elif kind == "int_log":
            p[k] = trial.suggest_int(k, lo, hi, log=True)
        elif kind == "float":
            p[k] = trial.suggest_float(k, lo, hi)
        elif kind == "float_log":
            p[k] = trial.suggest_float(k, lo, hi, log=True)
    return p


def run_cv(params, trial=None, keep_oof=False):
    tops, f1s, lls, fit_t = [], [], [], []
    oof = np.full(len(D), np.nan) if keep_oof else None
    for k, (tr_trips, va_trips) in enumerate(folds):
        tr = np.isin(g, tr_trips) & sm
        va = np.isin(g, va_trips)
        model, _ = make_model(name, params)
        t0 = time.time()
        fit_model(name, model, X[tr], y[tr], g[tr])
        fit_t.append(time.time() - t0)
        p = predict_pos(model, X[va])
        if keep_oof:
            oof[va] = p
        m, _, _ = score(D[va].reset_index(drop=True), p, PT[PT.trip_id.isin(va_trips)])
        tops.append(m["top1"]); f1s.append(m["macro_f1"]); lls.append(m["log_loss"])
        if trial is not None:
            trial.set_user_attr(f"fold{k}_top1", m["top1"])
            trial.report(float(np.mean(tops)), k)
            if trial.should_prune():
                trial.set_user_attr("cv_macro_f1_partial", float(np.mean(f1s)))
                trial.set_user_attr("cv_log_loss_partial", float(np.mean(lls)))
                raise optuna.TrialPruned()
    out = {"cv_top1": float(np.mean(tops)), "cv_top1_std": float(np.std(tops, ddof=1)),
           "cv_macro_f1": float(np.mean(f1s)), "cv_macro_f1_std": float(np.std(f1s, ddof=1)),
           "cv_log_loss": float(np.mean(lls)), "fit_s_mean": float(np.mean(fit_t)), "fold_top1": tops}
    return out, oof


def objective(trial):
    params = suggest(trial)
    t0 = time.time()
    r, _ = run_cv(params, trial)
    trial.set_user_attr("cv_macro_f1", r["cv_macro_f1"])
    trial.set_user_attr("cv_log_loss", r["cv_log_loss"])
    trial.set_user_attr("trial_s", time.time() - t0)
    print(f"trial {trial.number}: top1={r['cv_top1']:.4f} f1={r['cv_macro_f1']:.4f} ll={r['cv_log_loss']:.4f} "
          f"{time.time() - t0:.0f}s {params}", flush=True)
    return r["cv_top1"]


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", study_name=f"pathsnap_{name}",
                            sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
# trial 0 = the untuned defaults restricted to the search space, so the search starts from the baseline
_, _, defaults, _, _ = ZOO[name]
study.enqueue_trial({k: defaults[k] for k in space if k in defaults})
t_search = time.time()
study.optimize(objective, n_trials=args.trials)
t_search = time.time() - t_search
T = study.trials_dataframe()
T.to_csv(os.path.join(OUT, "trials.csv"), index=False)
best = study.best_trial
print(f"best trial {best.number}: {best.value:.4f} {best.params}; search took {t_search / 60:.1f} min")

# ------------------------------------------------------------- tuned vs untuned
rows = []
untuned = json.load(open(os.path.join(ROOT, "outputs", "models", "per_model", f"{name}.json")))
prepared_st = {tid: prepare_trip(load_trip("student_test", tid)) for tid in list_trips("student_test")}
cv_t, oof = run_cv(best.params, keep_oof=True)
np.save(os.path.join(PRED, "tuned_oof.npy"), oof)
model, params = make_model(name, best.params)
t0 = time.time(); fit_model(name, model, X[sm], y[sm], g[sm]); t_fit = time.time() - t0
t0 = time.time(); p_st = predict_pos(model, Xs); t_inf = time.time() - t0
np.save(os.path.join(PRED, "tuned_student_test.npy"), p_st)
m_st, per_trip, preds = score(S, p_st, PS)
cont = continuity_for(preds, prepared_st)
per_trip["continuity"] = per_trip.trip_id.map(cont)
per_trip.to_csv(os.path.join(OUT, "tuned_student_test_per_trip.csv"), index=False)
preds.to_csv(os.path.join(OUT, "tuned_student_test_point_preds.csv"), index=False)
# untuned inference time re-measured in this same process for a like-for-like comparison
m0, p0 = make_model(name)
fit_model(name, m0, X[sm], y[sm], g[sm])
t0 = time.time(); p0_st = predict_pos(m0, Xs); t_inf0 = time.time() - t0
rows.append({"model": f"{name} (untuned)", "params": json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in p0.items()}),
             "cv_macro_f1": untuned["cv_macro_f1_mean"], "cv_top1": untuned["cv_top1_mean"],
             "cv_log_loss": untuned["cv_log_loss_mean"], "st_top1": untuned["st_top1"],
             "st_macro_f1": untuned["st_macro_f1"], "st_log_loss": untuned["st_log_loss"],
             "st_route_continuity": untuned["st_route_continuity"],
             "train_time_s": untuned["final_fit_s"], "inference_s": t_inf0,
             "inference_ms_per_1000_obs": 1e6 * t_inf0 / len(PS)})
rows.append({"model": f"{name} (tuned, trial {best.number})", "params": json.dumps(params),
             "cv_macro_f1": cv_t["cv_macro_f1"], "cv_top1": cv_t["cv_top1"], "cv_log_loss": cv_t["cv_log_loss"],
             "st_top1": m_st["top1"], "st_macro_f1": m_st["macro_f1"], "st_log_loss": m_st["log_loss"],
             "st_route_continuity": float(np.average(per_trip.continuity, weights=per_trip.n_obs)),
             "train_time_s": t_fit, "inference_s": t_inf, "inference_ms_per_1000_obs": 1e6 * t_inf / len(PS)})
TU = pd.DataFrame(rows)
TU.to_csv(os.path.join(OUT, "tuned_vs_untuned.csv"), index=False)
print(TU.drop(columns=["params"]).round(4).to_string())
json.dump({"model": name, "best_trial": best.number, "best_cv_top1": best.value, "best_params": best.params,
           "final_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
           "n_trials": len(study.trials),
           "n_complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
           "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
           "search_minutes": t_search / 60, "tuned_cv": cv_t, "tuned_student_test": m_st,
           "tuned_confusion_matrix": confusion(ys, p_st).tolist()},
          open(os.path.join(OUT, "best_params.json"), "w"), indent=2, default=float)

# keep the better of tuned / untuned (by the same primary metric, CV Top-1) as the deployed model
use_tuned = cv_t["cv_top1"] >= untuned["cv_top1_mean"]
final_model = model if use_tuned else m0
joblib.dump({"model": final_model, "model_name": name, "tuned": bool(use_tuned),
             "params": params if use_tuned else p0, "features": FEATURES, "train_stride": TRAIN_STRIDE,
             "trained_on": "train partition, 78 trips (33 and 54 excluded)"},
            os.path.join(ART, "model.joblib"), compress=3)
print("saved artifacts/model.joblib (tuned=%s)" % use_tuned)

# ------------------------------------------------------------------ plots
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
ax = plot_optimization_history(study)
ax.set_title(f"Optuna TPE optimisation history - {name} (objective: CV Top-1, train)")
ax.figure.set_size_inches(8, 3.8); ax.figure.tight_layout()
ax.figure.savefig(os.path.join(OUT, "optimisation_history.png"), dpi=130); plt.close(ax.figure)
ax = plot_param_importances(study)
ax.set_title(f"Hyperparameter importance (fANOVA) - {name}")
ax.figure.set_size_inches(7, 3.6); ax.figure.tight_layout()
ax.figure.savefig(os.path.join(OUT, "param_importances.png"), dpi=130); plt.close(ax.figure)

fig, axs = plt.subplots(1, 5, figsize=(13, 3))
for ax, (col, lab) in zip(axs, [("cv_macro_f1", "CV Macro F1"), ("st_top1", "Student-test Top-1"),
                                ("st_macro_f1", "Student-test Macro F1"), ("st_log_loss", "Student-test log loss"),
                                ("inference_s", "Inference time [s]")]):
    ax.bar(["untuned", "tuned"], TU[col], color=["0.6", "tab:blue"])
    for i, v in enumerate(TU[col]):
        ax.text(i, v, f"{v:.4f}" if v < 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    lo = TU[col].min()
    if col not in ("inference_s", "st_log_loss"):
        ax.set_ylim(lo - 0.03, min(1.0, TU[col].max() + 0.02))
    ax.set_title(lab, fontsize=9)
fig.suptitle(f"{name}: untuned vs Bayesian-tuned (same partitions and metrics)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "tuned_vs_untuned.png"), dpi=130); plt.close(fig)
print("done")
