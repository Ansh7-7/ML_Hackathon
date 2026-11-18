"""Step 3b - consolidate per-model results of 03_train_compare.py into the
comparison table and the comparison plots (no training happens here).

Inputs : outputs/models/per_model/*.json, *_cv_folds.csv, outputs/models/mlp_history.csv
Outputs: outputs/models/comparison.csv, cv_folds_all.csv, model_selection.json
         outputs/figures/*.png
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.experiment import ROOT

OUT = os.path.join(ROOT, "outputs", "models")
FIG = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

ORDER = ["LogisticRegression", "GaussianNB", "KNN", "RandomForest", "ExtraTrees", "HistGradientBoosting",
         "LightGBM", "XGBoost", "MLP"]
res = {json.load(open(f))["model"]: json.load(open(f)) for f in glob.glob(os.path.join(OUT, "per_model", "*.json"))}
names = [n for n in ORDER if n in res]
F = pd.concat([pd.read_csv(os.path.join(OUT, "per_model", f"{n}_cv_folds.csv")) for n in names], ignore_index=True)
F.to_csv(os.path.join(OUT, "cv_folds_all.csv"), index=False)

KEYCFG = {
    "LogisticRegression": lambda p: f"StandardScaler; L2, C={p['C']}, lbfgs",
    "GaussianNB": lambda p: f"var_smoothing={p['var_smoothing']}",
    "KNN": lambda p: f"StandardScaler; k={p['n_neighbors']}, {p['weights']}-weighted; fit on 150k-row sample",
    "RandomForest": lambda p: f"{p['n_estimators']} trees, min_leaf={p['min_samples_leaf']}, max_feat={p['max_features']}",
    "ExtraTrees": lambda p: f"{p['n_estimators']} trees, min_leaf={p['min_samples_leaf']}, max_feat={p['max_features']}",
    "HistGradientBoosting": lambda p: f"lr={p['learning_rate']}, {p['max_iter']} iters, {p['max_leaf_nodes']} leaves",
    "LightGBM": lambda p: f"lr={p['learning_rate']}, {p['n_estimators']} trees, {p['num_leaves']} leaves, subsample {p['subsample']}",
    "XGBoost": lambda p: f"lr={p['learning_rate']}, {p['n_estimators']} trees, depth {p['max_depth']}, subsample {p['subsample']}",
    "MLP": lambda p: f"StandardScaler; {tuple(p['hidden_layer_sizes'])} ReLU, Adam lr={p['learning_rate_init']}, "
                     f"L2={p['alpha']}, early stop on val log loss",
}
rows = []
for n in names:
    r = res[n]
    rows.append({
        "Model": n, "Algorithm family": r["family"], "Key configuration": KEYCFG[n](r["params"]),
        "CV Macro F1 (mean)": r["cv_macro_f1_mean"], "CV Macro F1 (std)": r["cv_macro_f1_std"],
        "CV Top-1 (mean)": r["cv_top1_mean"], "CV Top-1 (std)": r["cv_top1_std"],
        "CV Log loss": r["cv_log_loss_mean"], "Train-fold Macro F1": r["train_macro_f1_mean"],
        "Student-test Top-1": r["st_top1"], "Student-test Macro F1": r["st_macro_f1"],
        "Student-test Precision (pos)": r["st_precision_pos"], "Student-test Recall (pos)": r["st_recall_pos"],
        "Student-test Precision (macro)": r["st_precision_macro"], "Student-test Recall (macro)": r["st_recall_macro"],
        "Student-test Accuracy (rows)": r["st_accuracy"],
        "Student-test Log loss": r["st_log_loss"], "Route continuity (student-test, classifier only)": r["st_route_continuity"],
        "Candidate miss rate (student-test)": r["st_miss_rate"],
        "Train time final fit [s]": r["final_fit_s"], "Mean CV fold fit time [s]": r["cv_fit_s_mean"],
        "Inference time student-test [s]": r["st_inference_s"],
        "Inference [ms / 1000 obs]": r["st_inference_ms_per_1000_obs"],
    })
C = pd.DataFrame(rows)
C.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
print(C.drop(columns=["Key configuration"]).round(4).to_string())

# ------------------------------------------------------------------ selection
best_top1 = C["CV Top-1 (mean)"].max()
std_best = float(C.loc[C["CV Top-1 (mean)"].idxmax(), "CV Top-1 (std)"])
contenders = C[C["CV Top-1 (mean)"] >= best_top1 - std_best].sort_values(
    ["CV Log loss", "Inference [ms / 1000 obs]"])
chosen = contenders.iloc[0].Model
sel = {"primary_metric": "mean CV observation-level Top-1 accuracy (GroupKFold by trip, train partition)",
       "tie_break": "models within one fold-std of the best CV Top-1 are ranked by CV log loss, then inference time",
       "best_cv_top1_model": C.loc[C["CV Top-1 (mean)"].idxmax(), "Model"], "best_cv_top1": float(best_top1),
       "fold_std_of_best": std_best, "contenders": contenders.Model.tolist(), "chosen": chosen}
json.dump(sel, open(os.path.join(OUT, "model_selection.json"), "w"), indent=2)
print(json.dumps(sel, indent=2))

# ------------------------------------------------------------------ plots
colors = {n: c for n, c in zip(ORDER, plt.cm.tab10(np.arange(9)))}
x = np.arange(len(names))

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(x, C["CV Macro F1 (mean)"], yerr=C["CV Macro F1 (std)"], capsize=3, color=[colors[n] for n in names])
for k, n in enumerate(names):
    ax.scatter(np.full(5, k) + np.linspace(-.2, .2, 5), F[F.model == n].val_macro_f1, s=8, c="k", zorder=3)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
ax.set_ylim(max(0, C["CV Macro F1 (mean)"].min() - .08), 1); ax.set_ylabel("Macro F1")
ax.set_title("5-fold GroupKFold CV Macro F1 (train, 78 trips; dots = folds, bars = mean ± std)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "cv_macro_f1.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.6))
w = .38
ax.bar(x - w / 2, C["Student-test Top-1"], w, label="Top-1 accuracy (observation level)", color="tab:blue")
ax.bar(x + w / 2, C["Student-test Macro F1"], w, label="Macro F1 (candidate level)", color="tab:orange")
ax.axhline(1 - C["Candidate miss rate (student-test)"].iloc[0], ls="--", c="k", lw=.8,
           label="Top-1 ceiling (1 - miss rate)")
for k in range(len(names)):
    ax.text(k - w / 2, C["Student-test Top-1"].iloc[k] + .003, f"{C['Student-test Top-1'].iloc[k]:.3f}",
            ha="center", fontsize=6.5, rotation=90, va="bottom")
    ax.text(k + w / 2, C["Student-test Macro F1"].iloc[k] + .003, f"{C['Student-test Macro F1'].iloc[k]:.3f}",
            ha="center", fontsize=6.5, rotation=90, va="bottom")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
ax.set_ylim(0.75, 1.04); ax.legend(fontsize=7, loc="lower right")
ax.set_title("Student-test (10 trips): Top-1 accuracy and Macro F1, classifier only")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "student_test_top1_macro_f1.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(x - w / 2, C["CV Log loss"], w, label="CV (train folds held out)", color="tab:purple")
ax.bar(x + w / 2, C["Student-test Log loss"], w, label="student-test", color="tab:green")
ax.set_yscale("log"); ax.set_ylabel("log loss (log scale, lower is better)")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right"); ax.legend()
ax.set_title("Candidate-level log loss")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "log_loss.png")); plt.close(fig)

fig, axs = plt.subplots(3, 3, figsize=(10, 9.5))
for ax, n in zip(axs.ravel(), names):
    cm = np.array(res[n]["st_confusion_matrix"])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cmn[i, j]:.3f})", ha="center", va="center",
                    color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred other", "pred ref arc"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true other", "true ref arc"])
    ax.set_title(n, fontsize=9)
for ax in axs.ravel()[len(names):]:
    ax.axis("off")
fig.suptitle("Candidate-level confusion matrices on student-test (threshold 0.5; row-normalised in brackets)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "confusion_matrices.png")); plt.close(fig)

if os.path.exists(os.path.join(OUT, "mlp_history.csv")):
    H = pd.read_csv(os.path.join(OUT, "mlp_history.csv"))
    folds = list(H.fold.unique())
    fig, axs = plt.subplots(1, len(folds), figsize=(2.3 * len(folds), 2.9), sharey=True)
    for ax, f in zip(np.atleast_1d(axs), folds):
        h = H[H.fold == f]
        ax.plot(h.epoch, h.train_log_loss, "o-", ms=2.5, label="train")
        ax.plot(h.epoch, h.val_log_loss, "s-", ms=2.5, label="validation")
        be = int(h.loc[h.val_log_loss.idxmin(), "epoch"])
        ax.axvline(be, ls=":", c="k", lw=.8)
        ax.set_title(f"fold {f}" if f != "final" else "final fit", fontsize=8); ax.set_xlabel("epoch")
    np.atleast_1d(axs)[0].set_ylabel("log loss"); np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.suptitle("MLP train / validation log loss per epoch (validation = 15% held-out training trips; dotted = best epoch)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "mlp_loss_curves.png")); plt.close(fig)

# over/under-fitting evidence: training-fold vs held-out Macro F1 and log loss
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].bar(x - w / 2, C["Train-fold Macro F1"], w, label="training folds (in-sample)", color="0.6")
ax[0].bar(x + w / 2, C["CV Macro F1 (mean)"], w, label="held-out folds", color="tab:blue")
ax[0].set_ylim(.75, 1.01); ax[0].set_ylabel("Macro F1")
tl = F.groupby("model").train_log_loss.mean().reindex(names)
ax[1].bar(x - w / 2, tl, w, label="training folds (in-sample)", color="0.6")
ax[1].bar(x + w / 2, C["CV Log loss"], w, label="held-out folds", color="tab:purple")
ax[1].set_yscale("log"); ax[1].set_ylabel("log loss")
for a in ax:
    a.set_xticks(x); a.set_xticklabels(names, rotation=25, ha="right"); a.legend(fontsize=7)
fig.suptitle("Generalisation gap: in-sample vs held-out (5-fold GroupKFold on train)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "train_vs_val_gap.png")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].boxplot([F[F.model == n].val_top1 for n in names], tick_labels=names)
ax[0].set_ylabel("held-out Top-1"); ax[0].set_title("CV Top-1 per fold (primary selection metric)")
ax[0].tick_params(axis="x", rotation=25)
ax[1].scatter(C["Inference [ms / 1000 obs]"], C["CV Top-1 (mean)"], c=[colors[n] for n in names], s=40)
for _, r in C.iterrows():
    ax[1].annotate(r.Model, (r["Inference [ms / 1000 obs]"], r["CV Top-1 (mean)"]), fontsize=7,
                   xytext=(3, 3), textcoords="offset points")
ax[1].set_xscale("log"); ax[1].set_xlabel("inference time [ms per 1000 observations] (log)")
ax[1].set_ylabel("CV Top-1"); ax[1].set_title("Accuracy vs inference cost")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "selection_top1_folds_and_speed.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.bar(x - w / 2, C["Train time final fit [s]"], w, label="training (final fit, 606k rows)")
ax.bar(x + w / 2, C["Inference time student-test [s]"], w, label="inference (student-test, 213k rows)")
ax.set_yscale("log"); ax.set_ylabel("seconds (log)"); ax.legend()
ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right"); ax.set_title("Computational cost")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "train_inference_time.png")); plt.close(fig)
print("plots written")
