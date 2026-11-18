"""Step 4b - HPO plots and hyperparameter importances, rebuilt from the saved
search record (outputs/hpo/trials.csv) without re-running the search.

Outputs -> outputs/hpo/optimisation_history.png, param_importances.png,
           param_importances.csv (PED-ANOVA and fANOVA, completed trials only)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from pathsnap.experiment import ROOT, SEED

OUT = os.path.join(ROOT, "outputs", "hpo")
ss = json.load(open(os.path.join(OUT, "search_space.json")))
T = pd.read_csv(os.path.join(OUT, "trials.csv"))
dists = {}
for k, (kind, lo, hi) in ss["space"].items():
    if kind == "int":
        dists[k] = optuna.distributions.IntDistribution(lo, hi)
    elif kind == "int_log":
        dists[k] = optuna.distributions.IntDistribution(lo, hi, log=True)
    elif kind == "float":
        dists[k] = optuna.distributions.FloatDistribution(lo, hi)
    else:
        dists[k] = optuna.distributions.FloatDistribution(lo, hi, log=True)
state = {"COMPLETE": optuna.trial.TrialState.COMPLETE, "PRUNED": optuna.trial.TrialState.PRUNED}
study = optuna.create_study(direction="maximize")
for _, r in T.iterrows():
    params = {k: (int(r[f"params_{k}"]) if isinstance(dists[k], optuna.distributions.IntDistribution)
                  else float(r[f"params_{k}"])) for k in dists}
    st = state[r.state]
    study.add_trial(optuna.trial.create_trial(params=params, distributions=dists, state=st,
                                              value=float(r.value) if st == optuna.trial.TrialState.COMPLETE else None,
                                              intermediate_values={} if st == optuna.trial.TrialState.COMPLETE
                                              else {1: float(r.value)}))
assert study.best_trial.number == int(T.loc[T[T.state == "COMPLETE"].value.idxmax(), "number"])

# optimisation history (completed trials; pruned trials shown at their partial value)
comp = T[T.state == "COMPLETE"]
pr = T[T.state == "PRUNED"]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.scatter(comp.number, comp.value, s=18, label="completed trial (5-fold CV Top-1)")
ax.scatter(pr.number, pr.value, s=18, marker="x", c="0.5", label="pruned trial (partial fold mean)")
best = comp.set_index("number").value.reindex(range(int(T.number.max()) + 1)).cummax().ffill()
ax.step(best.index, best.values, where="post", c="tab:red", label="best so far")
ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1 (train)")
ax.set_title(f"Optuna TPE optimisation history - {ss['model']} (objective: 5-fold CV Top-1, train)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "optimisation_history.png"), dpi=130); plt.close(fig)

# Two importance estimators (both on the completed trials only):
#  * PED-ANOVA - Optuna's default evaluator (deterministic); this is what 04_bayes_opt.py plotted
#  * fANOVA    - random-forest based, so it varies with the seed: mean +- std over 10 seeds
ped = optuna.importance.get_param_importances(study, evaluator=optuna.importance.PedAnovaImportanceEvaluator())
runs = [optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=s))
        for s in range(SEED, SEED + 10)]
F = pd.DataFrame(runs)
I = pd.DataFrame({"hyperparameter": list(dists)})
I["ped_anova"] = I.hyperparameter.map(ped)
I["fanova_mean"] = I.hyperparameter.map(F.mean())
I["fanova_std"] = I.hyperparameter.map(F.std())
I["rank_ped_anova"] = I.ped_anova.rank(ascending=False).astype(int)
I["rank_fanova"] = I.fanova_mean.rank(ascending=False).astype(int)
I = I.sort_values("ped_anova", ascending=False).reset_index(drop=True)
I.to_csv(os.path.join(OUT, "param_importances.csv"), index=False)
fig, ax = plt.subplots(figsize=(8, 3.8))
y = np.arange(len(I))[::-1]
ax.barh(y + 0.2, I.ped_anova, 0.4, label="PED-ANOVA (Optuna default)", color="tab:blue")
ax.barh(y - 0.2, I.fanova_mean, 0.4, xerr=I.fanova_std, capsize=2, label="fANOVA (mean +- std, 10 seeds)", color="tab:orange")
ax.set_yticks(y); ax.set_yticklabels(I.hyperparameter)
ax.set_xlabel("importance (share of objective variance explained), completed trials")
ax.set_title(f"Hyperparameter importance - {ss['model']} ({len(comp)} completed trials)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3, axis="x")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "param_importances.png"), dpi=130); plt.close(fig)
print(I.round(3).to_string(index=False))
print("pruned:", len(pr), "completed:", len(comp), "best:", study.best_trial.number, round(study.best_value, 4))
