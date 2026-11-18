"""Step 3c - what kind of mistakes does each classifier make? (student_test, classifier only)

For every observation whose Top-1 arc differs from the reference arc we classify
the error as
  * reverse_twin  - predicted arc is the opposite direction of the reference road
  * adjacent      - predicted arc shares a node with the reference arc (junction ambiguity)
  * missed        - the reference arc was not in the candidate set (no model can fix it)
  * other         - any other arc (parallel road, wrong street ...)
For reverse-twin errors we also count how often the GPS motion itself
contradicts the reference direction (angle between the +-5-point GPS heading and
the reference arc > 90 deg while moving), i.e. the "error" is more likely a
direction problem of the reference route than of the model.
Inputs : outputs/models/per_model/*_student_test_point_preds.csv, data/processed/student_test_points.parquet
Output : outputs/models/error_types_student_test.csv, outputs/figures/error_types.png
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.experiment import ROOT, load_partition
from pathsnap.features import prepare_trip
from pathsnap.io import list_trips, load_trip

S, PS = load_partition("student_test")
lab_feat = S[S.y == 1][["trip_id", "obs", "head_diff_w5", "low_motion"]]
nets = {t: prepare_trip(load_trip("student_test", t)).net for t in list_trips("student_test")}
ORDER = ["LogisticRegression", "GaussianNB", "KNN", "RandomForest", "ExtraTrees", "HistGradientBoosting",
         "LightGBM", "XGBoost", "MLP"]
rows = []
for name in ORDER:
    f = os.path.join(ROOT, "outputs", "models", "per_model", f"{name}_student_test_point_preds.csv")
    if not os.path.exists(f):
        continue
    pr = pd.read_csv(f, dtype={"trip_id": str}).merge(PS[["trip_id", "obs", "lab_arc", "covered"]], on=["trip_id", "obs"])
    pr = pr.merge(lab_feat, on=["trip_id", "obs"], how="left")
    cnt = {"reverse_twin": 0, "adjacent": 0, "missed": 0, "other": 0}
    twin_gps_opposes_ref = 0
    twin_by_trip = {}
    for tid, g in pr.groupby("trip_id"):
        net = nets[tid]
        a, b, cov = g.pred_arc.to_numpy(), g.lab_arc.to_numpy(), g.covered.to_numpy()
        wrong = a != b
        missed = wrong & ~cov
        ok = wrong & cov & (a >= 0)
        twin = ok & (net.u[np.maximum(a, 0)] == net.v[b]) & (net.v[np.maximum(a, 0)] == net.u[b])
        ua, va, ub, vb = net.u[np.maximum(a, 0)], net.v[np.maximum(a, 0)], net.u[b], net.v[b]
        adj = ok & ~twin & ((ua == ub) | (ua == vb) | (va == ub) | (va == vb))
        opp = twin & (g.head_diff_w5.to_numpy() > 90) & (g.low_motion.to_numpy() == 0)
        twin_gps_opposes_ref += int(opp.sum())
        twin_by_trip[tid] = int(twin.sum())
        cnt["reverse_twin"] += int(twin.sum()); cnt["adjacent"] += int(adj.sum())
        cnt["missed"] += int(missed.sum()); cnt["other"] += int((wrong & ~missed & ~twin & ~adj).sum())
    n = len(pr)
    rows.append({"model": name, "n_obs": n, "errors": sum(cnt.values()), **cnt,
                 **{f"{k}_share_of_obs": v / n for k, v in cnt.items()},
                 "twin_errors_gps_opposes_reference": twin_gps_opposes_ref,
                 "twin_errors_by_trip": ";".join(f"{t}:{c}" for t, c in twin_by_trip.items() if c)})
E = pd.DataFrame(rows)
E.to_csv(os.path.join(ROOT, "outputs", "models", "error_types_student_test.csv"), index=False)
print(E.to_string())

fig, ax = plt.subplots(figsize=(9, 3.6))
bottom = np.zeros(len(E))
for k, c in zip(["reverse_twin", "adjacent", "other", "missed"], ["tab:red", "tab:orange", "tab:purple", "0.4"]):
    v = E[f"{k}_share_of_obs"].to_numpy() * 100
    ax.bar(E.model, v, bottom=bottom, label=k.replace("_", " "), color=c)
    bottom += v
ax.set_ylabel("% of student-test observations"); ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("Top-1 error types per model (student-test, classifier only)")
ax.tick_params(axis="x", rotation=25)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "outputs", "figures", "error_types.png"), dpi=130); plt.close(fig)
