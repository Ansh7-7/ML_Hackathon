"""Step 3d - is the selected model really better than its closest rivals?

Paired comparison of the selected model against every other model:
  * CV (train): per-fold difference in held-out Top-1 (same folds for all models)
  * student_test: per-trip difference in Top-1, number of trips won/lost, and an
    observation-weighted mean difference with a 95% bootstrap CI over trips
    (5000 resamples of the 10 trips, seed 42).
Output: outputs/models/paired_comparison.csv
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from pathsnap.experiment import ROOT, SEED

M = os.path.join(ROOT, "outputs", "models")
chosen = json.load(open(os.path.join(M, "model_selection.json")))["chosen"]
F = pd.read_csv(os.path.join(M, "cv_folds_all.csv")).pivot(index="fold", columns="model", values="val_top1")
per = {m: pd.read_csv(os.path.join(M, "per_model", f"{m}_student_test_per_trip.csv"), dtype={"trip_id": str}).set_index("trip_id")
       for m in F.columns}
rows = []
for other in F.columns:
    if other == chosen:
        continue
    fd = F[chosen] - F[other]
    d = per[chosen].top1 - per[other].top1
    w = per[chosen].n_obs
    rng = np.random.RandomState(SEED)
    bs = []
    for _ in range(5000):
        s = rng.choice(d.index, len(d))
        bs.append(np.average(d.loc[s], weights=w.loc[s]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append({"selected": chosen, "other": other,
                 "cv_folds_selected_better": int((fd > 0).sum()), "cv_mean_diff": float(fd.mean()),
                 "cv_min_diff": float(fd.min()), "cv_max_diff": float(fd.max()),
                 "st_trips_selected_better": int((d > 0).sum()), "st_trips_selected_worse": int((d < 0).sum()),
                 "st_weighted_diff": float(np.average(d, weights=w)), "st_ci_low": float(lo), "st_ci_high": float(hi),
                 "st_ci_excludes_zero": bool(lo > 0 or hi < 0)})
R = pd.DataFrame(rows).sort_values("cv_mean_diff")
R.to_csv(os.path.join(M, "paired_comparison.csv"), index=False)
print(R.round(4).to_string(index=False))
