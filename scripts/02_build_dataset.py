"""Step 2 - candidate generation + feature table.

Partitions: the candidate radius R and cap K are chosen on TRAIN only
(78 trips: trips 33 and 54 excluded, see EXCLUDED_TRAIN). The student_test
miss rate is reported afterwards for information; it is not used to choose R/K.

Selection rule (fixed before looking at the curve):
  R = smallest radius in the grid whose uncapped miss rate is within
      0.5 percentage points of the miss rate at the largest radius (100 m);
  K = smallest cap whose miss rate at R is within 0.2 pp of the uncapped
      miss rate at R.

Outputs
  outputs/features/miss_rate_grid.csv, miss_rate_vs_radius.png, candidate_choice.json
  outputs/features/trip_summary.csv           per-trip rows / miss rate / label distance
  data/processed/{train,student_test}.parquet one row per (observation, candidate)
  data/processed/{train,student_test}_points.parquet one row per observation
  artifacts/feature_config.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.features import (FEATURES, FeatureConfig, build_features, label_rows, prepare_trip,
                               ranked_candidates, reference_labels)
from pathsnap.io import list_trips, load_trip

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "outputs", "features")
PROC = os.path.join(ROOT, "data", "processed")
ART = os.path.join(ROOT, "artifacts")
for d in (OUT, PROC, ART):
    os.makedirs(d, exist_ok=True)

# Trips whose GPS does not follow the reference route (EDA: median GPS->route
# distance 122 m and 140 m vs ~4.6 m elsewhere) -> labels unreliable.
EXCLUDED_TRAIN = ["00000033", "00000054"]
RADII = [10, 15, 20, 30, 40, 50, 75, 100]
KS = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32]

# ------------------------------------------------------------- 1. miss-rate curve
t0 = time.time()
recs = []
prepared = {}
for part in ("train", "student_test"):
    for tid in list_trips(part):
        tr = load_trip(part, tid)
        pt = prepare_trip(tr)
        lab_arc, lab_dist = reference_labels(pt, tr.route)
        obs, arc, d, _, rank = ranked_candidates(pt, max(RADII), None)
        hit = arc == lab_arc[obs]
        lab_rank = np.full(len(pt.px), -1)
        lab_rank[obs[hit]] = rank[hit]
        lab_cd = np.full(len(pt.px), np.inf)
        lab_cd[obs[hit]] = d[hit]
        # number of candidates per obs within each radius
        ncand = {r: np.bincount(obs[d <= r], minlength=len(pt.px)) for r in RADII}
        recs.append(pd.DataFrame({"partition": part, "trip_id": tid, "obs": np.arange(len(pt.px)),
                                  "lab_arc": lab_arc, "lab_dist": lab_dist, "lab_rank100": lab_rank,
                                  "lab_cand_dist": lab_cd, **{f"ncand_{r}": ncand[r] for r in RADII}}))
        prepared[(part, tid)] = (tr, pt, lab_arc)
        print(f"[candidates] {part} {tid} n={len(pt.px)}", flush=True)
P = pd.concat(recs, ignore_index=True)
print(f"candidate scan took {time.time() - t0:.0f}s")


def rank_within(Pp, r):
    """Rank of the label among candidates within radius r (ranking by distance is
    nested, so the rank among <=100 m candidates equals the rank among <=r m ones)."""
    return np.where(Pp.lab_cand_dist <= r, Pp.lab_rank100, -1)


rows = []
for part, sel in [("train(78, excl. 33/54)", (P.partition == "train") & ~P.trip_id.isin(EXCLUDED_TRAIN)),
                  ("train(all 80)", P.partition == "train"),
                  ("student_test", P.partition == "student_test")]:
    Pp = P[sel]
    for r in RADII:
        rk = rank_within(Pp, r)
        rows.append({"partition": part, "radius_m": r, "k": "all", "miss_rate": float(np.mean(rk < 0)),
                     "mean_candidates": float(Pp[f"ncand_{r}"].mean())})
        for k in KS:
            rows.append({"partition": part, "radius_m": r, "k": k,
                         "miss_rate": float(np.mean((rk < 0) | (rk >= k))),
                         "mean_candidates": float(np.minimum(Pp[f"ncand_{r}"], k).mean())})
G = pd.DataFrame(rows)
G.to_csv(os.path.join(OUT, "miss_rate_grid.csv"), index=False)

TR = G[G.partition == "train(78, excl. 33/54)"]
unc = TR[TR.k == "all"].set_index("radius_m").miss_rate
floor = unc[max(RADII)]
R = int(min(r for r in RADII if unc[r] <= floor + 0.005))
capped = TR[(TR.radius_m == R) & (TR.k != "all")].copy()
capped["k"] = capped.k.astype(int)
K = int(capped[capped.miss_rate <= unc[R] + 0.002].k.min())
choice = {"partition_used": "train (78 trips, 33 and 54 excluded)", "radius_m": R, "k_max": K,
          "miss_rate_uncapped_at_R": float(unc[R]), "miss_rate_at_100m": float(floor),
          "miss_rate_R_K": float(capped.set_index("k").miss_rate[K]),
          "mean_candidates_R_K": float(capped.set_index("k").mean_candidates[K]),
          "rule": "R: smallest radius within 0.5pp of the 100 m miss rate; K: smallest cap within 0.2pp of uncapped at R"}
ST = G[(G.partition == "student_test") & (G.radius_m == R) & (G.k == K)]
choice["student_test_miss_rate_R_K (info only)"] = float(ST.miss_rate.iloc[0])
json.dump(choice, open(os.path.join(OUT, "candidate_choice.json"), "w"), indent=2)
print(json.dumps(choice, indent=2))

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for part, c in [("train(78, excl. 33/54)", "tab:blue"), ("train(all 80)", "tab:gray")]:
    g = G[(G.partition == part) & (G.k == "all")]
    ax[0].plot(g.radius_m, g.miss_rate * 100, "o-", color=c, label=part)
ax[0].axvline(R, ls="--", color="k", lw=0.8)
ax[0].set_xlabel("candidate radius R [m]"); ax[0].set_ylabel("candidate miss rate [%]")
ax[0].set_title("Miss rate vs radius (no K cap)"); ax[0].legend(); ax[0].grid(alpha=.3)
for r, c in zip([20, 30, 50, 75, 100], plt.cm.viridis(np.linspace(0, 0.9, 5))):
    g = TR[(TR.radius_m == r) & (TR.k != "all")]
    ax[1].plot(g.k.astype(int), g.miss_rate * 100, "o-", ms=3, color=c, label=f"R={r} m")
ax[1].axvline(K, ls="--", color="k", lw=0.8)
ax[1].set_xlabel("top-K cap"); ax[1].set_ylabel("candidate miss rate [%]"); ax[1].set_yscale("log")
ax[1].set_title("Miss rate vs K (train, 78 trips)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
g = TR[TR.k == "all"]
ax[2].plot(g.radius_m, g.mean_candidates, "o-", color="tab:red", label="uncapped")
g = TR[TR.k.astype(str) == str(K)]
ax[2].plot(g.radius_m, g.mean_candidates, "s--", color="tab:purple", label=f"capped K={K}")
ax[2].set_xlabel("candidate radius R [m]"); ax[2].set_ylabel("mean candidates / observation")
ax[2].set_title("Candidate-set size (cost)"); ax[2].legend(); ax[2].grid(alpha=.3)
fig.suptitle(f"Candidate generation on train: chosen R = {R} m, K = {K}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "miss_rate_vs_radius.png"), dpi=140); plt.close(fig)

# ---------------------------------------------------------- 2. feature tables
cfg = FeatureConfig(radius_m=float(R), k_max=K, feature_names=list(FEATURES))
json.dump(cfg.to_dict(), open(os.path.join(ART, "feature_config.json"), "w"), indent=2)

summ = []
for part in ("train", "student_test"):
    dfs, pts = [], []
    t_feat = 0.0
    for tid in list_trips(part):
        tr, pt, lab_arc = prepared[(part, tid)]
        t1 = time.time()
        df = build_features(pt, cfg)
        t_feat += time.time() - t1
        df["y"] = label_rows(df, lab_arc)
        df.insert(0, "trip_id", tid)
        covered = np.zeros(len(pt.px), bool)
        covered[df.obs[df.y == 1].to_numpy()] = True
        Pq = P[(P.partition == part) & (P.trip_id == tid)]
        pts.append(pd.DataFrame({"trip_id": tid, "obs": np.arange(len(pt.px)), "lab_arc": lab_arc,
                                 "lab_dist": Pq.lab_dist.to_numpy(), "covered": covered,
                                 "n_cand": np.bincount(df.obs, minlength=len(pt.px)), "t": pt.t}))
        summ.append({"partition": part, "trip_id": tid, "excluded_from_training": tid in EXCLUDED_TRAIN,
                     "n_obs": len(pt.px), "n_rows": len(df), "mean_cand": len(df) / len(pt.px),
                     "miss_rate": float(1 - covered.mean()), "obs_no_candidates": int((pts[-1].n_cand == 0).sum()),
                     "median_gps_to_ref_m": float(np.median(Pq.lab_dist))})
        dfs.append(df)
    D = pd.concat(dfs, ignore_index=True)
    PT = pd.concat(pts, ignore_index=True)
    D.to_parquet(os.path.join(PROC, f"{part}.parquet"), index=False)
    PT.to_parquet(os.path.join(PROC, f"{part}_points.parquet"), index=False)
    print(f"[features] {part}: rows={len(D):,} points={len(PT):,} positives={int(D.y.sum()):,} "
          f"feature time {t_feat:.1f}s")
S = pd.DataFrame(summ)
S.to_csv(os.path.join(OUT, "trip_summary.csv"), index=False)
agg = S.groupby(["partition", "excluded_from_training"]).agg(
    trips=("trip_id", "count"), obs=("n_obs", "sum"), rows=("n_rows", "sum"),
    obs_no_candidates=("obs_no_candidates", "sum")).reset_index()
agg["miss_rate"] = [float(np.average(S[(S.partition == p) & (S.excluded_from_training == e)].miss_rate,
                                     weights=S[(S.partition == p) & (S.excluded_from_training == e)].n_obs))
                    for p, e in zip(agg.partition, agg.excluded_from_training)]
agg.to_csv(os.path.join(OUT, "partition_feature_summary.csv"), index=False)
print(agg)

# feature distributions (train, 78 trips) positives vs negatives - for the EDA section
D = pd.read_parquet(os.path.join(PROC, "train.parquet"))
D = D[~D.trip_id.isin(EXCLUDED_TRAIN)]
show = ["dist", "dist_minus_min", "head_diff_w2", "ctx_mean", "ctx_mean_minus_min", "dist_rank", "speed", "seg_len"]
fig, axs = plt.subplots(2, 4, figsize=(13, 5.5))
samp = D.sample(400_000, random_state=42) if len(D) > 400_000 else D
for ax, c in zip(axs.ravel(), show):
    for y, col, lab in [(0, "tab:gray", "other candidate"), (1, "tab:green", "reference arc")]:
        v = samp[samp.y == y][c]
        hi = np.nanpercentile(samp[c], 99)
        ax.hist(v.clip(upper=hi), bins=50, density=True, alpha=.55, color=col, label=lab)
    ax.set_title(c, fontsize=9)
axs[0, 0].legend(fontsize=7)
fig.suptitle("Feature distributions: reference arc vs other candidates (train, 78 trips, 400k-row sample)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "feature_distributions.png"), dpi=130); plt.close(fig)
print(f"done in {time.time() - t0:.0f}s")
