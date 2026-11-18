"""Step 5 - HMM / Viterbi route reconstruction: decoder tuning and comparison
with classifier-only predictions.

Emissions come from the final (tuned) classifier:
  * train trips (78): OUT-OF-FOLD probabilities from the 5-fold GroupKFold run
    in step 4 (data/processed/preds/tuned_oof.npy), so the decoder is tuned on
    probabilities that behave like unseen-trip probabilities;
  * student_test trips (10): probabilities of the final model fitted on train.

Decoder parameters (beta, emission weight, top-k, U-turn penalty, same-arc
backward tolerance) are chosen
on TRAIN by grid search (objective: observation-level Top-1). The full grid is
also scored on student_test and saved, so both permitted partitions are
visible; the choice itself uses train only.

Outputs -> outputs/hmm/ : grid.csv, per_trip.csv, summary.csv, breaks.csv,
           hmm_grid_heatmap.png, hmm_per_trip_delta.png, hmm_summary.png
           artifacts/hmm_config.json
"""
import itertools
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from pathsnap.evaluate import fill_route, route_continuity, route_pr, top1_arcs
from pathsnap.experiment import ROOT, load_partition
from pathsnap.features import prepare_trip
from pathsnap.hmm import HMMConfig, build_lattice, precompute_transitions, states_to_arcs, viterbi
from pathsnap.io import load_trip

OUT = os.path.join(ROOT, "outputs", "hmm")
ART = os.path.join(ROOT, "artifacts")
CACHE = os.path.join(ROOT, "data", "processed", "hmm_cache")
for d in (OUT, ART, CACHE):
    os.makedirs(d, exist_ok=True)

KMAX = 8
BACK_TOLS = [5.0, 15.0, 30.0]   # changes route distances -> one transition cache per value
GRID = {"back_tol_m": BACK_TOLS, "beta": [2.0, 5.0, 10.0, 30.0, 100.0], "emission_weight": [0.5, 1.0, 2.0],
        "top_k": [4, 8], "uturn_penalty": [0.0, 3.0]}

parts = {}
for part, pfile in [("train", "tuned_oof.npy"), ("student_test", "tuned_student_test.npy")]:
    D, PT = load_partition(part)
    p = np.load(os.path.join(ROOT, "data", "processed", "preds", pfile))
    assert len(p) == len(D) and np.isfinite(p).all()
    parts[part] = (D, PT, p)

# ------------------------------------------------- lattices + transition caches
t0 = time.time()
trips = []
for part, (D, PT, p) in parts.items():
    for tid, g in D.groupby("trip_id", sort=True):
        tr = load_trip(part, tid)
        pt = prepare_trip(tr)
        n = len(pt.px)
        pp = p[g.index.to_numpy()]
        lattice = build_lattice(g.obs.to_numpy(), g.arc.to_numpy(), g.t_proj.to_numpy(), pp, n, KMAX)
        steps_by_bt = {}
        for bt in BACK_TOLS:
            cpath = os.path.join(CACHE, f"{part}_{tid}_bt{int(bt)}.pkl")
            if os.path.exists(cpath):
                active, steps = pickle.load(open(cpath, "rb"))
            else:
                active, steps = precompute_transitions(pt, lattice, HMMConfig(top_k=KMAX, back_tol_m=bt))
                pickle.dump((active, steps), open(cpath, "wb"))
            steps_by_bt[bt] = steps
        clf_arc, clf_row = top1_arcs(g.obs.to_numpy(), g.arc.to_numpy(), pp, n)
        clf_t = np.where(clf_row >= 0, g.t_proj.to_numpy()[np.maximum(clf_row, 0)], 0.0)
        trips.append({"part": part, "tid": tid, "pt": pt, "route": tr.route, "lab": PT[PT.trip_id == tid].lab_arc.to_numpy(),
                      "g": g, "lattice": lattice, "active": active, "steps": steps_by_bt, "clf_arc": clf_arc, "clf_t": clf_t})
        print(f"[cache] {part} {tid} n={n} {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ grid search
rows = []
for bt, beta, w, k, up in itertools.product(*GRID.values()):
    cfg = HMMConfig(beta=beta, emission_weight=w, top_k=k, uturn_penalty=up, back_tol_m=bt)
    acc = {"train": [0, 0], "student_test": [0, 0]}
    nb = {"train": 0, "student_test": 0}
    for T in trips:
        st, br = viterbi(T["lattice"], T["active"], T["steps"][bt], cfg, k=k)
        arcs, _ = states_to_arcs(T["lattice"], T["active"], st, len(T["lab"]))
        acc[T["part"]][0] += int(np.sum(arcs == T["lab"])); acc[T["part"]][1] += len(T["lab"])
        nb[T["part"]] += len(br)
    rows.append({"back_tol_m": bt, "beta": beta, "emission_weight": w, "top_k": k, "uturn_penalty": up,
                 "train_top1": acc["train"][0] / acc["train"][1],
                 "student_test_top1": acc["student_test"][0] / acc["student_test"][1],
                 "train_breaks": nb["train"], "student_test_breaks": nb["student_test"]})
    print(rows[-1], flush=True)
G = pd.DataFrame(rows).sort_values("train_top1", ascending=False)
G.to_csv(os.path.join(OUT, "grid.csv"), index=False)
best = G.iloc[0]
cfg = HMMConfig(beta=float(best.beta), emission_weight=float(best.emission_weight), top_k=int(best.top_k),
                uturn_penalty=float(best.uturn_penalty), back_tol_m=float(best.back_tol_m))
hmm_cfg = cfg.to_dict()
hmm_cfg.update({"selected_on": "train partition (78 trips, out-of-fold classifier probabilities), objective Top-1",
                "grid": GRID, "train_top1": float(best.train_top1),
                "student_test_top1_at_selected": float(best.student_test_top1)})
json.dump(hmm_cfg, open(os.path.join(ART, "hmm_config.json"), "w"), indent=2)
print("selected", hmm_cfg)

# ------------------------------------------------ classifier-only vs classifier+HMM
per, brk = [], []
for T in trips:
    pt, lab, g = T["pt"], T["lab"], T["g"]
    t1 = time.time()
    st, br = viterbi(T["lattice"], T["active"], T["steps"][cfg.back_tol_m], cfg, k=cfg.top_k)
    arcs, ts = states_to_arcs(T["lattice"], T["active"], st, len(lab))
    t_dec = time.time() - t1
    for (i, reason) in br:
        brk.append({"partition": T["part"], "trip_id": T["tid"], "obs": i, "reason": reason,
                    "dt_s": float(pt.t[i] - pt.t[max(i - 1, 0)])})
    # decision-level Macro F1: exactly one candidate per observation predicted positive
    obs, arc, y = g.obs.to_numpy(), g.arc.to_numpy(), g.y.to_numpy()
    f1_clf = f1_score(y, (arc == T["clf_arc"][obs]).astype(int), average="macro")
    f1_hmm = f1_score(y, (arc == arcs[obs]).astype(int), average="macro")
    r = {"partition": T["part"], "trip_id": T["tid"], "n_obs": len(lab),
         "top1_clf": float(np.mean(T["clf_arc"] == lab)), "top1_hmm": float(np.mean(arcs == lab)),
         "macro_f1_decision_clf": f1_clf, "macro_f1_decision_hmm": f1_hmm,
         "continuity_clf": route_continuity(pt.net, T["clf_arc"], T["clf_t"], pt.px, pt.py),
         "continuity_hmm": route_continuity(pt.net, arcs, ts, pt.px, pt.py),
         "breaks_hmm": len(br), "decode_s": t_dec}
    for tag, a in [("clf", T["clf_arc"]), ("hmm", arcs)]:
        raw = route_pr(pt.net, a, T["route"])
        filled, unf = fill_route(pt.net, a, cfg.max_fill_m)
        fil = route_pr(pt.net, filled, T["route"])
        r.update({f"route_precision_raw_{tag}": raw["route_precision_len"], f"route_recall_raw_{tag}": raw["route_recall_len"],
                  f"route_precision_{tag}": fil["route_precision_len"], f"route_recall_{tag}": fil["route_recall_len"],
                  f"route_f1_{tag}": fil["route_f1_len"], f"unfillable_gaps_{tag}": unf})
    r["delta_top1"] = r["top1_hmm"] - r["top1_clf"]
    per.append(r)
    print(f"[{T['part']} {T['tid']}] top1 clf {r['top1_clf']:.4f} -> hmm {r['top1_hmm']:.4f} | cont "
          f"{r['continuity_clf']:.4f} -> {r['continuity_hmm']:.4f} | routeF1 {r['route_f1_clf']:.4f} -> "
          f"{r['route_f1_hmm']:.4f} | breaks {len(br)}", flush=True)
P = pd.DataFrame(per)
P.to_csv(os.path.join(OUT, "per_trip.csv"), index=False)
pd.DataFrame(brk, columns=["partition", "trip_id", "obs", "reason", "dt_s"]).to_csv(os.path.join(OUT, "breaks.csv"), index=False)

summ = []
for part in ("train", "student_test"):
    q = P[P.partition == part]
    wts = q.n_obs
    s = {"partition": part, "trips": len(q), "observations": int(wts.sum())}
    for c in ["top1", "macro_f1_decision", "continuity", "route_precision", "route_recall", "route_f1",
              "route_precision_raw", "route_recall_raw"]:
        for tag in ("clf", "hmm"):
            col = f"{c}_{tag}"
            s[col] = float(np.average(q[col], weights=wts)) if c in ("top1", "macro_f1_decision", "continuity") \
                else float(q[col].mean())
    s["trips_improved"] = int((q.delta_top1 > 0.001).sum())
    s["trips_worse"] = int((q.delta_top1 < -0.001).sum())
    s["trips_same"] = int(len(q) - s["trips_improved"] - s["trips_worse"])
    s["breaks_hmm"] = int(q.breaks_hmm.sum())
    s["decode_s_total"] = float(q.decode_s.sum())
    summ.append(s)
SU = pd.DataFrame(summ)
SU.to_csv(os.path.join(OUT, "summary.csv"), index=False)
print(SU.T.to_string())

# ------------------------------------------------------------------ plots
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, part in zip(axs, ["train", "student_test"]):
    sub = G[(G.top_k == cfg.top_k) & (G.uturn_penalty == cfg.uturn_penalty) & (G.back_tol_m == cfg.back_tol_m)]
    H = sub.pivot(index="emission_weight", columns="beta", values=f"{part}_top1")
    im = ax.imshow(H.values, cmap="viridis", aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H.values[i, j]:.4f}", ha="center", va="center", fontsize=7,
                    color="w" if H.values[i, j] < np.nanmean(H.values) else "k")
    ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns); ax.set_xlabel("beta [m]")
    ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index); ax.set_ylabel("emission weight w")
    ax.set_title(f"HMM Top-1 on {part} (top_k={cfg.top_k}, U-turn pen.={cfg.uturn_penalty}, back_tol={cfg.back_tol_m:g} m)", fontsize=8)
    plt.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hmm_grid_heatmap.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 3]})
for ax, part in zip(axs, ["student_test", "train"]):
    q = P[P.partition == part].sort_values("delta_top1")
    cols = ["tab:red" if d < -0.001 else ("tab:green" if d > 0.001 else "0.6") for d in q.delta_top1]
    ax.bar(range(len(q)), q.delta_top1 * 100, color=cols)
    ax.set_xticks(range(len(q))); ax.set_xticklabels([t[-2:] for t in q.trip_id], fontsize=6, rotation=90)
    ax.axhline(0, c="k", lw=.6); ax.set_ylabel("Top-1 change HMM - classifier [pp]")
    ax.set_title(f"{part}: per-trip effect of HMM decoding (green better, red worse)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hmm_per_trip_delta.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, (c, lab) in zip(axs, [("top1", "Top-1"), ("continuity", "Route continuity"),
                              ("route_precision", "Route precision (len)"), ("route_recall", "Route recall (len)")]):
    xx = np.arange(2)
    ax.bar(xx - .2, SU[f"{c}_clf"], .4, label="classifier only", color="tab:orange")
    ax.bar(xx + .2, SU[f"{c}_hmm"], .4, label="classifier + HMM", color="tab:red")
    for i in range(2):
        ax.text(i - .2, SU[f"{c}_clf"][i], f"{SU[f'{c}_clf'][i]:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + .2, SU[f"{c}_hmm"][i], f"{SU[f'{c}_hmm'][i]:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xx); ax.set_xticklabels(SU.partition); ax.set_ylim(.7, 1.02); ax.set_title(lab, fontsize=9)
axs[0].legend(fontsize=7, loc="lower left")
fig.suptitle("Classifier-only vs classifier + HMM/Viterbi (route P/R use gap-filled routes for both)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hmm_summary.png"), dpi=130); plt.close(fig)
print("done")
