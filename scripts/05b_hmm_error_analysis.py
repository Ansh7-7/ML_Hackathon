"""Step 5b - what does the HMM fix and what does it break? (train + student_test)

Re-decodes every trip with the selected decoder (artifacts/hmm_config.json) from
the transition caches written by 05_hmm_eval.py, then compares per observation:
  fixed  = classifier wrong, HMM right
  broken = classifier right, HMM wrong
and characterises both sets: error type of the wrong prediction (reverse twin of
the reference arc / adjacent arc / other), whether the vehicle is (nearly)
stationary (GPS displacement over +-2 fixes < 2 m), and whether the observation
lies within 30 fixes after a chain break.
Outputs: outputs/hmm/fix_break_analysis.csv (per partition and per trip)
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from pathsnap.evaluate import top1_arcs
from pathsnap.experiment import ROOT, load_partition
from pathsnap.features import gps_heading, prepare_trip
from pathsnap.hmm import HMMConfig, build_lattice, states_to_arcs, viterbi
from pathsnap.io import load_trip

cfg = HMMConfig.from_dict(json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json"))))
CACHE = os.path.join(ROOT, "data", "processed", "hmm_cache")
KMAX = 8


def kind(net, a, b):
    if a == b:
        return "correct"
    if net.u[a] == net.v[b] and net.v[a] == net.u[b]:
        return "reverse_twin"
    if len({net.u[a], net.v[a]} & {net.u[b], net.v[b]}):
        return "adjacent"
    return "other"


rows = []
for part, pfile in [("train", "tuned_oof.npy"), ("student_test", "tuned_student_test.npy")]:
    D, PT = load_partition(part)
    p = np.load(os.path.join(ROOT, "data", "processed", "preds", pfile))
    for tid, g in D.groupby("trip_id", sort=True):
        pt = prepare_trip(load_trip(part, tid))
        n = len(pt.px)
        pp = p[g.index.to_numpy()]
        lattice = build_lattice(g.obs.to_numpy(), g.arc.to_numpy(), g.t_proj.to_numpy(), pp, n, KMAX)
        active, steps = pickle.load(open(os.path.join(CACHE, f"{part}_{tid}_bt{int(cfg.back_tol_m)}.pkl"), "rb"))
        st, br = viterbi(lattice, active, steps, cfg, k=cfg.top_k)
        hmm, _ = states_to_arcs(lattice, active, st, n)
        clf, _ = top1_arcs(g.obs.to_numpy(), g.arc.to_numpy(), pp, n)
        lab = PT[PT.trip_id == tid].lab_arc.to_numpy()
        _, mv = gps_heading(pt.px, pt.py, 2)
        stationary = mv < 2.0
        near_break = np.zeros(n, bool)
        for i, _r in br:
            near_break[i:i + 30] = True
        net = pt.net
        for name, mask, wrong_arc in [("fixed", (clf != lab) & (hmm == lab), clf), ("broken", (clf == lab) & (hmm != lab), hmm)]:
            idx = np.where(mask)[0]
            kinds = pd.Series([kind(net, wrong_arc[i], lab[i]) if wrong_arc[i] >= 0 else "none" for i in idx], dtype=object)
            rows.append({"partition": part, "trip_id": tid, "set": name, "n": len(idx),
                         "reverse_twin": int((kinds == "reverse_twin").sum()), "adjacent": int((kinds == "adjacent").sum()),
                         "other": int((kinds == "other").sum()), "stationary": int(stationary[idx].sum()),
                         "near_break": int(near_break[idx].sum()), "n_obs": n})
        print(part, tid, flush=True)
R = pd.DataFrame(rows)
agg = R.groupby(["partition", "set"])[["n", "reverse_twin", "adjacent", "other", "stationary", "near_break"]].sum().reset_index()
agg["trip_id"] = "ALL"
out = pd.concat([agg, R], ignore_index=True)
out.to_csv(os.path.join(ROOT, "outputs", "hmm", "fix_break_analysis.csv"), index=False)
print(agg.to_string())
