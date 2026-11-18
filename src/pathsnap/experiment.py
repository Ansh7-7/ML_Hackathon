"""Shared experiment plumbing: loading the feature tables, group folds, scoring."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .evaluate import candidate_metrics, route_continuity, top1_arcs
from .features import FEATURES

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PROC = os.path.join(ROOT, "data", "processed")
EXCLUDED_TRAIN = ["00000033", "00000054"]   # GPS does not follow the reference route (see EDA)
SEED = 42
N_FOLDS = 5
TRAIN_STRIDE = 3   # train on every 3rd GPS observation (1 Hz data are strongly autocorrelated)


def load_partition(part):
    D = pd.read_parquet(os.path.join(PROC, f"{part}.parquet"))
    PT = pd.read_parquet(os.path.join(PROC, f"{part}_points.parquet"))
    if part == "train":
        D = D[~D.trip_id.isin(EXCLUDED_TRAIN)].reset_index(drop=True)
        PT = PT[~PT.trip_id.isin(EXCLUDED_TRAIN)].reset_index(drop=True)
    return D, PT


def xy(D):
    return D[FEATURES].to_numpy(np.float32), D["y"].to_numpy(np.int8), D["trip_id"].to_numpy()


def stride_mask(D, stride=TRAIN_STRIDE):
    return (D["obs"].to_numpy() % stride) == 0


def trip_folds(trip_ids, n_folds=N_FOLDS):
    """GroupKFold by trip -> list of (train_trips, val_trips)."""
    trips = np.array(sorted(set(trip_ids)))
    gkf = GroupKFold(n_splits=n_folds)
    return [(trips[a], trips[b]) for a, b in gkf.split(trips, groups=trips)]


def point_top1(D, p, PT):
    """Observation-level Top-1 over ALL observations of the trips in D.
    Returns (top1, per-trip DataFrame, predicted arc per point aligned to PT rows, chosen t_proj)."""
    res, preds, tps = [], [], []
    for tid, g in D.groupby("trip_id", sort=True):
        pts = PT[PT.trip_id == tid]
        n = len(pts)
        pred, prow = top1_arcs(g["obs"].to_numpy(), g["arc"].to_numpy(), p[g.index.to_numpy()], n)
        tp = np.where(prow >= 0, g["t_proj"].to_numpy()[np.maximum(prow, 0)], 0.0)
        lab = pts["lab_arc"].to_numpy()
        res.append({"trip_id": tid, "n_obs": n, "top1": float(np.mean(pred == lab)),
                    "miss_rate": float(1 - pts["covered"].mean())})
        preds.append(pd.DataFrame({"trip_id": tid, "obs": np.arange(n), "pred_arc": pred, "pred_t": tp}))
    R = pd.DataFrame(res)
    top1 = float(np.average(R.top1, weights=R.n_obs))
    return top1, R, pd.concat(preds, ignore_index=True)


def score(D, p, PT):
    """Candidate-level + observation-level metrics for probability vector p (aligned with D rows)."""
    m = candidate_metrics(D["y"].to_numpy(), p)
    top1, per_trip, preds = point_top1(D, p, PT)
    m["top1"] = top1
    m["miss_rate"] = float(1 - PT[PT.trip_id.isin(D.trip_id.unique())]["covered"].mean())
    return m, per_trip, preds


def continuity_for(preds, prepared):
    """Route continuity per trip given point predictions and {trip_id: PreparedTrip}."""
    out = {}
    for tid, g in preds.groupby("trip_id"):
        pt = prepared[tid]
        out[tid] = route_continuity(pt.net, g.pred_arc.to_numpy(), g.pred_t.to_numpy(), pt.px, pt.py)
    return out
