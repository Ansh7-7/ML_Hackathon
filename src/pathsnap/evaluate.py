"""Evaluation metrics shared by the training scripts, the HMM evaluation and the API.

Definitions (used consistently everywhere in the report):

* candidate-level metrics - binary classification of each (observation,
  candidate) row, positive = the candidate is the reference arc; threshold 0.5.
  Macro F1 = unweighted mean of the F1 of the two classes.
* observation-level Top-1 accuracy - for every GPS observation take the
  candidate with the highest score and compare its arc id with the reference
  arc of that observation. Observations with no candidate, or whose reference
  arc is not in the candidate set, count as errors (denominator = ALL
  observations of the evaluated trips).
* candidate miss rate - share of observations whose reference arc is not in
  the candidate set (upper bound on Top-1 = 1 - miss rate).
* route continuity - share of consecutive observation pairs (i, i+1) whose
  predicted arcs form a drivable transition: same arc, or arc b reachable from
  arc a along the directed network within 2 x great-circle distance + 100 m,
  and not an immediate flip onto the reverse twin of the same road.
* route precision / recall - length-weighted overlap between the predicted
  route (set of arcs, optionally gap-filled with shortest paths) and the arcs
  of the reference .route file (direction-aware).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score)

EPS = 1e-7


def candidate_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y, yhat)),
        "precision_pos": float(precision_score(y, yhat, zero_division=0)),
        "recall_pos": float(recall_score(y, yhat, zero_division=0)),
        "f1_pos": float(f1_score(y, yhat, zero_division=0)),
        "precision_macro": float(precision_score(y, yhat, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, yhat, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def confusion(y, p, thr=0.5):
    return confusion_matrix(np.asarray(y).astype(int), (np.asarray(p) >= thr).astype(int), labels=[0, 1])


def top1_arcs(obs, arc, score, n_obs):
    """Highest-scoring candidate arc per observation (-1 when no candidates).
    Also returns the row index of the chosen candidate (-1 if none)."""
    obs = np.asarray(obs)
    score = np.asarray(score, float)
    order = np.lexsort((-score, obs))
    first = np.r_[True, obs[order][1:] != obs[order][:-1]]
    rows = order[first]
    pred = np.full(n_obs, -1, dtype=np.int64)
    pred_row = np.full(n_obs, -1, dtype=np.int64)
    pred[obs[rows]] = np.asarray(arc)[rows]
    pred_row[obs[rows]] = rows
    return pred, pred_row


def top1_accuracy(pred_arc, lab_arc):
    return float(np.mean(np.asarray(pred_arc) == np.asarray(lab_arc)))


def transition_ok(net, a, ta, b, tb, gc):
    """Is the move arc a (pos ta) -> arc b (pos tb) drivable within 2*gc+100 m?"""
    if a < 0 or b < 0:
        return False
    if a == b:
        return True
    if net.u[a] == net.v[b] and net.v[a] == net.u[b]:  # flip onto reverse twin
        return False
    return bool(np.isfinite(net.route_distance(int(a), float(ta), int(b), float(tb), 2 * gc + 100.0)))


def route_continuity(net, pred_arc, pred_t, px, py):
    """Share of consecutive observation pairs with a drivable transition.
    Pairs where either observation has no prediction are counted as breaks."""
    n = len(pred_arc)
    if n < 2:
        return 1.0
    ok = 0
    for i in range(n - 1):
        gc = float(np.hypot(px[i + 1] - px[i], py[i + 1] - py[i]))
        ok += transition_ok(net, pred_arc[i], pred_t[i], pred_arc[i + 1], pred_t[i + 1], gc)
    return ok / (n - 1)


def dedupe(arcs):
    arcs = np.asarray(arcs)
    arcs = arcs[arcs >= 0]
    if len(arcs) == 0:
        return arcs
    return arcs[np.r_[True, arcs[1:] != arcs[:-1]]]


def fill_route(net, arcs, max_fill_m=1000.0):
    """Join consecutive distinct arcs with shortest paths (<= max_fill_m).
    Returns the gap-filled arc sequence and the number of unfillable breaks."""
    seq = dedupe(arcs)
    if len(seq) == 0:
        return seq, 0
    out, breaks = [int(seq[0])], 0
    for a, b in zip(seq[:-1], seq[1:]):
        p = net.path_between(int(a), int(b), max_fill_m)
        if p is None:
            breaks += 1
        else:
            out.extend(p)
        out.append(int(b))
    return np.asarray(out, dtype=np.int64), breaks


def route_pr(net, pred_arcs, ref_arcs):
    """Length-weighted and count-based precision / recall of predicted vs reference arc sets."""
    P = set(int(a) for a in np.asarray(pred_arcs) if a >= 0)
    R = set(int(a) for a in np.asarray(ref_arcs))
    inter = P & R
    L = net.length
    lp = sum(L[a] for a in P)
    lr = sum(L[a] for a in R)
    li = sum(L[a] for a in inter)
    out = {
        "route_precision_len": float(li / lp) if lp > 0 else 0.0,
        "route_recall_len": float(li / lr) if lr > 0 else 0.0,
        "route_precision_cnt": len(inter) / len(P) if P else 0.0,
        "route_recall_cnt": len(inter) / len(R) if R else 0.0,
    }
    p, r = out["route_precision_len"], out["route_recall_len"]
    out["route_f1_len"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return out
