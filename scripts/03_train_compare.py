"""Step 3 - train and compare candidate classifiers.

Partitions
  * cross-validation: TRAIN only (78 trips; 33 and 54 excluded), 5-fold
    GroupKFold by trip_id. Every fold is fitted on every 3rd observation of the
    training trips (TRAIN_STRIDE) and evaluated on ALL observations of the
    held-out trips.
  * model comparison: each model is refitted on all 78 train trips (stride 3)
    and evaluated on the 10 STUDENT_TEST trips.

Primary selection metric: mean CV observation-level Top-1 accuracy.
  Why: the deliverable is one road segment per GPS observation, and Top-1 is
  exactly that decision; it does not depend on the arbitrary 0.5 threshold that
  drives candidate-level Macro F1. Ties (within one fold standard deviation)
  are broken by CV log loss, because the HMM consumes the probabilities as
  emissions and needs them to be well calibrated, then by inference time.

Outputs -> outputs/models/, outputs/figures/
"""
import argparse
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

from pathsnap.evaluate import confusion
from pathsnap.experiment import (ROOT, SEED, TRAIN_STRIDE, continuity_for, load_partition, score, stride_mask,
                                 trip_folds, xy)
from pathsnap.features import prepare_trip
from pathsnap.io import list_trips, load_trip
from pathsnap.models import ZOO, fit_model, make_model, predict_pos

OUT = os.path.join(ROOT, "outputs", "models")
FIG = os.path.join(ROOT, "outputs", "figures")
PRED = os.path.join(ROOT, "data", "processed", "preds")
for d in (OUT, FIG, PRED, os.path.join(OUT, "per_model")):
    os.makedirs(d, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=list(ZOO))
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
np.random.seed(SEED)

D, PT = load_partition("train")
S, PS = load_partition("student_test")
X, y, g = xy(D)
Xs, ys, _ = xy(S)
sm = stride_mask(D)
folds = trip_folds(D.trip_id)
json.dump([{"fold": k, "train_trips": list(a), "val_trips": list(b)} for k, (a, b) in enumerate(folds)],
          open(os.path.join(OUT, "cv_folds_trips.json"), "w"), indent=1)
print(f"train rows {len(D):,} (fit rows with stride {TRAIN_STRIDE}: {sm.sum():,}), "
      f"student_test rows {len(S):,}, points {len(PS):,}")

prepared_st = {tid: prepare_trip(load_trip("student_test", tid)) for tid in list_trips("student_test")}

for name in args.models:
    res_path = os.path.join(OUT, "per_model", f"{name}.json")
    if os.path.exists(res_path) and not args.force:
        print(f"skip {name} (exists)")
        continue
    family = ZOO[name][0]
    fold_rows, oof = [], np.full(len(D), np.nan)
    histories = []
    for k, (tr_trips, va_trips) in enumerate(folds):
        tr = np.isin(g, tr_trips) & sm
        va = np.isin(g, va_trips)
        model, params = make_model(name)
        t0 = time.time()
        fit_model(name, model, X[tr], y[tr], g[tr])
        t_fit = time.time() - t0
        t0 = time.time()
        p_va = predict_pos(model, X[va])
        t_pred = time.time() - t0
        oof[va] = p_va
        Dva = D[va].reset_index(drop=True)
        m_va, _, _ = score(Dva, p_va, PT[PT.trip_id.isin(va_trips)])
        # in-sample (training-fold) metrics on up to 300k of the rows the model was fitted on
        tr_idx = np.where(tr)[0]
        if len(tr_idx) > 300_000:
            tr_idx = np.sort(np.random.RandomState(SEED).choice(tr_idx, 300_000, replace=False))
        p_tr = predict_pos(model, X[tr_idx])
        from pathsnap.evaluate import candidate_metrics
        m_tr = candidate_metrics(y[tr_idx], p_tr)
        row = {"model": name, "fold": k, "fit_s": t_fit, "predict_s": t_pred, "n_fit_rows": int(tr.sum()),
               **{f"val_{a}": b for a, b in m_va.items()},
               **{f"train_{a}": b for a, b in m_tr.items()}}
        if name == "MLP":
            h = model.history_.copy(); h["fold"] = k; histories.append(h)
            row["best_epoch"] = int(model.best_epoch_)
        fold_rows.append(row)
        print(f"[{name}] fold {k}: val top1={m_va['top1']:.4f} macroF1={m_va['macro_f1']:.4f} "
              f"logloss={m_va['log_loss']:.4f} | train macroF1={m_tr['macro_f1']:.4f} | fit {t_fit:.0f}s", flush=True)
    F = pd.DataFrame(fold_rows)

    # ---- final fit on all 78 train trips, evaluate on student_test
    model, params = make_model(name)
    t0 = time.time()
    fit_model(name, model, X[sm], y[sm], g[sm])
    t_fit_full = time.time() - t0
    t0 = time.time()
    p_st = predict_pos(model, Xs)
    t_inf = time.time() - t0
    m_st, per_trip, preds = score(S, p_st, PS)
    cont = continuity_for(preds, prepared_st)
    per_trip["continuity"] = per_trip.trip_id.map(cont)
    cont_all = float(np.average(per_trip.continuity, weights=per_trip.n_obs))
    cm = confusion(ys, p_st)
    np.save(os.path.join(PRED, f"{name}_student_test.npy"), p_st)
    np.save(os.path.join(PRED, f"{name}_oof.npy"), oof)
    preds.to_csv(os.path.join(OUT, "per_model", f"{name}_student_test_point_preds.csv"), index=False)
    per_trip.to_csv(os.path.join(OUT, "per_model", f"{name}_student_test_per_trip.csv"), index=False)
    F.to_csv(os.path.join(OUT, "per_model", f"{name}_cv_folds.csv"), index=False)
    if name == "MLP":
        H = pd.concat(histories, ignore_index=True)
        hf = model.history_.copy(); hf["fold"] = "final"
        pd.concat([H, hf], ignore_index=True).to_csv(os.path.join(OUT, "mlp_history.csv"), index=False)
    pjson = {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}
    result = {
        "model": name, "family": family, "params": pjson, "seed": SEED, "train_stride": TRAIN_STRIDE,
        "fit_row_cap": ZOO[name][3],
        "cv_top1_mean": float(F.val_top1.mean()), "cv_top1_std": float(F.val_top1.std()),
        "cv_macro_f1_mean": float(F.val_macro_f1.mean()), "cv_macro_f1_std": float(F.val_macro_f1.std()),
        "cv_log_loss_mean": float(F.val_log_loss.mean()), "cv_log_loss_std": float(F.val_log_loss.std()),
        "cv_precision_pos": float(F.val_precision_pos.mean()), "cv_recall_pos": float(F.val_recall_pos.mean()),
        "train_macro_f1_mean": float(F.train_macro_f1.mean()), "train_log_loss_mean": float(F.train_log_loss.mean()),
        "cv_fit_s_mean": float(F.fit_s.mean()),
        "st_top1": m_st["top1"], "st_macro_f1": m_st["macro_f1"], "st_log_loss": m_st["log_loss"],
        "st_accuracy": m_st["accuracy"], "st_precision_pos": m_st["precision_pos"], "st_recall_pos": m_st["recall_pos"],
        "st_f1_pos": m_st["f1_pos"], "st_precision_macro": m_st["precision_macro"],
        "st_recall_macro": m_st["recall_macro"], "st_miss_rate": m_st["miss_rate"],
        "st_route_continuity": cont_all,
        "st_confusion_matrix": cm.tolist(),
        "final_fit_s": t_fit_full, "final_fit_rows": int(min(sm.sum(), ZOO[name][3] or 10**12)),
        "st_inference_s": t_inf, "st_rows": int(len(S)), "st_points": int(len(PS)),
        "st_inference_ms_per_1000_obs": 1000 * t_inf / len(PS) * 1000,
    }
    if name == "MLP":
        result["final_best_epoch"] = int(model.best_epoch_)
        result["cv_best_epochs"] = F.best_epoch.tolist()
    json.dump(result, open(res_path, "w"), indent=2)
    print(f"[{name}] CV top1={result['cv_top1_mean']:.4f}±{result['cv_top1_std']:.4f} "
          f"CV macroF1={result['cv_macro_f1_mean']:.4f} | ST top1={m_st['top1']:.4f} "
          f"macroF1={m_st['macro_f1']:.4f} logloss={m_st['log_loss']:.4f} cont={cont_all:.4f} "
          f"| fit {t_fit_full:.0f}s inf {t_inf:.2f}s", flush=True)
