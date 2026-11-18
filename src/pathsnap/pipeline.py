"""End-to-end inference for one trip + server-side scoring.

predict_trip() never reads the reference route: it uses only the GPS track and
the trip's road network (nodes/arcs). score_prediction() is called afterwards,
on the server only, to compare the prediction with the reference route.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np

from .evaluate import fill_route, route_continuity, route_pr, top1_arcs
from .features import FEATURES, FeatureConfig, build_features, prepare_trip, reference_labels
from .hmm import HMMConfig, decode_trip


def load_model_bundle(art_dir):
    """Load artifacts/model.joblib; if the pickle cannot be read on this platform
    (pickled XGBoost boosters are not always portable, e.g. Linux -> Windows),
    fall back to the portable XGBoost JSON export model_xgb.json + model_meta.json,
    which contains the same booster and gives identical predictions."""
    try:
        return joblib.load(os.path.join(art_dir, "model.joblib"))
    except Exception as e:  # noqa: BLE001
        jpath, mpath = os.path.join(art_dir, "model_xgb.json"), os.path.join(art_dir, "model_meta.json")
        if not (os.path.isfile(jpath) and os.path.isfile(mpath)):
            raise
        from xgboost import XGBClassifier
        meta = json.load(open(mpath))
        model = XGBClassifier()
        model.load_model(jpath)
        print(f"[pathsnap] model.joblib could not be loaded ({type(e).__name__}: {e}); "
              f"using the portable export {os.path.basename(jpath)}", flush=True)
        return {**meta, "model": model}


class Artifacts:
    def __init__(self, art_dir):
        bundle = load_model_bundle(art_dir)
        self.model = bundle["model"]
        self.model_name = bundle["model_name"]
        self.tuned = bundle.get("tuned")
        self.fcfg = FeatureConfig.from_dict(json.load(open(os.path.join(art_dir, "feature_config.json"))))
        self.hcfg = HMMConfig.from_dict(json.load(open(os.path.join(art_dir, "hmm_config.json"))))
        assert list(bundle["features"]) == list(FEATURES), "feature list of model and code differ"


def predict_trip(trip, art: Artifacts):
    """trip: io.Trip (its .route is ignored). Returns predictions + timings."""
    tm = {}
    t0 = time.perf_counter()
    pt = prepare_trip(trip)
    tm["network_ms"] = 1000 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    df = build_features(pt, art.fcfg)
    tm["features_ms"] = 1000 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    X = df[FEATURES].to_numpy(np.float32)
    prob = art.model.predict_proba(X)[:, 1] if len(X) else np.empty(0)
    tm["classifier_ms"] = 1000 * (time.perf_counter() - t0)
    n = len(pt.px)
    obs, arc, tp = df["obs"].to_numpy(), df["arc"].to_numpy(), df["t_proj"].to_numpy()
    clf_arc, clf_row = top1_arcs(obs, arc, prob, n)
    clf_t = np.where(clf_row >= 0, tp[np.maximum(clf_row, 0)], 0.0)
    t0 = time.perf_counter()
    dec = decode_trip(pt, obs, arc, tp, prob, art.hcfg)
    tm["hmm_ms"] = 1000 * (time.perf_counter() - t0)
    clf_route, _ = fill_route(pt.net, clf_arc, art.hcfg.max_fill_m)
    return {"pt": pt, "n_points": n, "n_candidate_rows": int(len(df)),
            "hmm_arcs": dec["arcs"], "hmm_t": dec["t"], "hmm_route": dec["route"], "hmm_breaks": dec["breaks"],
            "clf_arcs": clf_arc, "clf_t": clf_t, "clf_route": clf_route,
            "n_no_candidates": dec["n_no_candidates"], "cand_obs": obs, "cand_arc": arc, "timings": tm}


def score_prediction(pred, route):
    """Server-side scoring against the reference route (labels derived by the
    monotone alignment, exactly as during development)."""
    pt = pred["pt"]
    lab, _ = reference_labels(pt, route)
    out = {}
    for tag, arcs, ts, rt in [("hmm", pred["hmm_arcs"], pred["hmm_t"], pred["hmm_route"]),
                              ("classifier", pred["clf_arcs"], pred["clf_t"], pred["clf_route"])]:
        pr = route_pr(pt.net, rt, route)
        out[tag] = {"top1_accuracy": float(np.mean(arcs == lab)),
                    "route_continuity": route_continuity(pt.net, arcs, ts, pt.px, pt.py),
                    "route_precision": pr["route_precision_len"], "route_recall": pr["route_recall_len"],
                    "route_f1": pr["route_f1_len"]}
    hit = np.zeros(len(lab), bool)
    hit[pred["cand_obs"][pred["cand_arc"] == lab[pred["cand_obs"]]]] = True
    out["candidate_miss_rate"] = float(1 - hit.mean())
    return out, lab
