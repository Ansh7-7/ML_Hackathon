"""PathSnap FastAPI backend (Laptop 1).

Run:   uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables
  EVAL_DIR       folder that contains one sub-folder per evaluation trip
                 (<EVAL_DIR>/<trip_id>/<trip_id>.track|.nodes|.arcs[|.route]).
                 Default: data/raw/student_test (development stand-in for the
                 evaluator trips, which are not part of the dataset we were given).
  ARTIFACTS_DIR  default: artifacts/  (model.joblib, feature_config.json, hmm_config.json)
  API_LOG_FILE   optional path; request log is also appended there.

Endpoints
  GET  /health   service status + loaded model
  GET  /trips    {"trips": [trip ids]}  - ids only, no data, no labels
  POST /predict  {"trip_id": "..."} -> runs the full pipeline (network + candidates +
                 features + classifier + HMM/Viterbi) and returns the predicted segment
                 per GPS point, the predicted route as lat/lon polylines, the GPS
                 points and server-side metrics. The reference route is read ONLY
                 after the prediction is complete, to compute the metrics, and is
                 never included in the response.
"""
from __future__ import annotations

import logging
import os
import re
import socket
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.io import load_trip
from pathsnap.maps import _chain
from pathsnap.pipeline import Artifacts, predict_trip, score_prediction

EVAL_DIR = os.path.abspath(os.environ.get("EVAL_DIR", os.path.join(ROOT, "data", "raw", "student_test")))
ART_DIR = os.path.abspath(os.environ.get("ARTIFACTS_DIR", os.path.join(ROOT, "artifacts")))

log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
_h = logging.StreamHandler(sys.stdout); _h.setFormatter(_fmt); log.addHandler(_h)
if os.environ.get("API_LOG_FILE"):
    _f = logging.FileHandler(os.environ["API_LOG_FILE"]); _f.setFormatter(_fmt); log.addHandler(_f)
log.propagate = False

ART = Artifacts(ART_DIR)
TRIP_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def list_eval_trips():
    if not os.path.isdir(EVAL_DIR):
        return []
    return sorted(d for d in os.listdir(EVAL_DIR)
                  if TRIP_RE.match(d) and os.path.isfile(os.path.join(EVAL_DIR, d, f"{d}.track")))


def _startup():
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "?"
    log.info("=" * 78)
    log.info("PathSnap API starting  host=%s  ip=%s", socket.gethostname(), ip)
    log.info("model=%s (tuned=%s)  artifacts=%s", ART.model_name, ART.tuned, ART_DIR)
    log.info("feature config: radius=%.0f m, k_max=%d | HMM: beta=%g, w=%g, top_k=%d, back_tol=%g m",
             ART.fcfg.radius_m, ART.fcfg.k_max, ART.hcfg.beta, ART.hcfg.emission_weight, ART.hcfg.top_k,
             ART.hcfg.back_tol_m)
    log.info("EVAL_DIR=%s  (%d trips)", EVAL_DIR, len(list_eval_trips()))
    log.info("=" * 78)


@asynccontextmanager
async def lifespan(_app):
    _startup()
    yield
    log.info("PathSnap API shutting down")


app = FastAPI(title="PathSnap map-matching API", version="1.0", lifespan=lifespan)
# the UI runs on a different laptop / origin
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if request.method != "OPTIONS":
        log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
                 request.client.host if request.client else "?", response.status_code,
                 1000 * (time.perf_counter() - t0))
    return response


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": ART.model_name, "tuned": ART.tuned, "eval_dir": EVAL_DIR,
            "n_trips": len(list_eval_trips())}


@app.get("/trips")
def trips():
    return {"trips": list_eval_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if not TRIP_RE.match(tid) or tid not in list_eval_trips():
        log.warning("predict: unknown trip_id %r", tid)
        raise HTTPException(status_code=404, detail=f"unknown trip_id {tid!r}")
    t_all = time.perf_counter()
    root, part = os.path.dirname(EVAL_DIR), os.path.basename(EVAL_DIR)
    log.info("predict trip=%s: loading GPS track + road network (reference route NOT loaded)", tid)
    t0 = time.perf_counter()
    trip = load_trip(part, tid, root=root, with_route=False)
    t_load = 1000 * (time.perf_counter() - t0)
    pred = predict_trip(trip, ART)
    pt = pred["pt"]
    tm = {"load_ms": t_load, **pred["timings"]}
    log.info("predict trip=%s: %d GPS points, %d arcs, %d candidate rows, %d HMM breaks | net %.0f ms, "
             "features %.0f ms, classifier %.0f ms, HMM %.0f ms", tid, pred["n_points"], pt.net.n_arcs,
             pred["n_candidate_rows"], len(pred["hmm_breaks"]), tm["network_ms"], tm["features_ms"],
             tm["classifier_ms"], tm["hmm_ms"])
    # ---- server-side scoring (after prediction)
    t0 = time.perf_counter()
    ref_path = os.path.join(EVAL_DIR, tid, f"{tid}.route")
    metrics = None
    if os.path.isfile(ref_path):
        route = load_trip(part, tid, root=root, with_route=True).route
        metrics, _ = score_prediction(pred, route)
    tm["scoring_ms"] = 1000 * (time.perf_counter() - t0)
    tm["total_ms"] = 1000 * (time.perf_counter() - t_all)
    if metrics:
        h, c = metrics["hmm"], metrics["classifier"]
        log.info("predict trip=%s: SCORE top1 HMM=%.4f (classifier %.4f) | continuity %.4f | route P=%.4f R=%.4f "
                 "| miss=%.4f | total %.0f ms", tid, h["top1_accuracy"], c["top1_accuracy"], h["route_continuity"],
                 h["route_precision"], h["route_recall"], metrics["candidate_miss_rate"], tm["total_ms"])
    else:
        log.info("predict trip=%s: no reference route on server -> metrics not available | total %.0f ms",
                 tid, tm["total_ms"])
    lat, lon = trip.track["lat"].to_numpy(), trip.track["lon"].to_numpy()
    r6 = lambda lines: [[[round(a, 6), round(b, 6)] for a, b in ln] for ln in lines]
    return {
        "trip_id": tid,
        "n_points": pred["n_points"],
        "model": ART.model_name,
        "gps_points": [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lat, lon)],
        "timestamps": trip.track["t"].round(3).tolist(),
        "predicted_segments": pred["hmm_arcs"].astype(int).tolist(),
        "classifier_segments": pred["clf_arcs"].astype(int).tolist(),
        "predicted_route": r6(_chain(pt.net, pred["hmm_route"])),
        "classifier_route": r6(_chain(pt.net, pred["clf_route"])),
        "hmm_breaks": [{"obs": int(i), "reason": r} for i, r in pred["hmm_breaks"]],
        "metrics": metrics,
        "timings_ms": {k: round(v, 1) for k, v in tm.items()},
    }
