"""Step 7 - interactive Folium maps (+ PNG screenshots for the report).

EDA maps (partition: train): GPS observations, nearby road network and the
reference route for
  * train 00000010 - a typical trip
  * train 00000033 - one of the two trips excluded from training (GPS does not
    follow the reference route)

Prediction maps (partition: student_test), produced with the DEPLOYED pipeline
(artifacts/model.joblib + hmm_config.json via pathsnap.pipeline.predict_trip,
the same code the API runs). Trips are picked automatically from
outputs/hmm/per_trip.csv: the student-test trip where the HMM helped most,
the one where it hurt most, and the trip with the median HMM Top-1. For each
one there is a full-extent map and a zoomed map centred on the 60-observation
window where classifier-only and HMM disagree most.

Layers: GPS points, nearby network, reference route, classifier-only route,
HMM route; distinct colours, legend, LayerControl, fit_bounds.

Usage: python scripts/07_maps.py [--offline-assets] [--no-screenshots]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from pathsnap.evaluate import fill_route
from pathsnap.experiment import ROOT
from pathsnap.features import prepare_trip
from pathsnap.io import load_trip
from pathsnap.maps import folium_map
from pathsnap.pipeline import Artifacts, predict_trip, score_prediction

ap = argparse.ArgumentParser()
ap.add_argument("--offline-assets", action="store_true", help="serve Leaflet locally for screenshots (no internet)")
ap.add_argument("--no-screenshots", action="store_true")
args = ap.parse_args()

OUT = os.path.join(ROOT, "outputs", "maps")
os.makedirs(OUT, exist_ok=True)
index = []


def shoot(html):
    if args.no_screenshots:
        return None
    from pathsnap.screenshot import screenshot
    png = html[:-5] + ".png"
    screenshot(html, png, width=1300, height=850, wait_ms=2500, offline_assets=args.offline_assets)
    return png


# ------------------------------------------------------------------ EDA maps
for tid, note in [("00000010", "typical train trip"),
                  ("00000033", "excluded from training: GPS far from reference route")]:
    tr = load_trip("train", tid)
    pt = prepare_trip(tr)
    html = os.path.join(OUT, f"eda_train_{tid}.html")
    folium_map(tr.track, pt.net, reference_arcs=tr.route, title=f"train trip {tid} (EDA)", subtitle=note,
               out_html=html)
    index.append({"map": os.path.basename(html), "partition": "train", "trip_id": tid, "purpose": note,
                  "png": os.path.basename(shoot(html) or "")})
    print("saved", html)

# ----------------------------------------------------------- prediction maps
art = Artifacts(os.path.join(ROOT, "artifacts"))
P = pd.read_csv(os.path.join(ROOT, "outputs", "hmm", "per_trip.csv"), dtype={"trip_id": str})
P = P[P.partition == "student_test"].sort_values("delta_top1")
picks = [(P.iloc[-1].trip_id, "largest HMM gain"), (P.iloc[0].trip_id, "largest HMM loss")]
med = P.iloc[(P.top1_hmm - P.top1_hmm.median()).abs().argsort().iloc[0]].trip_id
if med not in [p[0] for p in picks]:
    picks.append((med, "median HMM Top-1"))

for tid, why in picks:
    tr = load_trip("student_test", tid)
    trip_noref = load_trip("student_test", tid, with_route=False)
    pred = predict_trip(trip_noref, art)
    met, lab = score_prediction(pred, tr.route)
    pt = pred["pt"]
    sub = (f"student_test - {why}<br>Top-1: classifier {met['classifier']['top1_accuracy']:.3f}, "
           f"HMM {met['hmm']['top1_accuracy']:.3f}")
    html = os.path.join(OUT, f"student_test_{tid}.html")
    folium_map(tr.track, pt.net, reference_arcs=tr.route, predicted_arcs=pred["hmm_route"],
               classifier_arcs=pred["clf_route"], title=f"student_test trip {tid}", subtitle=sub,
               show_classifier=True, out_html=html)
    index.append({"map": os.path.basename(html), "partition": "student_test", "trip_id": tid, "purpose": why,
                  "top1_classifier": met["classifier"]["top1_accuracy"], "top1_hmm": met["hmm"]["top1_accuracy"],
                  "png": os.path.basename(shoot(html) or "")})
    # zoomed view on the window with most classifier/HMM disagreement
    dis = (pred["clf_arcs"] != pred["hmm_arcs"]).astype(float)
    W = min(60, len(dis))
    c = np.convolve(dis, np.ones(W), mode="valid")
    s = int(np.argmax(c))
    lat, lon = tr.track.lat.to_numpy()[s:s + W], tr.track.lon.to_numpy()[s:s + W]
    pad = 0.0015
    bounds = [[float(lat.min() - pad), float(lon.min() - pad)], [float(lat.max() + pad), float(lon.max() + pad)]]
    wtop_c = float(np.mean(pred["clf_arcs"][s:s + W] == lab[s:s + W]))
    wtop_h = float(np.mean(pred["hmm_arcs"][s:s + W] == lab[s:s + W]))
    html = os.path.join(OUT, f"student_test_{tid}_zoom.html")
    folium_map(tr.track, pt.net, reference_arcs=tr.route, predicted_arcs=pred["hmm_route"],
               classifier_arcs=pred["clf_route"], title=f"student_test trip {tid} - zoom obs {s}-{s + W - 1}",
               subtitle=f"window Top-1: classifier {wtop_c:.2f}, HMM {wtop_h:.2f}", show_classifier=True,
               bounds=bounds, out_html=html)
    index.append({"map": os.path.basename(html), "partition": "student_test", "trip_id": tid,
                  "purpose": f"{why} - zoom on obs {s}-{s + W - 1} (max classifier/HMM disagreement)",
                  "top1_classifier": wtop_c, "top1_hmm": wtop_h, "png": os.path.basename(shoot(html) or "")})
    print("saved", html, met["hmm"]["top1_accuracy"])

pd.DataFrame(index).to_csv(os.path.join(OUT, "maps_index.csv"), index=False)
print(pd.DataFrame(index).to_string())
