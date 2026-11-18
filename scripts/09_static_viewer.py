"""Step 9 - single-file interactive viewer of the deployed pipeline's results.

Runs the DEPLOYED pipeline (artifacts/ via pathsnap.pipeline.predict_trip +
score_prediction - the same code as POST /predict) on every trip in EVAL_DIR
(default data/raw/student_test) and writes ONE self-contained HTML file
(Leaflet inlined from ui/vendor) with a trip list, a map (OSM basemap when
online; GPS points, HMM route, classifier-only route, layer control, legend)
and the server-side metrics. It is a snapshot for viewing without Python; the
live system is the API + UI.

    python scripts/09_static_viewer.py  ->  outputs/maps/pathsnap_viewer.html
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathsnap.experiment import ROOT
from pathsnap.io import load_trip
from pathsnap.maps import _chain
from pathsnap.pipeline import Artifacts, predict_trip, score_prediction

EVAL_DIR = os.path.abspath(os.environ.get("EVAL_DIR", os.path.join(ROOT, "data", "raw", "student_test")))
art = Artifacts(os.path.join(ROOT, "artifacts"))
root, part = os.path.dirname(EVAL_DIR), os.path.basename(EVAL_DIR)
trips = sorted(d for d in os.listdir(EVAL_DIR) if os.path.isfile(os.path.join(EVAL_DIR, d, f"{d}.track")))
data = {}
for tid in trips:
    t0 = time.time()
    tr = load_trip(part, tid, root=root, with_route=False)
    pred = predict_trip(tr, art)
    met = None
    if os.path.isfile(os.path.join(EVAL_DIR, tid, f"{tid}.route")):
        met, _ = score_prediction(pred, load_trip(part, tid, root=root).route)
    r5 = lambda lines: [[[round(a, 6), round(b, 6)] for a, b in ln] for ln in lines]
    data[tid] = {"gps": [[round(float(a), 6), round(float(b), 6)] for a, b in zip(tr.track.lat, tr.track.lon)],
                 "hmm": r5(_chain(pred["pt"].net, pred["hmm_route"])),
                 "clf": r5(_chain(pred["pt"].net, pred["clf_route"])),
                 "metrics": met, "breaks": len(pred["hmm_breaks"]), "seconds": round(time.time() - t0, 2)}
    print(tid, "ok", data[tid]["seconds"], "s", flush=True)

vend = os.path.join(ROOT, "ui", "vendor", "leaflet")
css = open(os.path.join(vend, "leaflet.css")).read()
js = open(os.path.join(vend, "leaflet.js")).read()
page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PathSnap Viewer</title>
<style>%CSS%</style><style>
:root{--line:#dde2e8;--muted:#5f6b7a;--accent:#1f6feb}
*{box-sizing:border-box}html,body{margin:0;height:100%;font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#1d2433;background:#f5f6f8}
header{padding:10px 16px;background:#fff;border-bottom:1px solid var(--line)}header h1{font-size:18px;margin:0;display:inline}
header span{color:var(--muted);font-size:12px;margin-left:10px}
main{display:flex;height:calc(100% - 46px)}aside{width:360px;min-width:280px;overflow-y:auto;padding:10px 12px;background:#fff;border-right:1px solid var(--line)}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:12px 0 6px}
.trips{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}.trips button{padding:6px 8px;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer;font:13px ui-monospace,monospace;text-align:left}
.trips button.active{background:var(--accent);color:#fff;border-color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:4px 6px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
#map{flex:1;background:#e8ecef}.legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.25);font-size:12px;line-height:1.6}
.legend i{display:inline-block;width:22px;height:0;border-top:4px solid;vertical-align:middle;margin-right:6px}.legend i.dot{width:9px;height:9px;border:0;border-radius:50%;margin:0 12px 0 6px}
.legend i.dash{border-top-style:dashed}.note{font-size:11px;color:var(--muted)}
@media(max-width:800px){main{flex-direction:column;height:auto}aside{width:100%}#map{height:70vh}}
</style></head><body>
<header><h1>PathSnap viewer</h1><span>tuned XGBoost + HMM/Viterbi &middot; deployed pipeline &middot; snapshot of the API's results</span></header>
<main><aside><h2>Trips</h2><div id="trips" class="trips"></div><h2>Results <span id="lab"></span></h2><div id="metrics"></div>
<p class="note">Scores are computed server-side against the reference route, exactly as POST /predict does. The basemap needs internet.</p></aside>
<div id="map"></div></main>
<script>%JS%</script><script>
const DATA=%DATA%;const C={gps:"#1f77b4",hmm:"#d62728",clf:"#ff7f0e"};
const map=L.map("map",{preferCanvas:true});
const osm=L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{maxZoom:19,attribution:"&copy; OpenStreetMap contributors"}).addTo(map);
const g={gps:L.layerGroup().addTo(map),hmm:L.layerGroup().addTo(map),clf:L.layerGroup(),ends:L.layerGroup().addTo(map)};
L.control.layers({"OpenStreetMap":osm},{"GPS observations":g.gps,"Predicted route (classifier + HMM)":g.hmm,"Classifier-only route":g.clf,"Start / end":g.ends},{collapsed:false}).addTo(map);
L.control.scale().addTo(map);
const lg=L.control({position:"bottomright"});lg.onAdd=()=>{const d=L.DomUtil.create("div","legend");d.innerHTML=`<b>Legend</b><br><i class="dot" style="background:${C.gps}"></i>GPS observation<br><i style="border-color:${C.hmm}"></i>Predicted route (HMM)<br><i class="dash" style="border-color:${C.clf}"></i>Classifier-only route<br><i class="dot" style="background:#00c853"></i>start <i class="dot" style="background:#212121"></i>end`;return d};lg.addTo(map);
const pct=v=>v==null?"n/a":(100*v).toFixed(2)+" %";
function show(id){document.querySelectorAll(".trips button").forEach(b=>b.classList.toggle("active",b.dataset.t===id));
 Object.values(g).forEach(l=>l.clearLayers());const r=DATA[id];
 r.gps.forEach((p,i)=>L.circleMarker(p,{radius:2.5,color:C.gps,weight:0,fillOpacity:.85}).bindTooltip("obs "+i).addTo(g.gps));
 r.hmm.forEach(l=>L.polyline(l,{color:C.hmm,weight:4,opacity:.9}).addTo(g.hmm));
 r.clf.forEach(l=>L.polyline(l,{color:C.clf,weight:2.5,dashArray:"6 5"}).addTo(g.clf));
 L.circleMarker(r.gps[0],{radius:7,color:"#000",weight:1,fillColor:"#00c853",fillOpacity:1}).bindTooltip("start").addTo(g.ends);
 L.circleMarker(r.gps[r.gps.length-1],{radius:7,color:"#000",weight:1,fillColor:"#212121",fillOpacity:1}).bindTooltip("end").addTo(g.ends);
 map.fitBounds(L.latLngBounds(r.gps),{padding:[20,20]});document.getElementById("lab").textContent="- trip "+id;
 const m=r.metrics;let h=`<table><tr><td>GPS points</td><td colspan=2>${r.gps.length}</td></tr><tr><td>HMM chain breaks</td><td colspan=2>${r.breaks}</td></tr>`;
 if(m){h+=`<tr><th>metric</th><th>HMM</th><th>classifier</th></tr>`+[["Top-1 accuracy","top1_accuracy"],["Route continuity","route_continuity"],["Route precision","route_precision"],["Route recall","route_recall"],["Route F1","route_f1"]].map(([a,k])=>`<tr><td>${a}</td><td><b>${pct(m.hmm[k])}</b></td><td>${pct(m.classifier[k])}</td></tr>`).join("")+`<tr><td>Candidate miss rate</td><td colspan=2>${pct(m.candidate_miss_rate)}</td></tr>`}
 document.getElementById("metrics").innerHTML=h+"</table>"}
Object.keys(DATA).forEach(id=>{const b=document.createElement("button");b.textContent=id;b.dataset.t=id;b.onclick=()=>show(id);document.getElementById("trips").appendChild(b)});
show(Object.keys(DATA)[0]);
</script></body></html>"""
page = page.replace("%CSS%", css).replace("%JS%", js).replace("%DATA%", json.dumps(data, separators=(",", ":")))
out = os.path.join(ROOT, "outputs", "maps", "pathsnap_viewer.html")
open(out, "w").write(page)
print("written", out, round(os.path.getsize(out) / 1e6, 2), "MB")
