// PathSnap evaluator UI (Laptop 2). Talks only to the FastAPI backend:
//   GET /health, GET /trips, POST /predict {trip_id}
// The UI never receives reference routes or labels - only predictions and server-side scores.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const COLORS = { gps: "#1f77b4", hmm: "#d62728", clf: "#ff7f0e", start: "#00c853", end: "#212121" };

  function defaultApi() {
    const q = new URLSearchParams(location.search).get("api");
    if (q) return q;
    if (location.pathname.startsWith("/ui")) return location.origin;  // UI served by the API itself
    if (window.PATHSNAP_API_URL) return window.PATHSNAP_API_URL;
    return `${location.protocol === "https:" ? "https" : "http"}://${location.hostname || "localhost"}:8000`;
  }
  const api = () => $("apiUrl").value.replace(/\/+$/, "");

  function logLine(msg) {
    const t = new Date().toLocaleTimeString();
    $("log").textContent += `[${t}] ${msg}\n`;
    $("log").scrollTop = $("log").scrollHeight;
  }

  // ------------------------------------------------------------------ map
  const map = L.map("map", { preferCanvas: true }).setView([48.5, 9.5], 6);
  const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
  tiles.on("tileerror", () => { if (!tiles._warned) { tiles._warned = true; logLine("basemap tiles unavailable (offline) - overlays still shown"); } });
  const layers = { gps: L.layerGroup(), hmm: L.layerGroup(), clf: L.layerGroup(), ends: L.layerGroup() };
  layers.hmm.addTo(map); layers.gps.addTo(map); layers.ends.addTo(map);
  L.control.layers({ "OpenStreetMap": tiles }, {
    "GPS observations": layers.gps, "Predicted route (classifier + HMM)": layers.hmm,
    "Classifier-only route": layers.clf, "Start / end": layers.ends }, { collapsed: false }).addTo(map);
  L.control.scale().addTo(map);
  const legend = L.control({ position: "bottomright" });
  legend.onAdd = () => {
    const d = L.DomUtil.create("div", "legend");
    d.innerHTML = `<b>Legend</b><br>
      <i class="dot" style="background:${COLORS.gps}"></i>GPS observation<br>
      <i style="border-color:${COLORS.hmm}"></i>Predicted route (HMM)<br>
      <i class="dash" style="border-color:${COLORS.clf}"></i>Classifier-only route<br>
      <i class="dot" style="background:${COLORS.start}"></i>start&nbsp; <i class="dot" style="background:${COLORS.end}"></i>end`;
    return d;
  };
  legend.addTo(map);

  // ------------------------------------------------------------------ API calls
  async function getJSON(path, opts) {
    const r = await fetch(api() + path, opts);
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  }

  async function connect() {
    $("apiStatus").textContent = "connecting..."; $("apiStatus").className = "pill";
    try {
      const h = await getJSON("/health");
      $("apiStatus").textContent = `connected - ${h.model}`; $("apiStatus").className = "pill ok";
      logLine(`GET /health -> ok (model ${h.model}, ${h.n_trips} trips)`);
      const t = await getJSON("/trips");
      logLine(`GET /trips -> ${t.trips.length} trip ids`);
      const box = $("tripList"); box.innerHTML = "";
      t.trips.forEach((id) => {
        const b = document.createElement("button");
        b.textContent = id; b.dataset.trip = id;
        b.onclick = () => runTrip(id);
        box.appendChild(b);
      });
      if (!t.trips.length) box.innerHTML = "<em>The API has no trips in EVAL_DIR.</em>";
    } catch (e) {
      $("apiStatus").textContent = "connection failed"; $("apiStatus").className = "pill err";
      logLine(`connection error: ${e.message}`);
    }
  }

  async function runTrip(id) {
    document.querySelectorAll(".trips button").forEach((b) => { b.classList.toggle("active", b.dataset.trip === id); b.disabled = true; });
    $("busy").classList.remove("hidden");
    const t0 = performance.now();
    try {
      logLine(`POST /predict {"trip_id": "${id}"}`);
      const r = await getJSON("/predict", { method: "POST", headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ trip_id: id }) });
      const rt = performance.now() - t0;
      logLine(`  <- ${r.n_points} points, ${r.predicted_route.length} route polyline(s), ${(rt / 1000).toFixed(2)} s round trip`);
      draw(r); showMetrics(r, rt);
    } catch (e) {
      logLine(`  predict failed: ${e.message}`);
    } finally {
      $("busy").classList.add("hidden");
      document.querySelectorAll(".trips button").forEach((b) => (b.disabled = false));
    }
  }

  function draw(r) {
    Object.values(layers).forEach((l) => l.clearLayers());
    const pts = r.gps_points;
    pts.forEach((p, i) => L.circleMarker(p, { radius: 2.5, color: COLORS.gps, weight: 0, fillOpacity: 0.85 })
      .bindTooltip(`obs ${i} - segment ${r.predicted_segments[i]}`).addTo(layers.gps));
    r.predicted_route.forEach((ln) => L.polyline(ln, { color: COLORS.hmm, weight: 4, opacity: 0.9 }).addTo(layers.hmm));
    r.classifier_route.forEach((ln) => L.polyline(ln, { color: COLORS.clf, weight: 2.5, opacity: 0.9, dashArray: "6 5" }).addTo(layers.clf));
    L.circleMarker(pts[0], { radius: 7, color: "#000", weight: 1, fillColor: COLORS.start, fillOpacity: 1 }).bindTooltip("start").addTo(layers.ends);
    L.circleMarker(pts[pts.length - 1], { radius: 7, color: "#000", weight: 1, fillColor: COLORS.end, fillOpacity: 1 }).bindTooltip("end").addTo(layers.ends);
    map.fitBounds(L.latLngBounds(pts), { padding: [20, 20] });
  }

  const pct = (v) => (v === null || v === undefined ? "n/a" : (100 * v).toFixed(2) + " %");
  function showMetrics(r, rt) {
    $("tripLabel").textContent = `- trip ${r.trip_id}`;
    const m = r.metrics;
    let html = `<div class="kv"><span>GPS points</span><b>${r.n_points}</b><span>Model</span><b>${r.model} + HMM/Viterbi</b>
      <span>HMM chain breaks</span><b>${r.hmm_breaks.length}</b></div>`;
    if (m) {
      const rows = [["Top-1 accuracy (per point)", "top1_accuracy"], ["Route continuity", "route_continuity"],
                    ["Route precision (length)", "route_precision"], ["Route recall (length)", "route_recall"],
                    ["Route F1 (length)", "route_f1"]];
      html += `<table><tr><th>metric (server-side)</th><th>HMM</th><th>classifier only</th></tr>` +
        rows.map(([lab, k]) => {
          const a = m.hmm[k], b = m.classifier[k];
          return `<tr><td>${lab}</td><td class="${a >= b ? "best" : ""}">${pct(a)}</td><td class="${b > a ? "best" : ""}">${pct(b)}</td></tr>`;
        }).join("") + `<tr><td>Candidate miss rate</td><td colspan="2">${pct(m.candidate_miss_rate)}</td></tr></table>`;
    } else {
      html += "<p><em>No reference route on the server for this trip - metrics unavailable.</em></p>";
    }
    $("metrics").innerHTML = html;
    const t = r.timings_ms;
    $("timings").innerHTML = `<div class="kv">` + Object.entries(t).map(([k, v]) => `<span>${k}</span><b>${v.toFixed(0)} ms</b>`).join("") +
      `<span>round trip (UI)</span><b>${rt.toFixed(0)} ms</b></div>`;
  }

  $("apiUrl").value = defaultApi();
  $("connectBtn").onclick = connect;
  $("apiUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
  connect();
})();
