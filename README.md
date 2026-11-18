# PathSnap: map-matching and route reconstruction

PathSnap matches noisy 1 Hz GPS tracks to a road network and rebuilds the driven route.
For every GPS observation it generates candidate road segments, a classifier scores each
(observation, candidate) pair, and an HMM/Viterbi decoder picks one connected sequence of
segments. A FastAPI service (Laptop 1) runs the whole pipeline and scores the result against
the server-side reference route. An HTML/JS UI (Laptop 2) lists the trips, sends the chosen
trip ID and shows the predicted route and the scores.

The technical report is `report/PathSnap_Technical_Report.pdf`. Every number in it is produced by the
scripts below and stored under `outputs/`.

## 1. Installation

Python 3.11–3.13. Tested on Linux with Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Only screenshots need Playwright's browser (`scripts/07_maps.py`, `scripts/08_screenshots.py`):

```bash
playwright install chromium
```

## 2. Data placement

The raw data is not committed (see `.gitignore`). Unzip `Student_folder.zip` and move the two
partition folders so the layout is:

```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          80 trips
data/raw/student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   10 trips
```

Alternatively, `export PATHSNAP_DATA=/path/to/"Student folder"` (this is read by `src/pathsnap/io.py`).

Evaluator trips are **not** part of this dataset. On demo day, copy them to any folder with the
same per-trip layout (for example `data/eval/<trip_id>/...`) and point the API at it with `EVAL_DIR`
(section 4).

## 3. Reproduce every stage, in order

All scripts use fixed seeds (42). Run from the repository root with the venv active.
Times are from a 4-core laptop-class CPU.

| # | command | what it does | partition(s) | time |
|---|---|---|---|---|
| 1 | `python scripts/01_eda.py` | inventory, missing values, gaps, jumps, network connectivity, route checks, EDA plots | train, student_test | ~1 min |
| 2 | `python scripts/02_build_dataset.py` | candidate miss rate vs radius/K (selection on **train** only), feature tables, `artifacts/feature_config.json` | train (selection), both (tables) | ~1 min |
| 3 | `python scripts/03_train_compare.py` | 9 models, 5-fold GroupKFold by trip on train, refit, evaluate on student_test | train (CV), student_test (comparison) | ~1.5 h (RF, KNN, MLP dominate) |
| 3b | `python scripts/03b_plot_compare.py` | comparison table, model selection, all comparison plots | from step 3 outputs | seconds |
| 3c | `python scripts/03c_error_analysis.py` | Top-1 error types per model (reverse direction / adjacent / other / missed) | student_test | ~1 min |
| 3d | `python scripts/03d_paired_comparison.py` | paired per-fold / per-trip comparison of the selected model vs every other (bootstrap CI) | train (CV), student_test | seconds |
| 4 | `python scripts/04_bayes_opt.py --trials 40` | Optuna TPE on the selected model (objective CV Top-1, train only), tuned vs untuned, `artifacts/model.joblib` | train (search), student_test (after search) | ~30–60 min |
| 4b | `python scripts/04b_hpo_plots.py` | HPO history + importance plots (PED-ANOVA and fANOVA), rebuilt from `trials.csv` without re-running the search | train | seconds |
| 5 | `python scripts/05_hmm_eval.py` | HMM decoder grid (selection on train OOF), classifier-only vs HMM, `artifacts/hmm_config.json` | train, student_test | ~20 min |
| 5b | `python scripts/05b_hmm_error_analysis.py` | what the HMM fixed and broke (error type, stationary, near breaks) | train, student_test | ~2 min |
| 7 | `python scripts/07_maps.py` | Folium HTML maps and PNG screenshots | train (EDA maps), student_test | ~2 min |
| 8 | `python scripts/08_screenshots.py` | starts API and UI, runs every student_test trip, captures UI screenshots and the API console log | student_test | ~2 min |
| 9a | `python scripts/09_static_viewer.py` | single-file interactive viewer of the deployed pipeline's results for every trip in `EVAL_DIR` (`outputs/maps/pathsnap_viewer.html`, opens in any browser without Python) | student_test | ~15 s |
| 9 | `python report/build_report.py` | builds `report/PathSnap_Technical_Report.pdf` from `outputs/` | – | seconds |
| – | `python scripts/check_report_numbers.py` | cross-checks every decimal number in the PDF against `outputs/` (prints its own chance-match rate) | – | seconds |
| – | `pytest -q tests/test_api.py` | API contract: all trips predict, IDs only, no reference data in responses | EVAL_DIR | ~1 min |

Steps 3c, 3d, 4b and 5b are short analysis scripts that read the results of the step before them. `scripts/03_train_compare.py --models XGBoost LightGBM` re-runs selected models only. Results per model
are cached in `outputs/models/per_model/`; use `--force` to recompute.

Without internet access (no CDN or map tiles), add `--offline-assets` to step 7. Leaflet is then
served from `ui/vendor/` and the maps render without a basemap. If Playwright's own Chromium is
not installed, set `PATHSNAP_CHROMIUM=/path/to/chrome`.

## 4. Demo: API on Laptop 1, UI on Laptop 2

### Laptop 1: FastAPI service (model, HMM, road network, scoring)

```bash
source .venv/bin/activate
# folder containing the evaluator trips (default: data/raw/student_test as a stand-in)
export EVAL_DIR=/absolute/path/to/eval_trips        # Windows (cmd): set EVAL_DIR=C:\path\to\eval_trips
export API_LOG_FILE=outputs/logs/api_demo.log        # optional: also write the request log to a file
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `GET /trips` (IDs only)
- `POST /predict` with `{"trip_id": "00000007"}`

Every request is logged to the console: method, path, client IP, status, timings and the
server-side scores. Keep this window visible for the evaluator.

Find Laptop 1's IP address (`ipconfig` on Windows, `ip addr` or `ifconfig` on Linux/macOS), e.g. `192.168.1.20`.

### Laptop 2: evaluator UI (static HTML+JS, no Python packages needed)

```bash
python ui/serve.py --port 8501 --api http://192.168.1.20:8000
```

Open `http://localhost:8501` in a browser. The API URL can also be changed in the page header,
or passed as `http://localhost:8501/?api=http://192.168.1.20:8000`. You can also open `ui/index.html`
directly from disk. The UI lists the trips from `GET /trips`, sends the selected ID to
`POST /predict`, and draws the GPS points, the predicted route (HMM) and the classifier-only
route with a legend, along with the metrics. Leaflet is bundled in `ui/vendor/`, so the UI works
offline. The basemap tiles need internet; without it, only the basemap is missing.

### Evaluator trips

The evaluator trips were run with the frozen system (no retraining or tuning):

```bash
python scripts/08_screenshots.py --eval-dir data/eval --out-dir outputs/evaluator_run
```

The results are in `outputs/evaluator_run/`: UI screenshots per trip, `api_console.log`, `per_trip_scores.csv` and `summary.csv`.
The data itself (`data/eval/`) is not committed.

### Screenshot of the API console (report section 7)

`scripts/08_screenshots.py` captures the API's stdout to `outputs/screenshots/api_console.log` and renders it to a
PNG. On demo day, also take a real screenshot of the CMD/terminal window on Laptop 1 while the UI is used, as the
deliverables ask for it.

### Network and firewall notes

- Both laptops must be on the same network (same Wi-Fi or a phone hotspot). Guest Wi-Fi with
  client isolation blocks laptop-to-laptop traffic.
- Laptop 1 must accept inbound TCP 8000:
  - Windows: allow Python in the "Windows Defender Firewall" prompt on the first run, or
    `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`
  - macOS: System Settings → Network → Firewall → allow incoming connections for Python
  - Linux: `sudo ufw allow 8000/tcp`
- Test from Laptop 2 first: `curl http://192.168.1.20:8000/health` (or open that URL in the browser).
- CORS is open (`allow_origins=["*"]`) because the UI is served from a different host.
- `--host 0.0.0.0` is required. With `127.0.0.1`, uvicorn only accepts local connections.

### Leakage safeguards in the API

- `predict_trip()` loads the trip with `with_route=False`. The reference route is read only
  **after** the prediction is complete, by `score_prediction()`, to compute metrics.
- Responses contain predictions and aggregate scores only, never the reference route or per-point labels.
- No trip ID is hardcoded anywhere: `/trips` lists the folders in `EVAL_DIR`, and every
  `/predict` call runs the full pipeline (road network, candidates, features, classifier, Viterbi).

## 5. Folder map

```
api/main.py                  FastAPI backend (Laptop 1)
ui/                          evaluator UI (Laptop 2): index.html, app.js, style.css, config.js, serve.py, vendor/leaflet
src/pathsnap/
  io.py                      raw-file readers (PATHSNAP_DATA)
  geo.py                     local metric projection, haversine, point-segment distance, bearings
  network.py                 RoadNetwork: arc geometry, KD-tree, bounded Dijkstra, route_distance, path_between
  labels.py                  monotone DP alignment: route (arc list) -> per-observation reference arc
  features.py                candidate generation + per-(observation, candidate) features
  models.py                  model zoo (9 models) + epoch-wise MLP with train/val loss history
  experiment.py              data loading, GroupKFold by trip, scoring helpers
  evaluate.py                metrics: candidate-level, Top-1, miss rate, continuity, route precision/recall
  hmm.py                     HMM / Viterbi decoder
  pipeline.py                end-to-end inference + server-side scoring (shared by the API and the maps)
  maps.py                    Folium map builder (layers, legend, LayerControl, fit_bounds)
  screenshot.py              Playwright screenshots
scripts/                     01_eda ... 08_screenshots: one script per stage (section 3)
artifacts/                   model.joblib, feature_config.json, hmm_config.json (loaded by the API)
outputs/
  eda/                       inventory CSVs, data-quality log, EDA plots
  features/                  miss-rate grid and plot, candidate choice, feature distributions
  models/                    comparison.csv, per-model JSON/CSV (configs, folds, timings), MLP history
  figures/                   comparison plots (CV Macro F1, Top-1/Macro F1, log loss, confusion, MLP curves, ...)
  hpo/                       trials.csv, search space, best params, tuned vs untuned, Optuna plots
  hmm/                       decoder grid, per-trip classifier vs HMM, breaks, plots
  maps/                      Folium HTML maps + PNG screenshots
  screenshots/               UI screenshots, API console log
  logs/                      stdout of every pipeline run
report/                      build_report.py and PathSnap_Technical_Report.pdf
data/raw/                    (not committed) raw dataset
data/processed/              (not committed) feature tables, predictions, HMM caches (rebuilt by steps 2-5)
```

## 6. Notes

- Trips 33 and 54 are excluded from training and CV. Their GPS lies a median 122 m and 140 m from
  the reference route, compared with about 4.6 m elsewhere, and 84% of their points have no
  reference arc within 75 m. The labels are therefore unreliable (see the report, section 2.2).
- No secrets are needed or stored.
