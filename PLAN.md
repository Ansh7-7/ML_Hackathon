# PathSnap: handover plan for Claude Code

Read this file first. It lists every requirement from the five guideline PDFs (in `../docs/`), what is already built, the findings so far, and the remaining stages.

## 0. Setup

```
pip install -r requirements.txt
# put the dataset at data/raw/  so that data/raw/train/<id>/ and data/raw/student_test/<id>/ exist
#   (unzip ../dataset/Student_folder.zip and move "Student folder/*" into data/raw/)
# or: export PATHSNAP_DATA=/path/to/"Student folder"
python scripts/01_eda.py          # ~1 min, writes outputs/eda/
```

## 1. Deliverables (from Deliverables.pdf)

**D1. Public GitHub repo** with clearly named folders and:
- data processing, feature engineering
- training scripts for every model
- evaluation code (metrics, plots, confusion matrices)
- Bayesian optimisation code, search config and trial results
- HMM and Viterbi code
- FastAPI backend (endpoints and model loading)
- UI source
- deployment config, requirements and README (install, run, deploy)
- Folium map code and plotting scripts
- CSV results, experiment logs and configs
- saved model weights and configs (within size limits)
- no secrets

**D2. One PDF technical report** with these sections:
- **2.2 EDA:** how data was preprocessed; patterns seen; features chosen and why. Include missing-value analysis, GPS trajectory plots, road-network visualisations and at least one Folium map. **State which partition each analysis uses.**
- **3.1 Comparison table** with columns: Model, Algorithm family, Key configuration, CV Macro F1, Top-1 Accuracy, Route continuity, Student-test Macro F1, Log loss, Inference time. Write N/A plus a reason where a metric doesn't apply.
- **3.2 One subsection per model** answering:
  - A1: why it suits the problem
  - A2: strengths and limitations
  - B5: key hyperparameters and initial values
  - B6: loss or objective
  - C7: student-test performance
  - C8: Top-1, Macro F1 and log loss
  - C9: over- or underfitting, with evidence

  Answers must tie to the actual results.
- **3.3 Plots:**
  - CV Macro F1 for all models
  - student-test Top-1 and Macro F1
  - log loss
  - confusion matrices (optional)
  - train/validation loss curves **only** for models that train in epochs (e.g. the MLP); none faked for other models
  - plots that justify the model choice
- **4. Bayesian HPO on the chosen model:**
  - parameters searched, ranges, objective, number of trials, best values
  - tuned vs untuned on CV Macro F1, student-test Top-1, student-test Macro F1, log loss and inference time
- **5. HMM/Viterbi:**
  - hidden states, emission and transition probabilities
  - handling of missing candidates and large GPS gaps
  - consecutive points that land on disconnected segments
  - whether the HMM improved results, with evidence
  - classifier-only vs classifier + HMM comparison, including cases where the HMM didn't help
- **6. Folium maps** showing GPS points, road network, reference route and predicted route:
  - distinct colours, a legend, sensible zoom, and layer controls
  - a screenshot in the PDF
  - the HTML file or the code that generates it committed
- **7. API/UI:** screenshots of the working UI, and of successful API responses **in the CMD of the API laptop**
- **8. Integrity:** an AI-use disclosure, and every team member able to explain everything

## 2. Hard rules (Models, Assessment and Dataset PDFs)

- At least **5 models** from different families. Use **group-aware CV on the 80 train trips** (GroupKFold by trip), and the **10 student_test trips** for comparison.
- Record configs, random seeds, training time and inference time.
- Don't select a model on training accuracy alone. State the **primary selection metric** and why.
- Metrics:
  - accuracy, precision, recall, F1, confusion matrix, log loss
  - **observation-level Top-1** (highest-scoring candidate vs reference segment)
  - **candidate miss rate** (share of points where the reference segment isn't in the candidate set)
  - route continuity, sequence-level performance, training time, inference time
- Fit preprocessing only on training data.
- The evaluator test set is never used for training, feature selection, hyperparameters or thresholds. No hardcoded predictions.
- Reference labels stay **server-side only**. The API scores accuracy after predicting; the UI never sees labels.
- Final demo:
  - Laptop 1 runs the FastAPI service (model, HMM, road network and scoring).
  - Laptop 2 runs the UI, which lists the 10 evaluation trips, sends the chosen trip ID, and shows the route and scores.
  - It must work for all 10 trips, and the API log must be visible.
- **The evaluator trips are not in this dataset.** The API must load trips from a configurable folder (for example `data/eval/`). Use `student_test` as the stand-in during development.
- Viva topics: dataset, preprocessing, features, model comparison, model selection, Bayesian optimisation, HMM, Viterbi, API, UI and results.

## 3. Findings so far (outputs/eda/, partitions train + student_test)

- **Size:** 80 train and 10 student_test trips, 4 files each, all present.
  - GPS points: 208,417 train and 19,160 student_test.
  - Points per trip range from 262 to 19,313 (median about 1,500).
- **Timestamps:** no missing values, duplicate or out-of-order timestamps. About 94% of steps are exactly 1 s apart.
  - 1,216 gaps over 5 s and 133 over 30 s in train.
  - Extreme gaps: trip 90 at 153,064 s and trip 78 at 52,810 s; these trips have multiple sessions.
- **Implausible jumps** (over 60 m/s): 546 in train, concentrated in trips 32, 13 and 1, plus single jumps up to 888 m in trips 55 and 43.
- **The road network is local to each trip.**
  - Nodes per trip: 299 to 155,003. Road segments: 649 to 307,392.
  - About 92% of segments have a reverse twin (two-way roads), so the reverse direction is also a candidate and **heading features matter**.
  - No isolated nodes. The largest strongly connected component holds about 95% of nodes.
  - There are some self-loops and zero-length segments.
- **The route is segment-level only.**
  - It is a chronological list of segment IDs and does **not** say which GPS point belongs to which segment.
  - Per-point labels are derived in `src/pathsnap/labels.py::align_route`: a monotone DP that keeps route order and minimises the sum of point-to-segment distances. This is a key design decision for the report.
  - 14 small route breaks across 11 trips.
  - Median GPS-to-route distance is about 4.7 m.
- **Label outliers:** in trips **33** and **54** the GPS sits far from the reference route (medians 122 m and 138 m). Exclude them from training and say so in the report. Trip 2 also has a far section (95th percentile 242 m).
- **Candidate density** (teammate's EDA): on average about 6.8 segments within 30 m and about 12 within 50 m. About 97% of points are within 30 m of the route.
- The teammate's `eda_summary_teammate.csv` matches `trip_inventory.csv` exactly for point, segment and route counts.

## 4. Already built

| File | What it does |
|---|---|
| `src/pathsnap/io.py` | Trip loader; `PATHSNAP_DATA` sets the data path |
| `src/pathsnap/geo.py` | Local equirectangular projection to metres, haversine, point-to-segment distance, bearings |
| `src/pathsnap/network.py` | `RoadNetwork`: projected segments, KD-tree spatial index, bounded cached Dijkstra, `route_distance()` for HMM transitions, component checks |
| `src/pathsnap/labels.py` | Monotone DP labelling of GPS points to route segments |
| `src/pathsnap/maps.py` | Folium map builder (separate layers, legend, LayerControl, fit_bounds) plus a matplotlib static export of the same layers |
| `scripts/01_eda.py` | Full EDA covering every item in the Dataset PDF section 4; writes CSVs, plots and `data_quality_log.md` |
| `scripts/00_eda_quick_teammate.py`, `scripts/make_folium_map.py` | Teammate's scripts. The map script is fixed to keep segments with either endpoint in the map area |

## 5. Remaining stages (suggested design)

1. **`src/pathsnap/features.py` + `scripts/02_build_dataset.py`**
   - **Candidates:** segments within radius R. Choose R on train by plotting miss rate against R for 20/30/50/75/100 m; R = 50 m is likely, capped at K = 12 nearest. Save a miss-rate plot.
   - **One row per (point, candidate):**
     - distance; distance rank; distance minus the minimum; distance over (minimum + 1)
     - number of candidates
     - heading difference between the GPS heading (centred window of ±2 points; flag if movement is under 2 m) and the segment direction; same at ±5 points
     - speed; dt to the previous and next point
     - segment length; projection position t; distance to the nearer segment end
     - context distances from points i−5, i−2, i+2 and i+5 to the same segment, plus their mean
     - two-way flag; node degrees
   - **Label:** the candidate is the aligned reference segment.
   - Save parquet or npz for each partition.
2. **`scripts/03_train_compare.py`**
   - Models: LogReg, GaussianNB, KNN, RandomForest, ExtraTrees, HistGradientBoosting, MLP (and optionally XGBoost or LightGBM).
   - Pipeline with StandardScaler where needed. 5-fold GroupKFold by trip; seed 42.
   - Subsample training points (e.g. every 2nd or 3rd point, since 1 Hz data is highly autocorrelated) if compute is tight, and document it.
   - **Scores:**
     - candidate-level binary metrics: Macro F1, precision, recall, log loss, confusion matrix
     - point-level Top-1, by taking the highest probability within each point's candidate group
     - candidate miss rate
     - route continuity of the classifier-only path
     - training and inference time
   - Save `outputs/models/comparison.csv`, per-fold CSVs, the MLP `loss_curve_` and validation scores, and the plots.
3. **`scripts/04_bayes_opt.py`**
   - Optuna TPE (Bayesian) on the best model; objective is GroupKFold CV Macro F1 or Top-1 on train only.
   - About 30–50 trials. Save `trials.csv`, an optimisation-history plot and parameter importances.
   - Refit, then run the tuned vs untuned table.
4. **`src/pathsnap/hmm.py` + `scripts/05_hmm_eval.py`**
   - **States:** candidate segments at each point.
   - **Emission:** log of the classifier probability, optionally blended with a Gaussian distance term.
   - **Transition:** `exp(-|route_dist - gc_dist| / beta)` (Newson & Krumm), using `RoadNetwork.route_distance` with cutoff = max(2·gc + 100 m, v_max·dt).
   - **No candidates at a point:** skip it and carry the previous state.
   - **All transitions impossible:** break the chain and restart from emissions, then log the break.
   - **Gaps over a threshold:** relax beta or split the chain.
   - Tune beta and sigma on train and student_test only.
   - Report Top-1, Macro F1, continuity and route-level precision/recall for classifier-only vs HMM, plus per-trip wins and losses.
5. **`artifacts/`:** model `.joblib`, `feature_config.json`, `hmm_config.json`.
6. **`api/main.py` (FastAPI)**
   - `GET /trips`: IDs only, from `EVAL_DIR`.
   - `POST /predict {trip_id}`: runs the full pipeline, then scores against the server-side `.route`. Returns the predicted segment for each point, a polyline, Top-1, continuity and timings.
   - Log every request. Run with `uvicorn --host 0.0.0.0`.
7. **`ui/`:** Streamlit or a plain HTML+JS page with the API URL configurable. Lists the 10 trips, calls the API, and draws GPS points and the predicted route (Folium or Leaflet) with the scores.
8. **`scripts/07_maps.py`:** Folium HTML maps for selected student_test trips, with screenshots for the report.
9. **Report PDF** following section 1 (D2), and a README.

## 6. Contents of this handover zip

- `pathsnap/`: the code so far (this folder)
- `docs/`: the 5 guideline PDFs (Main Guidelines, Dataset Exploration, Models and Training, Deliverables, Assessment and Plagiarism Policy)
- `dataset/Student_folder.zip`: raw data (train + student_test)
- `teammate/`: teammate's original eda.py, make_folium_map.py, eda_summary.csv and trip_plots.png
- `original_uploads/PathSnap_student_materials.zip`: the original zip of the 5 PDFs
