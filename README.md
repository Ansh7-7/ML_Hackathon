# PathSnap: map-matching and route reconstruction

Work in progress. See **PLAN.md** for the requirements checklist, findings and remaining stages.

```
pip install -r requirements.txt
export PATHSNAP_DATA="/path/to/Student folder"   # or place it under data/raw/
python scripts/01_eda.py
```

Folders: `src/pathsnap/` library code · `scripts/` pipeline steps · `outputs/` results and plots · `artifacts/` exported models and configs · `api/` FastAPI backend (Laptop 1) · `ui/` evaluator UI (Laptop 2) · `report/` technical report
