"""API contract tests: every trip in EVAL_DIR predicts, IDs only from /trips,
no reference data in responses.   Run: pytest -q tests/test_api.py
(needs artifacts/ and data/raw/student_test, or EVAL_DIR)."""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app, list_eval_trips  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_trips_ids_only():
    r = client.get("/trips").json()
    assert set(r) == {"trips"} and all(isinstance(t, str) for t in r["trips"])
    assert r["trips"] == list_eval_trips() and len(r["trips"]) > 0


def test_unknown_trip_404():
    assert client.post("/predict", json={"trip_id": "../../etc"}).status_code == 404
    assert client.post("/predict", json={"trip_id": "nope"}).status_code == 404


def test_predict_all_trips():
    for tid in list_eval_trips():
        r = client.post("/predict", json={"trip_id": tid})
        assert r.status_code == 200, tid
        j = r.json()
        assert len(j["predicted_segments"]) == len(j["gps_points"]) == j["n_points"]
        assert len(j["predicted_route"]) >= 1
        forbidden = {"route", "reference", "reference_route", "labels", "lab_arc"}
        assert not forbidden & set(j), tid
        if j["metrics"] is not None:
            assert 0 <= j["metrics"]["hmm"]["top1_accuracy"] <= 1
