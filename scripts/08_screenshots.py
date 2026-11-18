"""Step 8 - end-to-end demo run + screenshots for the report (section 7).

1. starts the FastAPI service (uvicorn api.main:app --host 0.0.0.0 --port 8000) as a
   subprocess; its console output is captured verbatim to
   outputs/screenshots/api_console.log
2. starts the UI static server (ui/serve.py, port 8501)
3. drives the UI in headless Chromium: connects, then clicks EVERY trip listed
   by GET /trips, waits for the prediction, and saves a screenshot per trip
4. writes outputs/screenshots/ui_run_results.csv (status and the scores the UI
   displayed, per trip) and renders the captured API console log to PNG
   (outputs/screenshots/api_console.png). The PNG is a rendering of the real
   captured text, not a photo of a terminal window.

Usage: python scripts/08_screenshots.py [--offline-assets] [--eval-dir DIR]
"""
import argparse
import html
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import requests

from pathsnap.experiment import ROOT
from pathsnap.screenshot import _route_offline, launch

ap = argparse.ArgumentParser()
ap.add_argument("--offline-assets", action="store_true")
ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "raw", "student_test"))
ap.add_argument("--api-port", type=int, default=8000)
ap.add_argument("--ui-port", type=int, default=8501)
args = ap.parse_args()

OUT = os.path.join(ROOT, "outputs", "screenshots")
os.makedirs(OUT, exist_ok=True)
api_log = os.path.join(OUT, "api_console.log")
env = dict(os.environ, EVAL_DIR=args.eval_dir, PYTHONUNBUFFERED="1")
api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0",
                        "--port", str(args.api_port)], cwd=ROOT, env=env,
                       stdout=open(api_log, "w"), stderr=subprocess.STDOUT)
ui = subprocess.Popen([sys.executable, os.path.join(ROOT, "ui", "serve.py"), "--port", str(args.ui_port)],
                      cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
API = f"http://127.0.0.1:{args.api_port}"
try:
    for _ in range(120):
        try:
            if requests.get(API + "/health", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(1)
    else:
        raise RuntimeError("API did not start; see " + api_log)
    trips = requests.get(API + "/trips").json()["trips"]
    print("API up, trips:", trips)

    from playwright.sync_api import sync_playwright
    rows = []
    with sync_playwright() as p:
        b = launch(p)
        page = b.new_context(viewport={"width": 1500, "height": 900}).new_page()
        if args.offline_assets:
            page.route("**/*", _route_offline)
        page.goto(f"http://127.0.0.1:{args.ui_port}/?api={API}")
        page.wait_for_selector(".trips button", timeout=30000)
        page.screenshot(path=os.path.join(OUT, "ui_00_trip_list.png"))
        for k, tid in enumerate(trips, 1):
            t0 = time.time()
            n_before = page.locator("#log").inner_text().count("<-")
            page.click(f".trips button[data-trip='{tid}']")
            ok = True
            try:
                page.wait_for_function(f"document.getElementById('log').textContent.split('<-').length - 1 > {n_before}",
                                       timeout=300000)
            except Exception:
                ok = False
            page.wait_for_timeout(800)
            png = os.path.join(OUT, f"ui_{k:02d}_trip_{tid}.png")
            page.screenshot(path=png)
            cells = page.locator("#metrics table tr").all_inner_texts()
            log_txt = page.locator("#log").inner_text()
            failed = "predict failed" in log_txt.split(f'"{tid}"')[-1]
            rows.append({"trip_id": tid, "ui_success": ok and not failed, "seconds": round(time.time() - t0, 2),
                         "metrics_rows": " | ".join(c.replace("\t", " ") for c in cells), "screenshot": os.path.basename(png)})
            print(rows[-1], flush=True)
        b.close()
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ui_run_results.csv"), index=False)
finally:
    time.sleep(1)
    api.terminate(); ui.terminate()
    api.wait(timeout=10); ui.wait(timeout=10)

# render the captured console log as a terminal-style image
log_text = open(api_log).read()
lines = [l for l in log_text.splitlines() if l.strip()]
page_html = ("<html><body style='margin:0;background:#0c0c0c'><div style='font:12px/1.35 Consolas,ui-monospace,monospace;"
             "color:#cccccc;padding:10px 12px;white-space:pre'>"
             f"<div style='color:#8ab4f8'>Laptop 1 console - uvicorn api.main:app --host 0.0.0.0 --port {args.api_port}"
             " (captured stdout, rendered)</div>" +
             "\n".join(html.escape(l) for l in lines) + "</div></body></html>")
tmp = os.path.join(OUT, "_api_console.html")
open(tmp, "w").write(page_html)
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = launch(p)
    pg = b.new_page(viewport={"width": 1500, "height": 400})
    pg.goto("file://" + tmp)
    pg.screenshot(path=os.path.join(OUT, "api_console.png"), full_page=True)
    b.close()
os.remove(tmp)
print("saved", OUT)
