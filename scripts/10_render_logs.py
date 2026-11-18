"""Render captured terminal logs (outputs/logs/*.log) as terminal-style PNG images
for slides / the viva. The images show the real captured text of each run; they
are renderings, not photos of a terminal window.

    python scripts/10_render_logs.py            # default set of training logs
    python scripts/10_render_logs.py LOG [LOG ...]
Output: outputs/logs/terminal_<log name>.png
"""
import html
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathsnap.experiment import ROOT
from pathsnap.screenshot import launch

LOGS = os.path.join(ROOT, "outputs", "logs")
DEFAULT = [("03_train_compare_GaussianNB_LogReg_rerun.log", "python scripts/03_train_compare.py --models GaussianNB LogisticRegression (re-run)"),
           ("03_train_compare_rest.log", "python scripts/03_train_compare.py --models HistGradientBoosting LightGBM XGBoost ExtraTrees RandomForest MLP KNN"),
           ("03b_plot_compare.log", "python scripts/03b_plot_compare.py"),
           ("04_bayes_opt.log", "python scripts/04_bayes_opt.py --trials 40"),
           ("05_hmm_eval.log", "python scripts/05_hmm_eval.py")]
items = [(a, f"python ... {a}") for a in sys.argv[1:]] or DEFAULT

from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = launch(p)
    for name, cmd in items:
        path = os.path.join(LOGS, name)
        lines = [l for l in open(path).read().splitlines() if l.strip() and "Warning" not in l and "warnings.warn" not in l
                 and not l.lstrip().startswith("ax = plot_")]
        if name == "05_hmm_eval.log":   # 180 grid lines + 88 cache lines: keep the head of each block and all results
            keep = [l for l in lines if not l.startswith("[cache]") and not l.startswith("{'back_tol_m'")]
            lines = lines[:3] + ["... (88 trips cached) ..."] + [l for l in lines if l.startswith("{'back_tol_m'")][:5] + \
                ["... (180 decoder configurations in total, all in outputs/hmm/grid.csv) ..."] + keep
        body = "\n".join(html.escape(l[:230]) for l in lines)
        doc = ("<html><body style='margin:0;background:#0c0c0c'><div style='font:12px/1.35 Consolas,ui-monospace,monospace;"
               "color:#cccccc;padding:10px 12px;white-space:pre'>"
               f"<div style='color:#8ab4f8'>$ {html.escape(cmd)}   (captured output: outputs/logs/{name})</div>{body}</div></body></html>")
        tmp = os.path.join(LOGS, "_render.html")
        open(tmp, "w").write(doc)
        pg = b.new_page(viewport={"width": 1500, "height": 300})
        pg.goto("file://" + tmp)
        out = os.path.join(LOGS, "terminal_" + name.replace(".log", ".png"))
        pg.screenshot(path=out, full_page=True)
        pg.close()
        os.remove(tmp)
        print("saved", out)
    b.close()
