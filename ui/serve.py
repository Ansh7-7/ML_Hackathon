"""Serve the evaluator UI (Laptop 2) - static files only, standard library.

    python ui/serve.py --port 8501 --api http://<LAPTOP1_IP>:8000

then open http://localhost:8501 on Laptop 2. --api writes ui/config.js so the
page starts with the right API address (it can also be changed in the page).
"""
import argparse
import functools
import http.server
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8501)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--api", default=None, help="API base URL, e.g. http://192.168.1.20:8000")
a = ap.parse_args()
if a.api:
    with open(os.path.join(HERE, "config.js"), "w") as f:
        f.write("// written by ui/serve.py --api\nwindow.PATHSNAP_API_URL = %r;\n" % a.api.rstrip("/"))
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=HERE)
print(f"PathSnap UI on http://{a.host}:{a.port}  (API: {a.api or 'from config.js / same host:8000'})")
http.server.ThreadingHTTPServer((a.host, a.port), handler).serve_forever()
