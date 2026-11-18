"""PNG screenshots of HTML pages (Folium maps, the UI) with headless Chromium (Playwright).

If the machine has no internet access, pass offline_assets=True: requests for
Leaflet on a CDN are answered from ui/vendor/leaflet (committed), other CDN
assets (bootstrap, jquery, font-awesome) get an empty response, and basemap
tiles are not loaded (the map then has a plain background; all data layers
are drawn by Leaflet itself, so they are unaffected).

Set PATHSNAP_CHROMIUM to a Chromium binary if Playwright's own browser is not
installed (`playwright install chromium`).
"""
from __future__ import annotations

import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
VENDOR = os.path.join(ROOT, "ui", "vendor", "leaflet")


def _route_offline(route):
    url = route.request.url
    if url.startswith("file://") or "127.0.0.1" in url or "localhost" in url:
        return route.continue_()
    if "leaflet@" in url and "/dist/" in url:
        fname = url.split("/dist/")[-1].split("?")[0]
        path = os.path.join(VENDOR, fname)
        if os.path.isfile(path):
            ctype = "application/javascript" if fname.endswith(".js") else (
                "text/css" if fname.endswith(".css") else "image/png")
            return route.fulfill(path=path, content_type=ctype)
    if url.endswith(".js") or url.endswith(".css"):
        return route.fulfill(status=200, body="", content_type="text/plain")
    return route.abort()


def launch(p):
    exe = os.environ.get("PATHSNAP_CHROMIUM")
    return p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()


def screenshot(url_or_path, png, width=1400, height=900, wait_ms=1500, offline_assets=False, scale=1):
    from playwright.sync_api import sync_playwright
    url = url_or_path if "://" in url_or_path else "file://" + os.path.abspath(url_or_path)
    with sync_playwright() as p:
        b = launch(p)
        ctx = b.new_context(viewport={"width": width, "height": height}, device_scale_factor=scale)
        page = ctx.new_page()
        if offline_assets:
            page.route("**/*", _route_offline)
        page.goto(url, wait_until="load")
        page.wait_for_timeout(wait_ms)
        page.screenshot(path=png)
        b.close()
    return png
