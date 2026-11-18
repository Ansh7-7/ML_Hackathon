"""Folium map generation (interactive HTML) + matplotlib static export of the
same layers (used for the PDF report).

Layers: original GPS observations, nearby road network, reference route
(when available) and predicted route, each in its own toggleable FeatureGroup.
"""
from __future__ import annotations

import numpy as np

COLORS = {"gps": "#1f77b4", "network": "#9e9e9e", "reference": "#2ca02c",
          "predicted": "#d62728", "classifier": "#ff7f0e"}


def _arc_lines(net, arcs):
    lon_a, lat_a = net.proj.inv(net.ax[arcs], net.ay[arcs])
    lon_b, lat_b = net.proj.inv(net.bx[arcs], net.by[arcs])
    return [[(la, lo), (lb, lob)] for la, lo, lb, lob in zip(lat_a, lon_a, lat_b, lon_b)]


def _chain(net, arcs):
    """Join a sequence of arcs into polylines (break where arcs are not contiguous)."""
    lines, cur = [], []
    prev = None
    for a in arcs:
        lon_a, lat_a = net.proj.inv(net.ax[a], net.ay[a])
        lon_b, lat_b = net.proj.inv(net.bx[a], net.by[a])
        if prev is not None and net.v[prev] == net.u[a] and cur:
            cur.append((float(lat_b), float(lon_b)))
        else:
            if len(cur) > 1:
                lines.append(cur)
            cur = [(float(lat_a), float(lon_a)), (float(lat_b), float(lon_b))]
        prev = a
    if len(cur) > 1:
        lines.append(cur)
    return lines


def dedupe_consecutive(arcs):
    arcs = np.asarray(arcs)
    arcs = arcs[arcs >= 0]
    if len(arcs) == 0:
        return arcs
    keep = np.r_[True, arcs[1:] != arcs[:-1]]
    return arcs[keep]


def nearby_arcs(net, px, py, pad=150.0, max_arcs=25000):
    m = ((net.ax > px.min() - pad) & (net.ax < px.max() + pad) &
         (net.ay > py.min() - pad) & (net.ay < py.max() + pad))
    idx = np.where(m)[0]
    if len(idx) > max_arcs:  # keep arcs closest to the track
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([px, py]))
        d, _ = tree.query(np.column_stack([(net.ax[idx] + net.bx[idx]) / 2, (net.ay[idx] + net.by[idx]) / 2]))
        idx = idx[np.argsort(d)[:max_arcs]]
    # drop reverse duplicates for drawing
    seen, out = set(), []
    for a in idx:
        k = (min(net.u[a], net.v[a]), max(net.u[a], net.v[a]))
        if k not in seen:
            seen.add(k); out.append(a)
    return np.array(out, dtype=int)


def folium_map(track, net, reference_arcs=None, predicted_arcs=None, classifier_arcs=None,
               title="PathSnap", gps_every=1, out_html=None):
    import folium

    lat, lon = track["lat"].to_numpy(), track["lon"].to_numpy()
    px, py = net.proj.fwd(lon, lat)
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=13, tiles="OpenStreetMap",
                   control_scale=True)

    fg_net = folium.FeatureGroup(name="Road network (nearby)", show=True)
    for seg in _arc_lines(net, nearby_arcs(net, px, py)):
        folium.PolyLine(seg, color=COLORS["network"], weight=1, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)

    if reference_arcs is not None:
        fg = folium.FeatureGroup(name="Reference route", show=True)
        for line in _chain(net, dedupe_consecutive(reference_arcs)):
            folium.PolyLine(line, color=COLORS["reference"], weight=7, opacity=0.55, tooltip="reference").add_to(fg)
        fg.add_to(m)
    if classifier_arcs is not None:
        fg = folium.FeatureGroup(name="Classifier-only prediction", show=False)
        for line in _chain(net, dedupe_consecutive(classifier_arcs)):
            folium.PolyLine(line, color=COLORS["classifier"], weight=3, opacity=0.9, tooltip="classifier only").add_to(fg)
        fg.add_to(m)
    if predicted_arcs is not None:
        fg = folium.FeatureGroup(name="Predicted route (HMM/Viterbi)", show=True)
        for line in _chain(net, dedupe_consecutive(predicted_arcs)):
            folium.PolyLine(line, color=COLORS["predicted"], weight=3, opacity=0.95, tooltip="predicted").add_to(fg)
        fg.add_to(m)

    fg_gps = folium.FeatureGroup(name="GPS observations", show=True)
    tt = track["t"].to_numpy()
    for i in range(0, len(lat), gps_every):
        folium.CircleMarker([float(lat[i]), float(lon[i])], radius=2, color=COLORS["gps"], fill=True,
                            fill_opacity=0.8, weight=0, tooltip=f"obs {i}, t={tt[i]:.0f}s").add_to(fg_gps)
    fg_gps.add_to(m)
    folium.CircleMarker([float(lat[0]), float(lon[0])], radius=7, color="black", fill=True, fill_color="#00e676",
                        fill_opacity=1, tooltip="start").add_to(m)
    folium.CircleMarker([float(lat[-1]), float(lon[-1])], radius=7, color="black", fill=True, fill_color="#212121",
                        fill_opacity=1, tooltip="end").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white; padding: 8px 12px;
                border: 1px solid #999; border-radius: 6px; font: 12px sans-serif;">
      <b>{title}</b><br>
      <span style="color:{COLORS['gps']}">&#9679;</span> GPS observations<br>
      <span style="color:{COLORS['network']}">&#9473;</span> Road network<br>
      <span style="color:{COLORS['reference']}">&#9473;</span> Reference route<br>
      <span style="color:{COLORS['classifier']}">&#9473;</span> Classifier-only<br>
      <span style="color:{COLORS['predicted']}">&#9473;</span> Predicted (HMM)<br>
      <span style="color:#00e676">&#9679;</span> start &nbsp; <span style="color:#212121">&#9679;</span> end
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    if out_html:
        m.save(out_html)
    return m


def static_map(ax, track, net, reference_arcs=None, predicted_arcs=None, classifier_arcs=None,
               title="", zoom_box=None):
    """Matplotlib rendering of the same layers in lon/lat (aspect-corrected)."""
    lat, lon = track["lat"].to_numpy(), track["lon"].to_numpy()
    px, py = net.proj.fwd(lon, lat)
    from matplotlib.collections import LineCollection

    def lc(lines, **kw):
        ax.add_collection(LineCollection([[(p[1], p[0]) for p in ln] for ln in lines], **kw))

    lc(_arc_lines(net, nearby_arcs(net, px, py)), colors=COLORS["network"], linewidths=0.5, alpha=0.6, zorder=1)
    if reference_arcs is not None:
        lc(_chain(net, dedupe_consecutive(reference_arcs)), colors=COLORS["reference"], linewidths=5, alpha=0.5,
           zorder=2, label="Reference route")
    if classifier_arcs is not None:
        lc(_chain(net, dedupe_consecutive(classifier_arcs)), colors=COLORS["classifier"], linewidths=1.6,
           alpha=0.9, zorder=3, label="Classifier-only")
    if predicted_arcs is not None:
        lc(_chain(net, dedupe_consecutive(predicted_arcs)), colors=COLORS["predicted"], linewidths=1.6, zorder=4,
           label="Predicted (HMM/Viterbi)")
    ax.scatter(lon, lat, s=2, c=COLORS["gps"], zorder=5, label="GPS observations")
    ax.scatter([lon[0]], [lat[0]], s=50, c="#00e676", edgecolors="k", zorder=6, label="start")
    ax.scatter([lon[-1]], [lat[-1]], s=50, c="#212121", edgecolors="k", zorder=6, label="end")
    if zoom_box is None:
        padx = (lon.max() - lon.min()) * 0.05 + 1e-4
        pady = (lat.max() - lat.min()) * 0.05 + 1e-4
        zoom_box = (lon.min() - padx, lon.max() + padx, lat.min() - pady, lat.max() + pady)
    ax.set_xlim(zoom_box[0], zoom_box[1]); ax.set_ylim(zoom_box[2], zoom_box[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title(title)
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
