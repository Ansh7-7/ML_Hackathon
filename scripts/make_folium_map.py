"""
PathSnap - Folium map of one trip: GPS points, nearby road network, reference route.

Usage:
    pip install folium pandas numpy
    python make_folium_map.py "Student folder/train/00000010" trip_00000010_map.html
"""
import sys
import numpy as np
import pandas as pd
import folium

trip_dir, out_html = sys.argv[1], sys.argv[2]
tid = trip_dir.rstrip("/").split("/")[-1]

def read(ext, cols):
    return pd.read_csv(f"{trip_dir}/{tid}.{ext}", sep=r"\s+", header=None, names=cols)

track = read("track", ["lon", "lat", "t"])
nodes = read("nodes", ["lon", "lat"])
arcs = read("arcs", ["u", "v"])          # row index = arc_id
route = pd.read_csv(f"{trip_dir}/{tid}.route", header=None)[0].to_numpy()

# Folium wants (lat, lon) - the files store (lon, lat)
m = folium.Map(location=[track.lat.mean(), track.lon.mean()], zoom_start=14, tiles="cartodbpositron")

# Road network: only arcs near the trip, so the map stays light
pad = 0.003
lo0, lo1 = track.lon.min() - pad, track.lon.max() + pad
la0, la1 = track.lat.min() - pad, track.lat.max() + pad
u, v = arcs.u.to_numpy(), arcs.v.to_numpy()
def _in(ids):
    return ((nodes.lon.values[ids] > lo0) & (nodes.lon.values[ids] < lo1) &
            (nodes.lat.values[ids] > la0) & (nodes.lat.values[ids] < la1))
inside = _in(u) | _in(v)   # keep arcs with EITHER endpoint in the box (was: start node only)
net = folium.FeatureGroup(name="Road network (nearby)")
for a in np.where(inside)[0]:
    folium.PolyLine([(nodes.lat[u[a]], nodes.lon[u[a]]), (nodes.lat[v[a]], nodes.lon[v[a]])],
                    color="#999999", weight=1, opacity=0.6).add_to(net)
net.add_to(m)

# Reference route (ground truth)
ref = folium.FeatureGroup(name="Reference route")
for a in route:
    folium.PolyLine([(nodes.lat[u[a]], nodes.lon[u[a]]), (nodes.lat[v[a]], nodes.lon[v[a]])],
                    color="green", weight=5, opacity=0.8, tooltip=f"arc {a}").add_to(ref)
ref.add_to(m)

# GPS observations
gps = folium.FeatureGroup(name="GPS observations")
for r in track.itertuples():
    folium.CircleMarker((r.lat, r.lon), radius=2, color="blue", fill=True,
                        fill_opacity=0.7, weight=0, tooltip=f"t={r.t:.0f}s").add_to(gps)
gps.add_to(m)

folium.Marker((track.lat.iloc[0], track.lon.iloc[0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker((track.lat.iloc[-1], track.lon.iloc[-1]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)

# A predicted-route layer gets added here later (e.g. in red) once the HMM is built.
folium.LayerControl().add_to(m)
m.fit_bounds([[track.lat.min(), track.lon.min()], [track.lat.max(), track.lon.max()]])
m.save(out_html)
print("saved", out_html)
