"""Step 1 - Dataset inventory and exploratory data analysis.

Partitions used: train (80 trips) and student_test (10 trips). Evaluator-test
data is never read here.

Outputs -> outputs/eda/
  trip_inventory.csv      one row per trip: counts, sizes, gaps, extent, network, route
  partition_summary.csv   aggregates per partition
  data_quality_log.md     anomalies, assumptions, decisions
  *.png                   plots
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.geo import LocalProjection, haversine
from pathsnap.io import PARTITIONS, list_trips, load_trip
from pathsnap.labels import align_route
from pathsnap.network import RoadNetwork

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "eda")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

rows, dt_all, speed_all, d2route_all, d2route_part = [], [], [], [], []
per_trip_tracks = {}

for part in PARTITIONS:
    for tid in list_trips(part):
        tr = load_trip(part, tid)
        t = tr.track
        r = {"partition": part, "trip_id": tid}
        r.update({f"size_{k}_bytes": v for k, v in tr.meta["file_sizes"].items()})
        r["files_present"] = sum(v is not None for v in tr.meta["file_sizes"].values())
        # ---- track
        r["n_obs"] = len(t)
        r["n_nan"] = int(t.isna().sum().sum())
        r["n_dup_timestamps"] = int(t["t"].duplicated().sum())
        r["n_dup_positions"] = int(t[["lon", "lat"]].duplicated().sum())
        dt = np.diff(t["t"].to_numpy())
        r["n_nonmonotonic_t"] = int((dt <= 0).sum())
        r["dt_median"] = float(np.median(dt))
        r["dt_max"] = float(dt.max())
        r["share_dt_eq_1s"] = float(np.mean(np.isclose(dt, 1.0)))
        r["n_gaps_gt_5s"] = int((dt > 5).sum())
        r["n_gaps_gt_30s"] = int((dt > 30).sum())
        r["duration_s"] = float(t["t"].iloc[-1] - t["t"].iloc[0])
        r["lon_min"], r["lon_max"] = t["lon"].min(), t["lon"].max()
        r["lat_min"], r["lat_max"] = t["lat"].min(), t["lat"].max()
        valid_coords = t["lon"].between(-180, 180).all() and t["lat"].between(-90, 90).all()
        r["coords_valid_range"] = bool(valid_coords)
        step = haversine(t["lon"].values[:-1], t["lat"].values[:-1], t["lon"].values[1:], t["lat"].values[1:])
        speed = step / np.where(dt > 0, dt, np.nan)
        r["track_length_km"] = float(np.nansum(step) / 1000)
        r["bbox_diag_km"] = float(haversine(r["lon_min"], r["lat_min"], r["lon_max"], r["lat_max"]) / 1000)
        r["speed_median_mps"] = float(np.nanmedian(speed))
        r["speed_p99_mps"] = float(np.nanpercentile(speed, 99))
        r["n_speed_gt_60mps"] = int(np.nansum(speed > 60))   # > 216 km/h: implausible jump
        r["max_step_m"] = float(step.max())
        # ---- network
        lon0, lat0 = t["lon"].mean(), t["lat"].mean()
        proj = LocalProjection(lon0, lat0)
        net = RoadNetwork(tr.nodes, tr.arcs, proj)
        r["n_nodes"], r["n_arcs"] = net.n_nodes, net.n_arcs
        deg = net.out_deg + net.in_deg
        r["n_isolated_nodes"] = int((deg == 0).sum())
        nw, lw, ns, ls = net.components()
        r["n_weak_components"] = int(nw - r["n_isolated_nodes"])  # exclude singletons from isolated nodes
        r["largest_wcc_share"] = float(np.bincount(lw).max() / net.n_nodes)
        r["n_strong_components"] = int(ns)
        r["largest_scc_share"] = float(np.bincount(ls).max() / net.n_nodes)
        pairs = set(zip(net.u.tolist(), net.v.tolist()))
        r["share_two_way_arcs"] = float(np.mean([(b, a) in pairs for a, b in zip(net.u.tolist(), net.v.tolist())]))
        r["n_self_loops"] = int((net.u == net.v).sum())
        r["n_zero_len_arcs"] = int((net.length < 0.01).sum())
        r["arc_len_median_m"] = float(np.median(net.length))
        # ---- route
        route = tr.route
        r["route_available"] = route is not None
        r["route_len_arcs"] = len(route)
        r["route_ids_valid"] = bool((route >= 0).all() and (route < net.n_arcs).all())
        r["route_continuity"] = float(np.mean(net.v[route[:-1]] == net.u[route[1:]])) if len(route) > 1 else 1.0
        r["route_n_breaks"] = int(np.sum(net.v[route[:-1]] != net.u[route[1:]]))
        r["route_repeated_arcs"] = int(len(route) - len(set(route.tolist())))
        r["route_length_km"] = float(net.length[route].sum() / 1000)
        px, py = proj.fwd(t["lon"].to_numpy(), t["lat"].to_numpy())
        _, _, d = align_route(px, py, net, route)
        r["gps_to_route_median_m"] = float(np.median(d))
        r["gps_to_route_p95_m"] = float(np.percentile(d, 95))
        r["gps_to_route_max_m"] = float(d.max())
        r["share_obs_gt_50m_from_route"] = float(np.mean(d > 50))
        rows.append(r)
        dt_all.append(pd.DataFrame({"partition": part, "dt": dt}))
        speed_all.append(pd.DataFrame({"partition": part, "speed": speed}))
        d2route_part.append(pd.DataFrame({"partition": part, "d": d}))
        per_trip_tracks[(part, tid)] = (t, route, net)
        print(part, tid, r["n_obs"], r["n_arcs"], f'{r["gps_to_route_median_m"]:.1f}m', flush=True)

inv = pd.DataFrame(rows)
inv.to_csv(os.path.join(OUT, "trip_inventory.csv"), index=False)

agg = inv.groupby("partition").agg(
    trips=("trip_id", "count"), files=("files_present", "sum"),
    obs_total=("n_obs", "sum"), obs_min=("n_obs", "min"), obs_median=("n_obs", "median"), obs_max=("n_obs", "max"),
    nan_values=("n_nan", "sum"), dup_timestamps=("n_dup_timestamps", "sum"), nonmonotonic=("n_nonmonotonic_t", "sum"),
    gaps_gt_5s=("n_gaps_gt_5s", "sum"), gaps_gt_30s=("n_gaps_gt_30s", "sum"), max_gap_s=("dt_max", "max"),
    implausible_jumps=("n_speed_gt_60mps", "sum"),
    nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"), arcs_max=("n_arcs", "max"),
    isolated_nodes=("n_isolated_nodes", "sum"), route_breaks=("route_n_breaks", "sum"),
    gps_to_route_median_m=("gps_to_route_median_m", "median"),
)
agg.to_csv(os.path.join(OUT, "partition_summary.csv"))
print(agg.T)

DT = pd.concat(dt_all)
SP = pd.concat(speed_all)
DR = pd.concat(d2route_part)

# ---------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for part, c in zip(PARTITIONS, ["tab:blue", "tab:orange"]):
    s = inv[inv.partition == part]
    ax[0].hist(s.n_obs, bins=20, alpha=.7, color=c, label=part)
    ax[1].hist(s.track_length_km, bins=20, alpha=.7, color=c, label=part)
    ax[2].hist(s.duration_s / 60, bins=20, alpha=.7, color=c, label=part)
ax[0].set_xlabel("GPS observations per trip"); ax[1].set_xlabel("Track length [km]"); ax[2].set_xlabel("Duration [min]")
for a in ax: a.set_ylabel("trips"); a.legend()
fig.suptitle("Trip size distribution (train + student_test)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "trip_sizes.png")); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
v = DT.dt.clip(upper=200)
ax[0].hist(v, bins=np.logspace(-0.1, np.log10(200), 40), color="tab:blue")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("Sampling interval Δt [s] (log)"); ax[0].set_ylabel("count (log)")
ax[0].set_title(f"Δt = 1 s for {np.mean(np.isclose(DT.dt,1))*100:.1f}% of steps")
ax[1].hist(SP.speed.clip(upper=80) * 3.6, bins=60, color="tab:green")
ax[1].set_yscale("log"); ax[1].set_xlabel("Point-to-point speed [km/h] (clipped 288)"); ax[1].set_title("Implied speed")
ax[2].hist(DR.d.clip(upper=100), bins=60, color="tab:red")
ax[2].set_yscale("log"); ax[2].set_xlabel("GPS → aligned reference arc [m] (clipped 100)"); ax[2].set_title("GPS noise vs reference route")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "sampling_speed_noise.png")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for part, c in zip(PARTITIONS, ["tab:blue", "tab:orange"]):
    s = inv[inv.partition == part]
    ax[0].scatter(s.n_nodes, s.n_arcs, c=c, s=14, label=part)
    ax[1].scatter(s.n_arcs, s.largest_scc_share, c=c, s=14, label=part)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs (road segments)")
ax[0].set_title("Road-network size per trip"); ax[0].legend()
ax[1].set_xscale("log"); ax[1].set_xlabel("arcs"); ax[1].set_ylabel("share of nodes in largest SCC")
ax[1].set_title("Network connectivity"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "network_size_connectivity.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(inv.lon_min / 2 + inv.lon_max / 2, inv.lat_min / 2 + inv.lat_max / 2,
           c=inv.partition.map({"train": "tab:blue", "student_test": "tab:orange"}), s=18)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("Trip centroids (blue = train, orange = student_test)")
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(OUT, "trip_centroids.png")); plt.close(fig)

# missing-value / quality table as a figure
qcols = ["n_nan", "n_dup_timestamps", "n_nonmonotonic_t", "n_gaps_gt_5s", "n_gaps_gt_30s", "n_speed_gt_60mps",
         "n_isolated_nodes", "route_n_breaks", "route_repeated_arcs"]
q = inv.groupby("partition")[qcols].sum().T
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.axis("off")
tb = ax.table(cellText=q.values, rowLabels=q.index, colLabels=q.columns, loc="center")
tb.scale(1, 1.3); ax.set_title("Missing / invalid value counts per partition")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "missing_values_table.png")); plt.close(fig)

# example trajectories with road network and reference route
ex = [("train", inv[inv.partition == "train"].sort_values("n_gaps_gt_30s").trip_id.iloc[-1]),
      ("train", inv[inv.partition == "train"].trip_id.iloc[2]),
      ("student_test", inv[inv.partition == "student_test"].trip_id.iloc[1])]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, key in zip(axs, ex):
    t, route, net = per_trip_tracks[key]
    px, py = net.proj.fwd(t.lon.values, t.lat.values)
    pad = 300
    box = (px.min() - pad, px.max() + pad, py.min() - pad, py.max() + pad)
    m = ((net.ax > box[0]) & (net.ax < box[1]) & (net.ay > box[2]) & (net.ay < box[3]))
    for a in np.where(m)[0][:60000]:
        ax.plot([net.ax[a], net.bx[a]], [net.ay[a], net.by[a]], color="0.82", lw=.5, zorder=1)
    for a in route:
        ax.plot([net.ax[a], net.bx[a]], [net.ay[a], net.by[a]], color="tab:green", lw=2.2, zorder=2)
    ax.scatter(px, py, s=1.5, c="tab:blue", zorder=3)
    ax.set_aspect("equal"); ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    ax.set_title(f"{key[0]} trip {key[1]}"); ax.set_xlabel("x [m] (local proj.)"); ax.set_ylabel("y [m]")
fig.suptitle("GPS points (blue), road network (grey) and reference route (green)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "example_trajectories.png")); plt.close(fig)

# ---------------------------------------------------------------- quality log
bad = inv.sort_values("dt_max", ascending=False).head(8)[["partition", "trip_id", "dt_max", "n_gaps_gt_30s"]]
jumps = inv[inv.n_speed_gt_60mps > 0][["partition", "trip_id", "n_speed_gt_60mps", "max_step_m"]]
far = inv.sort_values("gps_to_route_p95_m", ascending=False).head(8)[["partition", "trip_id", "gps_to_route_median_m", "gps_to_route_p95_m", "gps_to_route_max_m"]]
breaks = inv[inv.route_n_breaks > 0][["partition", "trip_id", "route_len_arcs", "route_n_breaks"]]
with open(os.path.join(OUT, "data_quality_log.md"), "w") as f:
    f.write("# Data-quality log (generated by scripts/01_eda.py)\n\n")
    f.write("## Partition summary\n\n" + agg.T.to_markdown() + "\n\n")
    f.write("## Largest time gaps\n\n" + bad.to_markdown(index=False) + "\n\n")
    f.write("## Trips with implausible jumps (>60 m/s between consecutive fixes)\n\n" + (jumps.to_markdown(index=False) if len(jumps) else "none") + "\n\n")
    f.write("## Trips whose GPS lies furthest from the reference route\n\n" + far.to_markdown(index=False) + "\n\n")
    f.write("## Reference routes with discontinuities (arc i head != arc i+1 tail)\n\n" + (breaks.to_markdown(index=False) if len(breaks) else "none") + "\n")
print("EDA done")
