"""PathSnap first-pass EDA across all trips in train/ and student_test/."""
import sys, os, glob
import numpy as np, pandas as pd

ROOT = sys.argv[1]
R = 6371000.0

def to_xy(lon, lat, lat0):
    # local equirectangular projection in metres (fine for a city-sized area)
    return (np.radians(lon) * R * np.cos(np.radians(lat0)), np.radians(lat) * R)

def point_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1), 0)
    t = np.clip(t, 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))

rows = []
for part in ["train", "student_test"]:
    for d in sorted(glob.glob(os.path.join(ROOT, part, "*"))):
        tid = os.path.basename(d)
        f = lambda e: os.path.join(d, f"{tid}.{e}")
        tr = pd.read_csv(f("track"), sep=r"\s+", header=None, names=["lon", "lat", "t"])
        nd = pd.read_csv(f("nodes"), sep=r"\s+", header=None, names=["lon", "lat"])
        ar = pd.read_csv(f("arcs"), sep=r"\s+", header=None, names=["u", "v"])
        rt = pd.read_csv(f("route"), header=None, names=["arc"])["arc"].to_numpy()

        dt = np.diff(tr.t.to_numpy())
        lat0 = tr.lat.mean()
        px, py = to_xy(tr.lon.to_numpy(), tr.lat.to_numpy(), lat0)
        nx_, ny_ = to_xy(nd.lon.to_numpy(), nd.lat.to_numpy(), lat0)
        u, v = ar.u.to_numpy(), ar.v.to_numpy()
        ax, ay, bx, by = nx_[u], ny_[u], nx_[v], ny_[v]

        # route continuity: does arc i end where arc i+1 starts?
        cont = (v[rt[:-1]] == u[rt[1:]]).mean() if len(rt) > 1 else np.nan
        # reverse arcs present?
        pairs = set(zip(u, v))
        bidir = np.mean([(b, a) in pairs for a, b in zip(u, v)])

        # GPS noise: distance from each GPS point to nearest ROUTE arc
        ra = rt
        dmin_route = np.full(len(px), np.inf)
        for i in range(0, len(ra), 200):
            s = ra[i:i + 200]
            dd = point_seg_dist(px[:, None], py[:, None], ax[s][None], ay[s][None], bx[s][None], by[s][None])
            dmin_route = np.minimum(dmin_route, dd.min(1))
        # candidates within radius among ALL arcs (sample points to stay fast)
        idx = np.linspace(0, len(px) - 1, min(150, len(px))).astype(int)
        dd_all = point_seg_dist(px[idx, None], py[idx, None], ax[None], ay[None], bx[None], by[None])
        cand30 = (dd_all <= 30).sum(1).mean()
        cand50 = (dd_all <= 50).sum(1).mean()

        rows.append(dict(
            partition=part, trip=tid, n_gps=len(tr), duration_s=tr.t.iloc[-1] - tr.t.iloc[0],
            nan_rows=int(tr.isna().any(axis=1).sum()), non_monotonic_t=int((dt <= 0).sum()),
            pct_dt_1s=float((dt == 1).mean() * 100), max_gap_s=float(dt.max()), gaps_over_5s=int((dt > 5).sum()),
            n_nodes=len(nd), n_arcs=len(ar), route_len=len(rt), route_unique=len(set(rt)),
            route_continuity=cont, pct_arcs_with_reverse=bidir * 100,
            arc_id_out_of_range=int((rt >= len(ar)).sum()),
            gps_to_route_median_m=np.median(dmin_route), gps_to_route_p95_m=np.percentile(dmin_route, 95),
            gps_to_route_max_m=dmin_route.max(), pct_gps_within_30m=(dmin_route <= 30).mean() * 100,
            avg_cands_30m=cand30, avg_cands_50m=cand50,
            lon_min=tr.lon.min(), lon_max=tr.lon.max(), lat_min=tr.lat.min(), lat_max=tr.lat.max(),
        ))

df = pd.DataFrame(rows)
df.to_csv(sys.argv[2], index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(df.groupby("partition").size())
print(df.drop(columns=["trip"]).groupby("partition").describe().T.to_string())
