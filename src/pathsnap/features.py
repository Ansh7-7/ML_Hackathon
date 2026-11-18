"""Candidate generation and per-(observation, candidate) feature engineering.

Pipeline for one trip (identical at training and serving time):

1. project GPS + road network to a local metric frame (geo.LocalProjection)
2. candidate generation: every arc within `radius` metres of the GPS fix,
   ordered by (distance, heading difference) and capped at `k_max`
3. one feature row per (observation, candidate)
4. (training/evaluation only) label = candidate is the reference arc that the
   monotone route alignment (labels.align_route) attaches to the observation

Nothing in steps 1-3 reads the reference route, so the same code is used by
the API on evaluator trips. Step 4 is only called when labels are needed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .geo import LocalProjection, angle_diff, bearing_deg, point_segment
from .labels import align_route
from .network import RoadNetwork


@dataclass
class FeatureConfig:
    radius_m: float = 50.0          # candidate search radius (chosen in 02_build_dataset.py)
    k_max: int = 12                 # max candidates per observation
    head_windows: tuple = (2, 5)    # +-w points used for the GPS heading
    min_move_m: float = 2.0         # below this displacement the heading is unreliable
    ctx_offsets: tuple = (-5, -2, 2, 5)
    ctx_clip_m: float = 200.0
    version: str = "1.0"
    feature_names: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["head_windows"] = list(self.head_windows)
        d["ctx_offsets"] = list(self.ctx_offsets)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["head_windows"] = tuple(d["head_windows"])
        d["ctx_offsets"] = tuple(d["ctx_offsets"])
        return cls(**d)


FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand",
    "head_diff_w2", "head_diff_w5", "head_diff_w2_minus_min", "low_motion",
    "speed", "dt_prev", "dt_next",
    "seg_len", "t_proj", "dist_to_end", "zero_len",
    "ctx_d_m5", "ctx_d_m2", "ctx_d_p2", "ctx_d_p5", "ctx_mean", "ctx_mean_minus_min",
    "two_way", "in_deg_u", "out_deg_v",
]


# --------------------------------------------------------------------------- trip prep
@dataclass
class PreparedTrip:
    trip_id: str
    net: RoadNetwork
    px: np.ndarray
    py: np.ndarray
    t: np.ndarray
    lon: np.ndarray
    lat: np.ndarray


def prepare_trip(trip) -> PreparedTrip:
    lon = trip.track["lon"].to_numpy()
    lat = trip.track["lat"].to_numpy()
    proj = LocalProjection(lon.mean(), lat.mean())
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    px, py = proj.fwd(lon, lat)
    return PreparedTrip(trip.trip_id, net, px, py, trip.track["t"].to_numpy(), lon, lat)


def gps_heading(px, py, w):
    """Bearing of the displacement p[i-w] -> p[i+w] (indices clamped) and its length."""
    n = len(px)
    i0 = np.clip(np.arange(n) - w, 0, n - 1)
    i1 = np.clip(np.arange(n) + w, 0, n - 1)
    dx, dy = px[i1] - px[i0], py[i1] - py[i0]
    return bearing_deg(dx, dy), np.hypot(dx, dy)


# --------------------------------------------------------------------- candidates
def all_candidates(net: RoadNetwork, px, py, radius):
    """All (obs, arc) pairs with point-to-segment distance <= radius.

    Returns obs, arc, dist, t (unsorted)."""
    pts = np.column_stack([px, py])
    lists = net._tree.query_ball_point(pts, radius + net._half_step)
    lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
    if lens.sum() == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, np.empty(0), np.empty(0)
    obs = np.repeat(np.arange(len(px), dtype=np.int64), lens)
    samp = np.fromiter((j for x in lists for j in x), dtype=np.int64, count=int(lens.sum()))
    arc = net._sample_arc[samp].astype(np.int64)
    key = np.unique(obs * net.n_arcs + arc)
    obs, arc = key // net.n_arcs, key % net.n_arcs
    d, t = point_segment(px[obs], py[obs], net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc])
    keep = d <= radius
    return obs[keep], arc[keep], d[keep], t[keep]


def _group_rank(obs, n_obs):
    """Rank of each row within its (contiguous, sorted) obs group."""
    starts = np.searchsorted(obs, np.arange(n_obs))
    return np.arange(len(obs)) - starts[obs]


def ranked_candidates(pt: PreparedTrip, radius, k_max=None, head_w=2, min_move=2.0):
    """Candidates sorted by (obs, distance, heading difference) and capped at k_max.

    The heading tie-break matters because ~92% of arcs have a reverse twin with
    identical geometry: when the cap cuts through a twin pair we keep the twin
    whose direction agrees with the GPS motion (no labels involved).
    """
    net = pt.net
    obs, arc, d, t = all_candidates(net, pt.px, pt.py, radius)
    hd, mv = gps_heading(pt.px, pt.py, head_w)
    hdiff = angle_diff(hd[obs], net.bearing[arc])
    hdiff = np.where((mv[obs] < min_move) | (net.length[arc] < 0.01), 90.0, hdiff)
    order = np.lexsort((hdiff, np.round(d, 1), obs))
    obs, arc, d, t, hdiff = obs[order], arc[order], d[order], t[order], hdiff[order]
    rank = _group_rank(obs, len(pt.px))
    if k_max is not None:
        keep = rank < k_max
        obs, arc, d, t, hdiff, rank = obs[keep], arc[keep], d[keep], t[keep], hdiff[keep], rank[keep]
    return obs, arc, d, t, rank


# ----------------------------------------------------------------------- features
def _group_min(values, obs, n_obs):
    out = np.full(n_obs, np.inf)
    np.minimum.at(out, obs, values)
    return out[obs]


def build_features(pt: PreparedTrip, cfg: FeatureConfig) -> pd.DataFrame:
    """One row per (observation, candidate). No reference-route information is used."""
    net, px, py, ts = pt.net, pt.px, pt.py, pt.t
    n = len(px)
    obs, arc, d, tp, rank = ranked_candidates(pt, cfg.radius_m, cfg.k_max, cfg.head_windows[0], cfg.min_move_m)
    f = {"obs": obs.astype(np.int32), "arc": arc.astype(np.int64)}
    f["dist"] = d
    f["dist_rank"] = rank
    dmin = _group_min(d, obs, n)
    f["dist_minus_min"] = d - dmin
    f["dist_ratio"] = d / (dmin + 1.0)
    n_cand = np.bincount(obs, minlength=n)
    f["n_cand"] = n_cand[obs]

    zero_len = net.length[arc] < 0.01
    low_motion = None
    for w in cfg.head_windows:
        hd, mv = gps_heading(px, py, w)
        diff = angle_diff(hd[obs], net.bearing[arc])
        unreliable = (mv[obs] < cfg.min_move_m) | zero_len
        f[f"head_diff_w{w}"] = np.where(unreliable, 90.0, diff)
        if low_motion is None:
            low_motion = (mv[obs] < cfg.min_move_m).astype(np.int8)
    w0 = cfg.head_windows[0]
    f[f"head_diff_w{w0}_minus_min"] = f[f"head_diff_w{w0}"] - _group_min(f[f"head_diff_w{w0}"], obs, n)
    f["low_motion"] = low_motion

    dt = np.diff(ts)
    dt_prev = np.r_[0.0, dt]
    dt_next = np.r_[dt, 0.0]
    im1, ip1 = np.clip(np.arange(n) - 1, 0, n - 1), np.clip(np.arange(n) + 1, 0, n - 1)
    span = ts[ip1] - ts[im1]
    speed = np.hypot(px[ip1] - px[im1], py[ip1] - py[im1]) / np.where(span > 0, span, np.nan)
    f["speed"] = np.nan_to_num(speed, nan=0.0)[obs]
    f["dt_prev"] = np.minimum(dt_prev[obs], 600.0)
    f["dt_next"] = np.minimum(dt_next[obs], 600.0)

    L = net.length[arc]
    f["seg_len"] = L
    f["t_proj"] = tp
    f["dist_to_end"] = np.minimum(tp, 1 - tp) * L
    f["zero_len"] = zero_len.astype(np.int8)

    ctx = []
    for off in cfg.ctx_offsets:
        j = np.clip(obs + off, 0, n - 1)
        dc, _ = point_segment(px[j], py[j], net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc])
        dc = np.minimum(dc, cfg.ctx_clip_m)
        name = f"ctx_d_{'m' if off < 0 else 'p'}{abs(off)}"
        f[name] = dc
        ctx.append(dc)
    cm = np.mean(ctx, axis=0)
    f["ctx_mean"] = cm
    f["ctx_mean_minus_min"] = cm - _group_min(cm, obs, n)

    rev = getattr(net, "_has_reverse", None)
    if rev is None:
        pairs = pd.MultiIndex.from_arrays([net.u, net.v])
        rev = pd.MultiIndex.from_arrays([net.v, net.u]).isin(pairs)
        net._has_reverse = rev
    f["two_way"] = rev[arc].astype(np.int8)
    f["in_deg_u"] = net.in_deg[net.u[arc]]
    f["out_deg_v"] = net.out_deg[net.v[arc]]
    df = pd.DataFrame(f)
    for c in FEATURES:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


# ------------------------------------------------------------------------ labels
def reference_labels(pt: PreparedTrip, route: np.ndarray):
    """Per-observation reference arc from the monotone route alignment."""
    _, lab_arc, lab_dist = align_route(pt.px, pt.py, pt.net, route)
    return lab_arc, lab_dist


def label_rows(df: pd.DataFrame, lab_arc: np.ndarray) -> np.ndarray:
    return (df["arc"].to_numpy() == lab_arc[df["obs"].to_numpy()]).astype(np.int8)
