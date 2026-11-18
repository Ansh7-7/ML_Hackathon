"""Raw-data readers for the PathSnap dataset.

Each trip folder <partition>/<trip_id>/ holds four header-less, whitespace
separated files:

  <id>.track  lon  lat  t[s]        GPS observations (nominally 1 Hz)
  <id>.nodes  lon  lat              road-network nodes (row index = node_id)
  <id>.arcs   from_node  to_node    directed road segments (row index = arc_id)
  <id>.route  arc_id                reference route, chronological arc sequence

The road network is *local to each trip*: node/arc ids are only meaningful
inside the trip that owns them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PARTITIONS = ("train", "student_test")


def data_root() -> str:
    """Root folder that contains the partition folders.

    Override with the PATHSNAP_DATA environment variable.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.normpath(os.path.join(here, "..", "..", "data", "raw"))
    return os.environ.get("PATHSNAP_DATA", default)


def list_trips(partition: str, root: str | None = None) -> list[str]:
    root = root or data_root()
    pdir = os.path.join(root, partition)
    return sorted(d for d in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, d)))


def _read(path: str, names: list[str], dtype) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame({n: pd.Series(dtype=dtype) for n in names})
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame          # lon, lat, t
    nodes: pd.DataFrame          # lon, lat
    arcs: pd.DataFrame           # u, v
    route: np.ndarray | None     # arc ids (None when not supplied)
    meta: dict = field(default_factory=dict)


def load_trip(partition: str, trip_id: str, root: str | None = None,
              with_route: bool = True) -> Trip:
    root = root or data_root()
    tdir = os.path.join(root, partition, trip_id)
    base = os.path.join(tdir, trip_id)
    track = _read(base + ".track", ["lon", "lat", "t"], np.float64)
    nodes = _read(base + ".nodes", ["lon", "lat"], np.float64)
    arcs = _read(base + ".arcs", ["u", "v"], np.int64)
    route = None
    if with_route and os.path.exists(base + ".route"):
        route = _read(base + ".route", ["arc"], np.int64)["arc"].to_numpy()
    sizes = {ext: (os.path.getsize(base + "." + ext) if os.path.exists(base + "." + ext) else None)
             for ext in ("track", "nodes", "arcs", "route")}
    return Trip(trip_id, partition, track, nodes, arcs, route, {"file_sizes": sizes})
