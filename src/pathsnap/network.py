"""Road-network utilities: projected arc geometry, spatial index, shortest paths."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjection, bearing_deg, point_segment


class RoadNetwork:
    """Directed road graph for one trip, in a local metric projection."""

    def __init__(self, nodes_df, arcs_df, proj: LocalProjection, densify_m: float = 15.0):
        self.proj = proj
        self.nx, self.ny = proj.fwd(nodes_df["lon"].to_numpy(), nodes_df["lat"].to_numpy())
        self.u = arcs_df["u"].to_numpy()
        self.v = arcs_df["v"].to_numpy()
        self.n_nodes = len(nodes_df)
        self.n_arcs = len(arcs_df)
        self.ax, self.ay = self.nx[self.u], self.ny[self.u]
        self.bx, self.by = self.nx[self.v], self.ny[self.v]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing_deg(self.bx - self.ax, self.by - self.ay)
        # out-degree / in-degree
        self.out_deg = np.bincount(self.u, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.v, minlength=self.n_nodes)
        # adjacency (CSR-like) for Dijkstra: node -> outgoing arcs
        order = np.argsort(self.u, kind="stable")
        self._adj_arcs = order
        self._adj_ptr = np.concatenate([[0], np.cumsum(self.out_deg)])
        # spatial index over arc sample points (arcs can be long, so densify)
        nseg = np.maximum(1, np.ceil(self.length / densify_m).astype(int))
        arc_rep = np.repeat(np.arange(self.n_arcs), nseg + 1)
        frac = np.concatenate([np.linspace(0, 1, k + 1) for k in nseg]) if self.n_arcs else np.array([])
        sx = self.ax[arc_rep] + frac * (self.bx - self.ax)[arc_rep]
        sy = self.ay[arc_rep] + frac * (self.by - self.ay)[arc_rep]
        self._sample_arc = arc_rep
        self._tree = cKDTree(np.column_stack([sx, sy])) if self.n_arcs else None
        self._half_step = densify_m / 2.0
        self._sp_cache: dict[int, dict[int, float]] = {}

    # ---------------------------------------------------------------- spatial
    def arcs_near(self, px: float, py: float, radius: float) -> np.ndarray:
        """Arc ids whose geometry lies within `radius` metres of (px,py)."""
        idx = self._tree.query_ball_point([px, py], radius + self._half_step)
        if not idx:
            return np.empty(0, dtype=int)
        arcs = np.unique(self._sample_arc[idx])
        d, _ = point_segment(px, py, self.ax[arcs], self.ay[arcs], self.bx[arcs], self.by[arcs])
        return arcs[d <= radius]

    def project(self, px, py, arcs):
        """Distance and projection parameter of point(s) onto arcs."""
        return point_segment(px, py, self.ax[arcs], self.ay[arcs], self.bx[arcs], self.by[arcs])

    # ---------------------------------------------------------- shortest path
    def sp_from(self, src_node: int, cutoff: float) -> dict[int, float]:
        """Bounded Dijkstra from a node. Results are cached; a cached run with a
        larger cutoff is reused for smaller cutoffs."""
        cached = self._sp_cache.get(src_node)
        if cached is not None and cached.get(-1, 0.0) >= cutoff:
            return cached
        dist = {src_node: 0.0}
        heap = [(0.0, src_node)]
        done = set()
        ptr, adj, L, V = self._adj_ptr, self._adj_arcs, self.length, self.v
        while heap:
            d, n = heapq.heappop(heap)
            if n in done:
                continue
            done.add(n)
            for a in adj[ptr[n]:ptr[n + 1]]:
                nd = d + L[a]
                if nd > cutoff:
                    continue
                m = V[a]
                if nd < dist.get(m, np.inf):
                    dist[m] = nd
                    heapq.heappush(heap, (nd, m))
        dist[-1] = cutoff
        if len(self._sp_cache) > 20000:
            self._sp_cache.clear()
        self._sp_cache[src_node] = dist
        return dist

    def route_distance(self, a: int, ta: float, b: int, tb: float, cutoff: float,
                       back_tol: float = 5.0) -> float:
        """Driving distance from position ta on arc a to position tb on arc b.

        Returns np.inf if b is not reachable from a within `cutoff` metres.
        Small backward movement on the same arc (GPS jitter, <= back_tol m)
        is treated as zero distance.
        """
        La, Lb = self.length[a], self.length[b]
        if a == b:
            delta = (tb - ta) * La
            if delta >= -back_tol:
                return max(delta, 0.0)
        rem_a = (1.0 - ta) * La
        into_b = tb * Lb
        budget = cutoff - rem_a - into_b
        if budget < 0:
            return np.inf
        dmap = self.sp_from(int(self.v[a]), max(budget, 1.0))
        d = dmap.get(int(self.u[b]))
        if d is None or d > budget:
            return np.inf
        return rem_a + d + into_b

    # ----------------------------------------------------------- connectivity
    def components(self):
        """Weakly and strongly connected component labels (scipy csgraph)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        g = coo_matrix((np.ones(self.n_arcs), (self.u, self.v)), shape=(self.n_nodes, self.n_nodes)).tocsr()
        nw, lw = connected_components(g, directed=True, connection="weak")
        ns, ls = connected_components(g, directed=True, connection="strong")
        return nw, lw, ns, ls
