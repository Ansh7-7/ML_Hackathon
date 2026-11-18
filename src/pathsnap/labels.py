"""Derive an observation-level reference road segment from the arc-level route.

The `.route` file lists the arcs the vehicle traversed, in order, but does NOT
say which GPS observation belongs to which arc. We therefore align the GPS
sequence to the route sequence with a monotone dynamic programme:

    choose j_1 <= j_2 <= ... <= j_N  (indices into the route)
    minimising  sum_i  dist(p_i, arc(route[j_i]))

i.e. every observation is attached to a route arc, the order of the route is
respected, and the total point-to-segment distance is minimal. This is a
standard way of turning a path-level ground truth into per-point labels.

Only the training / student-test partitions are labelled this way during
development. At serving time the evaluator labels are used by the API only
*after* prediction, to score the output.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment


def align_route(px: np.ndarray, py: np.ndarray, net, route: np.ndarray):
    """Return (route_index_per_obs, arc_per_obs, distance_per_obs)."""
    N, M = len(px), len(route)
    ax, ay = net.ax[route], net.ay[route]
    bx, by = net.bx[route], net.by[route]
    cost, _ = point_segment(px[:, None], py[:, None], ax[None, :], ay[None, :], bx[None, :], by[None, :])
    back = np.empty((N, M), dtype=np.int32)
    D = cost[0].copy()
    back[0] = np.arange(M)
    for i in range(1, N):
        # prefix minimum of previous row (monotone constraint k <= j)
        acc = np.minimum.accumulate(D)
        changed = np.empty(M, dtype=bool)
        changed[0] = True
        changed[1:] = acc[1:] < acc[:-1]
        pm_idx = np.maximum.accumulate(np.where(changed, np.arange(M), 0)).astype(np.int32)
        back[i] = pm_idx
        D = cost[i] + acc
    j = np.empty(N, dtype=np.int64)
    j[-1] = int(np.argmin(D))
    for i in range(N - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    return j, route[j], cost[np.arange(N), j]
