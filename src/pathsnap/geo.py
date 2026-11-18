"""Geometry helpers.

Coordinates are WGS84 lon/lat in degrees. Each trip is projected to a local
equirectangular (tangent-plane) frame in metres centred on the trip. Trips span
up to ~80 km; the east-west scale error at the edge of the largest trip is
about 0.7 % (median trip ~0.05 %), i.e. < 0.6 m on the <= 75 m distances used for
candidates - below GPS noise - and it allows fast vectorised Euclidean maths. Great-circle (haversine) distance is used for whole-trip summaries.
"""
from __future__ import annotations

import numpy as np

R_EARTH = 6_371_008.8  # mean Earth radius [m]


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = np.deg2rad(1.0) * R_EARTH * np.cos(np.deg2rad(lat0))
        self.ky = np.deg2rad(1.0) * R_EARTH

    def fwd(self, lon, lat):
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def inv(self, x, y):
        return np.asarray(x) / self.kx + self.lon0, np.asarray(y) / self.ky + self.lat0


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (broadcasting).

    Returns (distance, t) where t in [0,1] is the projection parameter along AB.
    """
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = np.where(L2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t


def bearing_deg(dx, dy):
    """Compass bearing in degrees [0,360) of a vector in the local frame (x=east, y=north)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Absolute smallest difference between two bearings in degrees, in [0,180]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d
