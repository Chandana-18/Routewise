"""
Distance and travel-time matrices.

Default engine is haversine x circuity with a piecewise speed profile, so the
app runs with zero API calls. An OpenRouteService matrix path is included for
when you want real road network times.

LIMITATION, stated deliberately: true traffic-aware routing needs the travel
time of an arc to depend on the departure time, which is a decision variable
inside the solver. This pilot applies one congestion multiplier derived from
how much of the working shift overlaps peak periods. Swap in a live matrix API
to remove the approximation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import requests

EARTH_RADIUS_MI = 3958.7613
ORS_MATRIX_URL = "https://api.openrouteservice.org/v2/matrix/driving-hgv"


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def congestion_multiplier(shift_start_h: float, shift_end_h: float,
                          am_peak=(7.0, 9.5), pm_peak=(15.5, 18.0),
                          peak_mult: float = 1.30,
                          offpeak_mult: float = 1.00,
                          enabled: bool = True) -> float:
    if not enabled or shift_end_h <= shift_start_h:
        return 1.0
    total = shift_end_h - shift_start_h
    peak = 0.0
    for lo, hi in (am_peak, pm_peak):
        peak += max(0.0, min(shift_end_h, hi) - max(shift_start_h, lo))
    offpeak = max(0.0, total - peak)
    return (peak * peak_mult + offpeak * offpeak_mult) / total


def _effective_speed(dist_mi: float, urban_mph: float, hwy_mph: float,
                     short_leg_mi: float) -> float:
    """Short legs are stop-and-go; longer legs pick up highway speed."""
    if dist_mi <= short_leg_mi:
        return urban_mph
    t = short_leg_mi / urban_mph + (dist_mi - short_leg_mi) / hwy_mph
    return dist_mi / t if t > 0 else hwy_mph


def build_matrices(
    coords: Sequence[Tuple[float, float]],
    circuity: float = 1.32,
    urban_mph: float = 22.0,
    hwy_mph: float = 52.0,
    short_leg_mi: float = 4.0,
    time_multiplier: float = 1.0,
    ors_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return (distance_miles, travel_seconds, meta). coords[0] is the warehouse."""
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = haversine_miles(*coords[i], *coords[j]) * circuity

    time_s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                v = _effective_speed(dist[i, j], urban_mph, hwy_mph, short_leg_mi)
                time_s[i, j] = dist[i, j] / v * 3600.0

    meta = {"engine": "haversine", "circuity": circuity,
            "time_multiplier": round(time_multiplier, 3)}

    if ors_key:
        got = _try_ors_matrix(coords, ors_key)
        if got is not None:
            dist, base_time = got
            time_s = base_time
            meta["engine"] = "openrouteservice"

    return dist, time_s * time_multiplier, meta


def _try_ors_matrix(coords, api_key: str):
    """ORS free tier allows modest matrix sizes. Returns None on any failure."""
    try:
        locations = [[lon, lat] for lat, lon in coords]
        r = requests.post(
            ORS_MATRIX_URL,
            json={"locations": locations, "metrics": ["distance", "duration"],
                  "units": "mi"},
            headers={"Authorization": api_key,
                     "Content-Type": "application/json"},
            timeout=40,
        )
        r.raise_for_status()
        js = r.json()
        dist = np.array(js["distances"], dtype=float)
        dur = np.array(js["durations"], dtype=float)
        if np.isnan(dist).any() or np.isnan(dur).any():
            return None
        return dist, dur
    except Exception:                                             # noqa: BLE001
        return None
