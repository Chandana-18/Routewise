"""
Fleet sizing: can today's work be done with the trucks already owned, and how
often is the marginal truck actually needed?

Two analyses:

  fleet_sweep           Solve the same day with 1, 2, 3 ... k trucks and find
                        the smallest fleet that serves every stop legally.
                        Answers "do we need another truck for this day?"

  utilisation_study     Replay the operation over many simulated days with
                        demand variability and customer dropout, recording how
                        many trucks each day required. The resulting
                        distribution answers "how many days a year would a 4th
                        truck actually leave the yard?" - which is the input
                        the buy-versus-rent decision needs and which no generic
                        TCO calculator can supply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .routing import (RouteConfig, Solution, normalise_warehouses, solve_routes)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def submatrix(mat: np.ndarray, stop_rows: Sequence[int],
              n_warehouses: int = 1) -> np.ndarray:
    """Slice a full matrix down to every warehouse plus the given 0-based stop rows.

    Warehouses occupy rows 0..n_warehouses-1 and are always kept, so a simulated day
    that drops customers still routes from the same network.
    """
    idx = list(range(n_warehouses)) + [int(r) + n_warehouses for r in stop_rows]
    return mat[np.ix_(idx, idx)]


def build_fleet(template_trucks: pd.DataFrame, template_drivers: pd.DataFrame,
                k: int) -> tuple:
    """Take the first k trucks; if k exceeds the roster, clone the last one.

    A cloned truck inherits the warehouse of the truck it was copied from, so
    "add a truck" means adding one at an existing yard rather than silently
    inventing a new site.
    """
    trucks = template_trucks.copy().reset_index(drop=True)
    drivers = template_drivers.copy().reset_index(drop=True)

    if k <= len(trucks):
        t = trucks.iloc[:k].copy()
    else:
        extra = []
        for i in range(len(trucks), k):
            row = trucks.iloc[-1].copy()
            row["truck_id"] = f"T{i+1:02d}"
            row["truck_name"] = f"Truck {i+1} (added)"
            row["driver_id"] = f"D{i+1:02d}"
            extra.append(row)
        t = pd.concat([trucks, pd.DataFrame(extra)], ignore_index=True)

    needed = set(t["driver_id"])
    have = set(drivers["driver_id"])
    missing = needed - have
    if missing:
        clones = []
        for did in sorted(missing):
            row = drivers.iloc[-1].copy() if len(drivers) else pd.Series({
                "driver_name": "", "age": 35, "cdl_class": "A",
                "shift_start": "06:00", "shift_end": "18:00",
                "hours_used_week": 0.0, "max_weekly_hours": 70.0,
                "hourly_wage": 28.0})
            row["driver_id"] = did
            row["driver_name"] = f"Driver {did[-2:].lstrip('0') or did}"
            row["hours_used_week"] = 0.0
            clones.append(row)
        drivers = pd.concat([drivers, pd.DataFrame(clones)], ignore_index=True)

    return t.reset_index(drop=True), drivers


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #

@dataclass
class SweepRow:
    trucks: int
    served: int
    dropped: int
    all_served: bool
    miles: float
    hours: float
    cost: float
    cost_per_stop: float
    status: str


def fleet_sweep(stops: pd.DataFrame, trucks: pd.DataFrame,
                drivers: pd.DataFrame, dist_mi: np.ndarray, time_s: np.ndarray,
                cfg: Optional[RouteConfig] = None,
                max_trucks: Optional[int] = None,
                min_trucks: int = 1,
                progress=None, warehouses=None) -> pd.DataFrame:
    cfg = cfg or RouteConfig()
    max_trucks = max_trucks or (len(trucks) + 3)
    rows: List[SweepRow] = []

    for k in range(min_trucks, max_trucks + 1):
        t, d = build_fleet(trucks, drivers, k)
        sol = solve_routes(stops, t, d, dist_mi, time_s, cfg,
                           label=f"{k} trucks", warehouses=warehouses)
        rows.append(SweepRow(
            trucks=k,
            served=sol.stops_served,
            dropped=len(sol.dropped),
            all_served=sol.all_served,
            miles=round(sol.total_miles, 1),
            hours=round(sol.total_hours, 2),
            cost=round(sol.total_cost, 2),
            cost_per_stop=round(sol.total_cost / max(1, sol.stops_served), 2),
            status=sol.status,
        ))
        if progress is not None:
            progress((k - min_trucks + 1) / max(1, max_trucks - min_trucks + 1),
                     f"Testing {k} trucks")

    df = pd.DataFrame([r.__dict__ for r in rows])
    df["marginal_cost"] = df["cost"].diff().round(2)
    df["marginal_stops"] = df["served"].diff()
    return df


def minimum_fleet(sweep: pd.DataFrame) -> Optional[int]:
    ok = sweep[sweep["all_served"]]
    return int(ok["trucks"].min()) if len(ok) else None


def cheapest_fleet(sweep: pd.DataFrame, tolerance: float = 0.01) -> Optional[int]:
    """Smallest fleet whose cost is within `tolerance` of the best cost found.

    A heuristic solver returns slightly different costs run to run, so a larger
    fleet can look a few dollars cheaper purely by chance. Without this
    tolerance the app would recommend buying a truck to save $1.71.
    """
    ok = sweep[sweep["all_served"]]
    if not len(ok):
        return None
    best = float(ok["cost"].min())
    close = ok[ok["cost"] <= best * (1 + tolerance)]
    return int(close["trucks"].min())


# --------------------------------------------------------------------------- #
# utilisation across simulated days
# --------------------------------------------------------------------------- #

@dataclass
class UtilisationResult:
    days: pd.DataFrame = field(default_factory=pd.DataFrame)
    trucks_needed_counts: Dict[int, int] = field(default_factory=dict)
    days_per_year: Dict[int, float] = field(default_factory=dict)
    owned: int = 0
    n_days: int = 0

    def days_needing_at_least(self, k: int) -> float:
        """Days per year on which at least k trucks are required."""
        if not self.n_days:
            return 0.0
        hits = sum(c for n, c in self.trucks_needed_counts.items() if n >= k)
        return round(365.0 * hits / self.n_days, 1)

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        max_k = max(self.trucks_needed_counts) if self.trucks_needed_counts else 0
        for k in range(1, max_k + 1):
            rows.append({
                "Truck #": k,
                "Days/yr it is needed": self.days_needing_at_least(k),
                "% of days": round(100.0 * self.days_needing_at_least(k) / 365.0, 1),
                "Owned today": "yes" if k <= self.owned else "no",
            })
        return pd.DataFrame(rows)


def simulate_day(stops: pd.DataFrame, rng: np.random.Generator,
                 demand_cv: float, dropout: float,
                 surge_prob: float = 0.0, surge_factor: float = 1.4):
    """Perturb a base day: random demand noise, random customer dropout, and an
    occasional surge day. Returns (stops_subset, original_row_positions)."""
    n = len(stops)
    keep = rng.random(n) >= dropout
    if not keep.any():
        keep[rng.integers(0, n)] = True
    rows = np.where(keep)[0]

    s = stops.iloc[rows].copy().reset_index(drop=True)
    noise = rng.lognormal(mean=-0.5 * demand_cv ** 2, sigma=demand_cv, size=len(s))
    if surge_prob > 0 and rng.random() < surge_prob:
        noise = noise * surge_factor
    s["demand_kg"] = (s["demand_kg"].astype(float).values * noise).round(1)
    return s, rows


def utilisation_study(stops: pd.DataFrame, trucks: pd.DataFrame,
                      drivers: pd.DataFrame, dist_mi: np.ndarray,
                      time_s: np.ndarray, cfg: Optional[RouteConfig] = None,
                      n_days: int = 30, demand_cv: float = 0.30,
                      dropout: float = 0.20, surge_prob: float = 0.10,
                      surge_factor: float = 1.4, max_trucks: int = 8,
                      seed: int = 7, progress=None,
                      warehouses=None) -> UtilisationResult:
    cfg = cfg or RouteConfig()
    rng = np.random.default_rng(seed)
    n_warehouses = len(normalise_warehouses(warehouses)) if warehouses is not None else 1
    owned = len(trucks)
    records, counts = [], {}

    for day in range(1, n_days + 1):
        day_stops, rows = simulate_day(stops, rng, demand_cv, dropout,
                                       surge_prob, surge_factor)
        d_sub = submatrix(dist_mi, rows, n_warehouses)
        t_sub = submatrix(time_s, rows, n_warehouses)

        needed = None
        best = None
        for k in range(1, max_trucks + 1):
            t, dr = build_fleet(trucks, drivers, k)
            sol = solve_routes(day_stops, t, dr, d_sub, t_sub, cfg,
                               label=f"day{day}-k{k}", warehouses=warehouses)
            if sol.all_served:
                needed, best = k, sol
                break
        if needed is None:
            needed = max_trucks

        counts[needed] = counts.get(needed, 0) + 1
        records.append({
            "day": day,
            "stops": len(day_stops),
            "demand_kg": round(float(day_stops["demand_kg"].sum()), 1),
            "trucks_needed": needed,
            "miles": round(best.total_miles, 1) if best else np.nan,
            "cost": round(best.total_cost, 2) if best else np.nan,
            "within_owned_fleet": needed <= owned,
        })
        if progress is not None:
            progress(day / n_days, f"Simulating day {day} of {n_days}")

    res = UtilisationResult(days=pd.DataFrame(records),
                            trucks_needed_counts=counts,
                            owned=owned, n_days=n_days)
    res.days_per_year = {k: res.days_needing_at_least(k)
                         for k in range(1, (max(counts) if counts else 0) + 1)}
    return res
