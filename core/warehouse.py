"""
Warehouse siting: where should a warehouse be, and is a second one worth opening?

This is a facility location problem, not a routing problem, so it gets its own
model. Three levels are provided:

  center_of_gravity   demand-weighted centroid. One line of maths, useful as a
                      visual reference point only - it ignores roads, existing
                      sites, capacity and fixed costs.

  solve_cflp          Capacitated Facility Location, solved as a MILP with
                      PuLP/CBC. Opens a site only when the transport saving
                      beats its annualised fixed cost, so "open nothing" is a
                      legitimate answer. This is the model that actually
                      answers the question.

  validate_with_vrp   The honest check. Facility location uses out-and-back
                      distance; real cost comes from multi-stop routes. After
                      the MILP proposes a site, re-run the routing model with
                      that site as the warehouse and compare true routed cost.
                      A site that wins on the MILP objective can lose here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp

from .distance import haversine_miles


# --------------------------------------------------------------------------- #
# candidate generation
# --------------------------------------------------------------------------- #

def center_of_gravity(lats, lons, weights) -> Tuple[float, float]:
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    return (float(np.average(np.asarray(lats, float), weights=w)),
            float(np.average(np.asarray(lons, float), weights=w)))


def generate_candidates(stops: pd.DataFrame, weight_col: str = "annual_demand",
                        n_clusters: int = 5, grid_n: int = 3,
                        existing: Optional[pd.DataFrame] = None,
                        manual: Optional[pd.DataFrame] = None,
                        seed: int = 0) -> pd.DataFrame:
    """Build a discrete candidate set from the delivery points themselves.

    Sources: demand-weighted KMeans centroids, a coarse grid over the bounding
    box, the demand centre of gravity, any existing warehouses, and any sites
    the user pinned manually.
    """
    rows: List[dict] = []
    lats = stops["lat"].astype(float).values
    lons = stops["lon"].astype(float).values
    w = stops[weight_col].astype(float).values if weight_col in stops else np.ones(len(stops))

    # 1. demand-weighted clusters
    try:
        from sklearn.cluster import KMeans
        k = int(max(1, min(n_clusters, len(stops))))
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        km.fit(np.column_stack([lats, lons]), sample_weight=np.maximum(w, 1e-6))
        for i, c in enumerate(km.cluster_centers_, 1):
            rows.append({"site_id": f"C{i}", "name": f"Cluster centroid {i}",
                         "lat": float(c[0]), "lon": float(c[1]),
                         "source": "cluster"})
    except Exception:                                             # noqa: BLE001
        pass

    # 2. coarse grid over the service area
    if grid_n and grid_n > 1:
        lat_grid = np.linspace(lats.min(), lats.max(), grid_n)
        lon_grid = np.linspace(lons.min(), lons.max(), grid_n)
        n = 1
        for la in lat_grid:
            for lo in lon_grid:
                rows.append({"site_id": f"G{n}", "name": f"Grid point {n}",
                             "lat": float(la), "lon": float(lo),
                             "source": "grid"})
                n += 1

    # 3. centre of gravity
    cg = center_of_gravity(lats, lons, w)
    rows.append({"site_id": "CG", "name": "Centre of gravity",
                 "lat": cg[0], "lon": cg[1], "source": "centre of gravity"})

    # 4. existing warehouses
    if existing is not None and len(existing):
        for i, (_, r) in enumerate(existing.iterrows(), 1):
            rows.append({"site_id": f"E{i}", "name": str(r.get("name", f"Existing {i}")),
                         "lat": float(r["lat"]), "lon": float(r["lon"]),
                         "source": "existing"})

    # 5. manual pins
    if manual is not None and len(manual):
        for i, (_, r) in enumerate(manual.iterrows(), 1):
            rows.append({"site_id": f"M{i}", "name": str(r.get("name", f"Manual {i}")),
                         "lat": float(r["lat"]), "lon": float(r["lon"]),
                         "source": "manual"})

    df = pd.DataFrame(rows)
    # drop near-duplicates (within ~0.3 mi)
    keep = []
    for i, r in df.iterrows():
        dup = False
        for j in keep:
            if haversine_miles(r["lat"], r["lon"],
                               df.loc[j, "lat"], df.loc[j, "lon"]) < 0.3:
                dup = True
                break
        if not dup:
            keep.append(i)
    return df.loc[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# the MILP
# --------------------------------------------------------------------------- #

@dataclass
class CFLPResult:
    status: str = ""
    opened: pd.DataFrame = field(default_factory=pd.DataFrame)
    assignment: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_cost: float = 0.0
    transport_cost: float = 0.0
    fixed_cost: float = 0.0
    baseline_cost: float = 0.0
    baseline_transport: float = 0.0
    baseline_fixed: float = 0.0
    n_opened: int = 0
    solver_seconds: float = 0.0

    @property
    def annual_saving(self) -> float:
        return self.baseline_cost - self.total_cost

    def verdict(self) -> str:
        if self.annual_saving <= 0:
            return ("Keep the current network. No candidate configuration beats "
                    "it once fixed costs are counted.")
        names = ", ".join(self.opened["name"].tolist()) if len(self.opened) else "-"
        return (f"Open/keep: {names}. Modelled annual saving "
                f"${self.annual_saving:,.0f} versus the current network.")


def solve_cflp(stops: pd.DataFrame, candidates: pd.DataFrame,
               existing: Optional[pd.DataFrame] = None,
               weight_col: str = "annual_demand",
               cost_per_mile: float = 2.20,
               candidate_fixed_cost: float = 180_000.0,
               candidate_capacity: float = 1e9,
               max_sites: int = 1,
               must_keep_existing: bool = False,
               round_trip: bool = True,
               time_limit_s: int = 60) -> CFLPResult:
    """Choose which sites to open to minimise annual fixed + transport cost.

    Transport cost for stop i from site j is
        distance_ij  x  (2 if round_trip)  x  annual_deliveries_i  x  cost_per_mile
    which is an out-and-back approximation. validate_with_vrp() exists because
    of that approximation, not in spite of it.
    """
    import time as _t
    t0 = _t.time()

    cand = candidates.reset_index(drop=True).copy()
    if "fixed_cost" not in cand.columns:
        cand["fixed_cost"] = candidate_fixed_cost
    if "capacity" not in cand.columns:
        cand["capacity"] = candidate_capacity
    cand["fixed_cost"] = cand["fixed_cost"].fillna(candidate_fixed_cost).astype(float)
    cand["capacity"] = cand["capacity"].fillna(candidate_capacity).astype(float)

    if existing is not None and len(existing):
        ex_keys = set()
        for _, e in existing.iterrows():
            for ci, c in cand.iterrows():
                if haversine_miles(e["lat"], e["lon"], c["lat"], c["lon"]) < 0.3:
                    ex_keys.add(ci)
                    if "annual_fixed_cost" in existing.columns and not pd.isna(e.get("annual_fixed_cost")):
                        cand.loc[ci, "fixed_cost"] = float(e["annual_fixed_cost"])
                    if "capacity" in existing.columns and not pd.isna(e.get("capacity")):
                        cand.loc[ci, "capacity"] = float(e["capacity"])
        cand["is_existing"] = [i in ex_keys for i in range(len(cand))]
    else:
        cand["is_existing"] = False

    n_i, n_j = len(stops), len(cand)
    if n_i == 0 or n_j == 0:
        return CFLPResult(status="NO_DATA")

    w = (stops[weight_col].astype(float).values
         if weight_col in stops.columns else np.ones(n_i))
    freq = (stops["deliveries_per_year"].astype(float).values
            if "deliveries_per_year" in stops.columns else np.ones(n_i))

    mult = 2.0 if round_trip else 1.0
    dist = np.zeros((n_i, n_j))
    cost = np.zeros((n_i, n_j))
    for i in range(n_i):
        for j in range(n_j):
            d = haversine_miles(stops.iloc[i]["lat"], stops.iloc[i]["lon"],
                                cand.iloc[j]["lat"], cand.iloc[j]["lon"])
            dist[i, j] = d
            cost[i, j] = d * mult * freq[i] * cost_per_mile

    prob = pulp.LpProblem("CFLP", pulp.LpMinimize)
    y = {j: pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_j)}
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
         for i in range(n_i) for j in range(n_j)}

    prob += (pulp.lpSum(cand.iloc[j]["fixed_cost"] * y[j] for j in range(n_j))
             + pulp.lpSum(cost[i, j] * x[(i, j)]
                          for i in range(n_i) for j in range(n_j)))

    for i in range(n_i):
        prob += pulp.lpSum(x[(i, j)] for j in range(n_j)) == 1
    for i in range(n_i):
        for j in range(n_j):
            prob += x[(i, j)] <= y[j]
    for j in range(n_j):
        prob += (pulp.lpSum(w[i] * x[(i, j)] for i in range(n_i))
                 <= cand.iloc[j]["capacity"] * y[j])
    prob += pulp.lpSum(y[j] for j in range(n_j)) <= int(max_sites)
    prob += pulp.lpSum(y[j] for j in range(n_j)) >= 1

    if must_keep_existing:
        for j in range(n_j):
            if bool(cand.iloc[j]["is_existing"]):
                prob += y[j] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s))
    res = CFLPResult(status=pulp.LpStatus[prob.status],
                     solver_seconds=_t.time() - t0)
    if res.status not in ("Optimal", "Not Solved", "Undefined"):
        return res

    opened_idx = [j for j in range(n_j) if y[j].value() and y[j].value() > 0.5]
    res.opened = cand.iloc[opened_idx].reset_index(drop=True)
    res.n_opened = len(opened_idx)
    res.fixed_cost = float(sum(cand.iloc[j]["fixed_cost"] for j in opened_idx))

    rows = []
    for i in range(n_i):
        for j in opened_idx:
            if x[(i, j)].value() and x[(i, j)].value() > 0.5:
                rows.append({
                    "stop_id": stops.iloc[i].get("stop_id", f"S{i+1}"),
                    "stop": stops.iloc[i].get("label", ""),
                    "assigned_site": cand.iloc[j]["name"],
                    "site_id": cand.iloc[j]["site_id"],
                    "distance_mi": round(dist[i, j], 2),
                    "deliveries_per_year": float(freq[i]),
                    "annual_transport_$": round(cost[i, j], 2),
                })
    res.assignment = pd.DataFrame(rows)
    res.transport_cost = float(res.assignment["annual_transport_$"].sum()) \
        if len(res.assignment) else 0.0
    res.total_cost = res.fixed_cost + res.transport_cost

    # current network baseline: every stop served from its nearest existing site
    if existing is not None and len(existing):
        base_t = 0.0
        for i in range(n_i):
            best = min(haversine_miles(stops.iloc[i]["lat"], stops.iloc[i]["lon"],
                                       e["lat"], e["lon"])
                       for _, e in existing.iterrows())
            base_t += best * mult * freq[i] * cost_per_mile
        base_f = float(existing.get("annual_fixed_cost",
                                    pd.Series([candidate_fixed_cost] * len(existing))
                                    ).fillna(candidate_fixed_cost).sum())
        res.baseline_transport = base_t
        res.baseline_fixed = base_f
        res.baseline_cost = base_t + base_f
    return res


# --------------------------------------------------------------------------- #
# routed validation
# --------------------------------------------------------------------------- #

def validate_with_vrp(stops: pd.DataFrame, trucks: pd.DataFrame,
                      drivers: pd.DataFrame, warehouse_options: List[dict],
                      route_cfg, matrix_builder, label_key: str = "name"
                      ) -> pd.DataFrame:
    """Re-run the routing model once per candidate network and compare real
    routed cost.

    Each entry of warehouse_options is either a single {name, lat, lon} site or
    {name, warehouses: <DataFrame or list>} for a whole network of them. The
    second form is what lets a two-warehouse option be compared against one.

    matrix_builder(coords) -> (dist_mi, time_s)
    """
    from .routing import warehouse_coords, normalise_warehouses, solve_routes

    rows = []
    for opt in warehouse_options:
        if "warehouses" in opt:
            dep = normalise_warehouses(opt["warehouses"])
        else:
            dep = normalise_warehouses({"warehouse_name": opt.get(label_key, "Warehouse"),
                                    "lat": opt["lat"], "lon": opt["lon"]})

        # Every truck is re-based on the nearest site in the proposed network,
        # because a network comparison that kept trucks at a yard the option
        # does not contain would be measuring the wrong thing.
        t = trucks.copy()
        t["warehouse_id"] = [_nearest_warehouse_id(dep, r, stops) for _, r in t.iterrows()]

        coords = warehouse_coords(dep) + list(zip(stops["lat"].astype(float),
                                              stops["lon"].astype(float)))
        dist_mi, time_s = matrix_builder(coords)
        sol = solve_routes(stops, t, drivers, dist_mi, time_s, route_cfg,
                           label=str(opt.get(label_key, "network")),
                           warehouses=dep)
        rows.append({
            "Network": opt.get(label_key, "network"),
            "Sites": len(dep),
            "Trucks used": sol.trucks_used,
            "Stops served": sol.stops_served,
            "Stops dropped": len(sol.dropped),
            "Miles/day": round(sol.total_miles, 1),
            "Hours/day": round(sol.total_hours, 2),
            "Routed cost/day ($)": round(sol.total_cost, 2),
            "Status": sol.status,
        })
    df = pd.DataFrame(rows)
    if len(df):
        best = df["Routed cost/day ($)"].min()
        df["Δ vs best ($/day)"] = (df["Routed cost/day ($)"] - best).round(2)
    return df


def _nearest_warehouse_id(dep: pd.DataFrame, truck_row, stops: pd.DataFrame) -> str:
    """Keep a truck at its own site when that site is in this network,
    otherwise move it to the one nearest the centre of demand."""
    want = str(truck_row.get("warehouse_id", "") or "")
    if want and want in set(dep["warehouse_id"].astype(str)):
        return want
    clat = float(stops["lat"].astype(float).mean())
    clon = float(stops["lon"].astype(float).mean())
    best, best_d = str(dep.iloc[0]["warehouse_id"]), float("inf")
    for _, d in dep.iterrows():
        dist = haversine_miles(clat, clon, float(d["lat"]), float(d["lon"]))
        if dist < best_d:
            best, best_d = str(d["warehouse_id"]), dist
    return best
