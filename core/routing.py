"""
Vehicle Routing Problem with Time Windows, Hours-of-Service limits and driver
eligibility screening.

Formulation
-----------
Nodes      0 = warehouse, 1..n = delivery stops
Vehicles   heterogeneous trucks, each bound to one driver

Dimensions
  Capacity   kg, vehicle capacity = truck payload
  Time       seconds since midnight; transit = travel + service at origin,
             slack = waiting. Customer windows are hard bounds.
  Driving    travel seconds only, capped by the HOS driving limit so that
             service time cannot be used to dodge the 11-hour rule.

Objective (micro-dollars)
  arc        distance_mi * fuel_price / mpg
+ per truck  duty_seconds * (wage_per_sec)          <- labour, incl. waiting
+ per truck  fixed_cost_day when the truck is used
+ dropped    priority-weighted drop penalty
+ optional   global span coefficient for workload fairness

Driver age is NEVER a cost term. It is an eligibility filter applied before
the model is built, because under 49 CFR 391.11 it determines whether the
driver may legally take the run at all.

DECISION SUPPORT ONLY. This is a simplified model of 49 CFR 395 and is not a
compliance system of record.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

COST_SCALE = 1_000_000          # micro-dollars keeps integer rounding honest
KG_SCALE = 10                   # 0.1 kg resolution
HORIZON = 30 * 3600


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

@dataclass
class HOSRules:
    enabled: bool = True
    max_driving_hours: float = 11.0
    max_duty_span_hours: float = 14.0
    break_after_hours: float = 8.0
    break_minutes: int = 30
    min_interstate_age: int = 21
    min_cmv_age: int = 18
    enforce_weekly_cap: bool = True
    interstate_operation: bool = True


@dataclass
class CostModel:
    fuel_price_per_gal: float = 3.85
    ot_after_hours: float = 8.0
    ot_multiplier: float = 1.5
    cost_per_stop: float = 2.50
    co2_kg_per_gal: float = 10.18


@dataclass
class SolverSettings:
    time_limit_sec: int = 15
    first_solution: str = "PATH_CHEAPEST_ARC"
    metaheuristic: str = "GUIDED_LOCAL_SEARCH"
    drop_penalty_base: float = 5000.0
    fairness_coefficient: int = 0


@dataclass
class RouteConfig:
    hos: HOSRules = field(default_factory=HOSRules)
    cost: CostModel = field(default_factory=CostModel)
    solver: SolverSettings = field(default_factory=SolverSettings)


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

@dataclass
class RouteStop:
    seq: int
    stop_id: str
    label: str
    lat: float
    lon: float
    arrival_s: float
    service_start_s: float
    departure_s: float
    wait_s: float
    tw_open_s: float
    tw_close_s: float
    leg_miles: float
    cum_miles: float
    demand_kg: float
    load_remaining_kg: float


@dataclass
class Route:
    truck_id: str
    truck_name: str
    driver_id: str
    driver_name: str
    capacity_kg: float
    warehouse_id: str = ""
    warehouse_name: str = ""
    warehouse_lat: float = 0.0
    warehouse_lon: float = 0.0
    stops: List[RouteStop] = field(default_factory=list)
    start_s: float = 0.0
    end_s: float = 0.0
    total_miles: float = 0.0
    driving_s: float = 0.0
    service_s: float = 0.0
    wait_s: float = 0.0
    load_kg: float = 0.0
    fuel_gal: float = 0.0
    fuel_cost: float = 0.0
    labor_cost: float = 0.0
    fixed_cost: float = 0.0
    stop_cost: float = 0.0
    total_cost: float = 0.0
    co2_kg: float = 0.0
    breaks: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def n_stops(self) -> int:
        return len(self.stops)

    @property
    def duration_h(self) -> float:
        return max(0.0, (self.end_s - self.start_s) / 3600.0)

    @property
    def capacity_util(self) -> float:
        return self.load_kg / self.capacity_kg if self.capacity_kg > 0 else 0.0

    @property
    def time_util(self) -> float:
        return self.duration_h / 14.0 if self.duration_h else 0.0


@dataclass
class Solution:
    routes: List[Route] = field(default_factory=list)
    dropped: List[dict] = field(default_factory=list)
    excluded_trucks: List[dict] = field(default_factory=list)
    status: str = "NOT_SOLVED"
    objective: float = 0.0
    solve_seconds: float = 0.0
    label: str = "optimized"
    meta: dict = field(default_factory=dict)
    warehouses: Optional[pd.DataFrame] = None

    @property
    def warehouses_used(self) -> int:
        return len({r.warehouse_id for r in self.routes if r.n_stops > 0})

    @property
    def feasible(self) -> bool:
        return self.status in ("SUCCESS", "FAIL_TIMEOUT") and bool(self.routes)

    @property
    def all_served(self) -> bool:
        return self.feasible and not self.dropped

    @property
    def total_cost(self) -> float:
        return sum(r.total_cost for r in self.routes)

    @property
    def total_miles(self) -> float:
        return sum(r.total_miles for r in self.routes)

    @property
    def total_hours(self) -> float:
        return sum(r.duration_h for r in self.routes)

    @property
    def trucks_used(self) -> int:
        return sum(1 for r in self.routes if r.n_stops > 0)

    @property
    def stops_served(self) -> int:
        return sum(r.n_stops for r in self.routes)

    @property
    def co2_kg(self) -> float:
        return sum(r.co2_kg for r in self.routes)

    def summary(self) -> dict:
        return {
            "label": self.label,
            "status": self.status,
            "trucks_used": self.trucks_used,
            "stops_served": self.stops_served,
            "stops_dropped": len(self.dropped),
            "total_miles": round(self.total_miles, 1),
            "total_hours": round(self.total_hours, 2),
            "total_cost": round(self.total_cost, 2),
            "co2_kg": round(self.co2_kg, 1),
            "solve_seconds": round(self.solve_seconds, 2),
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def hhmm_to_seconds(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v) * 3600.0
    s = str(v).strip()
    if not s:
        return 0.0
    if ":" not in s:
        try:
            return float(s) * 3600.0
        except ValueError:
            return 0.0
    parts = s.split(":")
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60
    except ValueError:
        return 0.0


def seconds_to_hhmm(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "--:--"
    s = int(round(float(s)))
    return f"{(s // 3600) % 24:02d}:{(s % 3600) // 60:02d}"


def screen_drivers(trucks: pd.DataFrame, drivers: pd.DataFrame,
                   cfg: RouteConfig) -> Tuple[pd.DataFrame, List[dict]]:
    """Join trucks to drivers and remove pairs that cannot legally run today.

    Returns (eligible_fleet, exclusions). Exclusions carry a human-readable
    reason so the UI can explain why a truck is missing from the plan.
    """
    merged = trucks.merge(drivers, on="driver_id", how="left",
                          suffixes=("", "_drv"))
    keep, excluded = [], []
    hos = cfg.hos

    for _, r in merged.iterrows():
        reasons: List[str] = []
        driver_name = r.get("driver_name")
        if pd.isna(driver_name) or not str(driver_name).strip():
            reasons.append("no driver assigned to this truck")

        age = r.get("age", np.nan)
        if not pd.isna(age):
            age = float(age)
            if age < hos.min_cmv_age:
                reasons.append(
                    f"driver age {age:.0f} is below the {hos.min_cmv_age} "
                    f"minimum to operate a CMV")
            elif hos.interstate_operation and age < hos.min_interstate_age:
                reasons.append(
                    f"driver age {age:.0f} is below the interstate minimum of "
                    f"{hos.min_interstate_age} (49 CFR 391.11)")

        if hos.enabled and hos.enforce_weekly_cap:
            used = float(r.get("hours_used_week", 0) or 0)
            cap = float(r.get("max_weekly_hours", 70) or 70)
            if cap - used < 1.0:
                reasons.append(
                    f"only {max(0.0, cap - used):.1f} h remain on the "
                    f"{cap:.0f}-hour clock")

        start = hhmm_to_seconds(r.get("shift_start", "06:00"))
        end = hhmm_to_seconds(r.get("shift_end", "18:00"))
        if end - start < 3600:
            reasons.append("shift window is shorter than one hour")

        if reasons:
            excluded.append({
                "truck_id": r.get("truck_id"),
                "truck": r.get("truck_name", r.get("truck_id")),
                "driver": driver_name if not pd.isna(driver_name) else "-",
                "reason": "; ".join(reasons),
            })
        else:
            keep.append(r)

    fleet = pd.DataFrame(keep).reset_index(drop=True) if keep else merged.iloc[0:0]
    return fleet, excluded


def _status_name(code: int) -> str:
    return {0: "NOT_SOLVED", 1: "SUCCESS", 2: "FAIL_TIMEOUT",
            3: "FAIL", 4: "INVALID"}.get(code, f"CODE_{code}")


WAREHOUSE_COLUMNS = ["warehouse_id", "warehouse_name", "address", "lat", "lon"]


def normalise_warehouses(warehouses) -> pd.DataFrame:
    """Accept one warehouse or many, and always return a warehouse table.

    A single warehouse may be passed as the old {"lat","lon","label"} dict so that
    existing callers keep working. Warehouses occupy nodes 0..D-1 of every matrix.
    """
    if warehouses is None:
        raise ValueError("at least one warehouse is required")

    if isinstance(warehouses, dict):
        rows = [{
            "warehouse_id": warehouses.get("warehouse_id", "W01"),
            "warehouse_name": warehouses.get("warehouse_name", warehouses.get("label", "Warehouse")),
            "address": warehouses.get("address", ""),
            "lat": float(warehouses["lat"]), "lon": float(warehouses["lon"]),
        }]
        return pd.DataFrame(rows, columns=WAREHOUSE_COLUMNS)

    if isinstance(warehouses, (list, tuple)):
        warehouses = pd.DataFrame(list(warehouses))

    df = warehouses.copy().reset_index(drop=True)
    if "warehouse_id" not in df.columns:
        df["warehouse_id"] = [f"W{i+1:02d}" for i in range(len(df))]
    if "warehouse_name" not in df.columns:
        df["warehouse_name"] = df.get("label", pd.Series(
            [f"Warehouse {i+1}" for i in range(len(df))]))
    if "address" not in df.columns:
        df["address"] = ""
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    return df[WAREHOUSE_COLUMNS].reset_index(drop=True)


def warehouse_coords(warehouses) -> List[Tuple[float, float]]:
    d = normalise_warehouses(warehouses)
    return list(zip(d["lat"].astype(float), d["lon"].astype(float)))


def assign_truck_warehouses(fleet: pd.DataFrame, warehouses: pd.DataFrame) -> np.ndarray:
    """Map each truck to the row index of the warehouse it is based at.

    A truck names its base in `warehouse_id`. Anything missing or unrecognised
    falls back to the first warehouse, so a fleet table written before warehouses
    existed still solves.
    """
    lookup = {str(v): i for i, v in enumerate(warehouses["warehouse_id"])}
    name_lookup = {str(v).strip().lower(): i
                   for i, v in enumerate(warehouses["warehouse_name"])}
    out = []
    for _, r in fleet.iterrows():
        key = str(r.get("warehouse_id", "") or "").strip()
        idx = lookup.get(key)
        if idx is None:
            idx = name_lookup.get(key.lower())
        out.append(int(idx) if idx is not None else 0)
    return np.array(out, dtype=int)


def _stop_arrays(stops: pd.DataFrame, n_warehouses: int = 1) -> dict:
    n = len(stops) + n_warehouses
    service = np.zeros(n)
    demand = np.zeros(n)
    tw = np.zeros((n, 2))
    for d in range(n_warehouses):
        tw[d] = [0.0, HORIZON]
    for i, (_, s) in enumerate(stops.iterrows(), start=n_warehouses):
        service[i] = float(s.get("service_min", 10) or 0) * 60.0
        demand[i] = float(s.get("demand_kg", 0) or 0)
        tw[i, 0] = hhmm_to_seconds(s.get("tw_open", "06:00"))
        tw[i, 1] = hhmm_to_seconds(s.get("tw_close", "18:00"))
        if tw[i, 1] <= tw[i, 0]:
            tw[i, 1] = tw[i, 0] + 3600.0
    return {"service": service, "demand": demand, "tw": tw}


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #

def solve_routes(stops: pd.DataFrame, trucks: pd.DataFrame,
                 drivers: pd.DataFrame, dist_mi: np.ndarray,
                 time_s: np.ndarray, cfg: Optional[RouteConfig] = None,
                 label: str = "optimized", warehouses=None) -> Solution:
    """Solve the day.

    `warehouses` is one warehouse or a table of them. Warehouses occupy nodes 0..D-1 of
    dist_mi / time_s and stops follow, so the matrices must be built from
    warehouse_coords(warehouses) + the stop coordinates, in that order. Each truck
    starts and ends at the warehouse named in its `warehouse_id` column.
    """
    cfg = cfg or RouteConfig()
    t0 = _time.time()

    dep = normalise_warehouses(warehouses if warehouses is not None
                           else {"lat": 0.0, "lon": 0.0})
    n_warehouses = len(dep)

    fleet, excluded = screen_drivers(trucks, drivers, cfg)
    sol = Solution(excluded_trucks=excluded, label=label)
    if len(fleet) == 0:
        sol.status = "NO_ELIGIBLE_TRUCKS"
        sol.solve_seconds = _time.time() - t0
        return sol
    if len(stops) == 0:
        sol.status = "NO_STOPS"
        sol.solve_seconds = _time.time() - t0
        return sol

    arr = _stop_arrays(stops, n_warehouses)
    service, demand, tw = arr["service"], arr["demand"], arr["tw"]
    n_nodes = len(stops) + n_warehouses
    n_veh = len(fleet)

    if dist_mi.shape[0] != n_nodes:
        raise ValueError(
            f"matrix is {dist_mi.shape[0]}x{dist_mi.shape[1]} but the model "
            f"has {n_warehouses} warehouse(s) + {len(stops)} stops = {n_nodes} nodes. "
            f"Build it from warehouse_coords(warehouses) + the stop coordinates.")

    # Each truck starts and ends at its own warehouse. A truck is based at a yard;
    # it does not finish the day somewhere else.
    veh_warehouse = assign_truck_warehouses(fleet, dep)
    starts = [int(d) for d in veh_warehouse]
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_veh, starts, list(starts))
    routing = pywrapcp.RoutingModel(manager)

    shift_start = np.array([hhmm_to_seconds(r.get("shift_start", "06:00"))
                            for _, r in fleet.iterrows()])
    shift_end = np.array([hhmm_to_seconds(r.get("shift_end", "18:00"))
                          for _, r in fleet.iterrows()])

    # --- arc cost: fuel only; labour is charged on the time span ----------
    def make_arc_cb(fuel_per_mile: float):
        def cb(from_index, to_index):
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            return int(dist_mi[i, j] * fuel_per_mile * COST_SCALE)
        return cb

    for vi in range(n_veh):
        mpg = max(1.0, float(fleet.iloc[vi].get("mpg", 7.0) or 7.0))
        idx = routing.RegisterTransitCallback(
            make_arc_cb(cfg.cost.fuel_price_per_gal / mpg))
        routing.SetArcCostEvaluatorOfVehicle(idx, vi)
        routing.SetFixedCostOfVehicle(
            int(float(fleet.iloc[vi].get("fixed_cost_day", 0) or 0) * COST_SCALE),
            vi)

    # --- time dimension ---------------------------------------------------
    def time_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(round(time_s[i, j] + service[i]))

    t_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(t_idx, HORIZON, HORIZON, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # Warehouse nodes are vehicle starts/ends, not visitable indices, so only the
    # stop nodes get a customer window.
    for i in range(n_warehouses, n_nodes):
        index = manager.NodeToIndex(i)
        time_dim.CumulVar(index).SetRange(int(tw[i, 0]), int(tw[i, 1]))

    for vi in range(n_veh):
        row = fleet.iloc[vi]
        s_idx, e_idx = routing.Start(vi), routing.End(vi)
        time_dim.CumulVar(s_idx).SetRange(int(shift_start[vi]), int(shift_end[vi]))
        time_dim.CumulVar(e_idx).SetRange(int(shift_start[vi]), int(shift_end[vi]))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(s_idx))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(e_idx))

        span = shift_end[vi] - shift_start[vi]
        if cfg.hos.enabled:
            span = min(span, cfg.hos.max_duty_span_hours * 3600)
            if cfg.hos.enforce_weekly_cap:
                left = (float(row.get("max_weekly_hours", 70) or 70)
                        - float(row.get("hours_used_week", 0) or 0))
                span = min(span, max(3600.0, left * 3600))
        time_dim.SetSpanUpperBoundForVehicle(int(span), vi)

        wage_per_sec = float(row.get("hourly_wage", 28.0) or 28.0) / 3600.0
        time_dim.SetSpanCostCoefficientForVehicle(
            max(1, int(round(wage_per_sec * COST_SCALE))), vi)

    if cfg.solver.fairness_coefficient > 0:
        time_dim.SetGlobalSpanCostCoefficient(int(cfg.solver.fairness_coefficient))

    # --- driving-only dimension (the 11-hour rule) ------------------------
    if cfg.hos.enabled:
        def drive_cb(from_index, to_index):
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            return int(round(time_s[i, j]))
        d_idx = routing.RegisterTransitCallback(drive_cb)
        routing.AddDimensionWithVehicleCapacity(
            d_idx, 0, [int(cfg.hos.max_driving_hours * 3600)] * n_veh,
            True, "Driving")

    # --- capacity ---------------------------------------------------------
    def dem_cb(from_index):
        return int(round(demand[manager.IndexToNode(from_index)] * KG_SCALE))

    cap_idx = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(
        cap_idx, 0,
        [int(round(float(fleet.iloc[vi].get("capacity_kg", 1000) or 1000)
                   * KG_SCALE)) for vi in range(n_veh)],
        True, "Capacity")

    # --- HOS break --------------------------------------------------------
    if cfg.hos.enabled and cfg.hos.break_minutes > 0:
        node_visit = [0] * routing.Size()
        for idx in range(routing.Size()):
            node_visit[idx] = int(round(service[manager.IndexToNode(idx)]))
        slv = routing.solver()
        for vi in range(n_veh):
            dur = int(cfg.hos.break_minutes * 60)
            earliest = int(shift_start[vi] + cfg.hos.break_after_hours * 3600 * 0.45)
            latest = int(min(shift_start[vi] + cfg.hos.break_after_hours * 3600,
                             shift_end[vi] - dur))
            if latest <= earliest:
                continue
            br = slv.FixedDurationIntervalVar(
                earliest, latest, dur, True, f"Break_{vi}")
            time_dim.SetBreakIntervalsOfVehicle([br], vi, node_visit)

    # --- droppable stops so we get a diagnosis, not a failure -------------
    prio = (stops["priority"].astype(float).values
            if "priority" in stops.columns else np.ones(len(stops)))
    for i in range(n_warehouses, n_nodes):
        k = i - n_warehouses
        p = float(prio[k]) if k < len(prio) else 1.0
        penalty = cfg.solver.drop_penalty_base * (4.0 - min(3.0, p))
        routing.AddDisjunction([manager.NodeToIndex(i)],
                               int(penalty * COST_SCALE))

    # --- search -----------------------------------------------------------
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, cfg.solver.first_solution,
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, cfg.solver.metaheuristic,
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(max(1, int(cfg.solver.time_limit_sec)))

    assignment = routing.SolveWithParameters(params)
    sol.solve_seconds = _time.time() - t0
    sol.status = _status_name(routing.status())
    if assignment is None:
        return sol

    sol.objective = assignment.ObjectiveValue() / COST_SCALE
    sol.warehouses = dep
    sol.routes, sol.dropped = _extract(routing, manager, assignment, stops,
                                       fleet, cfg, dist_mi, time_s,
                                       service, demand, tw, dep, veh_warehouse)
    return sol


def _extract(routing, manager, assignment, stops, fleet, cfg,
             dist_mi, time_s, service, demand, tw, dep, veh_warehouse):
    n_warehouses = len(dep)
    time_dim = routing.GetDimensionOrDie("Time")
    sdf = stops.reset_index(drop=True)
    routes: List[Route] = []
    visited = set()

    for vi in range(len(fleet)):
        v = fleet.iloc[vi]
        home = int(veh_warehouse[vi])
        r = Route(
            truck_id=str(v.get("truck_id", f"T{vi+1}")),
            truck_name=str(v.get("truck_name", v.get("truck_id", f"T{vi+1}"))),
            driver_id=str(v.get("driver_id", "")),
            driver_name=str(v.get("driver_name", v.get("driver_id", ""))),
            capacity_kg=float(v.get("capacity_kg", 0) or 0),
            warehouse_id=str(dep.iloc[home]["warehouse_id"]),
            warehouse_name=str(dep.iloc[home]["warehouse_name"]),
            warehouse_lat=float(dep.iloc[home]["lat"]),
            warehouse_lon=float(dep.iloc[home]["lon"]),
        )

        index = routing.Start(vi)
        chain = []
        while not routing.IsEnd(index):
            chain.append((manager.IndexToNode(index), index))
            index = assignment.Value(routing.NextVar(index))
        chain.append((manager.IndexToNode(index), index))

        served_nodes = [n for n, _ in chain[1:-1]]
        r.load_kg = float(sum(demand[n] for n in served_nodes))
        remaining = r.load_kg
        r.start_s = float(assignment.Min(time_dim.CumulVar(routing.Start(vi))))

        prev, cum, seq = home, 0.0, 0
        for k, (node, idx) in enumerate(chain):
            if k == 0:
                continue
            leg = float(dist_mi[prev, node])
            cum += leg
            r.driving_s += float(time_s[prev, node])
            if routing.IsEnd(idx):
                r.end_s = float(assignment.Min(time_dim.CumulVar(idx)))
                break

            seq += 1
            visited.add(node)
            arrival = float(assignment.Min(time_dim.CumulVar(idx)))
            start_srv = max(arrival, tw[node, 0])
            wait = max(0.0, start_srv - arrival)
            depart = start_srv + service[node]
            r.service_s += service[node]
            r.wait_s += wait
            remaining -= demand[node]

            s = sdf.iloc[node - n_warehouses]
            r.stops.append(RouteStop(
                seq=seq,
                stop_id=str(s.get("stop_id", f"S{node}")),
                label=str(s.get("label", s.get("address", f"Stop {node}"))),
                lat=float(s["lat"]), lon=float(s["lon"]),
                arrival_s=arrival, service_start_s=start_srv,
                departure_s=depart, wait_s=wait,
                tw_open_s=float(tw[node, 0]), tw_close_s=float(tw[node, 1]),
                leg_miles=leg, cum_miles=cum,
                demand_kg=float(demand[node]),
                load_remaining_kg=round(max(0.0, remaining), 1),
            ))
            prev = node

        r.total_miles = cum
        if r.n_stops == 0:
            r.end_s = r.start_s

        try:
            for br in time_dim.GetBreakIntervalsOfVehicle(vi):
                if assignment.PerformedValue(br):
                    r.breaks.append((float(assignment.StartValue(br)),
                                     float(assignment.EndValue(br))))
        except Exception:                                         # noqa: BLE001
            pass

        _cost_route(r, v, cfg)
        routes.append(r)

    dropped = []
    for i in range(n_warehouses, len(stops) + n_warehouses):
        if i not in visited:
            s = sdf.iloc[i - n_warehouses]
            dropped.append({
                "stop_id": s.get("stop_id", f"S{i}"),
                "label": s.get("label", s.get("address", "")),
                "demand_kg": float(s.get("demand_kg", 0) or 0),
                "tw_open": s.get("tw_open", ""),
                "tw_close": s.get("tw_close", ""),
                "priority": s.get("priority", 1),
                "lat": float(s["lat"]), "lon": float(s["lon"]),
            })
    return routes, dropped


def _cost_route(r: Route, v, cfg: RouteConfig) -> None:
    mpg = max(1.0, float(v.get("mpg", 7.0) or 7.0))
    r.fuel_gal = r.total_miles / mpg
    r.fuel_cost = r.fuel_gal * cfg.cost.fuel_price_per_gal

    wage = float(v.get("hourly_wage", 28.0) or 28.0)
    duty = r.duration_h
    ot = max(0.0, duty - cfg.cost.ot_after_hours)
    r.labor_cost = (duty - ot) * wage + ot * wage * cfg.cost.ot_multiplier

    r.fixed_cost = float(v.get("fixed_cost_day", 0) or 0) if r.n_stops else 0.0
    r.stop_cost = r.n_stops * cfg.cost.cost_per_stop
    r.total_cost = r.fuel_cost + r.labor_cost + r.fixed_cost + r.stop_cost
    r.co2_kg = r.fuel_gal * cfg.cost.co2_kg_per_gal


# --------------------------------------------------------------------------- #
# naive baseline, for the savings comparison
# --------------------------------------------------------------------------- #

def baseline_routes(stops: pd.DataFrame, trucks: pd.DataFrame,
                    drivers: pd.DataFrame, dist_mi: np.ndarray,
                    time_s: np.ndarray,
                    cfg: Optional[RouteConfig] = None,
                    warehouses=None) -> Solution:
    """What a dispatcher does without optimisation: fill one truck at a time
    in entry order, then sequence each truck nearest-neighbour from its warehouse.

    Time windows are NOT enforced here. Violations are counted and reported,
    which is itself part of the comparison story.
    """
    cfg = cfg or RouteConfig()
    t0 = _time.time()
    dep = normalise_warehouses(warehouses if warehouses is not None
                           else {"lat": 0.0, "lon": 0.0})
    n_warehouses = len(dep)
    fleet, excluded = screen_drivers(trucks, drivers, cfg)
    sol = Solution(excluded_trucks=excluded, label="baseline", warehouses=dep)
    if len(fleet) == 0 or len(stops) == 0:
        sol.status = "NO_ELIGIBLE_TRUCKS" if len(fleet) == 0 else "NO_STOPS"
        return sol

    arr = _stop_arrays(stops, n_warehouses)
    service, demand, tw = arr["service"], arr["demand"], arr["tw"]
    sdf = stops.reset_index(drop=True)
    veh_warehouse = assign_truck_warehouses(fleet, dep)

    caps = [float(fleet.iloc[v].get("capacity_kg", 1000) or 1000)
            for v in range(len(fleet))]
    buckets: List[List[int]] = [[] for _ in range(len(fleet))]
    loads = [0.0] * len(fleet)
    unassigned = []

    for node in range(n_warehouses, len(stops) + n_warehouses):
        placed = False
        for v in range(len(fleet)):
            if loads[v] + demand[node] <= caps[v]:
                buckets[v].append(node)
                loads[v] += demand[node]
                placed = True
                break
        if not placed:
            unassigned.append(node)

    violations = 0
    for v, nodes in enumerate(buckets):
        row = fleet.iloc[v]
        home = int(veh_warehouse[v])
        r = Route(
            truck_id=str(row.get("truck_id", f"T{v+1}")),
            truck_name=str(row.get("truck_name", row.get("truck_id", f"T{v+1}"))),
            driver_id=str(row.get("driver_id", "")),
            driver_name=str(row.get("driver_name", "")),
            capacity_kg=caps[v],
            warehouse_id=str(dep.iloc[home]["warehouse_id"]),
            warehouse_name=str(dep.iloc[home]["warehouse_name"]),
            warehouse_lat=float(dep.iloc[home]["lat"]),
            warehouse_lon=float(dep.iloc[home]["lon"]),
        )
        r.start_s = hhmm_to_seconds(row.get("shift_start", "06:00"))
        r.load_kg = loads[v]
        remaining = loads[v]

        # nearest-neighbour sequencing from this truck's own warehouse
        order, pool, cur = [], list(nodes), home
        while pool:
            nxt = min(pool, key=lambda j: dist_mi[cur, j])
            order.append(nxt)
            pool.remove(nxt)
            cur = nxt

        clock, prev, cum, seq = r.start_s, home, 0.0, 0
        for node in order:
            leg = float(dist_mi[prev, node])
            cum += leg
            travel = float(time_s[prev, node])
            r.driving_s += travel
            arrival = clock + travel
            start_srv = max(arrival, tw[node, 0])
            if arrival > tw[node, 1]:
                violations += 1
            wait = max(0.0, start_srv - arrival)
            r.wait_s += wait
            r.service_s += service[node]
            clock = start_srv + service[node]
            remaining -= demand[node]
            seq += 1

            s = sdf.iloc[node - n_warehouses]
            r.stops.append(RouteStop(
                seq=seq, stop_id=str(s.get("stop_id", f"S{node}")),
                label=str(s.get("label", s.get("address", ""))),
                lat=float(s["lat"]), lon=float(s["lon"]),
                arrival_s=arrival, service_start_s=start_srv,
                departure_s=clock, wait_s=wait,
                tw_open_s=float(tw[node, 0]), tw_close_s=float(tw[node, 1]),
                leg_miles=leg, cum_miles=cum, demand_kg=float(demand[node]),
                load_remaining_kg=round(max(0.0, remaining), 1)))
            prev = node

        back = float(dist_mi[prev, home])
        cum += back
        r.driving_s += float(time_s[prev, home])
        clock += float(time_s[prev, home])
        r.total_miles = cum
        r.end_s = clock if nodes else r.start_s
        _cost_route(r, row, cfg)
        sol.routes.append(r)

    sol.dropped = [{
        "stop_id": sdf.iloc[n - n_warehouses].get("stop_id", f"S{n}"),
        "label": sdf.iloc[n - n_warehouses].get("label", ""),
        "demand_kg": float(demand[n]),
        "tw_open": sdf.iloc[n - n_warehouses].get("tw_open", ""),
        "tw_close": sdf.iloc[n - n_warehouses].get("tw_close", ""),
        "priority": sdf.iloc[n - n_warehouses].get("priority", 1),
        "lat": float(sdf.iloc[n - n_warehouses]["lat"]),
        "lon": float(sdf.iloc[n - n_warehouses]["lon"]),
    } for n in unassigned]

    sol.status = "SUCCESS"
    sol.solve_seconds = _time.time() - t0
    sol.meta["time_window_violations"] = violations
    return sol


# --------------------------------------------------------------------------- #
# tabular views
# --------------------------------------------------------------------------- #

def compliance_check(sol: Solution, cfg: Optional[RouteConfig] = None) -> List[dict]:
    """Post-solve HOS verification.

    The 30-minute break is modelled as an OPTIONAL interval, because a route
    with under 8 hours of driving does not require one and forcing the
    interval would stretch short routes artificially. That means the break can
    legitimately be absent, so compliance is verified here rather than assumed.
    """
    cfg = cfg or RouteConfig()
    issues: List[dict] = []
    if not cfg.hos.enabled:
        return issues

    for r in sol.routes:
        if r.n_stops == 0:
            continue
        drive_h = r.driving_s / 3600.0
        duty_h = r.duration_h

        if drive_h > cfg.hos.break_after_hours and not r.breaks:
            issues.append({
                "Truck": r.truck_name, "Driver": r.driver_name,
                "Severity": "Violation",
                "Issue": f"{drive_h:.1f} h driving with no 30-minute break "
                         f"scheduled (required after "
                         f"{cfg.hos.break_after_hours:.0f} h)"})
        if drive_h > cfg.hos.max_driving_hours + 0.01:
            issues.append({
                "Truck": r.truck_name, "Driver": r.driver_name,
                "Severity": "Violation",
                "Issue": f"{drive_h:.1f} h driving exceeds the "
                         f"{cfg.hos.max_driving_hours:.0f}-hour limit"})
        if duty_h > cfg.hos.max_duty_span_hours + 0.01:
            issues.append({
                "Truck": r.truck_name, "Driver": r.driver_name,
                "Severity": "Violation",
                "Issue": f"{duty_h:.1f} h duty span exceeds the "
                         f"{cfg.hos.max_duty_span_hours:.0f}-hour window"})
        if drive_h > cfg.hos.max_driving_hours * 0.9:
            issues.append({
                "Truck": r.truck_name, "Driver": r.driver_name,
                "Severity": "Watch",
                "Issue": f"{drive_h:.1f} h driving is within 10% of the limit"})

        late = [s for s in r.stops if s.arrival_s > s.tw_close_s + 1]
        for s in late:
            issues.append({
                "Truck": r.truck_name, "Driver": r.driver_name,
                "Severity": "Violation",
                "Issue": f"arrives at {s.label} at "
                         f"{seconds_to_hhmm(s.arrival_s)}, after the window "
                         f"closes at {seconds_to_hhmm(s.tw_close_s)}"})
    return issues


def routes_to_frame(sol: Solution) -> pd.DataFrame:
    rows = []
    for r in sol.routes:
        for s in r.stops:
            rows.append({
                "Truck": r.truck_name,
                "Warehouse": r.warehouse_name,
                "Driver": r.driver_name,
                "Seq": s.seq,
                "Stop": s.label,
                "Arrive": seconds_to_hhmm(s.arrival_s),
                "Start": seconds_to_hhmm(s.service_start_s),
                "Depart": seconds_to_hhmm(s.departure_s),
                "Window": f"{seconds_to_hhmm(s.tw_open_s)}-{seconds_to_hhmm(s.tw_close_s)}",
                "Late": s.arrival_s > s.tw_close_s + 1,
                "Wait (min)": round(s.wait_s / 60, 1),
                "Leg (mi)": round(s.leg_miles, 1),
                "Cum (mi)": round(s.cum_miles, 1),
                "Drop (kg)": round(s.demand_kg, 1),
                "Left on truck (kg)": s.load_remaining_kg,
            })
    return pd.DataFrame(rows)


def truck_summary_frame(sol: Solution) -> pd.DataFrame:
    rows = []
    for r in sol.routes:
        rows.append({
            "Truck": r.truck_name,
            "Warehouse": r.warehouse_name,
            "Driver": r.driver_name,
            "Stops": r.n_stops,
            "Miles": round(r.total_miles, 1),
            "Duty (h)": round(r.duration_h, 2),
            "Driving (h)": round(r.driving_s / 3600, 2),
            "Wait (h)": round(r.wait_s / 3600, 2),
            "Load (kg)": round(r.load_kg, 1),
            "Capacity (kg)": round(r.capacity_kg, 1),
            "Cap used %": round(100 * r.capacity_util, 1),
            "Start": seconds_to_hhmm(r.start_s),
            "End": seconds_to_hhmm(r.end_s),
            "Break": ", ".join(f"{seconds_to_hhmm(a)}-{seconds_to_hhmm(b)}"
                               for a, b in r.breaks) or "-",
            "Cost ($)": round(r.total_cost, 2),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# fuel price sensitivity
# --------------------------------------------------------------------------- #

def cost_breakdown(sol: Solution) -> pd.DataFrame:
    """Where the money goes on the solved day, one row per component."""
    fuel = sum(r.fuel_cost for r in sol.routes)
    labor = sum(r.labor_cost for r in sol.routes)
    fixed = sum(r.fixed_cost for r in sol.routes)
    stop = sum(r.stop_cost for r in sol.routes)
    total = fuel + labor + fixed + stop
    rows = [("Fuel", fuel), ("Driver wages", labor),
            ("Vehicle fixed cost", fixed), ("Handling at stops", stop)]
    return pd.DataFrame([{
        "Component": k,
        "Cost ($/day)": round(v, 2),
        "Share": f"{(100 * v / total if total else 0):.0f}%",
    } for k, v in rows])


def fuel_sensitivity(sol: Solution, cfg: RouteConfig,
                     prices: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Re-cost the SAME routes at a range of diesel prices.

    The route plan is held fixed and only the fuel bill is recomputed, which
    is the right question for an operator asking "what does this run cost me
    if diesel goes to $5?" - the stops and the sequence have not changed, only
    what it costs to drive them.

    Returns one row per price with cost per day, per stop, and the monthly
    cost of running this route once a week, twice a month and once a month.
    """
    if not sol.routes:
        return pd.DataFrame()
    if prices is None:
        base = cfg.cost.fuel_price_per_gal
        prices = [round(base + step, 2)
                  for step in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
                  if base + step > 0.2]

    gallons = sum(r.fuel_gal for r in sol.routes)
    non_fuel = sum(r.labor_cost + r.fixed_cost + r.stop_cost
                   for r in sol.routes)
    stops = max(1, sol.stops_served)

    rows = []
    for p in prices:
        day = non_fuel + gallons * p
        rows.append({
            "Diesel ($/gal)": round(float(p), 2),
            "Cost per run ($)": round(day, 2),
            "Cost per stop ($)": round(day / stops, 2),
            "Weekly ($/mo)": round(day * 4.33, 0),
            "Twice a month ($/mo)": round(day * 2, 0),
            "Once a month ($/mo)": round(day * 1, 0),
            "Is current": abs(float(p) - cfg.cost.fuel_price_per_gal) < 1e-9,
        })
    return pd.DataFrame(rows)


def frequency_tradeoff(sol: Solution, cfg: RouteConfig) -> dict:
    """What dropping from twice a month to once a month saves, and what it
    costs the customer in service.

    This is the decision an operator actually faces when fuel rises: the
    saving is real, and so is halving how often each store is served.
    """
    if not sol.routes:
        return {}
    day = sol.total_cost
    return {
        "cost_per_run": round(day, 2),
        "twice_monthly": round(day * 2, 2),
        "once_monthly": round(day, 2),
        "monthly_saving": round(day, 2),
        "annual_saving": round(day * 12, 2),
        "fuel_share": round(
            100 * sum(r.fuel_cost for r in sol.routes) / day, 1) if day else 0.0,
        "stops_affected": sol.stops_served,
    }
