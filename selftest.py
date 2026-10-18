"""
End-to-end check with no network calls and no Streamlit.

    python selftest.py

Exercises every module the app depends on. Run it after any change to the
core/ package; if this passes, the app will not fail for modelling reasons.
"""

import sys

import numpy as np
import pandas as pd

from core import demo_data, distance, finance, fleet, routing, warehouse

FAIL = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def main():
    print("=" * 70)
    print("FleetView self-test")
    print("=" * 70)

    stops = demo_data.demo_stops()
    main_wh = demo_data.demo_warehouse()   # not 'warehouse': that is the module
    trucks = demo_data.default_trucks(3)
    drivers = demo_data.default_drivers(3)
    check("demo data loads", len(stops) == 20 and len(trucks) == 3)

    warehouses1 = demo_data.demo_warehouses(1)
    coords = routing.warehouse_coords(warehouses1) + list(zip(stops.lat, stops.lon))
    mult = distance.congestion_multiplier(6, 18)
    dist_mi, time_s, meta = distance.build_matrices(coords, time_multiplier=mult)
    check("matrices built", dist_mi.shape == (21, 21),
          f"congestion x{mult:.3f}")
    check("matrix diagonal is zero", float(np.trace(dist_mi)) == 0.0)

    cfg = routing.RouteConfig()
    cfg.solver.time_limit_sec = 6

    # --- routing ----------------------------------------------------------
    sol = routing.solve_routes(stops, trucks, drivers, dist_mi, time_s, cfg,
                               warehouses=warehouses1)
    check("routing solves", sol.status == "SUCCESS", sol.status)
    check("all stops served", sol.all_served,
          f"{sol.stops_served}/{len(stops)} served, {len(sol.dropped)} dropped")
    check("capacity respected",
          all(r.load_kg <= r.capacity_kg + 1e-6 for r in sol.routes))
    check("time windows respected",
          all(s.arrival_s <= s.tw_close_s + 1
              for r in sol.routes for s in r.stops))
    check("duty span within limit",
          all(r.duration_h <= cfg.hos.max_duty_span_hours + 0.01
              for r in sol.routes))
    check("driving within limit",
          all(r.driving_s / 3600 <= cfg.hos.max_driving_hours + 0.01
              for r in sol.routes))

    base = routing.baseline_routes(stops, trucks, drivers, dist_mi, time_s, cfg,
                                   warehouses=warehouses1)
    check("baseline solves", base.status == "SUCCESS")
    saving = 100 * (base.total_miles - sol.total_miles) / base.total_miles
    check("optimised beats baseline", sol.total_miles < base.total_miles,
          f"{saving:.1f}% fewer miles")

    issues = routing.compliance_check(sol, cfg)
    hard = [i for i in issues if i["Severity"] == "Violation"]
    check("no HOS violations in optimised plan", not hard,
          f"{len(issues)} total notes")

    # --- driver eligibility ----------------------------------------------
    d2 = drivers.copy()
    d2.loc[0, "age"] = 19
    _, excl = routing.screen_drivers(trucks, d2, cfg)
    check("underage driver excluded", len(excl) == 1 and "21" in excl[0]["reason"],
          excl[0]["reason"] if excl else "none")

    d3 = drivers.copy()
    d3.loc[1, "hours_used_week"] = 69.8
    _, excl3 = routing.screen_drivers(trucks, d3, cfg)
    check("weekly cap exclusion works", len(excl3) == 1)

    # --- fleet sizing -----------------------------------------------------
    cfg.solver.time_limit_sec = 3
    sw = fleet.fleet_sweep(stops, trucks, drivers, dist_mi, time_s, cfg,
                           max_trucks=4, warehouses=warehouses1)
    need = fleet.minimum_fleet(sw)
    check("fleet sweep runs", len(sw) == 4 and need is not None,
          f"minimum fleet = {need}")
    check("sweep monotone in stops served",
          sw["served"].is_monotonic_increasing)

    cfg.solver.time_limit_sec = 2
    util = fleet.utilisation_study(stops, trucks, drivers, dist_mi, time_s, cfg,
                                   n_days=6, max_trucks=4, warehouses=warehouses1)
    check("utilisation study runs", len(util.days) == 6)
    check("days-per-year is bounded",
          0 <= util.days_needing_at_least(1) <= 365,
          f"truck 1 needed {util.days_needing_at_least(1):.0f} d/yr")

    # --- warehouse --------------------------------------------------------
    wk = stops.copy()
    wk["deliveries_per_year"] = wk["deliveries_per_month"] * 12
    wk["annual_demand"] = wk["demand_kg"] * wk["deliveries_per_year"]
    existing = pd.DataFrame([{"name": "Main", "lat": main_wh["lat"],
                              "lon": main_wh["lon"],
                              "annual_fixed_cost": 180_000, "capacity": 1e12}])
    cand = warehouse.generate_candidates(wk, existing=existing, n_clusters=4,
                                         grid_n=3)
    check("candidates generated", len(cand) >= 8, f"{len(cand)} sites")

    r1 = warehouse.solve_cflp(wk, cand, existing=existing, max_sites=1)
    check("CFLP solves", r1.status == "Optimal", r1.status)
    check("exactly one site opened", r1.n_opened == 1)
    check("every stop assigned", len(r1.assignment) == len(wk))

    r_expensive = warehouse.solve_cflp(wk, cand, existing=existing, max_sites=2,
                                       candidate_fixed_cost=5_000_000,
                                       must_keep_existing=True)
    check("prohibitive fixed cost -> no expansion", r_expensive.n_opened == 1,
          f"opened {r_expensive.n_opened}")

    def mb(c):
        d, t, _ = distance.build_matrices(c, time_multiplier=mult)
        return d, t

    val = warehouse.validate_with_vrp(
        stops, trucks, drivers,
        [{"name": "Current", "warehouses": warehouses1},
         {"name": "Proposed", "lat": float(r1.opened.iloc[0]["lat"]),
          "lon": float(r1.opened.iloc[0]["lon"])}], cfg, mb)
    check("routed validation runs", len(val) == 2)

    # --- multi-warehouse ------------------------------------------------------
    warehouses2 = demo_data.demo_warehouses(2)
    check("two warehouses defined", len(warehouses2) == 2)
    t2 = demo_data.default_trucks(4, "W01")
    t2.loc[2:, "warehouse_id"] = "W02"
    d2 = demo_data.default_drivers(4)
    coords2 = routing.warehouse_coords(warehouses2) + list(zip(stops.lat, stops.lon))
    dm2, tm2, _ = distance.build_matrices(coords2, time_multiplier=mult)
    check("multi-warehouse matrix sized right", dm2.shape == (22, 22),
          f"{dm2.shape}")

    cfg.solver.time_limit_sec = 6
    m = routing.solve_routes(stops, t2, d2, dm2, tm2, cfg, warehouses=warehouses2)
    check("multi-warehouse solves", m.status == "SUCCESS", m.status)
    check("multi-warehouse serves everything", m.all_served,
          f"{m.stops_served}/{len(stops)}")
    check("both warehouses used", m.warehouses_used == 2,
          f"{m.warehouses_used} warehouse(s) in use")

    homes = {(round(r.warehouse_lat, 4), round(r.warehouse_lon, 4)) for r in m.routes}
    want = {(round(float(d["lat"]), 4), round(float(d["lon"]), 4))
            for _, d in warehouses2.iterrows()}
    check("every route is based at a real warehouse", homes <= want)

    w02 = [r for r in m.routes if r.warehouse_id == "W02" and r.n_stops]
    check("west-yard trucks are based west",
          all(r.warehouse_lon < -90.5 for r in w02), f"{len(w02)} route(s)")
    check("multi-warehouse respects capacity",
          all(r.load_kg <= r.capacity_kg + 1e-6 for r in m.routes))
    check("multi-warehouse respects windows",
          all(s.arrival_s <= s.tw_close_s + 1
              for r in m.routes for s in r.stops))
    check("no HOS violations with two warehouses",
          not [i for i in routing.compliance_check(m, cfg)
               if i["Severity"] == "Violation"])

    mb2 = routing.baseline_routes(stops, t2, d2, dm2, tm2, cfg, warehouses=warehouses2)
    check("multi-warehouse baseline runs", mb2.status == "SUCCESS")
    check("multi-warehouse optimised beats its baseline",
          m.total_miles < mb2.total_miles,
          f"{m.total_miles:.0f} vs {mb2.total_miles:.0f} mi")

    cfg.solver.time_limit_sec = 2
    u2 = fleet.utilisation_study(stops, t2, d2, dm2, tm2, cfg, n_days=4,
                                 max_trucks=5, warehouses=warehouses2)
    check("utilisation study handles two warehouses", len(u2.days) == 4)

    sw2 = fleet.fleet_sweep(stops, t2, d2, dm2, tm2, cfg, max_trucks=5,
                            min_trucks=4, warehouses=warehouses2)
    check("sweep handles two warehouses", len(sw2) == 2)

    val2 = warehouse.validate_with_vrp(
        stops, t2, d2,
        [{"name": "Current network", "warehouses": warehouses2},
         {"name": "Single site", "warehouses": warehouses1}],
        cfg, lambda c: distance.build_matrices(c, time_multiplier=mult)[:2])
    check("network comparison runs", len(val2) == 2,
          f"2-site ${val2.iloc[0]['Routed cost/day ($)']:.0f} vs "
          f"1-site ${val2.iloc[1]['Routed cost/day ($)']:.0f}")

    # a truck pointed at a warehouse that does not exist must not crash
    t_bad = t2.copy()
    t_bad.loc[0, "warehouse_id"] = "NOPE"
    mbad = routing.solve_routes(stops, t_bad, d2, dm2, tm2, cfg, warehouses=warehouses2)
    check("unknown warehouse_id falls back to the first warehouse",
          mbad.status == "SUCCESS" and mbad.routes[0].warehouse_id == "W01",
          mbad.routes[0].warehouse_id if mbad.routes else "no routes")

    # legacy single-warehouse dict callers must still work
    legacy = routing.solve_routes(stops, trucks, drivers, dist_mi, time_s, cfg,
                                  warehouses=demo_data.demo_warehouse())
    check("legacy single-warehouse dict still solves", legacy.status == "SUCCESS")

    # --- cost reporting and fuel sensitivity ------------------------------
    cb = routing.cost_breakdown(sol)
    check("cost breakdown has four components", len(cb) == 4)
    check("cost breakdown sums to the solution total",
          abs(cb["Cost ($/day)"].sum() - sol.total_cost) < 0.05,
          f"{cb['Cost ($/day)'].sum():.2f} vs {sol.total_cost:.2f}")

    sens = routing.fuel_sensitivity(sol, cfg)
    check("fuel sensitivity returns a range", len(sens) >= 5)
    check("exactly one row is the current price",
          int(sens["Is current"].sum()) == 1)
    check("cost rises with diesel price",
          sens["Cost per run ($)"].is_monotonic_increasing)
    cur = sens[sens["Is current"]].iloc[0]
    check("current-price row matches the solved cost",
          abs(float(cur["Cost per run ($)"]) - sol.total_cost) < 0.05,
          f"{float(cur['Cost per run ($)']):.2f} vs {sol.total_cost:.2f}")
    check("per-stop cost is consistent",
          abs(float(cur["Cost per stop ($)"])
              - sol.total_cost / sol.stops_served) < 0.02)

    ft = routing.frequency_tradeoff(sol, cfg)
    check("frequency tradeoff computed", ft and ft["stops_affected"] == 20)
    check("twice-monthly is double one run",
          abs(ft["twice_monthly"] - 2 * ft["cost_per_run"]) < 0.05)
    check("annual saving is twelve monthly savings",
          abs(ft["annual_saving"] - 12 * ft["monthly_saving"]) < 0.05)
    check("fuel share is a sane percentage",
          0 < ft["fuel_share"] < 60, f"{ft['fuel_share']}%")

    # --- finance ----------------------------------------------------------
    e = finance.TruckEconomics()
    low = finance.recommend(e, 30)
    high = finance.recommend(e, 300)
    check("finance compares four options", len(low["table"]) == 4)
    check("low utilisation favours flexibility",
          low["best"] in ("Outsource to carrier", "Rent by the day"),
          low["best"])
    check("high utilisation favours commitment",
          high["best"] in ("Own (financed)", "Lease (full service)"),
          high["best"])
    curve = finance.breakeven_curve(e)
    check("break-even curve built", len(curve) > 5)
    check("NPV rises with utilisation",
          curve["Own (financed)"].is_monotonic_increasing)

    print("=" * 70)
    if FAIL:
        print(f"{len(FAIL)} CHECK(S) FAILED: {', '.join(FAIL)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
