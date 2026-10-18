"""
RouteWise - delivery route and cost planner for small fleets.

Run with:   streamlit run app.py

Tab 1  Start here         What the tool is for, how to use it step by step,
                          what the terms mean, and what it does not do.

Tab 2  Plan routes        Enter warehouses, stores, trucks and drivers; get a
                          legal optimised plan, what it costs, and how it
                          compares with the plan a dispatcher would draw by
                          hand.

Tab 3  Network & fleet    Should we add a truck (and buy, lease, rent or
                          outsource it)? Where should the warehouse be, and is
                          a second one worth opening?

Everything in Tab 2 is driven by the routing model in Tab 1, which is the
point: utilisation comes from actual solved routes, not from a guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import core
from core import demo_data, distance, finance, fleet, routing, viz, warehouse
from core.geocode import Geocoder, parse_bulk
from core.routing import (CostModel, HOSRules, RouteConfig, SolverSettings,
                          seconds_to_hhmm)

try:
    from streamlit_folium import st_folium
    HAVE_FOLIUM = True
except Exception:                                                 # noqa: BLE001
    HAVE_FOLIUM = False

APP_NAME = "RouteWise"
APP_TAGLINE = "Delivery route and cost planner for small fleets"

st.set_page_config(page_title=f"{APP_NAME} — {APP_TAGLINE}", page_icon="🚚",
                   layout="wide")


# --------------------------------------------------------------------------- #
# stale-upload guard
# --------------------------------------------------------------------------- #
# This app.py needs core/ at this version or newer. Deploying by uploading
# files one at a time is easy to get half-right, and the symptom on Streamlit
# Cloud is a redacted AttributeError that says nothing useful. Fail loudly
# instead, naming the files to re-upload.

NEEDS_CORE = 2
_core_version = getattr(core, "VERSION", 0)
_missing = [name for name in ("empty_warehouses", "demo_warehouses")
            if not hasattr(demo_data, name)]

if _core_version < NEEDS_CORE or _missing:
    st.error(
        f"**The `core/` folder is out of date.** This `app.py` needs core "
        f"version {NEEDS_CORE}, but the copy that loaded reports "
        f"`{_core_version}`."
        + (f"\n\nMissing from `core/demo_data.py`: "
           f"{', '.join('`' + m + '()`' for m in _missing)}." if _missing else "")
    )
    st.markdown(
        "You uploaded the new `app.py` without the matching `core/` files. "
        "Re-upload **all** of these, keeping them inside the `core/` folder:\n\n"
        "- `core/__init__.py`\n"
        "- `core/demo_data.py`\n"
        "- `core/routing.py`\n"
        "- `core/fleet.py`\n"
        "- `core/warehouse.py`\n"
        "- `core/viz.py`\n"
        "- `core/geocode.py`\n"
        "- `core/distance.py`\n"
        "- `core/finance.py`\n\n"
        "On GitHub, deleting the `core` folder and re-uploading it whole is "
        "more reliable than replacing files one by one."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# session state
# --------------------------------------------------------------------------- #

def init_state():
    ss = st.session_state
    ss.setdefault("stops_raw", demo_data.blank_rows(5))
    ss.setdefault("stops_geo", None)          # geocoded, solver-ready
    ss.setdefault("warehouses_raw", demo_data.empty_warehouses())
    ss.setdefault("warehouses", None)             # geocoded warehouse table
    ss.setdefault("trucks", demo_data.default_trucks(3))
    ss.setdefault("drivers", demo_data.default_drivers(3))
    ss.setdefault("geocoder", None)
    ss.setdefault("geo_table", None)
    ss.setdefault("solution", None)
    ss.setdefault("baseline", None)
    ss.setdefault("sweep", None)
    ss.setdefault("util", None)
    ss.setdefault("cflp", None)
    ss.setdefault("candidates", None)
    ss.setdefault("validation", None)


init_state()


def get_geocoder(key: str) -> Geocoder:
    gc = st.session_state.geocoder
    if gc is None or gc.ors_key != ((key or "").strip() or None):
        new = Geocoder(ors_key=key)
        if gc is not None:
            new._cache.update(gc._cache)       # keep what we already paid for
        st.session_state.geocoder = new
    return st.session_state.geocoder


@st.cache_data(show_spinner=False)
def cached_matrices(coords_key: tuple, circuity: float, urban: float,
                    hwy: float, mult: float, ors_key: str):
    coords = [(a, b) for a, b in coords_key]
    return distance.build_matrices(coords, circuity=circuity, urban_mph=urban,
                                   hwy_mph=hwy, time_multiplier=mult,
                                   ors_key=ors_key or None)


def coords_key(warehouses: pd.DataFrame, stops: pd.DataFrame) -> tuple:
    """Warehouses first, then stops — the node order the solver expects."""
    return tuple([(round(float(a), 6), round(float(b), 6))
                  for a, b in zip(warehouses["lat"], warehouses["lon"])]
                 + [(round(float(a), 6), round(float(b), 6))
                    for a, b in zip(stops["lat"], stops["lon"])])


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #

st.sidebar.title(f"🚚 {APP_NAME}")
st.sidebar.caption(APP_TAGLINE)

with st.sidebar.expander("Geocoding", expanded=False):
    ors_key = st.text_input(
        "OpenRouteService API key (optional)", value="", type="password",
        help="Free key from openrouteservice.org. Improves US address matching "
             "and unlocks real road-network distances. Without it the app "
             "falls back to Nominatim and straight-line distance.")
    use_ors_matrix = st.checkbox(
        "Use ORS road distances when a key is set", value=False,
        help="Off by default so demos never depend on a network call.")

with st.sidebar.expander("Hours of service", expanded=False):
    hos_on = st.checkbox("Enforce HOS rules", value=True)
    interstate = st.checkbox("Interstate operation", value=True,
                             help="Turns on the 21-year minimum driver age.")
    max_drive = st.number_input("Max driving hours", 1.0, 16.0, 11.0, 0.5)
    max_duty = st.number_input("Max duty span (hours)", 1.0, 24.0, 14.0, 0.5)
    break_after = st.number_input("Break required after (driving hours)",
                                  1.0, 12.0, 8.0, 0.5)
    break_min = st.number_input("Break length (minutes)", 0, 120, 30, 5)
    weekly_cap = st.checkbox("Enforce weekly hour cap", value=True)
st.sidebar.caption("Decision support only — a simplified model of 49 CFR 395, "
                   "not a compliance system of record.")

with st.sidebar.expander("Costs", expanded=False):
    fuel_price = st.number_input("Diesel ($/gal)", 1.0, 10.0, 3.85, 0.05)
    ot_after = st.number_input("Overtime after (hours)", 0.0, 14.0, 8.0, 0.5)
    ot_mult = st.number_input("Overtime multiplier", 1.0, 3.0, 1.5, 0.1)
    per_stop = st.number_input("Handling cost per stop ($)", 0.0, 50.0, 2.50, 0.25)

with st.sidebar.expander("Travel time", expanded=False):
    circuity = st.number_input("Road circuity factor", 1.0, 2.0, 1.32, 0.01,
                               help="Multiplier from straight-line to road distance.")
    urban_mph = st.number_input("Urban speed (mph)", 5.0, 50.0, 22.0, 1.0)
    hwy_mph = st.number_input("Highway speed (mph)", 20.0, 80.0, 52.0, 1.0)
    traffic_on = st.checkbox("Apply congestion multiplier", value=True)
    peak_mult = st.number_input("Peak multiplier", 1.0, 2.5, 1.30, 0.05)
    weather = st.selectbox("Weather", ["clear", "rain", "snow", "ice"], 0)
    wx_mult = {"clear": 1.0, "rain": 1.12, "snow": 1.35, "ice": 1.55}[weather]

with st.sidebar.expander("Solver", expanded=False):
    time_limit = st.slider("Time limit per solve (s)", 2, 60, 12)
    fairness = st.slider("Workload fairness weight", 0, 200, 0,
                         help="0 = pure efficiency. Higher values even out "
                              "route lengths across drivers, usually at the "
                              "cost of a few extra miles.")


def current_cfg() -> RouteConfig:
    return RouteConfig(
        hos=HOSRules(enabled=hos_on, max_driving_hours=max_drive,
                     max_duty_span_hours=max_duty,
                     break_after_hours=break_after, break_minutes=int(break_min),
                     enforce_weekly_cap=weekly_cap,
                     interstate_operation=interstate),
        cost=CostModel(fuel_price_per_gal=fuel_price, ot_after_hours=ot_after,
                       ot_multiplier=ot_mult, cost_per_stop=per_stop),
        solver=SolverSettings(time_limit_sec=time_limit,
                              fairness_coefficient=int(fairness)),
    )


def time_multiplier(shift_start=6.0, shift_end=18.0) -> float:
    cong = distance.congestion_multiplier(shift_start, shift_end,
                                          peak_mult=peak_mult,
                                          enabled=traffic_on)
    return cong * wx_mult


def build_matrices_for(warehouses: pd.DataFrame, stops: pd.DataFrame):
    key = ors_key if (use_ors_matrix and ors_key) else ""
    return cached_matrices(coords_key(warehouses, stops), circuity, urban_mph,
                           hwy_mph, time_multiplier(), key)


tab_about, tab_route, tab_network = st.tabs(
    ["① Start here", "② Plan routes", "③ Network & fleet decisions"])


# =========================================================================== #
# TAB 0 - START HERE
# =========================================================================== #

with tab_about:
    st.title(APP_NAME)
    st.subheader(APP_TAGLINE)

    st.markdown("""
A small food distributor runs three trucks out of one warehouse and delivers to
twenty grocery stores. Every morning someone decides which truck takes which
stores, and in what order. That decision is usually made from memory on a
whiteboard, and it is worth real money: the gap between a reasonable plan and a
good one is routinely 15\u201325% of the miles driven.

The routing systems that solve this well \u2014 Trimble, Descartes, ORTEC \u2014 are
priced for fleets of hundreds. Operators with three to fifteen trucks are left
with spreadsheets. **This tool is built for that gap.**

It answers three questions:

1. **What should today's routes be?** Given the stores to serve and the trucks
   and drivers available, what is the cheapest legal plan \u2014 and how much better
   is it than the plan you would have drawn by hand?
2. **Do we need another truck?** Not for today alone, but across many days of
   varying demand. And if we do, is it cheaper to buy, lease, rent by the day,
   or hand the overflow to a carrier?
3. **Is the warehouse in the right place?** And would a second one pay for
   itself, once you count what it costs to open and run?
""")

    st.info(
        "**What a delivery costs.** Diesel price, road speeds, traffic and "
        "weather all feed the cost of a run, and every plan reports cost per "
        "run and per stop. When fuel moves, you can see what the same route "
        "costs at the new price before deciding to change how often you serve "
        "a customer.")

    st.divider()
    st.markdown("### How to use it")

    st.markdown("""
**Step 1 \u2014 Open the Plan routes tab and load a demo.**
Two buttons sit at the top right: one warehouse, or two. Both come with twenty
St. Louis grocery stores and their coordinates already filled in, so nothing
needs typing and no address lookup happens. Use a demo the first time through
to see what the output looks like.

**Step 2 \u2014 Enter your own warehouses and stores.**
Add a row per warehouse: a short ID, a name, and a street address. Then fill
the stores table \u2014 name, address, how much the delivery weighs, the hours the
store can receive, how long the stop takes, and how often you serve that store
in a month. A paste box takes many addresses at once.

**Step 3 \u2014 Geocode, then check the matches.**
Addresses have to become map coordinates before anything can be routed. Press
**Geocode addresses**, then read the review table. A wrong pin produces a route
that looks perfectly reasonable and is wrong, so this step is not optional.
Green means the address landed on a building. Orange means it landed on the
street without a house number, usually fine. Red means it only matched a town
or a postcode and needs fixing.

**Step 4 \u2014 Enter your trucks and drivers.**
Each truck has a capacity, fuel economy, a daily fixed cost, a driver, and the
ID of the warehouse it is based at. Each driver has an age, a licence class, a
shift window, hours already worked this week, and an hourly wage.

**Step 5 \u2014 Press Build routes.**
You get a map, a schedule for each truck, and a comparison against an
unoptimised plan. Keep scrolling for the cost breakdown, the fuel price table,
the legality check, and run sheets you can download.

**Step 6 \u2014 Open the Network and fleet decisions tab.**
Run section A first. Sections B and C both build on what it produces.
""")

    st.divider()
    st.markdown("### What the words mean")

    st.dataframe(pd.DataFrame([
        {"Term": "Warehouse",
         "What it means": "Where trucks are based and loaded. Every truck "
                          "starts its day at one and returns to the same one. "
                          "Other people call this a depot, a yard or a "
                          "distribution centre; this tool always says "
                          "warehouse."},
        {"Term": "Stop",
         "What it means": "One delivery to one customer \u2014 here, a grocery or "
                          "retail store."},
        {"Term": "Route",
         "What it means": "The ordered list of stops one truck makes in one "
                          "day, out from its warehouse and back to it."},
        {"Term": "Time window",
         "What it means": "The hours a store can accept a delivery. The plan "
                          "never schedules an arrival outside it."},
        {"Term": "Service time",
         "What it means": "Minutes spent at a stop unloading and doing "
                          "paperwork. It counts against the driver's working "
                          "day."},
        {"Term": "Hours of Service",
         "What it means": "Federal limits on driving: at most 11 hours driving "
                          "inside a 14-hour working day, a 30-minute break "
                          "after 8 hours of driving, and a weekly cap."},
        {"Term": "Capacity",
         "What it means": "How much weight a truck can carry. The plan never "
                          "loads one past it."},
        {"Term": "Utilisation",
         "What it means": "How hard a truck works, as the number of days a "
                          "year it is actually needed. This is what decides "
                          "whether owning beats renting."},
        {"Term": "Baseline",
         "What it means": "An unoptimised plan: trucks filled in the order the "
                          "stores were entered, then each sequenced "
                          "nearest-first. Savings are measured against this."},
    ]), width="stretch", hide_index=True)

    st.divider()
    st.markdown("### What it does not do")
    st.markdown("""
- **Traffic is modelled, not live.** Travel times carry a congestion
  multiplier based on how much of the shift falls in peak hours, rather than a
  live feed. Peak and weather multipliers are adjustable in the sidebar.
- **It does not track trucks during the day.** This is a planning tool. There
  is no GPS, no driver app, and no customer notification.
- **It is decision support, not a compliance record.** The Hours of Service
  rules here are a simplified reading of 49 CFR 395. Keep your own logs.
- **The cost assumptions are defaults until you change them.** Fuel price,
  wages, insurance and lease rates ship with plausible industry figures.
  Replace them with yours before trusting any dollar amount.
""")

    st.caption("Built with Google OR-Tools, OpenStreetMap and "
               "OpenRouteService.")


# =========================================================================== #
# TAB 1 - ROUTING
# =========================================================================== #

with tab_route:
    st.header("Build today's routes")

    # ---- step 1: addresses ------------------------------------------------
    st.subheader("1 · Warehouses and store addresses")

    dc1, dc2 = st.columns([3, 1])
    with dc2:
        st.write("")
        if st.button("Load demo · 1 warehouse", width="stretch"):
            st.session_state.stops_geo = demo_data.demo_stops()
            st.session_state.warehouses = demo_data.demo_warehouses(1)
            st.session_state.warehouses_raw = st.session_state.warehouses[
                ["warehouse_id", "warehouse_name", "address"]].copy()
            st.session_state.trucks = demo_data.default_trucks(3, "W01")
            st.session_state.drivers = demo_data.default_drivers(3)
            st.session_state.stops_raw = st.session_state.stops_geo[
                ["label", "address", "demand_kg", "tw_open", "tw_close",
                 "service_min", "priority", "deliveries_per_month"]].copy()
            st.session_state.solution = None
            st.session_state.baseline = None
            st.success("Demo loaded — 20 grocery stores, 1 warehouse, "
                       "already geocoded.")
        if st.button("Load demo · 2 warehouses", width="stretch"):
            st.session_state.stops_geo = demo_data.demo_stops()
            st.session_state.warehouses = demo_data.demo_warehouses(2)
            st.session_state.warehouses_raw = st.session_state.warehouses[
                ["warehouse_id", "warehouse_name", "address"]].copy()
            t = demo_data.default_trucks(4, "W01")
            t.loc[2:, "warehouse_id"] = "W02"
            t.loc[2:, "truck_name"] = ["Truck 3 (west)", "Truck 4 (west)"]
            st.session_state.trucks = t
            st.session_state.drivers = demo_data.default_drivers(4)
            st.session_state.stops_raw = st.session_state.stops_geo[
                ["label", "address", "demand_kg", "tw_open", "tw_close",
                 "service_min", "priority", "deliveries_per_month"]].copy()
            st.session_state.solution = None
            st.session_state.baseline = None
            st.success("Demo loaded — 20 grocery stores, 2 warehouses, "
                       "4 trucks.")

    with dc1:
        st.markdown("**Warehouses** — add a row per warehouse or yard. Each truck "
                    "is based at one of these and returns to it.")
        st.session_state.warehouses_raw = st.data_editor(
            st.session_state.warehouses_raw, num_rows="dynamic", width="stretch",
            key="warehouses_editor",
            column_config={
                "warehouse_id": st.column_config.TextColumn(
                    "ID", width="small",
                    help="Trucks reference this in their warehouse_id column."),
                "warehouse_name": st.column_config.TextColumn("Name"),
                "address": st.column_config.TextColumn("Address", width="large"),
            })

    st.caption("Type your stores below, or paste a block of addresses. "
               "`deliveries_per_month` is what makes Tab 2 meaningful — it is "
               "how often this customer is served, not just today.")

    with st.expander("Paste addresses in bulk (one per line)"):
        bulk = st.text_area("Addresses", height=120, label_visibility="collapsed")
        bc1, bc2, bc3 = st.columns(3)
        b_demand = bc1.number_input("Default kg", 0.0, 50000.0, 200.0, 10.0)
        b_open = bc2.text_input("Default window open", "08:00")
        b_close = bc3.text_input("Default window close", "17:00")
        if st.button("Add pasted addresses"):
            lines = parse_bulk(bulk)
            if lines:
                add = pd.DataFrame([{
                    "label": ln.split(",")[0][:40], "address": ln,
                    "demand_kg": b_demand, "tw_open": b_open,
                    "tw_close": b_close, "service_min": 15, "priority": 1,
                    "deliveries_per_month": 4} for ln in lines])
                st.session_state.stops_raw = pd.concat(
                    [st.session_state.stops_raw, add], ignore_index=True)
                st.success(f"Added {len(lines)} addresses.")
            else:
                st.warning("Nothing to add.")

    edited = st.data_editor(
        st.session_state.stops_raw, num_rows="dynamic", width="stretch",
        key="stops_editor",
        column_config={
            "label": st.column_config.TextColumn("Store", width="medium"),
            "address": st.column_config.TextColumn("Address", width="large"),
            "demand_kg": st.column_config.NumberColumn("Load (kg)", min_value=0.0),
            "tw_open": st.column_config.TextColumn("Open", help="HH:MM"),
            "tw_close": st.column_config.TextColumn("Close", help="HH:MM"),
            "service_min": st.column_config.NumberColumn("Service (min)", min_value=0),
            "priority": st.column_config.NumberColumn("Priority 1-3", min_value=1, max_value=3),
            "deliveries_per_month": st.column_config.NumberColumn(
                "Deliveries/month", min_value=0,
                help="Drives the warehouse model in Tab 2."),
        })
    st.session_state.stops_raw = edited

    # ---- step 2: geocode --------------------------------------------------
    st.subheader("2 · Geocode and review")
    st.caption("A wrong pin produces a route that looks right and is not. "
               "Check the matches before solving.")

    if st.button("Geocode addresses", type="primary"):
        gc = get_geocoder(ors_key)
        rows = edited[edited["address"].astype(str).str.strip() != ""].copy()
        draw = st.session_state.warehouses_raw
        dep_rows = draw[draw["address"].astype(str).str.strip() != ""].copy()

        if not len(rows):
            st.error("No delivery addresses entered.")
        elif not len(dep_rows):
            st.error("Enter at least one warehouse address.")
        else:
            bar = st.progress(0.0, text="Starting")
            dep_results = gc.geocode_many(dep_rows["address"].tolist())
            results = gc.geocode_many(rows["address"].tolist(),
                                      progress=lambda p, t="": bar.progress(p, text=t))
            bar.empty()

            dep_ok = [r.ok for r in dep_results]
            failed = [a for a, o in zip(dep_rows["address"], dep_ok) if not o]
            if not any(dep_ok):
                st.error("No warehouse address could be geocoded: "
                         + "; ".join(str(f) for f in failed))
            else:
                if failed:
                    st.warning(f"{len(failed)} warehouse address(es) failed and "
                               f"were skipped: " + "; ".join(str(f) for f in failed))
                dkeep = dep_rows.loc[dep_ok].reset_index(drop=True)
                dkeep["lat"] = [r.lat for r, o in zip(dep_results, dep_ok) if o]
                dkeep["lon"] = [r.lon for r, o in zip(dep_results, dep_ok) if o]
                if "warehouse_id" not in dkeep.columns:
                    dkeep["warehouse_id"] = [f"W{i+1:02d}" for i in range(len(dkeep))]
                dkeep["warehouse_id"] = [
                    str(v).strip() or f"W{i+1:02d}"
                    for i, v in enumerate(dkeep["warehouse_id"])]
                if "warehouse_name" not in dkeep.columns:
                    dkeep["warehouse_name"] = dkeep["warehouse_id"]
                st.session_state.warehouses = dkeep

                table = pd.DataFrame([r.as_dict() for r in results])
                table.insert(0, "label", rows["label"].values)
                st.session_state.geo_table = table

                good = table["ok"].values
                keep = rows.loc[good].reset_index(drop=True)
                keep["lat"] = table.loc[good, "lat"].values
                keep["lon"] = table.loc[good, "lon"].values
                keep.insert(0, "stop_id",
                            [f"S{i:03d}" for i in range(1, len(keep) + 1)])
                st.session_state.stops_geo = keep
                st.session_state.solution = None
                st.session_state.baseline = None
                st.success(f"{int(good.sum())} of {len(table)} addresses matched.")

    if st.session_state.geo_table is not None:
        tbl = st.session_state.geo_table
        bad = tbl[~tbl["ok"]]
        low = tbl[(tbl["ok"]) & (tbl["confidence"] < 0.5)]
        if len(bad):
            st.error(f"{len(bad)} address(es) failed to match — they are "
                     f"excluded from the solve.")
        if len(low):
            st.warning(f"{len(low)} match(es) resolved only to a street or area "
                       f"centroid, not to a specific address. Check them on the "
                       f"map before trusting the route.")
        with st.expander("Match review", expanded=bool(len(bad) or len(low))):
            st.caption(
                "`confidence` is **precision** — did the match land on a "
                "building, a street, or just a city? Green ≥ 0.80 is "
                "deliverable, orange 0.50–0.79 is street-level and worth a "
                "look, red ≤ 0.40 is an area centroid and not a real "
                "address. `provider_score` is the geocoder's own number, "
                "shown for reference only: Pelias and Nominatim do not mean "
                "the same thing by it, so it is never what decides the "
                "colour.")
            cols = [c for c in ["label", "query", "matched", "precision",
                                "confidence", "provider_score", "provider",
                                "error"] if c in tbl.columns]
            st.dataframe(tbl[cols], width="stretch", hide_index=True)
            if HAVE_FOLIUM:
                st_folium(viz.geocode_review_map(tbl), height=330,
                          use_container_width=True, key="geomap")

    # ---- step 3: fleet ----------------------------------------------------
    st.subheader("3 · Trucks and drivers")
    fc1, fc2 = st.columns(2)
    with fc1:
        n_trucks = st.number_input("Number of trucks", 1, 20,
                                   len(st.session_state.trucks))
        if n_trucks != len(st.session_state.trucks):
            # New trucks default to the first warehouse in the table.
            ids = st.session_state.warehouses_raw.get("warehouse_id",
                                                  pd.Series(["W01"]))
            first = str(ids.iloc[0]).strip() if len(ids) else "W01"
            st.session_state.trucks = demo_data.default_trucks(
                int(n_trucks), first or "W01")
            st.session_state.drivers = demo_data.default_drivers(int(n_trucks))
        st.session_state.trucks = st.data_editor(
            st.session_state.trucks, width="stretch", key="trucks_editor",
            column_config={
                "truck_id": st.column_config.TextColumn("ID", width="small"),
                "truck_name": st.column_config.TextColumn("Truck"),
                "capacity_kg": st.column_config.NumberColumn("Capacity (kg)", min_value=0.0),
                "mpg": st.column_config.NumberColumn("MPG", min_value=1.0),
                "fixed_cost_day": st.column_config.NumberColumn("Fixed $/day", min_value=0.0),
                "driver_id": st.column_config.TextColumn("Driver ID", width="small"),
                "warehouse_id": st.column_config.TextColumn(
                    "Warehouse ID", width="small",
                    help="Which warehouse this truck is based at and returns to."),
            })
        known = set(st.session_state.warehouses_raw.get(
            "warehouse_id", pd.Series(dtype=str)).astype(str).str.strip())
        used = set(st.session_state.trucks.get(
            "warehouse_id", pd.Series(dtype=str)).astype(str).str.strip())
        unknown = {u for u in used if u and u not in known}
        if unknown:
            st.warning(
                f"Truck warehouse ID(s) not in the warehouse table: "
                f"{', '.join(sorted(unknown))}. Those trucks fall back to the "
                f"first warehouse.")
    with fc2:
        st.session_state.drivers = st.data_editor(
            st.session_state.drivers, width="stretch", key="drivers_editor",
            column_config={
                "driver_id": st.column_config.TextColumn("ID", width="small"),
                "driver_name": st.column_config.TextColumn("Driver"),
                "age": st.column_config.NumberColumn("Age", min_value=16, max_value=90),
                "cdl_class": st.column_config.SelectboxColumn("CDL", options=["A", "B", "C"]),
                "shift_start": st.column_config.TextColumn("Shift start"),
                "shift_end": st.column_config.TextColumn("Shift end"),
                "hours_used_week": st.column_config.NumberColumn("Hrs used/wk", min_value=0.0),
                "max_weekly_hours": st.column_config.NumberColumn("Weekly cap", min_value=1.0),
                "hourly_wage": st.column_config.NumberColumn("Wage $/h", min_value=0.0),
            })
        st.caption("Driver age is an eligibility filter, never a cost term. "
                   "Under 49 CFR 391.11 a driver under 21 cannot run "
                   "interstate, so the pairing is removed before the solve "
                   "and the reason is reported below.")

    # ---- step 4: solve ----------------------------------------------------
    st.subheader("4 · Optimise")
    stops_geo = st.session_state.stops_geo
    warehouses_geo = st.session_state.warehouses
    ready = (stops_geo is not None and len(stops_geo)
             and warehouses_geo is not None and len(warehouses_geo))

    if st.button("Build routes", type="primary", disabled=not ready):
        cfg = current_cfg()
        with st.spinner("Solving…"):
            d, t, meta = build_matrices_for(warehouses_geo, stops_geo)
            sol = routing.solve_routes(stops_geo, st.session_state.trucks,
                                       st.session_state.drivers, d, t, cfg,
                                       warehouses=warehouses_geo)
            base = routing.baseline_routes(stops_geo, st.session_state.trucks,
                                           st.session_state.drivers, d, t, cfg,
                                           warehouses=warehouses_geo)
        st.session_state.solution = sol
        st.session_state.baseline = base
        st.session_state.matrix_meta = meta

    sol = st.session_state.solution
    base = st.session_state.baseline

    if sol is not None:
        if not sol.routes:
            st.error(f"No solution produced (status: {sol.status}).")
        else:
            st.markdown("#### Optimised vs. the plan you'd build by hand")
            m1, m2, m3, m4, m5 = st.columns(5)
            d_mi = base.total_miles - sol.total_miles
            d_hr = base.total_hours - sol.total_hours
            d_cost = base.total_cost - sol.total_cost
            m1.metric("Miles", f"{sol.total_miles:,.0f}",
                      f"{-d_mi:,.0f} vs {base.total_miles:,.0f}")
            m2.metric("Driver hours", f"{sol.total_hours:,.1f}",
                      f"{-d_hr:,.1f} vs {base.total_hours:,.1f}")
            m3.metric("Cost/day", f"${sol.total_cost:,.0f}",
                      f"-${d_cost:,.0f}" if d_cost >= 0 else f"+${-d_cost:,.0f}")
            m4.metric("Trucks used", sol.trucks_used)
            m5.metric("CO₂ (kg)", f"{sol.co2_kg:,.0f}",
                      f"{sol.co2_kg - base.co2_kg:,.0f}")

            pct = 100 * d_mi / base.total_miles if base.total_miles else 0
            viol = base.meta.get("time_window_violations", 0)
            st.success(
                f"**{pct:.1f}% fewer miles** than the hand-built plan, and "
                f"the optimised plan lands every stop inside its window — the "
                f"naive plan misses {viol}.")

            if sol.excluded_trucks:
                st.warning("**Trucks excluded before the solve**")
                st.dataframe(pd.DataFrame(sol.excluded_trucks),
                             width="stretch", hide_index=True)

            issues = routing.compliance_check(sol, current_cfg())
            if issues:
                idf = pd.DataFrame(issues)
                hard = idf[idf["Severity"] == "Violation"]
                if len(hard):
                    st.error("**HOS / time-window issues**")
                    st.dataframe(hard, width="stretch", hide_index=True)
                soft = idf[idf["Severity"] == "Watch"]
                if len(soft):
                    with st.expander(f"{len(soft)} item(s) to watch"):
                        st.dataframe(soft, width="stretch", hide_index=True)
            else:
                st.info("No HOS or time-window issues detected in this plan.")

            if sol.dropped:
                st.error(f"**{len(sol.dropped)} stop(s) could not be served** — "
                         f"this is the diagnosis, not a failure. Add a truck, "
                         f"widen a window, or raise capacity.")
                st.dataframe(pd.DataFrame(sol.dropped),
                             width="stretch", hide_index=True)

            if HAVE_FOLIUM:
                st_folium(viz.route_map(sol),
                          height=520, use_container_width=True, key="routemap")

            st.markdown("#### Per-truck summary")
            st.dataframe(routing.truck_summary_frame(sol),
                         width="stretch", hide_index=True)

            # ---- what the run costs ---------------------------------------
            st.markdown("#### What this run costs")
            cfg_now = current_cfg()
            sens = routing.fuel_sensitivity(sol, cfg_now)

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Cost per run", f"${sol.total_cost:,.0f}")
            q2.metric("Cost per stop",
                      f"${sol.total_cost / max(1, sol.stops_served):,.2f}")
            q3.metric("Cost per mile",
                      f"${sol.total_cost / max(1.0, sol.total_miles):,.2f}")
            fuel_total = sum(r.fuel_cost for r in sol.routes)
            q4.metric("Fuel share of cost",
                      f"{100 * fuel_total / sol.total_cost:.0f}%"
                      if sol.total_cost else "—")

            k1, k2 = st.columns([1, 2])
            with k1:
                st.markdown("**Where the money goes**")
                st.dataframe(routing.cost_breakdown(sol),
                             width="stretch", hide_index=True)
            with k2:
                st.markdown("**Cost at other diesel prices**")
                st.dataframe(
                    sens.drop(columns=["Is current"]),
                    width="stretch", hide_index=True)
                st.caption("The route is held fixed and only the fuel bill is "
                           "recalculated, so these are what this same run "
                           "costs if diesel moves. The monthly columns are "
                           "that cost at each delivery frequency.")

            st.line_chart(sens.set_index("Diesel ($/gal)")[
                ["Cost per run ($)"]])

            st.markdown("#### Driver run sheets")
            sched = routing.routes_to_frame(sol)
            for truck in sched["Truck"].unique():
                with st.expander(f"{truck}"):
                    st.dataframe(sched[sched["Truck"] == truck],
                                 width="stretch", hide_index=True)
            st.download_button("Download run sheets (CSV)",
                               sched.to_csv(index=False).encode(),
                               "routewise_runsheets.csv", "text/csv")


# =========================================================================== #
# TAB 2 - NETWORK & FLEET
# =========================================================================== #

with tab_network:
    st.header("Network and fleet decisions")
    stops_geo = st.session_state.stops_geo

    ready_2 = (stops_geo is not None and len(stops_geo)
               and st.session_state.warehouses is not None
               and len(st.session_state.warehouses))
    if not ready_2:
        st.info("Enter and geocode addresses in Tab 1 first — every analysis "
                "here is driven by the routing model.")

# Guard the whole of Tab 2 on having data, without st.stop(), which would halt
# the entire script run rather than just this tab.
if ready_2:
  with tab_network:
    warehouses_geo = st.session_state.warehouses
    trucks = st.session_state.trucks
    drivers = st.session_state.drivers
    cfg = current_cfg()
    dist_mi, time_s, _ = build_matrices_for(warehouses_geo, stops_geo)

    sec = st.radio("Analysis", ["A · Do we need another truck?",
                                "B · Where should the warehouse be?",
                                "C · Buy, lease, rent or outsource?"],
                   horizontal=True, label_visibility="collapsed")

    # ---------------------------------------------------------------- A ----
    if sec.startswith("A"):
        st.subheader("A · Fleet sizing")
        st.caption("Two questions. First: what is the smallest fleet that can "
                   "serve today legally? Second, and more useful: across many "
                   "simulated days, how often is each truck actually needed?")

        a1, a2 = st.columns(2)
        max_k = a1.number_input("Test up to N trucks", 1, 15, len(trucks) + 3)
        quick = a2.slider("Solve time per configuration (s)", 2, 30, 6)

        if st.button("Run fleet sweep", type="primary"):
            c = current_cfg()
            c.solver.time_limit_sec = int(quick)
            bar = st.progress(0.0, text="Starting")
            sw = fleet.fleet_sweep(stops_geo, trucks, drivers, dist_mi, time_s,
                                   c, max_trucks=int(max_k),
                                   progress=lambda p, t="": bar.progress(p, text=t),
                                   warehouses=warehouses_geo)
            bar.empty()
            st.session_state.sweep = sw

        sw = st.session_state.sweep
        if sw is not None:
            need = fleet.minimum_fleet(sw)
            cheap = fleet.cheapest_fleet(sw)
            owned = len(trucks)
            c1, c2, c3 = st.columns(3)
            c1.metric("Trucks owned", owned)
            c2.metric("Minimum to serve everything", need if need else "—")
            c3.metric("Lowest-cost fleet", cheap if cheap else "—")

            if need is None:
                st.error("Even the largest fleet tested could not serve every "
                         "stop. Widen time windows or raise capacity.")
            elif need > owned:
                st.error(f"**Today's work does not fit in {owned} truck(s).** "
                         f"It needs {need}. Section C prices the options for "
                         f"covering the gap.")
            else:
                st.success(f"**Today's work fits inside the existing "
                           f"{owned}-truck fleet.** No extra truck required.")

            st.dataframe(sw, width="stretch", hide_index=True)
            st.caption("`marginal_cost` is what each additional truck adds. "
                       "Once it turns positive while `marginal_stops` is zero, "
                       "the extra truck is buying nothing.")
            st.line_chart(sw.set_index("trucks")[["cost", "miles"]])

            st.divider()
            st.markdown("#### How often is the marginal truck needed?")
            st.caption("A single day cannot answer this. The study replays the "
                       "operation over many days with demand noise and customer "
                       "dropout, and records how many trucks each day required. "
                       "The resulting days-per-year figure is the input Section "
                       "C needs — and it is something no generic calculator can "
                       "produce, because it depends on your routing.")

            u1, u2, u3, u4 = st.columns(4)
            n_days = u1.number_input("Days to simulate", 5, 120, 25)
            cv = u2.slider("Demand variability (CV)", 0.0, 1.0, 0.30, 0.05)
            dropout = u3.slider("Customer dropout/day", 0.0, 0.8, 0.20, 0.05)
            surge = u4.slider("Surge-day probability", 0.0, 0.5, 0.10, 0.05)

            if st.button("Run utilisation study"):
                c = current_cfg()
                c.solver.time_limit_sec = max(2, int(quick) // 2)
                bar = st.progress(0.0, text="Starting")
                res = fleet.utilisation_study(
                    stops_geo, trucks, drivers, dist_mi, time_s, c,
                    n_days=int(n_days), demand_cv=cv, dropout=dropout,
                    surge_prob=surge, max_trucks=int(max_k),
                    progress=lambda p, t="": bar.progress(p, text=t),
                    warehouses=warehouses_geo)
                bar.empty()
                st.session_state.util = res

            res = st.session_state.util
            if res is not None:
                st.dataframe(res.summary_frame(), width="stretch",
                             hide_index=True)
                nxt = len(trucks) + 1
                days_nxt = res.days_needing_at_least(nxt)
                st.metric(f"Days per year truck #{nxt} would be needed",
                          f"{days_nxt:.0f}",
                          f"{100*days_nxt/365:.0f}% of days")
                if days_nxt < 60:
                    st.info("Low utilisation — renting by the day or "
                            "outsourcing the overflow will almost certainly "
                            "beat owning. Confirm in Section C.")
                elif days_nxt > 200:
                    st.info("High utilisation — owning or leasing is likely to "
                            "win. Confirm in Section C.")
                st.bar_chart(res.days.set_index("day")["trucks_needed"])
                with st.expander("Day-by-day detail"):
                    st.dataframe(res.days, width="stretch",
                                 hide_index=True)

    # ---------------------------------------------------------------- B ----
    elif sec.startswith("B"):
        st.subheader("B · Warehouse siting")
        st.caption("A facility location problem, not a routing one. The model "
                   "opens a site only when transport savings beat its "
                   "annualised fixed cost, so *keep what you have* is a "
                   "legitimate answer.")

        wk = stops_geo.copy()
        if "deliveries_per_month" not in wk.columns:
            wk["deliveries_per_month"] = 4
        wk["deliveries_per_year"] = wk["deliveries_per_month"].astype(float) * 12
        wk["annual_demand"] = wk["demand_kg"].astype(float) * wk["deliveries_per_year"]

        b1, b2, b3 = st.columns(3)
        cost_mile = b1.number_input("Transport cost ($/mile)", 0.1, 20.0, 2.20, 0.05)
        fixed_cost = b2.number_input("Annual fixed cost of a new site ($)",
                                     0.0, 5_000_000.0, 180_000.0, 5_000.0)
        max_sites = b3.number_input("Max sites to operate", 1, 5, 1)

        b4, b5, b6 = st.columns(3)
        n_clusters = b4.slider("Cluster candidates", 1, 10, 5)
        grid_n = b5.slider("Grid density", 2, 6, 3)
        keep_existing = b6.checkbox("Must keep existing warehouse", value=False)

        existing = pd.DataFrame([{
            "name": str(d["warehouse_name"]),
            "lat": float(d["lat"]), "lon": float(d["lon"]),
            "annual_fixed_cost": fixed_cost, "capacity": 1e12}
            for _, d in warehouses_geo.iterrows()])
        if len(warehouses_geo) > 1:
            st.caption(f"Your current network has {len(warehouses_geo)} sites. "
                       f"The model is free to propose fewer, and the baseline "
                       f"it compares against carries all of their fixed costs.")

        if st.button("Find the best site(s)", type="primary"):
            cand = warehouse.generate_candidates(
                wk, n_clusters=int(n_clusters), grid_n=int(grid_n),
                existing=existing)
            res = warehouse.solve_cflp(
                wk, cand, existing=existing, cost_per_mile=cost_mile,
                candidate_fixed_cost=fixed_cost, max_sites=int(max_sites),
                must_keep_existing=keep_existing)
            st.session_state.candidates = cand
            st.session_state.cflp = res
            st.session_state.validation = None

        res = st.session_state.cflp
        cand = st.session_state.candidates
        if res is not None and cand is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Current network ($/yr)", f"{res.baseline_cost:,.0f}")
            c2.metric("Proposed ($/yr)", f"{res.total_cost:,.0f}",
                      f"{-res.annual_saving:,.0f}")
            c3.metric("Sites opened", res.n_opened)

            if res.annual_saving > 0:
                st.success(res.verdict())
            else:
                st.info(res.verdict())

            st.dataframe(
                res.opened[["site_id", "name", "lat", "lon", "source",
                            "fixed_cost"]],
                width="stretch", hide_index=True)

            if HAVE_FOLIUM:
                st_folium(viz.warehouse_map(wk, cand, res.opened, existing,
                                            res.assignment),
                          height=520, use_container_width=True, key="whmap")

            with st.expander("Stop-to-site assignment"):
                st.dataframe(res.assignment, width="stretch",
                             hide_index=True)

            st.divider()
            st.markdown("#### Validate against real routed cost")
            st.caption("The model above uses out-and-back distance. Real cost "
                       "comes from multi-stop routes, so the proposed site is "
                       "re-run through the full routing model. A site that "
                       "wins on the location objective can lose here — that "
                       "check is the point.")
            if st.button("Re-run routing for each network"):
                cur_label = ("Current network"
                             if len(warehouses_geo) > 1 else "Current warehouse")
                opts = [{"name": cur_label, "warehouses": warehouses_geo}]

                prop = res.opened.rename(columns={"name": "warehouse_name"})[
                    ["site_id", "warehouse_name", "lat", "lon"]].copy()
                prop = prop.rename(columns={"site_id": "warehouse_id"})
                same = (len(prop) == len(warehouses_geo) and all(
                    any(abs(float(p["lat"]) - float(d["lat"])) < 1e-6
                        and abs(float(p["lon"]) - float(d["lon"])) < 1e-6
                        for _, d in warehouses_geo.iterrows())
                    for _, p in prop.iterrows()))
                if not same:
                    name = ("Proposed network"
                            if len(prop) > 1
                            else f"Proposed: {prop.iloc[0]['warehouse_name']}")
                    opts.append({"name": name, "warehouses": prop})

                def mb(coords):
                    d, t, _ = distance.build_matrices(
                        coords, circuity=circuity, urban_mph=urban_mph,
                        hwy_mph=hwy_mph, time_multiplier=time_multiplier())
                    return d, t

                if len(opts) == 1:
                    st.info("The model proposed exactly your current network, "
                            "so there is nothing to compare against.")
                    st.session_state.validation = None
                else:
                    with st.spinner("Re-solving routes for each network…"):
                        st.session_state.validation = warehouse.validate_with_vrp(
                            stops_geo, trucks, drivers, opts, cfg, mb)

            val = st.session_state.validation
            if val is not None and len(val):
                st.dataframe(val, width="stretch", hide_index=True)
                best = val.loc[val["Routed cost/day ($)"].idxmin()]
                cur_rows = val[val["Network"].str.startswith("Current")]
                if len(cur_rows):
                    delta = float(cur_rows.iloc[0]["Routed cost/day ($)"]) - \
                        float(best["Routed cost/day ($)"])
                    if str(best["Network"]).startswith("Current"):
                        st.info("On real routed cost the current network still "
                                "wins. The location model's saving does not "
                                "survive multi-stop routing.")
                    else:
                        st.success(
                            f"**{best['Network']}** is cheaper by "
                            f"${delta:,.2f}/day on routed cost — roughly "
                            f"${delta*260:,.0f}/year over 260 operating days, "
                            f"before the site's fixed cost.")

    # ---------------------------------------------------------------- C ----
    else:
        st.subheader("C · Buy, lease, rent or outsource?")
        st.caption("Capital budgeting, not optimisation. The variable that "
                   "decides it is days used per year — which comes from the "
                   "utilisation study in Section A, not from a guess.")

        util = st.session_state.util
        default_days = 180.0
        if util is not None:
            nxt = len(trucks) + 1
            default_days = max(1.0, util.days_needing_at_least(nxt))
            st.success(f"Using **{default_days:.0f} days/year** from the "
                       f"utilisation study — the days truck #{nxt} would "
                       f"actually be needed.")
        else:
            st.warning("Run the utilisation study in Section A first for a "
                       "grounded number. Until then this uses a placeholder.")

        days = st.slider("Days used per year", 1, 365, int(default_days))

        with st.expander("Assumptions", expanded=False):
            e1, e2, e3 = st.columns(3)
            price = e1.number_input("Purchase price ($)", 0.0, 1e6, 145_000.0, 1000.0)
            down = e1.slider("Down payment %", 0, 100, 20) / 100
            rate = e1.number_input("Finance rate", 0.0, 0.3, 0.085, 0.005)
            fin_yrs = e1.number_input("Finance term (yr)", 1, 10, 5)
            resid = e1.slider("Residual % at horizon", 0, 90, 35) / 100

            ins = e2.number_input("Insurance ($/yr)", 0.0, 1e6, 9_500.0, 500.0)
            lic = e2.number_input("Licensing ($/yr)", 0.0, 1e6, 2_800.0, 100.0)
            maint = e2.number_input("Maintenance yr 1 ($)", 0.0, 1e6, 6_000.0, 500.0)
            maint_g = e2.slider("Maintenance growth/yr", 0.0, 0.5, 0.18, 0.01)
            lease_m = e2.number_input("Lease ($/month)", 0.0, 1e5, 2_450.0, 50.0)

            rent_d = e3.number_input("Daily rental ($)", 0.0, 1e4, 285.0, 5.0)
            out_load = e3.number_input("Carrier cost per load ($)", 0.0, 1e5, 540.0, 10.0)
            miles_day = e3.number_input("Miles per day", 0.0, 1000.0, 120.0, 5.0)
            hours_day = e3.number_input("Hours per day", 0.0, 24.0, 9.0, 0.5)
            horizon = e3.number_input("Horizon (years)", 1, 15, 5)
            disc = e3.number_input("Discount rate", 0.0, 0.3, 0.09, 0.005)

        econ = finance.TruckEconomics(
            purchase_price=price, down_payment_pct=down, finance_rate=rate,
            finance_years=int(fin_yrs), residual_pct_at_horizon=resid,
            insurance_annual=ins, licensing_annual=lic,
            maintenance_year1=maint, maintenance_growth=maint_g,
            lease_monthly=lease_m, rent_daily=rent_d,
            outsource_cost_per_load=out_load, miles_per_day=miles_day,
            hours_per_day=hours_day, fuel_price=fuel_price,
            horizon_years=int(horizon), discount_rate=disc)

        rec = finance.recommend(econ, float(days))
        st.markdown(f"### Recommendation: **{rec['best']}**")
        if rec["close_call"]:
            st.warning(f"Close call — only ${rec['gap']:,.0f} separates this "
                       f"from {rec['runner_up']} over {horizon} years. Treat "
                       f"them as equivalent and decide on flexibility.")
        else:
            st.success(f"Beats {rec['runner_up']} by ${rec['gap']:,.0f} in NPV "
                       f"over {horizon} years at {days} days/year of use.")

        st.dataframe(rec["table"], width="stretch", hide_index=True)

        st.markdown("#### Break-even on utilisation")
        st.caption("The answer flips with how hard the truck works. Below the "
                   "crossover, renting or outsourcing wins; above it, owning "
                   "or leasing does.")
        curve = finance.breakeven_curve(econ)
        st.line_chart(curve.set_index("days_per_year"))
        pts = finance.breakeven_points(curve)
        if pts:
            for p in pts:
                st.info(f"At about **{p['days_per_year']} days/year** the "
                        f"cheapest option switches from {p['from']} to "
                        f"{p['to']}.")
        else:
            st.info("One option stays cheapest across the whole utilisation "
                    "range tested.")

        with st.expander("Year-by-year cash flows"):
            for fn in finance.ALL_OPTIONS:
                r = fn(econ, float(days))
                st.markdown(f"**{r['option']}** — NPV ${r['npv']:,.0f}")
                st.dataframe(r["detail"], width="stretch",
                             hide_index=True)
