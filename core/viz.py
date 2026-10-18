"""Folium maps for routes, geocode review and warehouse candidates."""

from __future__ import annotations

from typing import List, Optional, Sequence

import folium
import numpy as np
import pandas as pd

from .routing import Solution, seconds_to_hhmm

PALETTE = ["#2E86C1", "#E67E22", "#27AE60", "#8E44AD", "#C0392B",
           "#16A085", "#D4AC0D", "#2C3E50", "#CB4335", "#148F77"]


def _center(lats, lons):
    return [float(np.mean(lats)), float(np.mean(lons))]


def route_map(sol: Solution, warehouse=None, show_lines: bool = True) -> folium.Map:
    """Draw the solved routes. Each truck's line starts and ends at its own
    warehouse, which is carried on the Route itself, so one or many warehouses draw
    the same way. `warehouse` is accepted for older callers and ignored when the
    solution already names its warehouses."""
    warehouses = []
    if sol.warehouses is not None and len(sol.warehouses):
        warehouses = [{"label": str(d["warehouse_name"]), "lat": float(d["lat"]),
                   "lon": float(d["lon"])} for _, d in sol.warehouses.iterrows()]
    elif isinstance(warehouse, dict):
        warehouses = [{"label": warehouse.get("label", "Warehouse"),
                   "lat": float(warehouse["lat"]), "lon": float(warehouse["lon"])}]

    lats = [d["lat"] for d in warehouses] + [s.lat for r in sol.routes for s in r.stops]
    lons = [d["lon"] for d in warehouses] + [s.lon for r in sol.routes for s in r.stops]
    for d in sol.dropped:
        lats.append(d["lat"]); lons.append(d["lon"])
    if not lats:
        return folium.Map(location=[38.63, -90.2], zoom_start=9,
                          tiles="OpenStreetMap")

    m = folium.Map(location=_center(lats, lons), zoom_start=10,
                   tiles="OpenStreetMap")

    for d in warehouses:
        folium.Marker(
            [d["lat"], d["lon"]], tooltip=f"Warehouse: {d['label']}",
            icon=folium.Icon(color="black", icon="home", prefix="fa"),
        ).add_to(m)

    for i, r in enumerate(sol.routes):
        if r.n_stops == 0:
            continue
        color = PALETTE[i % len(PALETTE)]
        home = ([r.warehouse_lat, r.warehouse_lon] if r.warehouse_name
                else [warehouses[0]["lat"], warehouses[0]["lon"]])
        name = (f"{r.truck_name} · {r.warehouse_name} ({r.n_stops} stops)"
                if r.warehouse_name else f"{r.truck_name} ({r.n_stops} stops)")
        group = folium.FeatureGroup(name=name)
        path = [list(home)]
        for s in r.stops:
            path.append([s.lat, s.lon])
            late = s.arrival_s > s.tw_close_s + 1
            folium.CircleMarker(
                [s.lat, s.lon], radius=7, color="#B03A2E" if late else color,
                weight=3 if late else 2, fill=True, fill_color=color,
                fill_opacity=0.85,
                tooltip=(f"<b>{r.truck_name} · stop {s.seq}</b><br>{s.label}<br>"
                         f"Arrive {seconds_to_hhmm(s.arrival_s)} "
                         f"(window {seconds_to_hhmm(s.tw_open_s)}–"
                         f"{seconds_to_hhmm(s.tw_close_s)})<br>"
                         f"Drop {s.demand_kg:.0f} kg · "
                         f"{s.load_remaining_kg:.0f} kg left on truck"
                         + ("<br><b>LATE</b>" if late else "")),
            ).add_to(group)
            folium.Marker(
                [s.lat, s.lon],
                icon=folium.DivIcon(html=(
                    f'<div style="font-size:10px;font-weight:700;color:#fff;'
                    f'background:{color};border-radius:9px;width:18px;'
                    f'height:18px;line-height:18px;text-align:center;'
                    f'box-shadow:0 0 2px #0006">{s.seq}</div>')),
            ).add_to(group)
        path.append(list(home))
        if show_lines:
            folium.PolyLine(path, color=color, weight=3, opacity=0.75).add_to(group)
        group.add_to(m)

    if sol.dropped:
        g = folium.FeatureGroup(name=f"Unserved ({len(sol.dropped)})")
        for d in sol.dropped:
            folium.CircleMarker(
                [d["lat"], d["lon"]], radius=8, color="#7B241C", weight=2,
                fill=True, fill_color="#E74C3C", fill_opacity=0.9,
                tooltip=f"UNSERVED: {d.get('label','')} · {d['demand_kg']:.0f} kg",
            ).add_to(g)
        g.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def geocode_review_map(df: pd.DataFrame) -> folium.Map:
    ok = df[df["ok"]] if "ok" in df.columns else df
    if not len(ok):
        return folium.Map(location=[38.63, -90.2], zoom_start=9,
                          tiles="OpenStreetMap")
    m = folium.Map(location=_center(ok["lat"], ok["lon"]), zoom_start=10,
                   tiles="OpenStreetMap")
    for _, r in ok.iterrows():
        conf = float(r.get("confidence", 0) or 0)
        color = "#27AE60" if conf >= 0.8 else "#E67E22" if conf >= 0.5 else "#C0392B"
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=7, color=color, fill=True,
            fill_color=color, fill_opacity=0.8,
            tooltip=(f"<b>{r.get('label','')}</b><br>typed: {r.get('query','')}<br>"
                     f"matched: {r.get('matched','')}<br>"
                     f"{r.get('precision','')}<br>"
                     f"confidence {conf:.2f} ({r.get('provider','')})"),
        ).add_to(m)
    return m


def warehouse_map(stops: pd.DataFrame, candidates: pd.DataFrame,
                  opened: Optional[pd.DataFrame] = None,
                  existing: Optional[pd.DataFrame] = None,
                  assignment: Optional[pd.DataFrame] = None) -> folium.Map:
    m = folium.Map(location=_center(stops["lat"], stops["lon"]), zoom_start=10,
                   tiles="OpenStreetMap")

    dg = folium.FeatureGroup(name="Delivery points")
    wmax = float(stops.get("annual_demand", pd.Series([1])).max() or 1)
    for _, s in stops.iterrows():
        w = float(s.get("annual_demand", 1) or 1)
        folium.CircleMarker(
            [s["lat"], s["lon"]], radius=4 + 8 * (w / wmax) ** 0.5,
            color="#5D6D7E", fill=True, fill_color="#5DADE2", fill_opacity=0.6,
            weight=1,
            tooltip=(f"{s.get('label','')}<br>"
                     f"{s.get('deliveries_per_year', 0):.0f} deliveries/yr<br>"
                     f"{w:,.0f} kg/yr"),
        ).add_to(dg)
    dg.add_to(m)

    cg = folium.FeatureGroup(name="Candidate sites")
    opened_ids = set(opened["site_id"]) if opened is not None and len(opened) else set()
    for _, c in candidates.iterrows():
        chosen = c["site_id"] in opened_ids
        folium.CircleMarker(
            [c["lat"], c["lon"]], radius=10 if chosen else 5,
            color="#1E8449" if chosen else "#AAB7B8",
            fill=True, fill_color="#28B463" if chosen else "#D5DBDB",
            fill_opacity=0.95 if chosen else 0.5, weight=3 if chosen else 1,
            tooltip=(f"<b>{c['name']}</b> ({c['source']})"
                     + ("<br><b>SELECTED</b>" if chosen else "")),
        ).add_to(cg)
    cg.add_to(m)

    if existing is not None and len(existing):
        eg = folium.FeatureGroup(name="Existing warehouses")
        for _, e in existing.iterrows():
            folium.Marker(
                [e["lat"], e["lon"]], tooltip=f"Existing: {e.get('name','')}",
                icon=folium.Icon(color="black", icon="home", prefix="fa"),
            ).add_to(eg)
        eg.add_to(m)

    if assignment is not None and len(assignment) and opened is not None:
        ag = folium.FeatureGroup(name="Assignments", show=False)
        site_xy = {r["site_id"]: (r["lat"], r["lon"]) for _, r in opened.iterrows()}
        lookup = {str(s.get("stop_id")): (s["lat"], s["lon"])
                  for _, s in stops.iterrows()}
        for _, a in assignment.iterrows():
            p = lookup.get(str(a["stop_id"]))
            q = site_xy.get(a["site_id"])
            if p and q:
                folium.PolyLine([list(p), list(q)], color="#85929E",
                                weight=1, opacity=0.5).add_to(ag)
        ag.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
