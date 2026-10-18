"""
Preloaded demo scenario so you can show the app without typing addresses live.

Coordinates are hard-coded, so loading the demo makes zero geocoding calls.
Replace with your own city by editing DEMO_WAREHOUSE and DEMO_STOPS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEMO_WAREHOUSE = {
    "address": "4100 Union Blvd, St. Louis, MO 63115",
    "label": "Main Warehouse",
    "lat": 38.6690, "lon": -90.2510,
}

# (label, address, lat, lon, demand_kg, open, close, service_min, priority, per_month)
_RAW = [
    ("Soulard Grocery",        "730 Carroll St, St. Louis, MO 63104",      38.6085, -90.2040, 420, "07:00", "11:00", 25, 1, 12),
    ("Delmar Food Market",     "6662 Delmar Blvd, University City, MO",     38.6560, -90.3090, 310, "08:00", "13:00", 20, 1, 8),
    ("Clayton Fresh Market",   "8000 Forsyth Blvd, Clayton, MO 63105",      38.6480, -90.3330, 180, "09:00", "14:00", 15, 2, 4),
    ("Kirkwood Grocery",       "139 W Argonne Dr, Kirkwood, MO 63122",      38.5830, -90.4070, 390, "07:30", "12:00", 22, 1, 10),
    ("Webster Corner Store",   "8148 Big Bend Blvd, Webster Groves, MO",    38.5920, -90.3550, 140, "08:00", "15:00", 12, 2, 6),
    ("Maplewood Food Mart",    "7326 Manchester Rd, Maplewood, MO 63143",   38.6120, -90.3240, 220, "08:30", "14:30", 18, 1, 8),
    ("Tower Grove Co-op",      "3137 Morgan Ford Rd, St. Louis, MO 63116",  38.5940, -90.2520, 350, "07:00", "12:30", 20, 1, 12),
    ("Central West End Market","4584 Laclede Ave, St. Louis, MO 63108",     38.6400, -90.2600, 260, "09:00", "16:00", 16, 2, 6),
    ("Ferguson Family Foods",  "101 S Florissant Rd, Ferguson, MO 63135",   38.7440, -90.3050, 480, "06:30", "11:30", 28, 1, 10),
    ("Florissant Supermarket", "1055 Rue St Francois, Florissant, MO",      38.7890, -90.3230, 300, "07:00", "13:00", 20, 1, 6),
    ("Chesterfield Supermarket","1600 Clarkson Rd, Chesterfield, MO 63017",  38.6300, -90.5700, 540, "08:00", "15:00", 30, 1, 8),
    ("Ballwin Grocery",        "14890 Manchester Rd, Ballwin, MO 63011",    38.5950, -90.5460, 280, "08:00", "14:00", 18, 2, 6),
    ("Creve Coeur Market",     "11401 Olive Blvd, Creve Coeur, MO 63141",   38.6730, -90.4530, 160, "09:00", "16:00", 12, 3, 4),
    ("Affton Food Center",     "9666 Gravois Rd, St. Louis, MO 63123",      38.5500, -90.3300, 330, "07:30", "12:30", 20, 1, 8),
    ("Arnold Grocery",         "2010 Michigan Ave, Arnold, MO 63010",       38.4330, -90.3770, 450, "07:00", "11:00", 26, 1, 4),
    ("Belleville Food Market", "100 W Main St, Belleville, IL 62220",       38.5200, -89.9840, 390, "08:00", "14:00", 24, 2, 4),
    ("Edwardsville Supermarket","100 N Main St, Edwardsville, IL 62025",     38.8110, -89.9530, 300, "08:30", "15:00", 20, 2, 4),
    ("Granite City Grocery",   "1800 Delmar Ave, Granite City, IL 62040",   38.7020, -90.1490, 240, "07:00", "13:00", 18, 2, 6),
    ("St. Charles Fresh Market","300 N Main St, St Charles, MO 63301",       38.7830, -90.4820, 420, "07:00", "12:00", 25, 1, 8),
    ("O'Fallon Grocery",       "2630 Hwy K, O Fallon, MO 63368",            38.7640, -90.6980, 360, "08:00", "14:00", 22, 2, 4),
]


def demo_stops() -> pd.DataFrame:
    rows = []
    for i, (label, addr, lat, lon, kg, o, c, svc, pr, per_month) in enumerate(_RAW, 1):
        rows.append({
            "stop_id": f"S{i:03d}", "label": label, "address": addr,
            "lat": lat, "lon": lon, "demand_kg": float(kg),
            "tw_open": o, "tw_close": c, "service_min": svc,
            "priority": pr, "deliveries_per_month": per_month,
        })
    return pd.DataFrame(rows)


def demo_warehouse() -> dict:
    return dict(DEMO_WAREHOUSE)


# A second yard in west county, for trying the multi-warehouse path.
DEMO_WAREHOUSE_WEST = {
    "warehouse_id": "W02",
    "warehouse_name": "West Yard",
    "address": "16100 Chesterfield Pkwy W, Chesterfield, MO 63017",
    "lat": 38.6540, "lon": -90.5770,
}


def demo_warehouses(n: int = 1) -> pd.DataFrame:
    """n=1 gives the single main warehouse, n=2 adds the west county yard."""
    rows = [{"warehouse_id": "W01", "warehouse_name": "Main Warehouse",
             "address": DEMO_WAREHOUSE["address"],
             "lat": DEMO_WAREHOUSE["lat"], "lon": DEMO_WAREHOUSE["lon"]}]
    if n >= 2:
        rows.append(dict(DEMO_WAREHOUSE_WEST))
    return pd.DataFrame(rows)


def empty_warehouses() -> pd.DataFrame:
    return pd.DataFrame([{"warehouse_id": "W01", "warehouse_name": "Main Warehouse",
                          "address": ""}])


def default_trucks(n: int = 3, warehouse_id: str = "W01") -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "truck_id": f"T{i:02d}",
            "truck_name": f"Truck {i}",
            "capacity_kg": 3200.0,
            "mpg": 7.0,
            "fixed_cost_day": 95.0,
            "driver_id": f"D{i:02d}",
            "warehouse_id": warehouse_id,
        })
    return pd.DataFrame(rows)


def default_drivers(n: int = 3) -> pd.DataFrame:
    ages = [34, 27, 45, 52, 23, 38, 29, 41]
    wages = [28.0, 26.5, 31.0, 29.5, 25.0, 30.0, 27.0, 32.0]
    used = [38.0, 44.0, 22.0, 51.0, 30.0, 18.0, 46.0, 25.0]
    rows = []
    for i in range(1, n + 1):
        k = (i - 1) % len(ages)
        rows.append({
            "driver_id": f"D{i:02d}",
            "driver_name": f"Driver {i}",
            "age": ages[k],
            "cdl_class": "A",
            "shift_start": "06:00",
            "shift_end": "18:00",
            "hours_used_week": used[k],
            "max_weekly_hours": 70.0,
            "hourly_wage": wages[k],
        })
    return pd.DataFrame(rows)


def empty_stops() -> pd.DataFrame:
    return pd.DataFrame({
        "label": pd.Series(dtype="object"),
        "address": pd.Series(dtype="object"),
        "demand_kg": pd.Series(dtype="float"),
        "tw_open": pd.Series(dtype="object"),
        "tw_close": pd.Series(dtype="object"),
        "service_min": pd.Series(dtype="int"),
        "priority": pd.Series(dtype="int"),
        "deliveries_per_month": pd.Series(dtype="int"),
    })


def blank_rows(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame([{
        "label": "", "address": "", "demand_kg": 200.0,
        "tw_open": "08:00", "tw_close": "17:00", "service_min": 15,
        "priority": 1, "deliveries_per_month": 4,
    } for _ in range(n)])
