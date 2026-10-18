"""
Buy versus rent versus outsource for an additional truck.

This is capital budgeting, not optimisation. Four options are compared on
discounted cash flow over a planning horizon:

  OWN            purchase with financing, depreciating asset, residual value
  LEASE          long-term full-service lease, fixed monthly, mileage cap
  SHORT_RENT     rent by the day, only on days the truck is actually needed
  OUTSOURCE      hand the volume to a carrier, per load

The variable that decides the answer is DAYS USED PER YEAR, which comes from
the routing model's utilisation study rather than from a guess. That linkage
is the whole point: a truck needed 240 days a year is a purchase, a truck
needed 45 days a year is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TruckEconomics:
    # purchase
    purchase_price: float = 145_000.0
    down_payment_pct: float = 0.20
    finance_rate: float = 0.085
    finance_years: int = 5
    residual_pct_at_horizon: float = 0.35
    # annual ownership costs
    insurance_annual: float = 9_500.0
    licensing_annual: float = 2_800.0
    maintenance_year1: float = 6_000.0
    maintenance_growth: float = 0.18          # compounding per year of age
    # lease
    lease_monthly: float = 2_450.0
    lease_included_miles_per_year: float = 30_000.0
    lease_overage_per_mile: float = 0.14
    # short-term rental
    rent_daily: float = 285.0
    rent_included_miles_per_day: float = 150.0
    rent_overage_per_mile: float = 0.35
    # outsourcing
    outsource_cost_per_load: float = 540.0
    loads_per_day: float = 1.0
    # shared operating assumptions
    miles_per_day: float = 120.0
    hours_per_day: float = 9.0
    driver_hourly: float = 28.0
    driver_benefit_load: float = 0.28         # payroll tax + benefits uplift
    fuel_price: float = 3.85
    mpg: float = 7.0
    # analysis
    horizon_years: int = 5
    discount_rate: float = 0.09


def loan_payment(principal: float, annual_rate: float, years: int) -> float:
    """Level annual payment on an amortising loan."""
    if years <= 0:
        return 0.0
    if annual_rate <= 0:
        return principal / years
    r = annual_rate
    return principal * r / (1 - (1 + r) ** (-years))


def npv(cashflows: List[float], rate: float) -> float:
    """cashflows[0] is time zero. Costs are positive numbers here."""
    return float(sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows)))


# --------------------------------------------------------------------------- #

def _operating_cost(e: TruckEconomics, days: float) -> float:
    """Driver plus fuel for one year, charged only on days the truck runs."""
    driver = (e.driver_hourly * (1 + e.driver_benefit_load)
              * e.hours_per_day * days)
    fuel = e.miles_per_day * days / max(1.0, e.mpg) * e.fuel_price
    return driver + fuel


def option_own(e: TruckEconomics, days_per_year: float) -> Dict:
    down = e.purchase_price * e.down_payment_pct
    financed = e.purchase_price - down
    pmt = loan_payment(financed, e.finance_rate, e.finance_years)

    flows = [down]
    detail = []
    for yr in range(1, e.horizon_years + 1):
        maint = e.maintenance_year1 * (1 + e.maintenance_growth) ** (yr - 1)
        loan = pmt if yr <= e.finance_years else 0.0
        fixed = loan + e.insurance_annual + e.licensing_annual + maint
        var = _operating_cost(e, days_per_year)
        total = fixed + var
        if yr == e.horizon_years:
            total -= e.purchase_price * e.residual_pct_at_horizon
        flows.append(total)
        detail.append({"year": yr, "loan": round(loan, 0),
                       "maintenance": round(maint, 0),
                       "insurance+licensing": round(e.insurance_annual + e.licensing_annual, 0),
                       "operating": round(var, 0), "net": round(total, 0)})
    return {"option": "Own (financed)", "npv": npv(flows, e.discount_rate),
            "flows": flows, "detail": pd.DataFrame(detail)}


def option_lease(e: TruckEconomics, days_per_year: float) -> Dict:
    flows = [0.0]
    detail = []
    annual_miles = e.miles_per_day * days_per_year
    for yr in range(1, e.horizon_years + 1):
        lease = e.lease_monthly * 12
        over = max(0.0, annual_miles - e.lease_included_miles_per_year) * \
            e.lease_overage_per_mile
        var = _operating_cost(e, days_per_year)
        total = lease + over + var
        flows.append(total)
        detail.append({"year": yr, "lease": round(lease, 0),
                       "mileage overage": round(over, 0),
                       "operating": round(var, 0), "net": round(total, 0)})
    return {"option": "Lease (full service)", "npv": npv(flows, e.discount_rate),
            "flows": flows, "detail": pd.DataFrame(detail)}


def option_short_rent(e: TruckEconomics, days_per_year: float) -> Dict:
    flows = [0.0]
    detail = []
    for yr in range(1, e.horizon_years + 1):
        rent = e.rent_daily * days_per_year
        over = max(0.0, e.miles_per_day - e.rent_included_miles_per_day) * \
            e.rent_overage_per_mile * days_per_year
        var = _operating_cost(e, days_per_year)
        total = rent + over + var
        flows.append(total)
        detail.append({"year": yr, "rental": round(rent, 0),
                       "mileage overage": round(over, 0),
                       "operating": round(var, 0), "net": round(total, 0)})
    return {"option": "Rent by the day", "npv": npv(flows, e.discount_rate),
            "flows": flows, "detail": pd.DataFrame(detail)}


def option_outsource(e: TruckEconomics, days_per_year: float) -> Dict:
    flows = [0.0]
    detail = []
    for yr in range(1, e.horizon_years + 1):
        total = e.outsource_cost_per_load * e.loads_per_day * days_per_year
        flows.append(total)
        detail.append({"year": yr, "carrier charges": round(total, 0),
                       "operating": 0, "net": round(total, 0)})
    return {"option": "Outsource to carrier", "npv": npv(flows, e.discount_rate),
            "flows": flows, "detail": pd.DataFrame(detail)}


ALL_OPTIONS = [option_own, option_lease, option_short_rent, option_outsource]


def compare(e: TruckEconomics, days_per_year: float) -> pd.DataFrame:
    rows = []
    for fn in ALL_OPTIONS:
        r = fn(e, days_per_year)
        per_day = r["npv"] / max(1.0, days_per_year * e.horizon_years)
        rows.append({
            "Option": r["option"],
            f"NPV over {e.horizon_years} yr ($)": round(r["npv"], 0),
            "Cost per day used ($)": round(per_day, 2),
            "Cost per mile ($)": round(
                per_day / max(1.0, e.miles_per_day), 2),
        })
    df = pd.DataFrame(rows).sort_values(
        f"NPV over {e.horizon_years} yr ($)").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def breakeven_curve(e: TruckEconomics, days_grid: Optional[np.ndarray] = None
                    ) -> pd.DataFrame:
    """NPV of every option across a sweep of days-used-per-year."""
    if days_grid is None:
        days_grid = np.arange(10, 311, 10)
    rows = []
    for d in days_grid:
        rec = {"days_per_year": int(d)}
        for fn in ALL_OPTIONS:
            r = fn(e, float(d))
            rec[r["option"]] = round(r["npv"], 0)
        rows.append(rec)
    return pd.DataFrame(rows)


def breakeven_points(curve: pd.DataFrame) -> List[dict]:
    """Where the cheapest option changes as utilisation rises."""
    opts = [c for c in curve.columns if c != "days_per_year"]
    out, prev = [], None
    for _, row in curve.iterrows():
        best = min(opts, key=lambda c: row[c])
        if prev is not None and best != prev:
            out.append({"days_per_year": int(row["days_per_year"]),
                        "from": prev, "to": best})
        prev = best
    return out


def recommend(e: TruckEconomics, days_per_year: float) -> dict:
    df = compare(e, days_per_year)
    best = df.iloc[0]
    second = df.iloc[1] if len(df) > 1 else None
    col = f"NPV over {e.horizon_years} yr ($)"
    gap = (second[col] - best[col]) if second is not None else 0.0
    margin = gap / max(1.0, abs(best[col]))
    return {
        "days_per_year": days_per_year,
        "best": best["Option"],
        "best_npv": float(best[col]),
        "runner_up": second["Option"] if second is not None else None,
        "gap": float(gap),
        "close_call": bool(margin < 0.07),
        "table": df,
    }
