# RouteWise

Delivery route and cost planner for small fleets. Three tabs:

**① Start here** — what the tool is for, the problem it solves, a step-by-step
walkthrough, a glossary, and an honest list of what it does not do. This is
the tab to show someone who has never seen the tool before.

**② Plan routes** — type warehouse and delivery addresses, set trucks and drivers, get
optimised routes that respect capacity, customer time windows, driver shifts
and Hours-of-Service limits. Compared against the naive plan a dispatcher
would build by hand, so the saving is a number, not a claim. One warehouse or
several: each truck is based at a yard and returns to it.

**③ Network & fleet decisions**
- **A · Do we need another truck?** Smallest legal fleet for today, plus a
  multi-day study of how often the marginal truck is actually needed.
- **B · Where should the warehouse be?** Capacitated facility location, then
  re-validated through the full routing model.
- **C · Buy, lease, rent or outsource?** Discounted cash flow over a horizon,
  driven by the utilisation number from A.

The three sections are linked on purpose. Section A's routing output feeds
Section C's utilisation, which is the variable that decides the answer. A
generic TCO calculator cannot do that, because it does not know your routes.

---

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open **① Start here**, then click **Load demo · 1 warehouse** or **Load demo · 2 warehouses** on the Plan routes tab — 20 St. Louis
stops with coordinates already baked in, so the demo makes zero network calls
and always works.

There are two demo buttons — one warehouse, or two — so you can see multi-warehouse
routing without typing anything.

Verify the models independently at any time:

```bash
python selftest.py                 # 60 checks, no network, no Streamlit
```

---

## Project layout

```
app.py                  Streamlit UI, both tabs
core/
  geocode.py            address -> lat/lon, ORS + Nominatim, cached
  distance.py           distance and travel-time matrices
  routing.py            OR-Tools VRPTW, HOS, driver eligibility, baseline
  fleet.py              fleet sizing sweep and multi-day utilisation study
  warehouse.py          candidate generation, CFLP via PuLP, routed validation
  finance.py            own / lease / rent / outsource NPV and break-even
  viz.py                Folium maps
  demo_data.py          preloaded demo scenario and fleet defaults
selftest.py             end-to-end model checks
```

---

## How the models work

### Routing (`core/routing.py`)

Nodes are the warehouses (`0..D-1`) and then the stops (`D..D+n-1`). Build every
matrix as `warehouse_coords(warehouses) + stop coordinates`, in that order — the
solver raises a clear error if the matrix size does not match.

Each truck names its home yard in a `warehouse_id` column, and
`RoutingIndexManager` is given per-vehicle starts and ends, so a truck leaves
from and returns to its own warehouse. An unrecognised `warehouse_id` falls back to
the first warehouse rather than failing, so a fleet table written before warehouses
existed still solves. Passing a single `{"lat","lon"}` dict also still works.

Four dimensions:

| Dimension | Meaning | Bound |
|---|---|---|
| `Capacity` | kg carried | truck payload |
| `Time` | seconds since midnight, transit = travel + service | customer windows, shift window, duty span |
| `Driving` | travel seconds only | HOS driving limit |

Keeping `Driving` separate from `Time` matters: without it, service time could
be used to dodge the 11-hour driving rule.

Objective, in micro-dollars so integer rounding stays honest:

```
  arc:        distance_mi x fuel_price / mpg
+ per truck:  duty_seconds x wage_per_second        (covers waiting too)
+ per truck:  fixed_cost_day when used
+ dropped:    priority-weighted penalty
+ optional:   global span coefficient for workload fairness
```

Every stop carries a **disjunction**, so when the problem is over-constrained
the model drops stops and reports them rather than failing with no explanation.
That is a diagnosis you can act on: add a truck, widen a window, raise capacity.

### Driver age is a filter, not a cost

Under 49 CFR 391.11 a driver under 21 may not operate interstate, and under 18
may not operate a CMV at all. Age therefore decides *whether the pairing is
allowed*, not how expensive it is. `screen_drivers()` removes ineligible
truck–driver pairs before the model is built and reports the reason in the UI.
The same screen handles the weekly hour clock and impossible shift windows.

### The 30-minute break

Modelled as an **optional** interval via `SetBreakIntervalsOfVehicle`. A route
with under 8 hours of driving does not require a break, and forcing the
interval would stretch short routes artificially. Because the break can
legitimately be absent, `compliance_check()` verifies after the solve that no
route exceeded the driving threshold without one, rather than assuming it.

### Warehouse siting (`core/warehouse.py`)

A facility location problem, not a routing one, so it gets its own MILP:

```
minimise   sum_j  fixed_j · y_j  +  sum_ij  cost_ij · x_ij
s.t.       sum_j x_ij = 1           every stop served
           x_ij <= y_j              only from an open site
           sum_i w_i x_ij <= cap_j · y_j
           sum_j y_j <= max_sites
```

`y_j` opens a site, `x_ij` assigns a stop to it, both binary. Because fixed
cost is in the objective, **"open nothing new" is a legitimate answer** — and
with realistic fixed costs it is frequently the right one.

Candidates are generated from the delivery points themselves: demand-weighted
KMeans centroids, a coarse grid over the bounding box, the centre of gravity,
and any existing sites. You never have to supply a candidate list.

**The validation step most tools skip.** The MILP uses out-and-back distance.
Real cost comes from multi-stop routes. So the proposed site is re-run through
the full routing model and compared on actual routed cost. On the demo data the
location model shows about $8.5k/year of saving; routed validation cuts that to
roughly $2.5k. A site that wins the MILP can lose here, and showing that is the
most valuable thing in the tab.

### Buy vs rent (`core/finance.py`)

Four options over a horizon, discounted: own (financed, with residual value),
full-service lease, daily rental, and outsourcing to a carrier. The deciding
variable is **days used per year**, taken from Section A's utilisation study.

On the defaults the crossover lands near 200 days/year: below it, outsourcing
wins; above it, leasing does. A truck needed 45 days a year is not a purchase.

---

## Known limitations — state these, do not hide them

**Traffic is static.** True traffic-aware routing needs arc travel time to
depend on departure time, which is a decision variable inside the solver. This
pilot applies one congestion multiplier derived from how much of the shift
overlaps peak periods. Swap in a live matrix API (Mapbox, HERE, TomTom) to
remove the approximation.

**Distances are estimated by default.** Haversine × a circuity factor with a
piecewise speed profile, so the app runs with zero API calls. Supply an
OpenRouteService key and tick the box in the sidebar for real road distances.

**Geocoding is the weakest link.** A wrong pin produces a route that looks
plausible and is wrong, which is worse than failing. The app shows the matched
address, a confidence score and a review map before solving. Do not skip it.

**Not a compliance system of record.** The HOS model is a simplified reading of
49 CFR 395. It is decision support.

**Solve time.** 20–60 stops solve in seconds. Past roughly 150 stops the UI
feels frozen, because Streamlit blocks during computation. Cap demo instances.

---

## Deploying for free

Hugging Face Spaces is a better fit than Streamlit Community Cloud here: more
RAM (OR-Tools is memory-hungry during search) and private Spaces on the free
tier. Push `app.py`, `core/`, and `requirements.txt`; set the Space SDK to
Streamlit.

If you demo with real customer addresses, run locally instead — anything on a
public Space is public.
