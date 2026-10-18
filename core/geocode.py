"""
Address -> coordinates.

Two free providers, tried in order:
  1. OpenRouteService (Pelias)  - needs a free API key, better US coverage,
                                  returns a confidence score.
  2. Nominatim (OpenStreetMap)  - no key at all, 1 request/second, requires a
                                  descriptive User-Agent header.

Every successful lookup is cached by the raw address string so repeated
Streamlit reruns never re-query the provider. This is the single most
important thing for staying inside a free tier.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
ORS_URL = "https://api.openrouteservice.org/geocode/search"

# Nominatim's usage policy requires a real identifying User-Agent.
USER_AGENT = "RouteWise-Pilot/1.0 (student routing project; contact via app)"


@dataclass
class GeoResult:
    query: str
    ok: bool
    lat: Optional[float] = None
    lon: Optional[float] = None
    matched: str = ""
    # Normalised precision score, comparable across providers. This is what
    # the map colours and the warnings key off.
    confidence: float = 0.0
    precision: str = ""
    # The provider's own raw score, kept for reference only. Pelias and
    # Nominatim do not mean the same thing by it, so it is never used to
    # decide whether a match is good enough.
    provider_score: float = 0.0
    provider: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class Geocoder:
    """Caching geocoder. Keep one instance in st.session_state."""

    def __init__(self, ors_key: Optional[str] = None,
                 country: str = "US", min_interval_s: float = 1.05):
        self.ors_key = (ors_key or "").strip() or None
        self.country = country
        self.min_interval_s = min_interval_s
        self._cache: Dict[str, GeoResult] = {}
        self._last_call = 0.0

    # -- cache -------------------------------------------------------------
    @property
    def cache(self) -> Dict[str, GeoResult]:
        return self._cache

    def prime(self, address: str, lat: float, lon: float,
              matched: str = "manual entry") -> None:
        """Insert a known coordinate (manual correction, demo data, re-upload)."""
        self._cache[self._key(address)] = GeoResult(
            query=address, ok=True, lat=float(lat), lon=float(lon),
            matched=matched, confidence=1.0, precision="supplied coordinates",
            provider="manual")

    @staticmethod
    def _key(address: str) -> str:
        return " ".join(str(address).split()).lower()

    def _throttle(self) -> None:
        delta = time.time() - self._last_call
        if delta < self.min_interval_s:
            time.sleep(self.min_interval_s - delta)
        self._last_call = time.time()

    # -- public ------------------------------------------------------------
    def geocode(self, address: str) -> GeoResult:
        address = str(address).strip()
        if not address:
            return GeoResult(query=address, ok=False, error="empty address")

        key = self._key(address)
        if key in self._cache:
            return self._cache[key]

        res = None
        if self.ors_key:
            res = self._try_ors(address)
        if res is None or not res.ok:
            fallback = self._try_nominatim(address)
            if fallback.ok or res is None:
                res = fallback

        self._cache[key] = res
        return res

    def geocode_many(self, addresses: List[str], progress=None) -> List[GeoResult]:
        out = []
        total = max(1, len(addresses))
        for i, a in enumerate(addresses):
            out.append(self.geocode(a))
            if progress is not None:
                try:
                    progress((i + 1) / total, f"Geocoding {i+1} of {total}")
                except TypeError:
                    progress((i + 1) / total)
        return out

    # -- providers ---------------------------------------------------------
    def _try_ors(self, address: str) -> GeoResult:
        try:
            r = requests.get(
                ORS_URL,
                params={"api_key": self.ors_key, "text": address,
                        "boundary.country": self.country, "size": 1},
                timeout=20,
            )
            r.raise_for_status()
            feats = r.json().get("features", [])
            if not feats:
                return GeoResult(query=address, ok=False, provider="ors",
                                 error="no match")
            f = feats[0]
            lon, lat = f["geometry"]["coordinates"]
            props = f.get("properties", {})
            conf, precision = _ors_precision(props)
            return GeoResult(
                query=address, ok=True, lat=float(lat), lon=float(lon),
                matched=props.get("label", ""),
                confidence=conf, precision=precision,
                provider_score=float(props.get("confidence", 0.0) or 0.0),
                provider="ors")
        except Exception as e:                                    # noqa: BLE001
            return GeoResult(query=address, ok=False, provider="ors",
                             error=str(e)[:160])

    def _try_nominatim(self, address: str) -> GeoResult:
        try:
            self._throttle()
            r = requests.get(
                NOMINATIM_URL,
                params={"q": address, "format": "json", "limit": 1,
                        "countrycodes": self.country.lower(),
                        "addressdetails": 1},
                headers={"User-Agent": USER_AGENT},
                timeout=20,
            )
            r.raise_for_status()
            js = r.json()
            if not js:
                return GeoResult(query=address, ok=False, provider="nominatim",
                                 error="no match")
            top = js[0]
            conf, precision = _nominatim_precision(top)
            return GeoResult(
                query=address, ok=True,
                lat=float(top["lat"]), lon=float(top["lon"]),
                matched=top.get("display_name", ""),
                confidence=conf, precision=precision,
                provider="nominatim")
        except Exception as e:                                    # noqa: BLE001
            return GeoResult(query=address, ok=False, provider="nominatim",
                             error=str(e)[:160])


def _ors_precision(props: dict) -> tuple:
    """Score a Pelias/ORS hit by the LAYER it resolved to.

    Pelias reports `match_type` as exact | interpolated | fallback. "fallback"
    only says it gave up on the requested layer — it does not say what it fell
    back TO, and falling back to a street is usable while falling back to a
    city centroid is not. `layer` is the field that actually tells you, so it
    drives the score here.

    The returned score is deliberately on the SAME scale as the Nominatim one,
    so a single colour threshold means the same thing whichever provider
    answered. Pelias's own confidence is kept separately as provider_score.
    """
    layer = str(props.get("layer", "")).lower()
    match_type = str(props.get("match_type", "")).lower()
    accuracy = str(props.get("accuracy", "")).lower()

    if layer == "venue":
        return 0.85, "named premises"
    if layer == "address":
        if match_type == "interpolated":
            return 0.75, "interpolated along street"
        return 0.95, "rooftop / exact address"
    if layer in ("street", "intersection"):
        return 0.60, "street centroid (no house number)"
    if layer in ("neighbourhood", "borough", "macrohood", "microhood"):
        return 0.40, "neighbourhood centroid"
    if layer in ("locality", "localadmin", "county", "macrocounty",
                 "region", "macroregion", "postalcode", "country", "coarse"):
        return 0.20, f"{layer} centroid — not a delivery address"
    if accuracy == "centroid":
        return 0.35, "centroid of a larger area"
    return 0.50, f"unclassified match ({layer or 'no layer'})"


def _nominatim_precision(top: dict) -> tuple:
    """Score a Nominatim hit by how precisely it resolved, not how famous it is.

    Nominatim returns no confidence field. Its `importance` value measures the
    notability of a place (a capital city scores high, a street address scores
    almost nothing), so using it as a confidence proxy marks every ordinary
    delivery address as suspect. What actually matters for routing is the
    PRECISION of the match: a rooftop/house-number hit is usable, a street
    centroid is roughly right, and a city or postcode centroid is not a
    delivery location at all.
    """
    addr = top.get("address", {}) or {}
    cls = str(top.get("class", ""))
    typ = str(top.get("type", ""))
    atype = str(top.get("addresstype", ""))

    if addr.get("house_number"):
        return 0.95, "rooftop / house number"
    if cls in ("amenity", "shop", "building", "office", "leisure",
               "tourism", "healthcare") or atype in ("amenity", "shop",
                                                     "building", "office"):
        return 0.85, "named premises"
    if atype in ("road", "street") or typ in ("road", "residential",
                                              "primary", "secondary",
                                              "tertiary", "service",
                                              "trunk", "motorway"):
        return 0.60, "street centroid (no house number)"
    if atype in ("suburb", "neighbourhood", "quarter", "hamlet", "village"):
        return 0.40, "neighbourhood centroid"
    if atype in ("city", "town", "municipality", "county", "state",
                 "postcode") or typ in ("city", "town", "administrative",
                                        "postcode"):
        return 0.20, "area centroid — too coarse to deliver to"
    return 0.50, "unclassified match"


def parse_bulk(text: str) -> List[str]:
    """One address per line; blank lines and '#' comments ignored."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip().strip(",")
        if line and not line.startswith("#"):
            out.append(line)
    return out
