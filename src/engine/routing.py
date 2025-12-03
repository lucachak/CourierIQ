"""
routing.py - Routing service abstraction

Provides a simple interface to get distance matrices and route directions.
Supports OSRM (public) and Google Maps (if API key provided).

This implementation is intentionally lightweight and synchronous. It can be
replaced later with an async client or more robust retry/quotas handling for
production use.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class RoutingService:
    """
    Minimal routing wrapper for OSRM and Google Maps.

    Methods:
      - get_distance_matrix(origins, destinations, provider='osrm'|'google')
        -> returns {"distance_matrix": [[m,...],...], "duration_matrix": [[s,...],...]}
      - get_directions(origin, destination, provider='osrm'|'google')
        -> returns {"distance": m, "duration": s, "polyline": "..."}

    Notes:
      - origins/destinations: sequences of (lat, lon)
      - provider: if None, will prefer Google when an API key is configured,
        otherwise fall back to OSRM.
      - This class uses a requests.Session with simple retry strategy.
    """

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        osrm_url: str = "http://router.project-osrm.org",
        timeout: float = 10.0,
        max_retries: int = 2,
    ):
        self.google_api_key = google_api_key
        self.osrm_url = osrm_url.rstrip("/")
        self.timeout = float(timeout)

        # Create a requests session with a small retry policy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            backoff_factor=0.3,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @staticmethod
    def _format_coord(coord: Tuple[float, float]) -> str:
        """Format (lat, lon) to OSRM-style 'lon,lat' string."""
        lat, lon = coord
        return f"{lon:.6f},{lat:.6f}"

    def _choose_provider(self, provider: Optional[str]) -> str:
        if provider:
            return provider.lower()
        return "google" if self.google_api_key else "osrm"

    def get_distance_matrix(
        self,
        origins: Sequence[Tuple[float, float]],
        destinations: Sequence[Tuple[float, float]],
        provider: Optional[str] = None,
    ) -> Optional[Dict[str, List[List[float]]]]:
        """
        Return distance and duration matrices between origins and destinations.

        Returns:
            {
                "distance_matrix": [[meters,...], ...],
                "duration_matrix": [[seconds,...], ...]
            } or None on failure
        """
        provider = self._choose_provider(provider)

        if provider == "osrm":
            # OSRM Table API expects coordinates in the same list; we pass origins+destinations
            try:
                # Build coordinates string
                coords = list(origins) + list(destinations)
                coord_str = ";".join(self._format_coord(c) for c in coords)

                url = f"{self.osrm_url}/table/v1/driving/{coord_str}"
                params = {"annotations": "distance,duration"}

                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()

                distances = payload.get("distances")
                durations = payload.get("durations")
                if distances is None or durations is None:
                    logger.error("OSRM response missing distances/durations")
                    return None

                o = len(origins)
                d = len(destinations)
                # Slice top-left origin->destination submatrix
                dist_matrix = [row[o : o + d] for row in distances[:o]]
                dur_matrix = [row[o : o + d] for row in durations[:o]]

                return {"distance_matrix": dist_matrix, "duration_matrix": dur_matrix}
            except Exception as exc:  # keep broad for robustness
                logger.exception("OSRM distance matrix request failed: %s", exc)
                return None

        elif provider == "google":
            if not self.google_api_key:
                logger.error("Google provider requested but no API key configured")
                return None
            try:
                base = "https://maps.googleapis.com/maps/api/distancematrix/json"
                origins_str = "|".join(f"{lat},{lon}" for lat, lon in origins)
                dests_str = "|".join(f"{lat},{lon}" for lat, lon in destinations)
                params = {
                    "origins": origins_str,
                    "destinations": dests_str,
                    "key": self.google_api_key,
                    "units": "metric",
                }
                resp = self.session.get(base, params=params, timeout=self.timeout)
                resp.raise_for_status()
                j = resp.json()

                dist_matrix: List[List[float]] = []
                dur_matrix: List[List[float]] = []
                for row in j.get("rows", []):
                    elements = row.get("elements", [])
                    dist_row = [
                        el.get("distance", {}).get("value", 0) for el in elements
                    ]
                    dur_row = [
                        el.get("duration", {}).get("value", 0) for el in elements
                    ]
                    dist_matrix.append(dist_row)
                    dur_matrix.append(dur_row)

                return {"distance_matrix": dist_matrix, "duration_matrix": dur_matrix}
            except Exception as exc:
                logger.exception("Google Distance Matrix request failed: %s", exc)
                return None
        else:
            logger.error("Unknown provider for distance matrix: %s", provider)
            return None
    @property
    def providers(self) -> List[str]:
        """List available providers"""
        available = ['osrm']
        if self.google_api_key:
            available.append('google')
        return available

    def get_directions(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        provider: Optional[str] = None,
    ) -> Optional[Dict[str, object]]:
        """
        Return a simple directions dict with keys:
            - distance (meters)
            - duration (seconds)
            - polyline (string or None)
        """
        provider = self._choose_provider(provider)

        if provider == "osrm":
            try:
                coord_str = (
                    f"{self._format_coord(origin)};{self._format_coord(destination)}"
                )
                url = f"{self.osrm_url}/route/v1/driving/{coord_str}"
                params = {"overview": "full", "geometries": "polyline"}
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                j = resp.json()
                routes = j.get("routes") or []
                if not routes:
                    return None
                r0 = routes[0]
                return {
                    "distance": r0.get("distance", 0),
                    "duration": r0.get("duration", 0),
                    "polyline": r0.get("geometry"),
                }
            except Exception as exc:
                logger.exception("OSRM directions request failed: %s", exc)
                return None

        elif provider == "google":
            if not self.google_api_key:
                logger.error("Google provider requested but no API key configured")
                return None
            try:
                base = "https://maps.googleapis.com/maps/api/directions/json"
                params = {
                    "origin": f"{origin[0]},{origin[1]}",
                    "destination": f"{destination[0]},{destination[1]}",
                    "key": self.google_api_key,
                    "mode": "driving",
                }
                resp = self.session.get(base, params=params, timeout=self.timeout)
                resp.raise_for_status()
                j = resp.json()
                routes = j.get("routes") or []
                if not routes:
                    return None
                r0 = routes[0]
                legs = r0.get("legs", []) or []
                leg0 = legs[0] if legs else {}
                return {
                    "distance": leg0.get("distance", {}).get("value", 0),
                    "duration": leg0.get("duration", {}).get("value", 0),
                    "polyline": r0.get("overview_polyline", {}).get("points"),
                }
            except Exception as exc:
                logger.exception("Google directions request failed: %s", exc)
                return None
        else:
            logger.error("Unknown provider for directions: %s", provider)
            return None
