"""
scorer.py - Route scoring utilities
Provides a flexible RouteScorer used by EngineService.score_route
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.utils.helpers import estimate_fuel_cost


@dataclass
class ScoringWeights:
    """
    Weights for the different scoring components.
    Sum is expected to be ~1.0 but not strictly enforced.
    Tune these to change importance of each metric.
    """

    distance: float = 0.25
    time: float = 0.25
    fuel: float = 0.15
    violations: float = 0.15
    priority: float = 0.1
    traffic: float = 0.05
    weather: float = 0.05


@dataclass
class RouteScore:
    """
    Aggregated route scoring output.
    - total_score: numeric score where lower is better (0 best, 1 worst in our convention)
    - efficiency_rating: human-friendly 0..100 (higher is better)
    - individual *_score fields are normalized 0..1 where 1 = best
    - breakdown: raw dict of component scores
    """

    total_score: float
    efficiency_rating: float
    distance_score: float
    time_score: float
    fuel_score: float
    violation_score: float
    priority_score: float
    traffic_score: float
    weather_score: float
    breakdown: Dict[str, float]


class RouteScorer:
    """
    Heuristic route scorer.

    The scoring tries to normalize input metrics into 0..1 (1 = best) for each
    component then composes them with configurable weights.

    Notes:
    - Input `route_data` is expected to be a dict with (optionally):
        - 'distance' (meters)
        - 'duration' (seconds)
        - 'duration_in_traffic' (seconds)
        - 'violations' (count)
        - 'late_count' (count)
      but the function is defensive and will accept missing fields.
    - `stops` is expected to be a list of dict-like objects that may contain
      a 'priority' key (1..N). Lower numeric priority means more important.
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()

        # Normalization ceilings (tunable)
        self.max_distance_km = 200.0  # distances above this saturate
        self.max_time_hours = 8.0  # times above this saturate
        self.max_fuel_cost = 50.0  # fuel cost above this saturate

    def score_route(
        self,
        route_data: Dict,
        stops: List[Dict],
        current_time: float = 0.0,
        time_of_day: Optional[int] = None,
        weather_conditions: Optional[Dict] = None,
    ) -> RouteScore:
        """
        Compute a composite RouteScore.

        Returns:
            RouteScore
        """
        # Defensive extraction with defaults
        distance_m = float(route_data.get("distance", 0.0) or 0.0)
        duration_s = float(route_data.get("duration", 0.0) or 0.0)

        distance_km = distance_m / 1000.0
        time_hours = duration_s / 3600.0

        # Distance: smaller is better -> normalize to 0..1 (1 best)
        distance_score = 1.0 - min(distance_km / max(self.max_distance_km, 1e-6), 1.0)

        # Time: smaller is better
        time_score = 1.0 - min(time_hours / max(self.max_time_hours, 1e-6), 1.0)

        # Fuel: estimate fuel cost and prefer lower cost
        try:
            fuel_cost = estimate_fuel_cost(distance_km)
        except Exception:
            fuel_cost = 0.0
        fuel_score = 1.0 - min(fuel_cost / max(self.max_fuel_cost, 1e-6), 1.0)

        # Violations: explicit 'violations' or 'late_count'
        violations = float(route_data.get("violations", 0.0) or 0.0)
        if violations > 0:
            violation_score = 1.0 - min(violations / max(len(stops), 1), 1.0)
        else:
            late_count = float(route_data.get("late_count", 0.0) or 0.0)
            violation_score = 1.0 - min(late_count / max(len(stops), 1), 1.0)

        # Priority: average priority across stops. Lower numeric priority is more important.
        priorities = []
        for s in stops:
            try:
                pri = (
                    s.get("priority", 1)
                    if isinstance(s, dict)
                    else getattr(s, "priority", 1)
                )
                priorities.append(float(pri))
            except Exception:
                continue

        if priorities:
            avg_priority = sum(priorities) / len(priorities)
            # Map average priority 1 -> 1.0, high values -> approach 0.0
            priority_score = 1.0 - min(max((avg_priority - 1.0) / 9.0, 0.0), 1.0)
        else:
            priority_score = 1.0

        # Traffic: if duration_in_traffic is provided, compute penalty ratio
        if route_data.get("duration_in_traffic"):
            dit = float(route_data.get("duration_in_traffic") or duration_s)
            if dit > 0 and duration_s > 0:
                traffic_penalty = max(0.0, (dit - duration_s) / max(dit, 1e-6))
                traffic_score = 1.0 - min(traffic_penalty, 1.0)
            else:
                traffic_score = 1.0
        else:
            traffic_score = 1.0

        # Weather: simple mapping - penalize heavy weather
        weather_score = 1.0
        if weather_conditions:
            cond = None
            if isinstance(weather_conditions, dict):
                cond = weather_conditions.get("condition")
            else:
                try:
                    cond = str(weather_conditions)
                except Exception:
                    cond = None

            if cond:
                cond = cond.lower()
                if (
                    "rain" in cond
                    or "storm" in cond
                    or "snow" in cond
                    or "sleet" in cond
                ):
                    weather_score = 0.6
                elif "wind" in cond or "cloud" in cond or "fog" in cond:
                    weather_score = 0.85
                else:
                    weather_score = 1.0

        # Compose weighted composite (higher is better)
        w = self.weights
        composite = (
            w.distance * distance_score
            + w.time * time_score
            + w.fuel * fuel_score
            + w.violations * violation_score
            + w.priority * priority_score
            + w.traffic * traffic_score
            + w.weather * weather_score
        )

        # Clamp to [0,1]
        composite = max(0.0, min(float(composite), 1.0))

        # total_score convention: lower is better -> invert composite
        total_score = 1.0 - composite
        efficiency_rating = composite * 100.0

        breakdown = {
            "distance": float(distance_score),
            "time": float(time_score),
            "fuel": float(fuel_score),
            "violations": float(violation_score),
            "priority": float(priority_score),
            "traffic": float(traffic_score),
            "weather": float(weather_score),
        }

        return RouteScore(
            total_score=float(total_score),
            efficiency_rating=float(efficiency_rating),
            distance_score=float(distance_score),
            time_score=float(time_score),
            fuel_score=float(fuel_score),
            violation_score=float(violation_score),
            priority_score=float(priority_score),
            traffic_score=float(traffic_score),
            weather_score=float(weather_score),
            breakdown=breakdown,
        )
