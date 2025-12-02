"""
engine_service.py - High-level service that integrates all engine components
Place this in: src/api/services/engine_service.py
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.engine.heatmap import DeliveryZone, HeatmapGenerator
from src.engine.optimizer import Route, RouteOptimizer, Stop
from src.engine.routing import RoutingService
from src.engine.scorer import RouteScore, RouteScorer, ScoringWeights

logger = logging.getLogger(__name__)


class EngineService:
    """
    High-level service that orchestrates all routing engine components
    """

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        osrm_url: str = "http://router.project-osrm.org",
        default_algorithm: str = "simulated_annealing",
    ):
        """
        Initialize the engine service with all components

        Args:
            google_api_key: Google Maps API key (optional)
            osrm_url: OSRM server URL for routing
            default_algorithm: Default optimization algorithm
        """
        self.optimizer = RouteOptimizer()
        # RoutingService constructor does not accept `default_provider`; remove that kwarg.
        self.routing_service = RoutingService(
            google_api_key=google_api_key,
            osrm_url=osrm_url,
        )
        self.scorer = RouteScorer()
        self.heatmap = HeatmapGenerator(grid_size_km=1.0)
        self.default_algorithm = default_algorithm

        logger.info("EngineService initialized")

    def optimize_route_simple(
        self,
        depot_lat: float,
        depot_lon: float,
        stops: List[Dict],
        algorithm: Optional[str] = None,
    ) -> Dict:
        """
        Simple route optimization without external routing APIs
        Uses haversine distance only

        Args:
            depot_lat, depot_lon: Depot coordinates
            stops: List of dicts with 'lat', 'lon', 'priority', etc.
            algorithm: Optimization algorithm to use

        Returns:
            Optimized route with metrics
        """
        # Convert to Stop objects
        depot = Stop(id="depot", lat=depot_lat, lon=depot_lon)
        stop_objects = []

        for idx, stop in enumerate(stops):
            stop_obj = Stop(
                id=stop.get("id", f"stop_{idx}"),
                lat=stop["lat"],
                lon=stop["lon"],
                priority=stop.get("priority", 1),
                time_window=stop.get("time_window"),
                service_time=stop.get("service_time", 300),
            )
            stop_objects.append(stop_obj)

        # Optimize
        algo = algorithm or self.default_algorithm
        route = self.optimizer.optimize(stop_objects, depot, algorithm=algo)

        # Convert back to dict format
        return {
            "success": True,
            "algorithm": algo,
            "depot": {"lat": depot_lat, "lon": depot_lon},
            "stops": [
                {
                    "id": stop.id,
                    "lat": stop.lat,
                    "lon": stop.lon,
                    "priority": stop.priority,
                }
                for stop in route.stops
            ],
            "total_distance_km": route.total_distance / 1000,
            "total_time_hours": route.total_time / 3600,
            "score": route.score,
            "num_stops": len(route.stops),
        }

    def optimize_route_with_apis(
        self,
        depot_lat: float,
        depot_lon: float,
        stops: List[Dict],
        provider: Optional[str] = None,
    ) -> Dict:
        """
        Optimize route using external routing APIs
        Gets real distance and time estimates

        Args:
            depot_lat, depot_lon: Depot coordinates
            stops: List of stop dictionaries
            provider: 'google' or 'osrm'

        Returns:
            Optimized route with real API data
        """
        depot_coords = (depot_lat, depot_lon)
        stop_coords = [(s["lat"], s["lon"]) for s in stops]

        # Get distance matrix
        distance_matrix_result = self.routing_service.get_distance_matrix(
            origins=[depot_coords] + stop_coords,
            destinations=[depot_coords] + stop_coords,
            provider=provider,
        )

        if not distance_matrix_result:
            logger.error("Failed to get distance matrix")
            return {"success": False, "error": "Failed to get distance matrix"}

        # Create custom distance function that uses the API data
        distance_matrix = distance_matrix_result["distance_matrix"]

        def api_distance_func(stop1: Stop, stop2: Stop) -> float:
            # Map stops to matrix indices
            all_stops = [Stop(id="depot", lat=depot_lat, lon=depot_lon)] + [
                Stop(id=s.get("id", f"stop_{i}"), lat=s["lat"], lon=s["lon"])
                for i, s in enumerate(stops)
            ]

            idx1 = next((i for i, s in enumerate(all_stops) if s.id == stop1.id), 0)
            idx2 = next((i for i, s in enumerate(all_stops) if s.id == stop2.id), 0)

            return distance_matrix[idx1][idx2]

        # Run optimization with API distance function
        depot = Stop(id="depot", lat=depot_lat, lon=depot_lon)
        stop_objects = [
            Stop(
                id=s.get("id", f"stop_{idx}"),
                lat=s["lat"],
                lon=s["lon"],
                priority=s.get("priority", 1),
                time_window=s.get("time_window"),
                service_time=s.get("service_time", 300),
            )
            for idx, s in enumerate(stops)
        ]

        # Temporarily replace distance function
        original_func = self.optimizer.distance_func
        self.optimizer.distance_func = api_distance_func

        route = self.optimizer.optimize(
            stop_objects, depot, algorithm=self.default_algorithm
        )

        # Restore original function
        self.optimizer.distance_func = original_func

        return {
            "success": True,
            "provider": provider or "default",
            "depot": {"lat": depot_lat, "lon": depot_lon},
            "stops": [
                {
                    "id": stop.id,
                    "lat": stop.lat,
                    "lon": stop.lon,
                    "priority": stop.priority,
                }
                for stop in route.stops
            ],
            "total_distance_km": route.total_distance / 1000,
            "total_time_hours": route.total_time / 3600,
            "score": route.score,
            "num_stops": len(route.stops),
        }

    def score_route(
        self,
        route_data: Dict,
        stops: List[Dict],
        current_time: float = 0.0,
        time_of_day: Optional[int] = None,
        weather_conditions: Optional[Dict] = None,
    ) -> Dict:
        """
        Score an existing route

        Returns:
            Route score breakdown
        """
        score = self.scorer.score_route(
            route_data, stops, current_time, time_of_day, weather_conditions
        )

        return {
            "total_score": score.total_score,
            "efficiency_rating": score.efficiency_rating,
            "breakdown": {
                "distance": score.distance_score,
                "time": score.time_score,
                "fuel": score.fuel_score,
                "violations": score.violation_score,
                "priority": score.priority_score,
                "traffic": score.traffic_score,
                "weather": score.weather_score,
            },
            "metrics": score.breakdown,
        }

    def compare_algorithms(
        self, depot_lat: float, depot_lon: float, stops: List[Dict]
    ) -> Dict:
        """
        Compare different optimization algorithms

        Returns:
            Results from all algorithms with rankings
        """
        algorithms = [
            "nearest_neighbor",
            "two_opt",
            "simulated_annealing",
            "tabu_search",
        ]
        results = []

        for algo in algorithms:
            try:
                result = self.optimize_route_simple(
                    depot_lat, depot_lon, stops, algorithm=algo
                )
                result["algorithm"] = algo
                results.append(result)
            except Exception as e:
                logger.error(f"Algorithm {algo} failed: {e}")

        # Sort by score (lower is better)
        results.sort(key=lambda x: x["score"])

        return {
            "comparison": results,
            "best_algorithm": results[0]["algorithm"] if results else None,
            "worst_algorithm": results[-1]["algorithm"] if results else None,
        }

    def generate_delivery_heatmap(self, deliveries: List[Dict]) -> Dict:
        """
        Generate heatmap from historical delivery data

        Args:
            deliveries: List of delivery dicts with 'lat', 'lon', etc.

        Returns:
            Heatmap data and high-efficiency zones
        """
        heatmap = self.heatmap.generate_delivery_heatmap(deliveries)
        zones = self.heatmap.identify_high_efficiency_zones(heatmap, top_n=10)

        # Convert to JSON-serializable format
        heatmap_list = [
            {
                "cell": {"lat": k[0], "lon": k[1]},
                "center": self.heatmap._get_cell_center(k),
                "count": v["count"],
                "avg_time": v["avg_time"],
                "success_rate": v["success_rate"],
                "density_score": v["density_score"],
                "efficiency_score": v["efficiency_score"],
            }
            for k, v in heatmap.items()
        ]

        zones_list = [
            {
                "center": {"lat": z.center[0], "lon": z.center[1]},
                "radius_km": z.radius / 1000,
                "num_deliveries": z.num_deliveries,
                "avg_delivery_time": z.avg_delivery_time,
                "success_rate": z.success_rate,
                "efficiency_score": z.efficiency_score,
            }
            for z in zones
        ]

        return {
            "heatmap": heatmap_list,
            "high_efficiency_zones": zones_list,
            "total_cells": len(heatmap),
            "total_deliveries": sum(v["count"] for v in heatmap.values()),
        }

    def dynamic_reoptimize(
        self,
        current_route: List[Dict],
        new_orders: List[Dict],
        current_position: Dict,
        depot: Dict,
    ) -> Dict:
        """
        Dynamically reoptimize when new orders arrive

        Args:
            current_route: Current planned route (remaining stops)
            new_orders: New orders to insert
            current_position: Current driver position {'lat', 'lon'}
            depot: Depot coordinates

        Returns:
            New optimized route
        """
        # Combine remaining stops with new orders
        all_stops = current_route + new_orders

        # Mark new orders with higher priority for quick insertion
        for order in new_orders:
            order["priority"] = order.get("priority", 1) + 2  # Boost priority

        # Optimize from current position
        result = self.optimize_route_simple(
            depot_lat=current_position["lat"],
            depot_lon=current_position["lon"],
            stops=all_stops,
            algorithm="simulated_annealing",  # Fast and good quality
        )

        result["recalculated"] = True
        result["new_orders_count"] = len(new_orders)

        return result
