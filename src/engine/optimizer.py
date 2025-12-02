"""
optimizer.py - Multi-stop route optimization algorithms
Implements A*, Simulated Annealing, and Tabu Search for delivery route optimization
"""

import heapq
import logging
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Stop:
    """Represents a delivery stop"""

    id: str
    lat: float
    lon: float
    priority: int = 1  # Higher priority = must deliver sooner
    time_window: Optional[Tuple[float, float]] = (
        None  # (earliest, latest) delivery time
    )
    service_time: float = 300  # seconds to complete delivery

    def __hash__(self):
        return hash(self.id)


@dataclass
class Route:
    """Represents a complete delivery route"""

    stops: List[Stop]
    total_distance: float = 0.0
    total_time: float = 0.0
    score: float = 0.0  # Lower is better

    def __lt__(self, other):
        return self.score < other.score


class RouteOptimizer:
    """Main optimization engine with multiple algorithms"""

    def __init__(self, distance_func: Callable = None):
        """
        Args:
            distance_func: Function to calculate distance between two stops
                          Should accept (stop1, stop2) and return distance in meters
        """
        self.distance_func = distance_func or self._haversine_distance

    def _haversine_distance(self, stop1: Stop, stop2: Stop) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371000  # Earth radius in meters

        lat1, lon1 = np.radians(stop1.lat), np.radians(stop1.lon)
        lat2, lon2 = np.radians(stop2.lat), np.radians(stop2.lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def calculate_route_metrics(
        self, route: List[Stop], depot: Stop
    ) -> Dict[str, float]:
        """Calculate total distance, time, and penalties for a route"""
        if not route:
            return {
                "total_distance": 0.0,
                "total_time": 0.0,
                "penalty": 0.0,
                "score": 0.0,
            }

        total_distance = 0.0
        total_time = 0.0
        penalty = 0.0

        # From depot to first stop
        current = depot
        for stop in route:
            dist = self.distance_func(current, stop)
            total_distance += dist

            # Estimate time (assuming 30 km/h average speed + service time)
            travel_time = (dist / 1000) / 30 * 3600  # seconds
            total_time += travel_time + stop.service_time

            # Time window penalty
            if stop.time_window:
                earliest, latest = stop.time_window
                if total_time < earliest:
                    penalty += (earliest - total_time) * 0.1  # Wait penalty
                elif total_time > latest:
                    penalty += (total_time - latest) * 2.0  # Late penalty (worse)

            current = stop

        # Return to depot
        dist = self.distance_func(current, depot)
        total_distance += dist
        total_time += (dist / 1000) / 30 * 3600

        # Priority penalty (high priority stops should be early)
        for idx, stop in enumerate(route):
            penalty += stop.priority * idx * 10  # Penalty increases with position

        score = total_distance + penalty * 1000  # Weight penalty heavily

        return {
            "distance": total_distance,
            "time": total_time,
            "total_distance": total_distance,
            "total_time": total_time,
            "penalty": penalty,
            "score": score,
        }

    def nearest_neighbor(self, stops: List[Stop], depot: Stop) -> Route:
        """
        Greedy nearest neighbor algorithm - fast but not optimal
        Good for initial solution
        """
        if not stops:
            return Route(stops=[], total_distance=0.0, total_time=0.0, score=0.0)

        route = []
        remaining = set(stops)
        current = depot

        while remaining:
            # Find nearest unvisited stop
            nearest = min(remaining, key=lambda s: self.distance_func(current, s))
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest

        metrics = self.calculate_route_metrics(route, depot)
        return Route(
            stops=route,
            total_distance=metrics["total_distance"],
            total_time=metrics["total_time"],
            score=metrics["score"],
        )

    def two_opt(
        self, route: List[Stop], depot: Stop, max_iterations: int = 1000
    ) -> Route:
        """
        2-opt local search optimization
        Iteratively removes crossing edges to improve route
        """
        if len(route) < 4:
            metrics = self.calculate_route_metrics(route, depot)
            return Route(
                stops=route,
                total_distance=metrics["total_distance"],
                total_time=metrics["total_time"],
                score=metrics["score"],
            )

        best_route = route.copy()
        best_score = self.calculate_route_metrics(best_route, depot)["score"]
        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route)):
                    # Try reversing segment [i:j]
                    new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                    new_score = self.calculate_route_metrics(new_route, depot)["score"]

                    if new_score < best_score:
                        best_route = new_route
                        best_score = new_score
                        improved = True
                        break

                if improved:
                    break

        metrics = self.calculate_route_metrics(best_route, depot)
        return Route(
            stops=best_route,
            total_distance=metrics["total_distance"],
            total_time=metrics["total_time"],
            score=metrics["score"],
        )

    def simulated_annealing(
        self,
        stops: List[Stop],
        depot: Stop,
        initial_temp: float = 10000,
        cooling_rate: float = 0.995,
        min_temp: float = 1,
        max_iterations: int = 10000,
    ) -> Route:
        """
        Simulated Annealing optimization
        Probabilistically accepts worse solutions to escape local optima
        """
        if not stops:
            return Route(stops=[], total_distance=0.0, total_time=0.0, score=0.0)

        # If there's only one stop, sampling/swaps below will fail.
        # Return early with computed metrics for a single-stop route.
        if len(stops) < 2:
            metrics = self.calculate_route_metrics(stops, depot)
            return Route(
                stops=stops,
                total_distance=metrics.get("total_distance", 0.0),
                total_time=metrics.get("total_time", 0.0),
                score=metrics.get("score", 0.0),
            )

        # Start with nearest neighbor solution
        current_route = self.nearest_neighbor(stops, depot).stops
        current_score = self.calculate_route_metrics(current_route, depot)["score"]

        best_route = current_route.copy()
        best_score = current_score

        temperature = initial_temp
        iterations = 0

        while temperature > min_temp and iterations < max_iterations:
            iterations += 1

            # Generate neighbor solution using random swap
            new_route = current_route.copy()
            i, j = random.sample(range(len(new_route)), 2)
            new_route[i], new_route[j] = new_route[j], new_route[i]

            new_score = self.calculate_route_metrics(new_route, depot)["score"]
            delta = new_score - current_score

            # Accept if better, or with probability based on temperature
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_route = new_route
                current_score = new_score

                if current_score < best_score:
                    best_route = current_route
                    best_score = current_score

            temperature *= cooling_rate

        logger.info(f"Simulated Annealing completed in {iterations} iterations")
        metrics = self.calculate_route_metrics(best_route, depot)
        return Route(
            stops=best_route,
            total_distance=metrics["total_distance"],
            total_time=metrics["total_time"],
            score=metrics["score"],
        )

    def tabu_search(
        self,
        stops: List[Stop],
        depot: Stop,
        tabu_size: int = 20,
        max_iterations: int = 1000,
        max_no_improve: int = 100,
    ) -> Route:
        """
        Tabu Search optimization
        Maintains a memory of recent moves to avoid cycling
        """
        if not stops:
            return Route(stops=[], total_distance=0.0, total_time=0.0, score=0.0)

        # Start with nearest neighbor
        current_route = self.nearest_neighbor(stops, depot).stops
        current_score = self.calculate_route_metrics(current_route, depot)["score"]

        best_route = current_route.copy()
        best_score = current_score

        tabu_list = []
        iterations = 0
        no_improve_count = 0

        while iterations < max_iterations and no_improve_count < max_no_improve:
            iterations += 1

            # Generate all possible swaps
            neighbors = []
            for i in range(len(current_route)):
                for j in range(i + 1, len(current_route)):
                    if (i, j) not in tabu_list:
                        new_route = current_route.copy()
                        new_route[i], new_route[j] = new_route[j], new_route[i]
                        score = self.calculate_route_metrics(new_route, depot)["score"]
                        neighbors.append((new_route, score, (i, j)))

            if not neighbors:
                break

            # Select best non-tabu neighbor
            neighbors.sort(key=lambda x: x[1])
            current_route, current_score, move = neighbors[0]

            # Update tabu list
            tabu_list.append(move)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

            # Update best if improved
            if current_score < best_score:
                best_route = current_route.copy()
                best_score = current_score
                no_improve_count = 0
            else:
                no_improve_count += 1

        logger.info(f"Tabu Search completed in {iterations} iterations")
        metrics = self.calculate_route_metrics(best_route, depot)
        return Route(
            stops=best_route,
            total_distance=metrics["total_distance"],
            total_time=metrics["total_time"],
            score=metrics["score"],
        )

    def optimize(
        self, stops: List[Stop], depot: Stop, algorithm: str = "simulated_annealing"
    ) -> Route:
        """
        Main optimization method - selects algorithm based on input

        Args:
            stops: List of delivery stops to optimize
            depot: Starting/ending depot location
            algorithm: One of ['nearest_neighbor', 'two_opt', 'simulated_annealing', 'tabu_search']

        Returns:
            Optimized Route object
        """
        if algorithm == "nearest_neighbor":
            route = self.nearest_neighbor(stops, depot)
        elif algorithm == "two_opt":
            nn_route = self.nearest_neighbor(stops, depot)
            route = self.two_opt(nn_route.stops, depot)
        elif algorithm == "simulated_annealing":
            route = self.simulated_annealing(stops, depot)
        elif algorithm == "tabu_search":
            route = self.tabu_search(stops, depot)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        logger.info(
            f"Optimization complete: {len(stops)} stops, "
            f"distance: {route.total_distance / 1000:.2f}km, "
            f"time: {route.total_time / 60:.1f}min"
        )

        return route
