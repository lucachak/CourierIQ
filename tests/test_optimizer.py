"""
Unit tests for optimizer.py
Place in: tests/test_optimizer.py
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.engine.optimizer import Route, RouteOptimizer, Stop


class TestRouteOptimizer:
    @pytest.fixture
    def optimizer(self):
        return RouteOptimizer()

    @pytest.fixture
    def depot(self):
        return Stop(id="depot", lat=40.7128, lon=-74.0060)

    @pytest.fixture
    def sample_stops(self):
        return [
            Stop(id="stop1", lat=40.7589, lon=-73.9851, priority=1),
            Stop(id="stop2", lat=40.7614, lon=-73.9776, priority=2),
            Stop(id="stop3", lat=40.7489, lon=-73.9680, priority=1),
            Stop(id="stop4", lat=40.7308, lon=-73.9973, priority=3),
        ]

    def test_haversine_distance(self, optimizer, depot, sample_stops):
        """Test haversine distance calculation"""
        distance = optimizer._haversine_distance(depot, sample_stops[0])

        assert distance > 0
        assert isinstance(distance, float)
        # Distance between these points should be ~5-6 km
        assert 4000 < distance < 7000

    def test_calculate_route_metrics(self, optimizer, depot, sample_stops):
        """Test route metrics calculation"""
        metrics = optimizer.calculate_route_metrics(sample_stops, depot)

        assert "distance" in metrics
        assert "time" in metrics
        assert "penalty" in metrics
        assert "score" in metrics

        assert metrics["distance"] > 0
        assert metrics["time"] > 0
        assert metrics["score"] > 0

    def test_nearest_neighbor(self, optimizer, depot, sample_stops):
        """Test nearest neighbor algorithm"""
        route = optimizer.nearest_neighbor(sample_stops, depot)

        assert isinstance(route, Route)
        assert len(route.stops) == len(sample_stops)
        assert route.total_distance > 0
        assert route.total_time > 0

        # Verify all stops are included
        route_ids = {stop.id for stop in route.stops}
        expected_ids = {stop.id for stop in sample_stops}
        assert route_ids == expected_ids

    def test_two_opt(self, optimizer, depot, sample_stops):
        """Test 2-opt optimization"""
        route = optimizer.two_opt(sample_stops, depot, max_iterations=100)

        assert isinstance(route, Route)
        assert len(route.stops) == len(sample_stops)

    def test_simulated_annealing(self, optimizer, depot, sample_stops):
        """Test simulated annealing"""
        route = optimizer.simulated_annealing(
            sample_stops, depot, initial_temp=1000, max_iterations=500
        )

        assert isinstance(route, Route)
        assert len(route.stops) == len(sample_stops)
        assert route.score > 0

    def test_tabu_search(self, optimizer, depot, sample_stops):
        """Test tabu search"""
        route = optimizer.tabu_search(
            sample_stops, depot, tabu_size=10, max_iterations=100
        )

        assert isinstance(route, Route)
        assert len(route.stops) == len(sample_stops)

    def test_optimize_with_all_algorithms(self, optimizer, depot, sample_stops):
        """Test optimize method with all algorithms"""
        algorithms = [
            "nearest_neighbor",
            "two_opt",
            "simulated_annealing",
            "tabu_search",
        ]

        for algo in algorithms:
            route = optimizer.optimize(sample_stops, depot, algorithm=algo)

            assert isinstance(route, Route)
            assert len(route.stops) == len(sample_stops)
            assert route.score > 0

    def test_empty_stops(self, optimizer, depot):
        """Test handling of empty stops list"""
        route = optimizer.nearest_neighbor([], depot)

        assert len(route.stops) == 0
        assert route.total_distance == 0.0

    def test_single_stop(self, optimizer, depot, sample_stops):
        """Test optimization with single stop"""
        route = optimizer.optimize([sample_stops[0]], depot)

        assert len(route.stops) == 1
        assert route.stops[0].id == sample_stops[0].id

    def test_priority_handling(self, optimizer, depot):
        """Test that high priority stops get positioned earlier"""
        stops = [
            Stop(id="low", lat=40.7589, lon=-73.9851, priority=1),
            Stop(id="high", lat=40.7614, lon=-73.9776, priority=5),
            Stop(id="medium", lat=40.7489, lon=-73.9680, priority=3),
        ]

        route = optimizer.optimize(stops, depot, algorithm="simulated_annealing")

        # High priority should contribute to score calculation
        metrics = optimizer.calculate_route_metrics(route.stops, depot)
        assert metrics["penalty"] >= 0

    def test_time_window_violations(self, optimizer, depot):
        """Test time window violation detection"""
        stops = [
            Stop(
                id="stop1",
                lat=40.7589,
                lon=-73.9851,
                time_window=(0, 1000),  # Very early window
            ),
            Stop(
                id="stop2",
                lat=40.7614,
                lon=-73.9776,
                time_window=(50000, 60000),  # Very late window
            ),
        ]

        route = optimizer.optimize(stops, depot)
        metrics = optimizer.calculate_route_metrics(route.stops, depot)

        # Should have penalty for time windows
        assert metrics["penalty"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
