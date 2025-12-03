#!/usr/bin/env python3
"""
CourierIQ Demo Scenarios
Real-world use cases for user acceptance testing
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List

import requests


class CourierIQDemo:
    """Interactive demo scenarios"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def print_scenario(self, title: str, description: str):
        print("\n" + "=" * 70)
        print(f"📦 SCENARIO: {title}")
        print("=" * 70)
        print(f"📋 Description: {description}\n")

    def print_result(self, label: str, value: any):
        print(f"  ✓ {label}: {value}")

    def wait_for_input(self, message: str = "Press Enter to continue..."):
        input(f"\n{message}")

    def scenario_morning_delivery_planning(self):
        """Scenario 1: Morning delivery planning"""
        self.print_scenario(
            "Morning Delivery Planning",
            "A driver starts their day with 8 deliveries. The system optimizes the route.",
        )

        # Realistic NYC addresses (approximated as coordinates)
        request = {
            "depot": {
                "lat": 40.7128,  # Manhattan warehouse
                "lon": -74.0060,
            },
            "stops": [
                {
                    "lat": 40.7589,
                    "lon": -73.9851,
                    "priority": 1,
                    "id": "Order #1001",
                },  # Times Square area
                {
                    "lat": 40.7614,
                    "lon": -73.9776,
                    "priority": 2,
                    "id": "Order #1002",
                },  # Central Park South
                {
                    "lat": 40.7489,
                    "lon": -73.9680,
                    "priority": 1,
                    "id": "Order #1003",
                },  # East Side
                {
                    "lat": 40.7308,
                    "lon": -73.9973,
                    "priority": 3,
                    "id": "Order #1004",
                },  # West Village
                {
                    "lat": 40.7580,
                    "lon": -73.9855,
                    "priority": 1,
                    "id": "Order #1005",
                },  # Near Times Square
                {
                    "lat": 40.7410,
                    "lon": -73.9897,
                    "priority": 2,
                    "id": "Order #1006",
                },  # Chelsea
                {
                    "lat": 40.7505,
                    "lon": -73.9934,
                    "priority": 1,
                    "id": "Order #1007",
                },  # Hell's Kitchen
                {
                    "lat": 40.7282,
                    "lon": -73.9942,
                    "priority": 2,
                    "id": "Order #1008",
                },  # Greenwich Village
            ],
            "algorithm": "simulated_annealing",
        }

        print("📤 Sending optimization request...")
        start_time = time.time()

        response = requests.post(
            f"{self.base_url}/routes/optimize", json=request, timeout=30
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()

            self.print_result("Optimization Time", f"{elapsed:.2f} seconds")
            self.print_result("Total Stops", data["num_stops"])
            self.print_result("Total Distance", f"{data['total_distance_km']:.2f} km")
            self.print_result(
                "Estimated Time", f"{data['total_time_minutes']:.1f} minutes"
            )
            self.print_result("Route Score", f"{data['score']:.1f}")

            print("\n📍 Optimized Stop Order:")
            for i, stop in enumerate(data["stops"], 1):
                print(f"    {i}. {stop['id']} (Priority: {stop['priority']})")

            return data
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None

    def scenario_rush_hour_delivery(self):
        """Scenario 2: Rush hour delivery with high traffic"""
        self.print_scenario(
            "Rush Hour Delivery",
            "It's 5 PM rush hour. Calculate ETA considering heavy traffic.",
        )

        request = {
            "origin": {"lat": 40.7128, "lon": -74.0060},
            "destination": {"lat": 40.7589, "lon": -73.9851},
            "num_stops": 3,
            "traffic_level": 0.9,  # Very high traffic
        }

        print("🚦 Calculating ETA with rush hour traffic...")

        response = requests.post(
            f"{self.base_url}/routes/eta", json=request, timeout=10
        )

        if response.status_code == 200:
            data = response.json()

            self.print_result("ETA", f"{data['eta_minutes']:.1f} minutes")
            self.print_result("ETA (seconds)", f"{data['eta_seconds']:.0f} seconds")

            # Compare with low traffic
            request["traffic_level"] = 0.2
            response2 = requests.post(
                f"{self.base_url}/routes/eta", json=request, timeout=10
            )

            if response2.status_code == 200:
                data2 = response2.json()
                difference = data["eta_minutes"] - data2["eta_minutes"]
                self.print_result("Traffic Impact", f"+{difference:.1f} minutes")

            return data
        else:
            print(f"❌ Error: {response.status_code}")
            return None

    def scenario_new_order_arrives(self):
        """Scenario 3: New urgent order arrives mid-route"""
        self.print_scenario(
            "Dynamic Reoptimization",
            "Driver is mid-route when a new urgent order comes in. Route must be recalculated.",
        )

        print("📍 Current situation:")
        print("   • Driver has completed 3 deliveries")
        print("   • 4 deliveries remaining")
        print("   • 1 new urgent order just arrived")

        request = {
            "current_route": [
                {"id": "stop4", "lat": 40.7308, "lon": -73.9973, "priority": 1},
                {"id": "stop5", "lat": 40.7580, "lon": -73.9855, "priority": 1},
                {"id": "stop6", "lat": 40.7410, "lon": -73.9897, "priority": 2},
                {"id": "stop7", "lat": 40.7505, "lon": -73.9934, "priority": 1},
            ],
            "new_orders": [
                {
                    "id": "URGENT",
                    "lat": 40.7450,
                    "lon": -73.9900,
                    "priority": 5,
                }  # High priority
            ],
            "current_position": {
                "lat": 40.7489,
                "lon": -73.9680,
            },  # Just finished stop3
            "depot": {"lat": 40.7128, "lon": -74.0060},
        }

        print("\n🔄 Recalculating optimal route...")
        start_time = time.time()

        response = requests.post(
            f"{self.base_url}/routes/dynamic-reoptimize", json=request, timeout=30
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()

            self.print_result("Recalculation Time", f"{elapsed:.2f} seconds")
            self.print_result("New Stop Count", data["num_stops"])
            self.print_result("New Distance", f"{data['total_distance_km']:.2f} km")
            self.print_result("New ETA", f"{data['total_time_minutes']:.1f} minutes")

            print("\n📍 Updated Stop Order:")
            for i, stop in enumerate(data["stops"], 1):
                marker = "🔴 NEW!" if stop["id"] == "URGENT" else ""
                print(f"    {i}. {stop['id']} {marker}")

            return data
        else:
            print(f"❌ Error: {response.status_code}")
            return None

    def scenario_algorithm_comparison(self):
        """Scenario 4: Compare different optimization strategies"""
        self.print_scenario(
            "Algorithm Comparison",
            "Compare all 4 optimization algorithms to find the best route.",
        )

        request = {
            "depot": {"lat": 40.7128, "lon": -74.0060},
            "stops": [
                {"lat": 40.7589, "lon": -73.9851, "priority": 1},
                {"lat": 40.7614, "lon": -73.9776, "priority": 2},
                {"lat": 40.7489, "lon": -73.9680, "priority": 1},
                {"lat": 40.7308, "lon": -73.9973, "priority": 3},
                {"lat": 40.7580, "lon": -73.9855, "priority": 1},
            ],
        }

        print("🔬 Testing all algorithms...")

        response = requests.post(
            f"{self.base_url}/routes/compare-algorithms", json=request, timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            comparison = data.get("comparison", [])

            print("\n📊 Algorithm Performance:")
            print(f"{'Algorithm':<25} {'Distance':<12} {'Time':<12} {'Score':<10}")
            print("-" * 60)

            for result in comparison:
                algo = result.get("algorithm", "Unknown")
                distance = f"{result.get('total_distance_km', 0):.2f} km"
                time_str = f"{result.get('total_time_minutes', 0):.1f} min"
                score = f"{result.get('score', 0):.1f}"

                marker = "⭐" if algo == data.get("best_algorithm") else "  "
                print(f"{marker} {algo:<23} {distance:<12} {time_str:<12} {score:<10}")

            self.print_result("\nBest Algorithm", data.get("best_algorithm"))

            return data
        else:
            print(f"❌ Error: {response.status_code}")
            return None

    def scenario_weekly_heatmap_analysis(self):
        """Scenario 5: Analyze delivery patterns from past week"""
        self.print_scenario(
            "Weekly Heatmap Analysis",
            "Analyze 100+ deliveries from the past week to identify hot zones.",
        )

        print("📈 Generating synthetic delivery data...")

        # Generate realistic delivery data
        deliveries = []
        base_locations = [
            (40.7589, -73.9851),  # Times Square
            (40.7614, -73.9776),  # Central Park
            (40.7489, -73.9680),  # East Side
            (40.7308, -73.9973),  # West Village
            (40.7580, -73.9855),  # Midtown
        ]

        for day in range(7):
            for i in range(15):  # 15 deliveries per day
                base_lat, base_lon = random.choice(base_locations)
                deliveries.append(
                    {
                        "lat": base_lat + random.uniform(-0.01, 0.01),
                        "lon": base_lon + random.uniform(-0.01, 0.01),
                        "delivery_time": random.uniform(600, 1800),
                        "distance": random.uniform(1000, 5000),
                        "status": "success" if random.random() > 0.1 else "failed",
                        "hour": 9 + i % 10,
                    }
                )

        request = {"deliveries": deliveries, "grid_size_km": 0.5}

        print(f"🗺️  Processing {len(deliveries)} deliveries...")

        response = requests.post(
            f"{self.base_url}/routes/heatmap", json=request, timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            heatmap = data.get("heatmap", [])
            zones = data.get("high_efficiency_zones", [])

            self.print_result("Total Cells", len(heatmap))
            self.print_result("High-Efficiency Zones", len(zones))
            self.print_result("Total Deliveries Analyzed", data.get("total_deliveries"))

            print("\n🎯 Top 3 High-Efficiency Zones:")
            for i, zone in enumerate(zones[:3], 1):
                center = zone.get("center", {})
                metrics = zone.get("metrics", {})
                print(f"\n   Zone {i}:")
                print(
                    f"     Location: ({center.get('lat', 0):.4f}, {center.get('lon', 0):.4f})"
                )
                print(f"     Deliveries: {metrics.get('num_deliveries', 0)}")
                print(f"     Success Rate: {metrics.get('success_rate', 0):.1%}")
                print(f"     Efficiency: {metrics.get('efficiency_score', 0):.3f}")

            return data
        else:
            print(f"❌ Error: {response.status_code}")
            return None

    def scenario_route_scoring(self):
        """Scenario 6: Score and compare two routes"""
        self.print_scenario(
            "Route Quality Scoring", "Evaluate route quality based on multiple factors."
        )

        routes = [
            {
                "name": "Route A (Short but slow)",
                "params": {
                    "distance_km": 8.5,
                    "duration_hours": 0.75,
                    "num_stops": 5,
                    "time_of_day": 10,
                    "traffic_level": 0.3,
                },
            },
            {
                "name": "Route B (Long but fast)",
                "params": {
                    "distance_km": 12.0,
                    "duration_hours": 0.55,
                    "num_stops": 5,
                    "time_of_day": 10,
                    "traffic_level": 0.3,
                },
            },
        ]

        print("⚖️  Comparing two route options...\n")

        for route in routes:
            print(f"📊 {route['name']}:")

            response = requests.get(
                f"{self.base_url}/routes/score", params=route["params"], timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                self.print_result("  Total Score", f"{data.get('total_score', 0):.2f}")
                self.print_result(
                    "  Efficiency Rating", data.get("efficiency_rating", "N/A")
                )

                breakdown = data.get("breakdown", {})
                print("  Score Breakdown:")
                for metric, value in breakdown.items():
                    if value > 0:
                        print(f"    • {metric}: {value:.2f}")

                route["score_data"] = data

            print()

        # Determine winner
        if all("score_data" in r for r in routes):
            scores = [(r["name"], r["score_data"]["total_score"]) for r in routes]
            winner = min(scores, key=lambda x: x[1])  # Lower score is better

            print(f"🏆 Winner: {winner[0]} (Score: {winner[1]:.2f})")

        return routes


def run_interactive_demo():
    """Run interactive demo with user"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║          CourierIQ Interactive Demo                               ║
║          Real-World Delivery Scenarios                            ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    print("This demo will walk you through real-world delivery scenarios.")
    print("Make sure the API server is running: uvicorn src.api.main:app --reload\n")

    input("Press Enter to start the demo...")

    demo = CourierIQDemo()

    scenarios = [
        ("Morning Delivery Planning", demo.scenario_morning_delivery_planning),
        ("Rush Hour Delivery", demo.scenario_rush_hour_delivery),
        ("Dynamic Reoptimization", demo.scenario_new_order_arrives),
        ("Algorithm Comparison", demo.scenario_algorithm_comparison),
        ("Weekly Heatmap Analysis", demo.scenario_weekly_heatmap_analysis),
        ("Route Quality Scoring", demo.scenario_route_scoring),
    ]

    for i, (name, scenario_func) in enumerate(scenarios, 1):
        print(f"\n{'=' * 70}")
        print(f"Scenario {i}/{len(scenarios)}")
        print(f"{'=' * 70}")

        try:
            scenario_func()
            demo.wait_for_input()
        except requests.exceptions.ConnectionError:
            print("\n❌ Error: Cannot connect to API server")
            print("Make sure the server is running:")
            print("   uvicorn src.api.main:app --reload")
            break
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error in scenario: {e}")
            demo.wait_for_input("Press Enter to continue to next scenario...")

    print("""
╔════════════════════════════════════════════════════════════════════╗
║                    Demo Complete!                                  ║
║                                                                    ║
║  CourierIQ is ready for production use!                          ║
║                                                                    ║
║  Next steps:                                                      ║
║    • Run production tests: python test_production.py             ║
║    • Check API docs: http://localhost:8000/docs                  ║
║    • Start building your delivery app!                           ║
╚════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    run_interactive_demo()
