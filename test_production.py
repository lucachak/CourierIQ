#!/usr/bin/env python3
"""
CourierIQ Production Testing Suite
Complete end-to-end tests simulating real-world usage
"""

import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests


# Colors for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class ProductionTester:
    """Complete production-level testing suite"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {"passed": [], "failed": [], "warnings": []}
        self.server_process = None

    def print_header(self, text: str):
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}")
        print(f"  {text}")
        print(f"{'=' * 70}{Colors.RESET}\n")

    def print_test(self, name: str):
        print(f"{Colors.BLUE}▶ {name}{Colors.RESET}")

    def print_success(self, text: str):
        print(f"  {Colors.GREEN}✓ {text}{Colors.RESET}")

    def print_error(self, text: str):
        print(f"  {Colors.RED}✗ {text}{Colors.RESET}")

    def print_warning(self, text: str):
        print(f"  {Colors.YELLOW}⚠ {text}{Colors.RESET}")

    def start_server(self) -> bool:
        """Start the API server in background"""
        self.print_header("Starting API Server")

        try:
            # Check if server is already running
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    self.print_warning("Server already running")
                    return True
            except:
                pass

            # Start server
            print("Starting uvicorn server...")
            self.server_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "src.api.main:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to start
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get(f"{self.base_url}/health-check", timeout=2)
                    if response.status_code == 200:
                        self.print_success("Server started successfully")
                        return True
                except:
                    time.sleep(1)

            self.print_error("Server failed to start")
            return False

        except Exception as e:
            self.print_error(f"Failed to start server: {e}")
            return False

    def stop_server(self):
        """Stop the API server"""
        if self.server_process:
            self.print_header("Stopping API Server")
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
            self.print_success("Server stopped")

    def test_health_check(self) -> bool:
        """Test basic health endpoint"""
        self.print_test("Health Check")

        try:
            response = requests.get(f"{self.base_url}/health-check", timeout=5)

            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Status: {data.get('status')}")
                self.print_success(f"Version: {data.get('version')}")
                self.results["passed"].append("Health check")
                return True
            else:
                self.print_error(f"Status code: {response.status_code}")
                self.results["failed"].append("Health check")
                return False

        except Exception as e:
            self.print_error(f"Health check failed: {e}")
            self.results["failed"].append("Health check")
            return False

    def test_route_optimization(self) -> bool:
        """Test route optimization with realistic data"""
        self.print_test("Route Optimization - Multiple Scenarios")

        scenarios = [
            {
                "name": "Small delivery (3 stops)",
                "data": {
                    "depot": {"lat": 40.7128, "lon": -74.0060},
                    "stops": [
                        {"lat": 40.7589, "lon": -73.9851, "priority": 1},
                        {"lat": 40.7614, "lon": -73.9776, "priority": 2},
                        {"lat": 40.7489, "lon": -73.9680, "priority": 1},
                    ],
                    "algorithm": "simulated_annealing",
                },
            },
            {
                "name": "Medium delivery (7 stops)",
                "data": {
                    "depot": {"lat": 40.7128, "lon": -74.0060},
                    "stops": [
                        {
                            "lat": 40.7589 + i * 0.005,
                            "lon": -73.9851 + i * 0.005,
                            "priority": i % 3 + 1,
                        }
                        for i in range(7)
                    ],
                    "algorithm": "simulated_annealing",
                },
            },
            {
                "name": "Nearest neighbor (fast)",
                "data": {
                    "depot": {"lat": 40.7128, "lon": -74.0060},
                    "stops": [
                        {"lat": 40.7589, "lon": -73.9851, "priority": 1},
                        {"lat": 40.7614, "lon": -73.9776, "priority": 2},
                    ],
                    "algorithm": "nearest_neighbor",
                },
            },
        ]

        all_passed = True

        for scenario in scenarios:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/routes/optimize",
                    json=scenario["data"],
                    timeout=30,
                )
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    self.print_success(
                        f"{scenario['name']}: {data.get('num_stops')} stops, "
                        f"{data.get('total_distance_km', 0):.2f}km, "
                        f"{data.get('total_time_minutes', 0):.1f}min "
                        f"({elapsed:.2f}s)"
                    )

                    # Validate response structure
                    required_fields = ["success", "depot", "stops", "total_distance_km"]
                    missing = [f for f in required_fields if f not in data]
                    if missing:
                        self.print_warning(f"Missing fields: {missing}")

                    self.results["passed"].append(f"Optimization: {scenario['name']}")
                else:
                    self.print_error(
                        f"{scenario['name']}: Status {response.status_code}"
                    )
                    self.print_error(f"Response: {response.text[:200]}")
                    all_passed = False
                    self.results["failed"].append(f"Optimization: {scenario['name']}")

            except Exception as e:
                self.print_error(f"{scenario['name']}: {e}")
                all_passed = False
                self.results["failed"].append(f"Optimization: {scenario['name']}")

        return all_passed

    def test_algorithm_comparison(self) -> bool:
        """Test algorithm comparison endpoint"""
        self.print_test("Algorithm Comparison")

        try:
            data = {
                "depot": {"lat": 40.7128, "lon": -74.0060},
                "stops": [
                    {"lat": 40.7589, "lon": -73.9851, "priority": 1},
                    {"lat": 40.7614, "lon": -73.9776, "priority": 2},
                    {"lat": 40.7489, "lon": -73.9680, "priority": 1},
                ],
            }

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/routes/compare-algorithms", json=data, timeout=60
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                comparison = result.get("comparison", [])
                best = result.get("best_algorithm")

                self.print_success(
                    f"Compared {len(comparison)} algorithms in {elapsed:.2f}s"
                )
                self.print_success(f"Best algorithm: {best}")

                # Show results
                for algo_result in comparison[:3]:  # Top 3
                    self.print_success(
                        f"  {algo_result.get('algorithm')}: "
                        f"{algo_result.get('total_distance_km', 0):.2f}km, "
                        f"score: {algo_result.get('score', 0):.1f}"
                    )

                self.results["passed"].append("Algorithm comparison")
                return True
            else:
                self.print_error(f"Status: {response.status_code}")
                self.results["failed"].append("Algorithm comparison")
                return False

        except Exception as e:
            self.print_error(f"Failed: {e}")
            self.results["failed"].append("Algorithm comparison")
            return False

    def test_eta_prediction(self) -> bool:
        """Test ETA prediction"""
        self.print_test("ETA Prediction")

        test_cases = [
            {
                "name": "Short distance",
                "data": {
                    "origin": {"lat": 40.7128, "lon": -74.0060},
                    "destination": {"lat": 40.7589, "lon": -73.9851},
                    "num_stops": 1,
                    "traffic_level": 0.3,
                },
            },
            {
                "name": "High traffic",
                "data": {
                    "origin": {"lat": 40.7128, "lon": -74.0060},
                    "destination": {"lat": 40.7589, "lon": -73.9851},
                    "num_stops": 3,
                    "traffic_level": 0.9,
                },
            },
        ]

        all_passed = True

        for case in test_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/routes/eta", json=case["data"], timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    eta_minutes = data.get("eta_minutes", 0)
                    self.print_success(f"{case['name']}: {eta_minutes:.1f} minutes")

                    # Sanity check - ETA should be reasonable
                    if eta_minutes < 1 or eta_minutes > 180:
                        self.print_warning(
                            f"ETA seems unrealistic: {eta_minutes:.1f}min"
                        )

                    self.results["passed"].append(f"ETA: {case['name']}")
                else:
                    self.print_error(f"{case['name']}: Status {response.status_code}")
                    all_passed = False
                    self.results["failed"].append(f"ETA: {case['name']}")

            except Exception as e:
                self.print_error(f"{case['name']}: {e}")
                all_passed = False
                self.results["failed"].append(f"ETA: {case['name']}")

        return all_passed

    def test_heatmap_generation(self) -> bool:
        """Test heatmap generation"""
        self.print_test("Heatmap Generation")

        try:
            # Create realistic delivery data
            deliveries = [
                {
                    "lat": 40.7128 + i * 0.01,
                    "lon": -74.0060 + i * 0.01,
                    "delivery_time": 600 + i * 100,
                    "distance": 1000 + i * 500,
                    "status": "success" if i % 5 != 0 else "failed",
                    "hour": 9 + (i % 12),
                }
                for i in range(30)
            ]

            data = {"deliveries": deliveries, "grid_size_km": 1.0}

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/routes/heatmap", json=data, timeout=30
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                heatmap = result.get("heatmap", [])
                zones = result.get("high_efficiency_zones", [])

                self.print_success(
                    f"Heatmap generated: {len(heatmap)} cells ({elapsed:.2f}s)"
                )
                self.print_success(f"Identified {len(zones)} high-efficiency zones")

                if zones:
                    top_zone = zones[0]
                    self.print_success(
                        f"Top zone: {top_zone.get('num_deliveries')} deliveries, "
                        f"efficiency: {top_zone.get('metrics', {}).get('efficiency_score', 0):.3f}"
                    )

                self.results["passed"].append("Heatmap generation")
                return True
            else:
                self.print_error(f"Status: {response.status_code}")
                self.results["failed"].append("Heatmap generation")
                return False

        except Exception as e:
            self.print_error(f"Failed: {e}")
            self.results["failed"].append("Heatmap generation")
            return False

    def test_dynamic_reoptimization(self) -> bool:
        """Test dynamic route reoptimization"""
        self.print_test("Dynamic Reoptimization")

        try:
            # Simulate: driver is mid-route, new orders arrive
            data = {
                "current_route": [
                    {"id": "stop2", "lat": 40.7614, "lon": -73.9776, "priority": 1},
                    {"id": "stop3", "lat": 40.7489, "lon": -73.9680, "priority": 1},
                ],
                "new_orders": [
                    {"id": "new1", "lat": 40.7550, "lon": -73.9800, "priority": 3}
                ],
                "current_position": {"lat": 40.7589, "lon": -73.9851},
                "depot": {"lat": 40.7128, "lon": -74.0060},
            }

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/routes/dynamic-reoptimize", json=data, timeout=30
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                self.print_success(f"Reoptimized in {elapsed:.2f}s")
                self.print_success(f"New route: {result.get('num_stops')} stops")
                self.print_success(
                    f"New orders inserted: {result.get('new_orders_count')}"
                )

                self.results["passed"].append("Dynamic reoptimization")
                return True
            else:
                self.print_error(f"Status: {response.status_code}")
                self.results["failed"].append("Dynamic reoptimization")
                return False

        except Exception as e:
            self.print_error(f"Failed: {e}")
            self.results["failed"].append("Dynamic reoptimization")
            return False

    def test_route_scoring(self) -> bool:
        """Test route scoring"""
        self.print_test("Route Scoring")

        try:
            response = requests.get(
                f"{self.base_url}/routes/score",
                params={
                    "distance_km": 10.5,
                    "duration_hours": 0.5,
                    "num_stops": 5,
                    "time_of_day": 17,
                    "traffic_level": 0.8,
                },
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                score = result.get("total_score")
                rating = result.get("efficiency_rating")

                self.print_success(f"Score: {score:.2f}, Rating: {rating}")

                breakdown = result.get("breakdown", {})
                for metric, value in breakdown.items():
                    if value > 0:
                        self.print_success(f"  {metric}: {value:.2f}")

                self.results["passed"].append("Route scoring")
                return True
            else:
                self.print_error(f"Status: {response.status_code}")
                self.results["failed"].append("Route scoring")
                return False

        except Exception as e:
            self.print_error(f"Failed: {e}")
            self.results["failed"].append("Route scoring")
            return False

    def test_performance_under_load(self) -> bool:
        """Test system performance under load"""
        self.print_test("Performance Under Load")

        try:
            num_requests = 10
            request_data = {
                "depot": {"lat": 40.7128, "lon": -74.0060},
                "stops": [
                    {"lat": 40.7589, "lon": -73.9851, "priority": 1},
                    {"lat": 40.7614, "lon": -73.9776, "priority": 2},
                ],
                "algorithm": "nearest_neighbor",
            }

            self.print_success(f"Sending {num_requests} concurrent requests...")

            start_time = time.time()
            responses = []

            import concurrent.futures

            def make_request():
                return requests.post(
                    f"{self.base_url}/routes/optimize", json=request_data, timeout=30
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(num_requests)]
                responses = [
                    f.result() for f in concurrent.futures.as_completed(futures)
                ]

            elapsed = time.time() - start_time
            successful = sum(1 for r in responses if r.status_code == 200)

            avg_time = elapsed / num_requests
            requests_per_sec = num_requests / elapsed

            self.print_success(f"Completed: {successful}/{num_requests} requests")
            self.print_success(f"Total time: {elapsed:.2f}s")
            self.print_success(f"Average: {avg_time:.3f}s per request")
            self.print_success(f"Throughput: {requests_per_sec:.2f} req/s")

            if successful == num_requests:
                self.results["passed"].append("Performance under load")
                return True
            else:
                self.print_warning(f"Some requests failed: {num_requests - successful}")
                self.results["warnings"].append("Performance under load")
                return False

        except Exception as e:
            self.print_error(f"Failed: {e}")
            self.results["failed"].append("Performance under load")
            return False

    def test_error_handling(self) -> bool:
        """Test error handling with invalid inputs"""
        self.print_test("Error Handling & Validation")

        test_cases = [
            {
                "name": "Invalid coordinates",
                "endpoint": "/routes/optimize",
                "data": {"depot": {"lat": 999, "lon": 999}, "stops": []},
                "expect_error": True,
            },
            {
                "name": "Missing required fields",
                "endpoint": "/routes/optimize",
                "data": {
                    "depot": {"lat": 40.7128, "lon": -74.0060}
                    # Missing 'stops'
                },
                "expect_error": True,
            },
            {
                "name": "Invalid algorithm",
                "endpoint": "/routes/optimize",
                "data": {
                    "depot": {"lat": 40.7128, "lon": -74.0060},
                    "stops": [{"lat": 40.7589, "lon": -73.9851, "priority": 1}],
                    "algorithm": "invalid_algorithm",
                },
                "expect_error": True,
            },
        ]

        all_passed = True

        for case in test_cases:
            try:
                response = requests.post(
                    f"{self.base_url}{case['endpoint']}", json=case["data"], timeout=10
                )

                if case["expect_error"]:
                    if response.status_code >= 400:
                        self.print_success(
                            f"{case['name']}: Correctly rejected (status {response.status_code})"
                        )
                        self.results["passed"].append(f"Error handling: {case['name']}")
                    else:
                        self.print_error(f"{case['name']}: Should have been rejected")
                        all_passed = False
                        self.results["failed"].append(f"Error handling: {case['name']}")

            except Exception as e:
                self.print_error(f"{case['name']}: {e}")
                all_passed = False
                self.results["failed"].append(f"Error handling: {case['name']}")

        return all_passed

    def test_api_documentation(self) -> bool:
        """Test API documentation availability"""
        self.print_test("API Documentation")

        try:
            # Test OpenAPI docs
            response = requests.get(f"{self.base_url}/docs", timeout=5)
            if response.status_code == 200:
                self.print_success("Swagger UI available at /docs")
            else:
                self.print_warning("Swagger UI not accessible")

            # Test OpenAPI JSON
            response = requests.get(f"{self.base_url}/openapi.json", timeout=5)
            if response.status_code == 200:
                openapi_spec = response.json()
                num_endpoints = len(openapi_spec.get("paths", {}))
                self.print_success(
                    f"OpenAPI spec available: {num_endpoints} endpoints documented"
                )
                self.results["passed"].append("API documentation")
                return True
            else:
                self.print_warning("OpenAPI spec not accessible")
                self.results["warnings"].append("API documentation")
                return False

        except Exception as e:
            self.print_error(f"Failed: {e}")
            self.results["failed"].append("API documentation")
            return False

    def generate_report(self):
        """Generate comprehensive test report"""
        self.print_header("TEST REPORT")

        total = (
            len(self.results["passed"])
            + len(self.results["failed"])
            + len(self.results["warnings"])
        )
        passed = len(self.results["passed"])
        failed = len(self.results["failed"])
        warnings = len(self.results["warnings"])

        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"{Colors.BOLD}Total Tests: {total}{Colors.RESET}")
        print(f"{Colors.GREEN}✓ Passed: {passed} ({pass_rate:.1f}%){Colors.RESET}")
        print(f"{Colors.YELLOW}⚠ Warnings: {warnings}{Colors.RESET}")
        print(f"{Colors.RED}✗ Failed: {failed}{Colors.RESET}")

        print(f"\n{Colors.BOLD}Test Coverage:{Colors.RESET}")
        categories = {
            "Core Functionality": ["Health check", "Optimization", "ETA", "Heatmap"],
            "Advanced Features": [
                "Algorithm comparison",
                "Dynamic reoptimization",
                "Route scoring",
            ],
            "Quality Assurance": ["Error handling", "Performance under load"],
            "Documentation": ["API documentation"],
        }

        for category, tests in categories.items():
            category_tests = [
                t for t in self.results["passed"] if any(test in t for test in tests)
            ]
            print(f"  {category}: {len(category_tests)}/{len(tests)}")

        if failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.RESET}")
            print(f"{Colors.GREEN}Your CourierIQ is production-ready! 🚀{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.RESET}")
            print(f"{Colors.RED}Please review the failures above.{Colors.RESET}")

        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "pass_rate": pass_rate,
            },
            "results": self.results,
        }

        report_file = Path("test_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(
            f"\n{Colors.CYAN}📄 Detailed report saved to: {report_file}{Colors.RESET}"
        )


def main():
    """Run complete production test suite"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║          CourierIQ Production Testing Suite                       ║")
    print("║              Testing like a real product!                         ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")

    tester = ProductionTester()

    try:
        # Start server
        if not tester.start_server():
            print(f"\n{Colors.RED}Cannot start server. Exiting.{Colors.RESET}")
            return

        print("\n⏳ Waiting for server to stabilize...")
        time.sleep(2)

        # Run all tests
        tester.print_header("RUNNING PRODUCTION TESTS")

        tests = [
            ("Health Check", tester.test_health_check),
            ("Route Optimization", tester.test_route_optimization),
            ("Algorithm Comparison", tester.test_algorithm_comparison),
            ("ETA Prediction", tester.test_eta_prediction),
            ("Heatmap Generation", tester.test_heatmap_generation),
            ("Dynamic Reoptimization", tester.test_dynamic_reoptimization),
            ("Route Scoring", tester.test_route_scoring),
            ("Performance Under Load", tester.test_performance_under_load),
            ("Error Handling", tester.test_error_handling),
            ("API Documentation", tester.test_api_documentation),
        ]

        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                tester.print_error(f"{test_name} crashed: {e}")
                tester.results["failed"].append(test_name)

            time.sleep(0.5)  # Small delay between tests

        # Generate report
        tester.generate_report()

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"\n\n{Colors.RED}Testing failed: {e}{Colors.RESET}")
        import traceback

        traceback.print_exc()
    finally:
        # Always stop server
        tester.stop_server()


if __name__ == "__main__":
    main()
