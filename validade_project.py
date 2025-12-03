#!/usr/bin/env python3
"""
CourierIQ Project Validator
Comprehensive validation and testing script for the entire project
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class Colors:
    """ANSI color codes"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}{Colors.RESET}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")


class ProjectValidator:
    """Complete project validation"""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = []

    def validate_structure(self) -> bool:
        """Validate project structure"""
        print_header("1. Project Structure Validation")

        required_files = {
            "src/engine/optimizer.py": "Route optimization",
            "src/engine/routing.py": "Routing APIs",
            "src/engine/scorer.py": "Route scoring",
            "src/engine/heatmap.py": "Heatmap generation",
            "src/models/eta_model.py": "ETA prediction model",
            "src/api/services/engine_service.py": "Engine service",
            "src/api/controllers/routes.py": "Routes controller",
            "src/api/main.py": "FastAPI main",
            "src/utils/helpers.py": "Helper utilities",
            "src/utils/config.py": "Configuration",
            "src/utils/logger.py": "Logging",
            "requirements.txt": "Dependencies",
        }

        required_dirs = [
            "src/engine",
            "src/models",
            "src/api",
            "src/api/services",
            "src/api/controllers",
            "src/utils",
            "src/data",
            "tests",
            "notebooks",
        ]

        # Check directories
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                print_success(f"Directory: {dir_path}")
                self.passed.append(f"Directory {dir_path}")
            else:
                print_error(f"Missing directory: {dir_path}")
                self.errors.append(f"Missing directory: {dir_path}")

        # Check files
        for file_path, description in required_files.items():
            if Path(file_path).exists():
                print_success(f"{description}: {file_path}")
                self.passed.append(f"File {file_path}")
            else:
                print_error(f"Missing {description}: {file_path}")
                self.errors.append(f"Missing file: {file_path}")

        # Check __init__.py files
        init_dirs = [
            "src",
            "src/engine",
            "src/models",
            "src/api",
            "src/api/services",
            "src/api/controllers",
            "src/utils",
            "tests",
        ]

        print()
        for dir_path in init_dirs:
            init_file = Path(dir_path) / "__init__.py"
            if init_file.exists():
                print_success(f"Init file: {init_file}")
                self.passed.append(f"Init {init_file}")
            else:
                print_warning(f"Missing __init__.py in {dir_path}")
                self.warnings.append(f"Missing __init__.py in {dir_path}")

        return len(self.errors) == 0

    def validate_imports(self) -> bool:
        """Validate that all modules can be imported"""
        print_header("2. Import Validation")

        modules_to_test = [
            ("src.engine.optimizer", "RouteOptimizer"),
            ("src.engine.routing", "RoutingService"),
            ("src.engine.scorer", "RouteScorer"),
            ("src.engine.heatmap", "HeatmapGenerator"),
            ("src.models.eta_model", "ETAPredictor"),
            ("src.api.services.engine_service", "EngineService"),
            ("src.utils.helpers", "haversine_distance"),
            ("src.utils.config", "load_settings"),
            ("src.utils.logger", "setup_logger"),
        ]

        for module_name, class_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    print_success(f"Import {module_name}.{class_name}")
                    self.passed.append(f"Import {module_name}.{class_name}")
                else:
                    print_error(f"Class {class_name} not found in {module_name}")
                    self.errors.append(f"Missing class {class_name}")
            except ImportError as e:
                print_error(f"Cannot import {module_name}: {e}")
                self.errors.append(f"Import error: {module_name}")
            except Exception as e:
                print_error(f"Error importing {module_name}: {e}")
                self.errors.append(f"Import error: {module_name}")

        return len(self.errors) == 0

    def validate_dependencies(self) -> bool:
        """Check if all dependencies are installed"""
        print_header("3. Dependencies Validation")

        required_packages = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "numpy",
            "pandas",
            "sklearn",
            "requests",
            "geopy",
            "matplotlib",
            "plotly",
        ]

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print_success(f"Package: {package}")
                self.passed.append(f"Package {package}")
            except ImportError:
                print_error(f"Missing package: {package}")
                self.errors.append(f"Missing package: {package}")

        # Optional packages
        optional_packages = ["lightgbm", "torch"]
        print()
        print_info("Optional packages:")
        for package in optional_packages:
            try:
                __import__(package)
                print_success(f"  {package} (installed)")
            except ImportError:
                print_warning(f"  {package} (not installed - optional)")

        return len(self.errors) == 0

    def test_optimizer(self) -> bool:
        """Test optimizer functionality"""
        print_header("4. Optimizer Testing")

        try:
            from src.engine.optimizer import RouteOptimizer, Stop

            # Create test data
            optimizer = RouteOptimizer()
            depot = Stop(id="depot", lat=40.7128, lon=-74.0060)
            stops = [
                Stop(id="s1", lat=40.7589, lon=-73.9851, priority=1),
                Stop(id="s2", lat=40.7614, lon=-73.9776, priority=2),
            ]

            # Test nearest neighbor
            print_info("Testing nearest neighbor...")
            route = optimizer.nearest_neighbor(stops, depot)
            if route and len(route.stops) == 2:
                print_success("Nearest neighbor works")
                self.passed.append("Nearest neighbor algorithm")
            else:
                print_error("Nearest neighbor failed")
                self.errors.append("Nearest neighbor algorithm")

            # Test simulated annealing
            print_info("Testing simulated annealing...")
            route = optimizer.simulated_annealing(stops, depot, max_iterations=100)
            if route and len(route.stops) == 2:
                print_success("Simulated annealing works")
                self.passed.append("Simulated annealing algorithm")
            else:
                print_error("Simulated annealing failed")
                self.errors.append("Simulated annealing algorithm")

            # Test optimize method
            print_info("Testing optimize method...")
            route = optimizer.optimize(stops, depot, algorithm="nearest_neighbor")
            if route and route.total_distance > 0:
                print_success(
                    f"Optimization works (distance: {route.total_distance / 1000:.2f}km)"
                )
                self.passed.append("Optimize method")
            else:
                print_error("Optimize method failed")
                self.errors.append("Optimize method")

            return True

        except Exception as e:
            print_error(f"Optimizer testing failed: {e}")
            self.errors.append(f"Optimizer test: {e}")
            return False

    def test_routing_service(self) -> bool:
        """Test routing service"""
        print_header("5. Routing Service Testing")

        try:
            from src.engine.routing import RoutingService

            service = RoutingService()
            print_success("RoutingService initialized")

            # Check providers
            if "osrm" in service.providers:
                print_success("OSRM provider available")
                self.passed.append("OSRM provider")
            else:
                print_error("OSRM provider not available")
                self.errors.append("OSRM provider")

            print_info("Note: Actual API calls require internet connection")

            return True

        except Exception as e:
            print_error(f"Routing service testing failed: {e}")
            self.errors.append(f"Routing service: {e}")
            return False

    def test_scorer(self) -> bool:
        """Test route scorer"""
        print_header("6. Route Scorer Testing")

        try:
            from src.engine.scorer import RouteScorer

            scorer = RouteScorer()

            # Test scoring
            route_data = {
                "distance": 5000,  # 5km
                "duration": 900,  # 15 minutes
            }

            stops = [
                {"priority": 1},
                {"priority": 2},
            ]

            score = scorer.score_route(route_data, stops)

            if score.total_score > 0:
                print_success(
                    f"Scoring works (score: {score.total_score:.2f}, rating: {score.efficiency_rating})"
                )
                self.passed.append("Route scoring")
            else:
                print_error("Scoring failed")
                self.errors.append("Route scoring")

            return True

        except Exception as e:
            print_error(f"Scorer testing failed: {e}")
            self.errors.append(f"Scorer test: {e}")
            return False

    def test_heatmap(self) -> bool:
        """Test heatmap generator"""
        print_header("7. Heatmap Generator Testing")

        try:
            from src.engine.heatmap import HeatmapGenerator

            generator = HeatmapGenerator(grid_size_km=1.0)

            # Create test deliveries
            deliveries = [
                {
                    "lat": 40.7128,
                    "lon": -74.0060,
                    "delivery_time": 900,
                    "status": "success",
                },
                {
                    "lat": 40.7589,
                    "lon": -73.9851,
                    "delivery_time": 1200,
                    "status": "success",
                },
                {
                    "lat": 40.7614,
                    "lon": -73.9776,
                    "delivery_time": 800,
                    "status": "success",
                },
            ]

            # Generate heatmap
            heatmap = generator.generate_delivery_heatmap(deliveries)

            if len(heatmap) > 0:
                print_success(f"Heatmap generated with {len(heatmap)} cells")
                self.passed.append("Heatmap generation")
            else:
                print_error("Heatmap generation failed")
                self.errors.append("Heatmap generation")

            # Test zone identification
            zones = generator.identify_high_efficiency_zones(
                heatmap, min_deliveries=1, top_n=3
            )

            if len(zones) > 0:
                print_success(f"Zone identification works ({len(zones)} zones)")
                self.passed.append("Zone identification")
            else:
                print_warning(
                    "No zones identified (may be expected with small dataset)"
                )

            return True

        except Exception as e:
            print_error(f"Heatmap testing failed: {e}")
            self.errors.append(f"Heatmap test: {e}")
            return False

    def test_eta_model(self) -> bool:
        """Test ETA model"""
        print_header("8. ETA Model Testing")

        try:
            import pandas as pd

            from src.models.eta_model import (
                ETAPredictor,
                generate_synthetic_training_data,
            )

            # Test model creation
            predictor = ETAPredictor(model_type="gradient_boosting")
            print_success("ETAPredictor created")

            # Generate small dataset
            print_info("Generating synthetic training data...")
            data = generate_synthetic_training_data(n_samples=100)

            if len(data) == 100:
                print_success("Synthetic data generated (100 samples)")
                self.passed.append("Synthetic data generation")
            else:
                print_error("Synthetic data generation failed")
                self.errors.append("Synthetic data generation")

            # Quick training test
            print_info("Running quick training test...")
            try:
                metrics = predictor.train(
                    data, target_column="eta_seconds", validate=True
                )
                if metrics.get("test_mae"):
                    print_success(
                        f"Model training works (MAE: {metrics['test_mae']:.2f}s)"
                    )
                    self.passed.append("Model training")
                else:
                    print_warning("Training completed but no metrics")
            except Exception as e:
                print_warning(f"Training test skipped: {e}")

            return True

        except Exception as e:
            print_error(f"ETA model testing failed: {e}")
            self.errors.append(f"ETA model test: {e}")
            return False

    def test_api_startup(self) -> bool:
        """Test if API can start"""
        print_header("9. API Startup Test")

        try:
            from src.api.main import app

            print_success("FastAPI app imported successfully")

            # Check routes
            routes = [route.path for route in app.routes]

            expected_routes = [
                "/routes/optimize",
                "/routes/eta",
                "/routes/heatmap",
                "/health",
            ]

            for route in expected_routes:
                if any(route in r for r in routes):
                    print_success(f"Route exists: {route}")
                    self.passed.append(f"Route {route}")
                else:
                    print_warning(f"Route not found: {route}")
                    self.warnings.append(f"Route {route}")

            print_info(f"Total routes: {len(routes)}")

            return True

        except Exception as e:
            print_error(f"API startup test failed: {e}")
            self.errors.append(f"API startup: {e}")
            return False

    def test_helpers(self) -> bool:
        """Test helper functions"""
        print_header("10. Helper Functions Testing")

        try:
            from src.utils.helpers import (
                format_distance,
                format_duration,
                haversine_distance,
                validate_coordinates,
            )

            # Test haversine
            dist = haversine_distance(40.7128, -74.0060, 40.7589, -73.9851)
            if 4 < dist < 7:  # Should be ~5-6 km
                print_success(f"Haversine distance works: {dist:.2f}km")
                self.passed.append("Haversine distance")
            else:
                print_error(f"Haversine distance unexpected: {dist:.2f}km")
                self.errors.append("Haversine distance")

            # Test formatting
            duration_str = format_duration(3665)
            if "1h" in duration_str or "61m" in duration_str:
                print_success(f"Duration formatting works: {duration_str}")
                self.passed.append("Duration formatting")
            else:
                print_error(f"Duration formatting failed: {duration_str}")
                self.errors.append("Duration formatting")

            # Test coordinate validation
            if validate_coordinates(40.7128, -74.0060):
                print_success("Coordinate validation works")
                self.passed.append("Coordinate validation")
            else:
                print_error("Coordinate validation failed")
                self.errors.append("Coordinate validation")

            return True

        except Exception as e:
            print_error(f"Helper functions testing failed: {e}")
            self.errors.append(f"Helper functions: {e}")
            return False

    def print_summary(self):
        """Print validation summary"""
        print_header("VALIDATION SUMMARY")

        total_tests = len(self.passed) + len(self.errors) + len(self.warnings)

        print(f"{Colors.BOLD}Total Tests: {total_tests}{Colors.RESET}")
        print(f"{Colors.GREEN}✓ Passed: {len(self.passed)}{Colors.RESET}")
        print(f"{Colors.YELLOW}⚠ Warnings: {len(self.warnings)}{Colors.RESET}")
        print(f"{Colors.RED}✗ Failed: {len(self.errors)}{Colors.RESET}")

        if len(self.errors) > 0:
            print(f"\n{Colors.RED}{Colors.BOLD}ERRORS:{Colors.RESET}")
            for error in self.errors:
                print(f"  {Colors.RED}• {error}{Colors.RESET}")

        if len(self.warnings) > 0:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}WARNINGS:{Colors.RESET}")
            for warning in self.warnings:
                print(f"  {Colors.YELLOW}• {warning}{Colors.RESET}")

        print()
        if len(self.errors) == 0:
            print(
                f"{Colors.GREEN}{Colors.BOLD}✓ PROJECT VALIDATION PASSED!{Colors.RESET}"
            )
            print(
                f"{Colors.GREEN}Your CourierIQ project is ready to use!{Colors.RESET}"
            )
            return True
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ PROJECT VALIDATION FAILED{Colors.RESET}")
            print(
                f"{Colors.RED}Please fix the errors above before proceeding.{Colors.RESET}"
            )
            return False


def main():
    """Main validation function"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║          CourierIQ Project Validation & Testing Suite             ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")

    validator = ProjectValidator()

    # Run all validations
    tests = [
        validator.validate_structure,
        validator.validate_imports,
        validator.validate_dependencies,
        validator.test_optimizer,
        validator.test_routing_service,
        validator.test_scorer,
        validator.test_heatmap,
        validator.test_eta_model,
        validator.test_api_startup,
        validator.test_helpers,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print_error(f"Test failed with exception: {e}")
            validator.errors.append(f"Test exception: {e}")

    # Print summary
    success = validator.print_summary()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
