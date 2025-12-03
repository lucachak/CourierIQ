#!/usr/bin/env python3
"""
CourierIQ Quick Start
One command to set up and test everything
"""

import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a shell command"""
    print(f"\n{'=' * 60}")
    print(f"▶ {description}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            cmd, shell=True, check=check, capture_output=True, text=True
        )

        if result.stdout:
            print(result.stdout)

        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            if result.stderr:
                print(f"✗ Error: {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def create_example_data():
    """Create example data files"""
    print("\n" + "=" * 60)
    print("▶ Creating example data")
    print("=" * 60)

    # Create example delivery data
    example_deliveries = [
        {
            "lat": 40.7128 + i * 0.01,
            "lon": -74.0060 + i * 0.01,
            "delivery_time": 600 + i * 100,
            "distance": 1000 + i * 500,
            "status": "success" if i % 5 != 0 else "failed",
            "hour": 9 + (i % 12),
        }
        for i in range(50)
    ]

    data_file = Path("example_deliveries.json")
    with open(data_file, "w") as f:
        json.dump(example_deliveries, f, indent=2)

    print(f"✓ Created {data_file} with 50 sample deliveries")

    # Create example route request
    example_request = {
        "depot": {"lat": 40.7128, "lon": -74.0060},
        "stops": [
            {"lat": 40.7589, "lon": -73.9851, "priority": 1},
            {"lat": 40.7614, "lon": -73.9776, "priority": 2},
            {"lat": 40.7489, "lon": -73.9680, "priority": 1},
            {"lat": 40.7308, "lon": -73.9973, "priority": 3},
        ],
        "algorithm": "simulated_annealing",
    }

    request_file = Path("example_route_request.json")
    with open(request_file, "w") as f:
        json.dump(example_request, f, indent=2)

    print(f"✓ Created {request_file}")


def test_api_endpoint():
    """Test the API endpoint"""
    print("\n" + "=" * 60)
    print("▶ Testing API endpoint")
    print("=" * 60)

    try:
        import time

        import requests

        # Give server time to start
        print("Waiting for server to start...")
        time.sleep(3)

        # Test health endpoint
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✓ Health check passed")
                print(f"  Response: {response.json()}")
            else:
                print(f"✗ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"⚠ Could not reach server: {e}")
            print("  (This is normal if running validation only)")

    except ImportError:
        print("⚠ requests library not installed, skipping API test")


def quick_test():
    """Run quick functionality tests"""
    print("\n" + "=" * 60)
    print("▶ Running quick functionality tests")
    print("=" * 60)

    try:
        # Test optimizer
        from src.engine.optimizer import RouteOptimizer, Stop

        optimizer = RouteOptimizer()
        depot = Stop(id="depot", lat=40.7128, lon=-74.0060)
        stops = [
            Stop(id="s1", lat=40.7589, lon=-73.9851, priority=1),
            Stop(id="s2", lat=40.7614, lon=-73.9776, priority=2),
        ]

        route = optimizer.optimize(stops, depot, algorithm="nearest_neighbor")
        print(
            f"✓ Optimization works: {len(route.stops)} stops, {route.total_distance / 1000:.2f}km"
        )

        # Test ETA model
        from src.models.eta_model import ETAPredictor, generate_synthetic_training_data

        print("\n  Testing ETA model with tiny dataset...")
        data = generate_synthetic_training_data(n_samples=50)
        predictor = ETAPredictor("gradient_boosting")
        metrics = predictor.train(data, target_column="eta_seconds")

        print(f"✓ ETA model trained: MAE={metrics.get('test_mae', 'N/A'):.1f}s")

        # Test prediction
        eta = predictor.predict_single(
            origin_lat=40.7128,
            origin_lon=-74.0060,
            dest_lat=40.7589,
            dest_lon=-73.9851,
            num_stops=2,
            traffic_level=0.5,
        )
        print(f"✓ Prediction works: {eta / 60:.1f} minutes")

        # Test heatmap
        from src.engine.heatmap import HeatmapGenerator

        generator = HeatmapGenerator()
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
        ]

        heatmap = generator.generate_delivery_heatmap(deliveries)
        print(f"✓ Heatmap generated: {len(heatmap)} cells")

        print("\n✓ All quick tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Quick test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def print_next_steps():
    """Print next steps"""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE! 🎉")
    print("=" * 60)

    print("""
Next steps:

1. Start the API server:
   uvicorn src.api.main:app --reload

2. Open the API documentation:
   http://localhost:8000/docs

3. Try the example endpoints:

   # Optimize a route
   curl -X POST "http://localhost:8000/routes/optimize" \\
     -H "Content-Type: application/json" \\
     -d @example_route_request.json

   # Get ETA prediction
   curl -X POST "http://localhost:8000/routes/eta" \\
     -H "Content-Type: application/json" \\
     -d '{"origin":{"lat":40.7128,"lon":-74.0060},"destination":{"lat":40.7589,"lon":-73.9851}}'

   # Generate heatmap
   curl -X POST "http://localhost:8000/routes/heatmap" \\
     -H "Content-Type: application/json" \\
     -d @example_deliveries.json

4. Train a better model with more data:
   jupyter notebook notebooks/03_model_training.ipynb

5. Run the full test suite:
   pytest tests/ -v

6. Check the implementation guide:
   See IMPLEMENTATION_GUIDE.md for detailed documentation

Happy routing! 🚀
""")


def main():
    """Main quick start function"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                  CourierIQ Quick Start                             ║
║             One command to set up everything!                      ║
╚════════════════════════════════════════════════════════════════════╝
""")

    print("\nThis script will:")
    print("  1. Validate project structure")
    print("  2. Check dependencies")
    print("  3. Run quick functionality tests")
    print("  4. Create example data files")
    print("\n")

    input("Press Enter to continue...")

    # Run validation
    validation_script = Path("validate_project.py")
    if validation_script.exists():
        success = run_command(
            f"{sys.executable} {validation_script}",
            "Running project validation",
            check=False,
        )

        if not success:
            print("\n⚠ Validation found issues. Continue anyway? (y/n): ", end="")
            if input().lower() != "y":
                print("Exiting...")
                return

    # Create example data
    create_example_data()

    # Run quick tests
    if not quick_test():
        print("\n⚠ Some tests failed. The project may not be fully functional.")
        print("Please check the errors above and review the setup.")
        return

    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nSetup failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
