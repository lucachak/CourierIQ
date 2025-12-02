#!/usr/bin/env python3
"""
CourierIQ Project Setup Script
Run this after copying all the artifact files to set up the project structure
"""

import os
import subprocess
import sys
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def check_python_version():
    """Ensure Python 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")


def create_directories():
    """Create necessary directories"""
    print_header("Creating Directory Structure")

    dirs = [
        "src/engine",
        "src/models",
        "src/api/services",
        "src/data/raw",
        "src/data/processed",
        "src/data/geo",
        "logs",
        "cache",
        "tests",
        "notebooks",
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")


def create_init_files():
    """Create __init__.py files"""
    print_header("Creating __init__.py Files")

    init_dirs = [
        "src",
        "src/engine",
        "src/models",
        "src/api",
        "src/api/services",
        "src/api/controllers",
        "src/utils",
        "src/data",
        "src/ml",
        "tests",
    ]

    for dir_path in init_dirs:
        init_file = Path(dir_path) / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"✅ Created: {init_file}")


def create_env_file():
    """Create .env template"""
    print_header("Creating .env File")

    env_content = """# CourierIQ Environment Variables

# API Keys
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
MAX_REQUESTS_PER_MINUTE=60

# OSRM Server (optional - use public or local)
OSRM_URL=http://router.project-osrm.org

# Model Settings
MODEL_TYPE=gradient_boosting
MODEL_PATH=src/models/eta_regressor.pkl
"""

    env_file = Path(".env")
    if not env_file.exists():
        env_file.write_text(env_content)
        print("✅ Created .env template")
    else:
        print("⚠️  .env already exists, skipping")


def create_gitignore():
    """Create .gitignore"""
    print_header("Creating .gitignore")

    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Logs
logs/
*.log

# Data
src/data/raw/*.csv
src/data/processed/*.csv
src/data/geo/*.geojson
cache/

# Models
src/models/*.pkl
!src/models/__init__.py

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Testing
.coverage
htmlcov/
.pytest_cache/
"""

    gitignore_file = Path(".gitignore")
    if not gitignore_file.exists():
        gitignore_file.write_text(gitignore_content)
        print("✅ Created .gitignore")
    else:
        print("⚠️  .gitignore already exists, skipping")


def check_required_files():
    """Check if required files exist"""
    print_header("Checking Required Files")

    required_files = {
        "src/engine/optimizer.py": "Route optimization algorithms",
        "src/engine/routing.py": "Routing API integration",
        "src/engine/scorer.py": "Route scoring system",
        "src/engine/heatmap.py": "Geospatial analysis",
        "src/models/eta_model.py": "ETA prediction model",
        "src/api/services/engine_service.py": "Engine service layer",
        "src/utils/helpers.py": "Utility functions",
        "src/utils/logger.py": "Logging configuration",
        "requirements.txt": "Project dependencies",
    }

    missing = []
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ Missing: {file_path} ({description})")
            missing.append(file_path)

    if missing:
        print("\n⚠️  Please copy the missing files from the artifacts")
        return False

    return True


def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")

    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False

    try:
        print("Installing dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def create_sample_data():
    """Create sample data for testing"""
    print_header("Creating Sample Data")

    try:
        from src.models.eta_model import generate_synthetic_training_data

        data = generate_synthetic_training_data(n_samples=100)
        data_path = Path("src/data/processed/sample_data.csv")
        data.to_csv(data_path, index=False)
        print(f"✅ Created sample data: {data_path}")
        return True
    except Exception as e:
        print(f"⚠️  Could not create sample data: {e}")
        return False


def run_tests():
    """Run test suite"""
    print_header("Running Tests")

    try:
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)
        print("✅ All tests passed")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Some tests failed")
        return False
    except FileNotFoundError:
        print("⚠️  pytest not installed, skipping tests")
        return False


def print_next_steps():
    """Print next steps"""
    print_header("Setup Complete! 🎉")

    print("""
Next Steps:

1. Configure your environment:
   - Edit .env and add your Google Maps API key (optional)
   - Adjust settings as needed

2. Train the ML model:
   jupyter notebook notebooks/03_model_training.ipynb

   Or run programmatically:
   python -c "from src.models.eta_model import ETAPredictor, generate_synthetic_training_data; \\
              data = generate_synthetic_training_data(5000); \\
              model = ETAPredictor('gradient_boosting'); \\
              model.train(data); \\
              model.save('src/models/eta_regressor.pkl')"

3. Start the API server:
   uvicorn src.api.main:app --reload

4. Visit the API documentation:
   http://localhost:8000/docs

5. Test an optimization:
   curl -X POST "http://localhost:8000/routes/optimize" \\
     -H "Content-Type: application/json" \\
     -d '{"depot": {"lat": 40.7128, "lon": -74.0060}, \\
          "stops": [{"lat": 40.7589, "lon": -73.9851, "priority": 1}]}'

6. Check the implementation guide:
   See IMPLEMENTATION_GUIDE.md for detailed documentation

For questions or issues, refer to the documentation or create an issue.

Happy routing! 🚀
""")


def main():
    """Main setup function"""
    print_header("CourierIQ Project Setup")
    print("This script will set up your CourierIQ project structure\n")

    # Check Python version
    check_python_version()

    # Create structure
    create_directories()
    create_init_files()
    create_env_file()
    create_gitignore()

    # Check files
    if not check_required_files():
        print("\n⚠️  Please copy all required files before continuing")
        return

    # Install dependencies
    print("\nDo you want to install dependencies now? (y/n): ", end="")
    if input().lower() == "y":
        install_dependencies()

    # Create sample data
    print("\nDo you want to create sample data? (y/n): ", end="")
    if input().lower() == "y":
        create_sample_data()

    # Run tests
    print("\nDo you want to run tests? (y/n): ", end="")
    if input().lower() == "y":
        run_tests()

    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
