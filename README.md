# CourierIQ — Intelligent Delivery Route Optimizer

CourierIQ is an intelligent routing engine designed for modern delivery workflows.  
It uses geospatial analysis, machine learning, and real-time map data to compute the fastest and most efficient delivery path across multiple stops.

## 🚀 Features

- 📍 Real-time routing via Google Maps or OpenStreetMap APIs  
- 🧠 ETA prediction model (regression-based or neural network)  
- 🗺️ Heatmap of high-efficiency delivery zones  
- 🔄 Dynamic route recalculation when new orders appear  
- 📊 Interactive dashboard built with FastAPI + Plotly  
- 🧭 Multi-stop route optimization with heuristics (A*, Simulated Annealing, or Tabu Search)

---

## 🧰 Tech Stack

**Backend / Engine**  
- Python 3.12  
- FastAPI  
- NumPy, Pandas  
- Scikit-learn or PyTorch  
- Geopy / OSRM / Google Maps API  
- Matplotlib or Plotly

**Data**  
- CSV datasets  
- Pickled ML models  
- JSON config + cache local

**Optional**  
- Docker  
- Redis for caching  
- PostGIS for geospatial data

---

## 🗂️ Project Structure



### Directory Overview

- **`src/data/`** - Data storage and management
  - `raw/`: Original, unprocessed datasets
  - `processed/`: Cleaned and transformed data ready for model training
  - `geo/`: Geospatial files (shapefiles, GeoJSON, etc.)

- **`src/engine/`** - Core routing and optimization logic
  - `optimizer.py`: Implements multi-stop route optimization algorithms
  - `routing.py`: Handles external routing API calls (Google Maps, OSRM)
  - `scorer.py`: Scores and ranks alternative route candidates
  - `heatmap.py`: Performs geospatial analysis and generates heatmaps

- **`src/api/`** - Web API layer
  - `main.py`: FastAPI application setup and route definitions
  - `schemas.py`: Pydantic models for request/response validation
  - `controllers.py`: Business logic and request handling

- **`src/models/`** - Machine learning models
  - `eta_regressor.pkl`: Serialized ETA prediction model
  - Model training and inference utilities

- **`src/utils/`** - Shared utilities
  - `config.py`: Configuration management and environment variables
  - `logger.py`: Logging configuration and custom loggers
  - `helpers.py`: Reusable helper functions

- **`notebooks/`** - Jupyter notebooks for analysis and prototyping
  - Exploratory data analysis and visualization
  - Feature engineering experiments
  - Model training and evaluation

- **`tests/`** - Test suite
  - Unit tests for all major components
  - Integration tests for API endpoints

---

## ▶️ Getting Started

### Install dependencies

pip install -r requirements.txt

### Run the API locally

uvicorn src.api.main:app --reload

### Access the dashboard / API docs  
- API Docs: `http://localhost:8000/docs`  
- Home: `http://localhost:8000/`

---

## 📊 The ETA Model

The prediction model uses engineered features such as:

- distance metrics (Haversine, route distance)  
- traffic-level proxies  
- time-of-day / day-of-week cycles  
- historical wait times  
- restaurant preparation patterns  
- weather data (optional)

Model types supported:

- Gradient Boosting (recommended)  
- LightGBM  
- RandomForest  
- Lightweight Neural Networks  

Model retraining is handled through Jupyter notebooks located in `/notebooks`.

---

## 🛣️ Roadmap

- [ ] Mobile-friendly PWA  
- [ ] GPS integration through device sensors  
- [ ] Support for electric scooters / bicycles  
- [ ] Self-learning ETA model  
- [ ] Offline routing mode (OSRM local docker)  
- [ ] Driver statistics panel (acceptance rate, speed, efficiency)  

---
