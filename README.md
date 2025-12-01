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


courieriq/
│
├── src/
│ ├── data/
│ │ ├── raw/ # original datasets
│ │ ├── processed/ # cleaned + ready for training
│ │ └── geo/ # geospatial reference files
│ │
│ ├── models/
│ │ ├── eta_regressor.pkl
│ │ └── init.py
│ │
│ ├── engine/
│ │ ├── optimizer.py # multi-stop route optimization logic
│ │ ├── routing.py # API calls (Google/OSRM)
│ │ ├── scorer.py # scoring & ranking of route candidates
│ │ ├── heatmap.py # geospatial analysis
│ │ └── init.py
│ │
│ ├── api/
│ │ ├── main.py # FastAPI app
│ │ ├── schemas.py # validation models
│ │ └── controllers.py
│ │
│ ├── utils/
│ │ ├── config.py
│ │ ├── logger.py
│ │ └── helpers.py
│ │
│ └── init.py
│
├── notebooks/
│ ├── 01_exploration.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_model_training.ipynb
│
├── tests/
│ ├── test_optimizer.py
│ ├── test_routing.py
│ └── test_api.py
│
├── requirements.txt
├── README.md
└── LICENSE



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

## 📜 License

MIT License.
