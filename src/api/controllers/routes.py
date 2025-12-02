"""
Updated routes controller with full engine integration
Replace your existing src/api/controllers/routes.py with this
"""

import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.services.engine_service import EngineService

router = APIRouter(prefix="/routes", tags=["Routes"])

# Initialize engine service
# Get API key from environment if available
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
engine = EngineService(
    google_api_key=GOOGLE_API_KEY, default_algorithm="simulated_annealing"
)


# Request/Response Models
class Coordinate(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class StopRequest(BaseModel):
    id: Optional[str] = None
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    priority: int = Field(default=1, ge=1, le=10)
    time_window: Optional[tuple[float, float]] = None
    service_time: float = Field(default=300, ge=0)  # seconds


class RouteOptimizationRequest(BaseModel):
    depot: Coordinate
    stops: List[StopRequest]
    algorithm: Optional[str] = Field(
        default="simulated_annealing",
        description="Algorithm: nearest_neighbor, two_opt, simulated_annealing, tabu_search",
    )
    use_api: bool = Field(
        default=False, description="Use external routing API for real distances"
    )
    provider: Optional[str] = Field(
        default=None, description="Routing provider: google or osrm"
    )


class RouteSegment(BaseModel):
    stop_id: str
    lat: float
    lon: float
    priority: int


class RouteOptimizationResponse(BaseModel):
    success: bool
    depot: Coordinate
    stops: List[RouteSegment]
    total_distance_km: float
    total_time_hours: float
    total_time_minutes: float
    score: float
    num_stops: int
    algorithm: Optional[str] = None
    provider: Optional[str] = None


class ETARequest(BaseModel):
    origin: Coordinate
    destination: Coordinate
    num_stops: Optional[int] = 1
    traffic_level: Optional[float] = Field(default=0.5, ge=0, le=1)
    timestamp: Optional[datetime] = None


class ETAResponse(BaseModel):
    eta_seconds: float
    eta_minutes: float
    distance_km: float


class HeatmapRequest(BaseModel):
    deliveries: List[dict]
    grid_size_km: float = Field(default=1.0, gt=0)


@router.post("/optimize", response_model=RouteOptimizationResponse)
async def optimize_route(request: RouteOptimizationRequest):
    """
    Generate an optimized delivery route

    - **depot**: Starting point coordinates
    - **stops**: List of delivery stops with priorities
    - **algorithm**: Optimization algorithm to use
    - **use_api**: Whether to use external routing APIs
    """
    try:
        stops_data = [
            {
                "id": stop.id,
                "lat": stop.lat,
                "lon": stop.lon,
                "priority": stop.priority,
                "time_window": stop.time_window,
                "service_time": stop.service_time,
            }
            for stop in request.stops
        ]

        if request.use_api:
            result = engine.optimize_route_with_apis(
                depot_lat=request.depot.lat,
                depot_lon=request.depot.lon,
                stops=stops_data,
                provider=request.provider,
            )
        else:
            result = engine.optimize_route_simple(
                depot_lat=request.depot.lat,
                depot_lon=request.depot.lon,
                stops=stops_data,
                algorithm=request.algorithm,
            )

        if not result.get("success", False):
            raise HTTPException(
                status_code=500, detail=result.get("error", "Optimization failed")
            )

        return RouteOptimizationResponse(
            success=True,
            depot=request.depot,
            stops=[
                RouteSegment(
                    stop_id=stop["id"],
                    lat=stop["lat"],
                    lon=stop["lon"],
                    priority=stop.get("priority", 1),
                )
                for stop in result["stops"]
            ],
            total_distance_km=result["total_distance_km"],
            total_time_hours=result["total_time_hours"],
            total_time_minutes=result["total_time_hours"] * 60,
            score=result["score"],
            num_stops=result["num_stops"],
            algorithm=result.get("algorithm"),
            provider=result.get("provider"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-algorithms")
async def compare_algorithms(request: RouteOptimizationRequest):
    """
    Compare all available optimization algorithms
    Returns results from each algorithm sorted by performance
    """
    try:
        stops_data = [
            {
                "id": stop.id,
                "lat": stop.lat,
                "lon": stop.lon,
                "priority": stop.priority,
                "time_window": stop.time_window,
                "service_time": stop.service_time,
            }
            for stop in request.stops
        ]

        result = engine.compare_algorithms(
            depot_lat=request.depot.lat, depot_lon=request.depot.lon, stops=stops_data
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/eta", response_model=ETAResponse)
async def calculate_eta(request: ETARequest):
    """
    Calculate ETA for a delivery
    Uses ML model if available, otherwise uses distance estimation
    """
    try:
        # Try to load ML model
        from pathlib import Path

        from src.models.eta_model import ETAPredictor

        model_path = (
            Path(__file__).parent.parent.parent / "models" / "eta_regressor.pkl"
        )

        if model_path.exists():
            # Use ML model
            predictor = ETAPredictor.load(str(model_path))
            eta_seconds = predictor.predict_single(
                origin_lat=request.origin.lat,
                origin_lon=request.origin.lon,
                dest_lat=request.destination.lat,
                dest_lon=request.destination.lon,
                timestamp=request.timestamp or datetime.now(),
                num_stops=request.num_stops,
                traffic_level=request.traffic_level,
            )
        else:
            # Fallback to distance-based estimation
            from src.engine.optimizer import RouteOptimizer

            optimizer = RouteOptimizer()

            from src.engine.optimizer import Stop

            origin = Stop(id="origin", lat=request.origin.lat, lon=request.origin.lon)
            dest = Stop(
                id="dest", lat=request.destination.lat, lon=request.destination.lon
            )

            distance = optimizer._haversine_distance(origin, dest)

            # Simple estimation: distance/speed + service time
            avg_speed_kph = 30
            travel_time = (distance / 1000) / avg_speed_kph * 3600
            service_time = 300 * request.num_stops
            traffic_penalty = travel_time * request.traffic_level

            eta_seconds = travel_time + service_time + traffic_penalty

        return ETAResponse(
            eta_seconds=float(eta_seconds),
            eta_minutes=float(eta_seconds / 60),
            distance_km=float(distance / 1000) if "distance" in locals() else 0.0,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/heatmap")
async def generate_heatmap(request: HeatmapRequest):
    """
    Generate delivery efficiency heatmap

    Deliveries should have format:
    ```json
    {
      "lat": 40.7128,
      "lon": -74.0060,
      "delivery_time": 1200,
      "status": "success",
      "distance": 5000
    }
    ```
    """
    try:
        result = engine.generate_delivery_heatmap(request.deliveries)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dynamic-reoptimize")
async def dynamic_reoptimize(
    current_route: List[StopRequest],
    new_orders: List[StopRequest],
    current_position: Coordinate,
    depot: Coordinate,
):
    """
    Dynamically reoptimize route when new orders arrive
    """
    try:
        current_stops = [
            {"id": stop.id, "lat": stop.lat, "lon": stop.lon, "priority": stop.priority}
            for stop in current_route
        ]

        new_stops = [
            {"id": stop.id, "lat": stop.lat, "lon": stop.lon, "priority": stop.priority}
            for stop in new_orders
        ]

        result = engine.dynamic_reoptimize(
            current_route=current_stops,
            new_orders=new_stops,
            current_position={"lat": current_position.lat, "lon": current_position.lon},
            depot={"lat": depot.lat, "lon": depot.lon},
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/score")
async def score_route(
    distance_km: float = Query(..., description="Total route distance in km"),
    duration_hours: float = Query(..., description="Total route duration in hours"),
    num_stops: int = Query(..., description="Number of stops"),
    time_of_day: Optional[int] = Query(
        None, ge=0, le=23, description="Hour of day (0-23)"
    ),
    traffic_level: Optional[float] = Query(
        0.5, ge=0, le=1, description="Traffic level 0-1"
    ),
):
    """
    Score a route based on metrics
    """
    try:
        route_data = {
            "distance": distance_km * 1000,  # Convert to meters
            "duration": duration_hours * 3600,  # Convert to seconds
            "duration_in_traffic": duration_hours * 3600 * (1 + traffic_level * 0.5),
        }

        stops_data = [{"priority": 1} for _ in range(num_stops)]

        score = engine.score_route(
            route_data=route_data, stops=stops_data, time_of_day=time_of_day
        )

        return score

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
