from typing import List, Optional

from fastapi import APIRouter, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/routes", tags=["Routes"])


class Coordinate(BaseModel):
    lat: float
    lon: float


class RouteRequest(BaseModel):
    origin: Coordinate
    destinations: List[Coordinate]


class RouteSegment(BaseModel):
    start: Coordinate
    end: Coordinate
    distance_km: float
    eta_minutes: float


class RouteResponse(BaseModel):
    total_distance_km: float
    total_eta_minutes: float
    segments: List[RouteSegment]


@router.post("/optimize", response_model=RouteResponse)
async def optimize_route(request: RouteRequest):
    """
    Generates an optimized delivery route (mock).
    Later this will call `engine/optimizer.py`
    """
    segments = []

    for idx, dst in enumerate(request.destinations):
        seg = RouteSegment(
            start=request.origin if idx == 0 else request.destinations[idx - 1],
            end=dst,
            distance_km=1.2 * (idx + 1),
            eta_minutes=4.5 * (idx + 1),
        )
        segments.append(seg)

    return RouteResponse(
        total_distance_km=sum(s.distance_km for s in segments),
        total_eta_minutes=sum(s.eta_minutes for s in segments),
        segments=segments,
    )


@router.get("/eta")
async def get_eta(
    origin_lat: float = Query(...),
    origin_lon: float = Query(...),
    dest_lat: float = Query(...),
    dest_lon: float = Query(...),
):
    """
    Returns a mock ETA (in minutes).
    In the future this calls `models/eta_regressor.pkl`
    """

    eta = abs(dest_lat - origin_lat) * 10 + abs(dest_lon - origin_lon) * 8
    return {"eta_minutes": round(eta, 2)}


@router.get("/heatmap")
async def heatmap():
    return {
        "clusters": [
            {"lat": 47.49, "lon": 19.04, "intensity": 0.85},
            {"lat": 47.50, "lon": 19.02, "intensity": 0.63},
        ]
    }


@router.get("/preview")
async def preview_route():
    """
    Returns a mock preview link (later → real static map or canvas).
    """
    return {"image_url": "https://example.com/mock-route-preview.png"}


@router.get("/routes/{route_id}", tags=["Routes"])
async def get_route(route_id: int = Path(...)):
    return {"route_id": route_id, "status": "mock"}


@router.get("/routes/history", tags=["Routes"])
async def get_route_history():
    return {"history": []}
