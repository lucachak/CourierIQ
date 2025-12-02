"""
Helper utilities for CourierIQ
Place in: src/utils/helpers.py
"""

import json
import logging
import math

# datetime/timedelta are not required in this module
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float, unit: str = "km"
) -> float:
    """
    Calculate haversine distance between two points

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        unit: 'km' or 'miles'

    Returns:
        Distance in specified unit
    """
    R = 6371 if unit == "km" else 3959  # Earth radius

    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing between two points

    Returns:
        Bearing in degrees (0-360)
    """
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        dlon
    )

    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def calculate_bounding_box(
    coordinates: List[Tuple[float, float]], padding: float = 0.01
) -> Dict[str, float]:
    """
    Calculate bounding box for a set of coordinates

    Args:
        coordinates: List of (lat, lon) tuples
        padding: Extra padding around box (degrees)

    Returns:
        Dict with 'min_lat', 'max_lat', 'min_lon', 'max_lon'
    """
    if not coordinates:
        return {"min_lat": 0, "max_lat": 0, "min_lon": 0, "max_lon": 0}

    lats = [coord[0] for coord in coordinates]
    lons = [coord[1] for coord in coordinates]

    return {
        "min_lat": min(lats) - padding,
        "max_lat": max(lats) + padding,
        "min_lon": min(lons) - padding,
        "max_lon": max(lons) + padding,
    }


def point_in_polygon(
    point: Tuple[float, float], polygon: List[Tuple[float, float]]
) -> bool:
    """
    Check if a point is inside a polygon using ray casting

    Args:
        point: (lat, lon) tuple
        polygon: List of (lat, lon) tuples forming polygon

    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    xinters = None
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    # Only evaluate x <= xinters when xinters was computed;
                    # if the edge is vertical (p1x == p2x) use that test instead.
                    if p1x == p2x or (xinters is not None and x <= xinters):
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string

    Examples:
        format_duration(65) -> "1m 5s"
        format_duration(3661) -> "1h 1m"
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"


def format_distance(meters: float, unit: str = "auto") -> str:
    """
    Format distance in meters to human-readable string

    Args:
        meters: Distance in meters
        unit: 'auto', 'metric', or 'imperial'

    Examples:
        format_distance(500) -> "500m"
        format_distance(1500) -> "1.5km"
    """
    if unit == "imperial":
        feet = meters * 3.28084
        if feet < 5280:
            return f"{int(feet)}ft"
        miles = feet / 5280
        return f"{miles:.1f}mi"
    else:  # metric or auto
        if meters < 1000:
            return f"{int(meters)}m"
        km = meters / 1000
        return f"{km:.1f}km"


def estimate_fuel_cost(
    distance_km: float,
    fuel_efficiency_l_per_100km: float = 8.0,
    fuel_price_per_liter: float = 1.5,
) -> float:
    """
    Estimate fuel cost for a distance

    Args:
        distance_km: Distance in kilometers
        fuel_efficiency_l_per_100km: Liters per 100km
        fuel_price_per_liter: Price per liter

    Returns:
        Estimated cost in currency units
    """
    liters_needed = (distance_km / 100) * fuel_efficiency_l_per_100km
    return liters_needed * fuel_price_per_liter


def calculate_carbon_emissions(distance_km: float, vehicle_type: str = "car") -> float:
    """
    Calculate estimated CO2 emissions

    Args:
        distance_km: Distance in kilometers
        vehicle_type: 'car', 'van', 'truck', 'bike', 'ebike'

    Returns:
        CO2 emissions in kg
    """
    # Emissions in kg CO2 per km
    emissions_factors = {
        "car": 0.171,
        "van": 0.256,
        "truck": 0.391,
        "bike": 0.0,
        "ebike": 0.022,
        "scooter": 0.025,
    }

    factor = emissions_factors.get(vehicle_type, emissions_factors["car"])
    return distance_km * factor


def get_time_of_day_category(hour: int) -> str:
    """
    Categorize hour into time of day

    Args:
        hour: Hour (0-23)

    Returns:
        Category: 'early_morning', 'morning', 'afternoon', 'evening', 'night'
    """
    if 0 <= hour < 6:
        return "early_morning"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 22:
        return "evening"
    else:
        return "night"


def is_rush_hour(hour: int, day_of_week: int) -> bool:
    """
    Check if given time is rush hour

    Args:
        hour: Hour (0-23)
        day_of_week: Day (0=Monday, 6=Sunday)

    Returns:
        True if rush hour
    """
    if day_of_week >= 5:  # Weekend
        return False

    return (7 <= hour <= 9) or (17 <= hour <= 19)


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude

    Returns:
        True if valid
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def calculate_route_polyline(stops: List[Dict]) -> List[Tuple[float, float]]:
    """
    Create simple polyline from stops

    Args:
        stops: List of stop dicts with 'lat', 'lon'

    Returns:
        List of (lat, lon) coordinates
    """
    return [(stop["lat"], stop["lon"]) for stop in stops]


def simplify_polyline(
    polyline: List[Tuple[float, float]], tolerance: float = 0.001
) -> List[Tuple[float, float]]:
    """
    Simplify polyline using Ramer-Douglas-Peucker algorithm

    Args:
        polyline: List of (lat, lon) coordinates
        tolerance: Simplification tolerance

    Returns:
        Simplified polyline
    """
    if len(polyline) < 3:
        return polyline

    def perpendicular_distance(point, line_start, line_end):
        if line_start == line_end:
            return haversine_distance(point[0], point[1], line_start[0], line_start[1])

        # Calculate perpendicular distance
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        return num / den if den > 0 else 0

    def rdp(points, epsilon):
        dmax = 0
        index = 0

        for i in range(1, len(points) - 1):
            d = perpendicular_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            rec1 = rdp(points[: index + 1], epsilon)
            rec2 = rdp(points[index:], epsilon)
            return rec1[:-1] + rec2
        else:
            return [points[0], points[-1]]

    return rdp(polyline, tolerance)


def cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from arguments

    Returns:
        String cache key
    """
    import hashlib

    key_str = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)

    return hashlib.md5(key_str.encode()).hexdigest()


def retry_on_failure(func, max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure

    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
    """

    def wrapper(*args, **kwargs):
        import time

        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(delay)

    return wrapper


def batch_process(
    items: List[Any], batch_size: int, process_func: callable
) -> List[Any]:
    """
    Process items in batches

    Args:
        items: List of items to process
        batch_size: Size of each batch
        process_func: Function to process each batch

    Returns:
        List of results
    """
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_result = process_func(batch)
        results.extend(
            batch_result if isinstance(batch_result, list) else [batch_result]
        )

    return results


# Example usage
if __name__ == "__main__":
    # Test haversine
    dist = haversine_distance(40.7128, -74.0060, 40.7589, -73.9851)
    print(f"Distance: {format_distance(dist * 1000)}")

    # Test duration formatting
    print(f"Duration: {format_duration(3725)}")

    # Test bearing
    bearing = calculate_bearing(40.7128, -74.0060, 40.7589, -73.9851)
    print(f"Bearing: {bearing:.1f}°")
