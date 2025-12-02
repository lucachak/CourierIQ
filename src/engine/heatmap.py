"""
heatmap.py - Geospatial analysis and heatmap generation
Identifies high-efficiency delivery zones and patterns
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DeliveryZone:
    """Represents a geographic delivery zone with statistics"""

    center: Tuple[float, float]  # (lat, lon)
    radius: float  # meters
    num_deliveries: int = 0
    avg_delivery_time: float = 0.0
    success_rate: float = 1.0
    avg_distance_between_stops: float = 0.0
    density_score: float = 0.0  # Higher = more deliveries per km²
    efficiency_score: float = 0.0  # Overall efficiency rating


class HeatmapGenerator:
    """Generates delivery efficiency heatmaps and zone analysis"""

    def __init__(self, grid_size_km: float = 1.0):
        """
        Args:
            grid_size_km: Size of grid cells for heatmap (km)
        """
        self.grid_size_km = grid_size_km
        self.grid_size_deg = grid_size_km / 111  # Approximate km to degrees

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in meters"""
        R = 6371000  # Earth radius in meters

        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def _get_grid_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """Get grid cell indices for a coordinate"""
        cell_lat = int(lat / self.grid_size_deg)
        cell_lon = int(lon / self.grid_size_deg)
        return (cell_lat, cell_lon)

    def _get_cell_center(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        """Get center coordinates of a grid cell"""
        lat = (cell[0] + 0.5) * self.grid_size_deg
        lon = (cell[1] + 0.5) * self.grid_size_deg
        return (lat, lon)

    def generate_delivery_heatmap(
        self, deliveries: List[Dict]
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Generate heatmap of delivery frequency and metrics

        Args:
            deliveries: List of delivery dicts with 'lat', 'lon', 'delivery_time', etc.

        Returns:
            Dict mapping grid cells to aggregated statistics
        """
        heatmap = defaultdict(
            lambda: {
                "count": 0,
                "total_time": 0.0,
                "total_distance": 0.0,
                "successful": 0,
                "failed": 0,
                "coordinates": [],
            }
        )

        for delivery in deliveries:
            lat = delivery.get("lat")
            lon = delivery.get("lon")

            if lat is None or lon is None:
                continue

            cell = self._get_grid_cell(lat, lon)
            cell_data = heatmap[cell]

            cell_data["count"] += 1
            cell_data["total_time"] += delivery.get("delivery_time", 0)
            cell_data["total_distance"] += delivery.get("distance", 0)
            cell_data["coordinates"].append((lat, lon))

            if delivery.get("status") == "success":
                cell_data["successful"] += 1
            else:
                cell_data["failed"] += 1

        # Calculate averages and scores
        for cell, data in heatmap.items():
            count = data["count"]

            # Average metrics
            data["avg_time"] = data["total_time"] / count if count > 0 else 0
            data["avg_distance"] = data["total_distance"] / count if count > 0 else 0
            data["success_rate"] = data["successful"] / count if count > 0 else 0.0

            # Calculate density score (deliveries per km²)
            cell_area = self.grid_size_km**2
            data["density_score"] = count / cell_area

            # Calculate efficiency score (combination of metrics)
            # Lower time = better, higher success rate = better, higher density = better
            avg_time_score = 1.0 / (data["avg_time"] + 1) if data["avg_time"] > 0 else 0
            success_score = data["success_rate"]
            density_score = min(data["density_score"] / 10, 1.0)  # Normalize to 0-1

            data["efficiency_score"] = (
                avg_time_score * 0.3 + success_score * 0.4 + density_score * 0.3
            )

        return dict(heatmap)

    def identify_high_efficiency_zones(
        self,
        heatmap: Dict[Tuple[int, int], Dict],
        min_deliveries: int = 10,
        top_n: int = 10,
    ) -> List[DeliveryZone]:
        """
        Identify the most efficient delivery zones from heatmap

        Args:
            heatmap: Output from generate_delivery_heatmap
            min_deliveries: Minimum deliveries required to consider a zone
            top_n: Number of top zones to return

        Returns:
            List of DeliveryZone objects, sorted by efficiency
        """
        zones = []

        for cell, data in heatmap.items():
            if data["count"] < min_deliveries:
                continue

            center = self._get_cell_center(cell)
            radius = (
                self.grid_size_km * 1000 / 2
            )  # Convert to meters, radius = half of cell size

            # Calculate avg distance between stops in this zone
            coords = data["coordinates"]
            avg_distance = 0.0
            if len(coords) > 1:
                distances = []
                # Sample pairs to avoid O(n²) for large datasets
                sample_size = min(len(coords), 50)
                sampled_coords = coords[:sample_size]

                for i in range(len(sampled_coords) - 1):
                    for j in range(i + 1, len(sampled_coords)):
                        dist = self._haversine_distance(
                            sampled_coords[i][0],
                            sampled_coords[i][1],
                            sampled_coords[j][0],
                            sampled_coords[j][1],
                        )
                        distances.append(dist)
                avg_distance = np.mean(distances) if distances else 0.0

            zone = DeliveryZone(
                center=center,
                radius=radius,
                num_deliveries=data["count"],
                avg_delivery_time=data["avg_time"],
                success_rate=data["success_rate"],
                avg_distance_between_stops=avg_distance,
                density_score=data["density_score"],
                efficiency_score=data["efficiency_score"],
            )
            zones.append(zone)

        # Sort by efficiency score (descending)
        zones.sort(key=lambda z: z.efficiency_score, reverse=True)

        return zones[:top_n]

    def generate_time_based_heatmap(
        self,
        deliveries: List[Dict],
        hour_ranges: Optional[List[Tuple[str, int, int]]] = None,
    ) -> Dict[str, Dict]:
        """
        Generate heatmaps segmented by time of day

        Args:
            deliveries: List of deliveries with 'timestamp' or 'hour' field
            hour_ranges: List of (name, start_hour, end_hour) tuples
                        Default: [morning, afternoon, evening, night]

        Returns:
            Dict mapping time period names to heatmap data
        """
        if hour_ranges is None:
            hour_ranges = [
                ("morning", 6, 12),
                ("afternoon", 12, 18),
                ("evening", 18, 22),
                ("night", 22, 6),
            ]

        time_based_heatmaps = {}

        for period_name, start_hour, end_hour in hour_ranges:
            # Filter deliveries by time
            period_deliveries = []
            for delivery in deliveries:
                hour = delivery.get("hour")

                # Try to extract hour from timestamp if not provided
                if hour is None and "timestamp" in delivery:
                    try:
                        from datetime import datetime

                        timestamp = delivery["timestamp"]
                        if isinstance(timestamp, (int, float)):
                            dt = datetime.fromtimestamp(timestamp)
                        else:
                            dt = datetime.fromisoformat(str(timestamp))
                        hour = dt.hour
                    except Exception as e:
                        logger.warning(f"Could not extract hour from timestamp: {e}")
                        continue

                if hour is not None:
                    # Handle ranges that wrap around midnight
                    if start_hour < end_hour:
                        if start_hour <= hour < end_hour:
                            period_deliveries.append(delivery)
                    else:  # Wraps around midnight (e.g., 22-6)
                        if hour >= start_hour or hour < end_hour:
                            period_deliveries.append(delivery)

            # Generate heatmap for this period
            if period_deliveries:
                heatmap = self.generate_delivery_heatmap(period_deliveries)
                time_based_heatmaps[period_name] = heatmap
            else:
                time_based_heatmaps[period_name] = {}

        return time_based_heatmaps

    def calculate_route_coverage(
        self,
        route_coordinates: List[Tuple[float, float]],
        heatmap: Dict[Tuple[int, int], Dict],
    ) -> Dict[str, any]:
        """
        Calculate how well a route covers high-efficiency zones

        Args:
            route_coordinates: List of (lat, lon) tuples for the route
            heatmap: Output from generate_delivery_heatmap

        Returns:
            Coverage statistics
        """
        cells_covered = set()
        efficiency_scores = []

        for lat, lon in route_coordinates:
            cell = self._get_grid_cell(lat, lon)
            cells_covered.add(cell)

            if cell in heatmap:
                efficiency_scores.append(heatmap[cell]["efficiency_score"])

        avg_eff = np.mean(efficiency_scores) if efficiency_scores else 0.0

        return {
            "num_cells_covered": len(cells_covered),
            "avg_efficiency": float(avg_eff),
            "total_efficiency": float(sum(efficiency_scores)),
            "coverage_quality": (
                "high" if avg_eff > 0.7 else "medium" if avg_eff > 0.4 else "low"
            ),
        }

    def export_heatmap_geojson(
        self, heatmap: Dict[Tuple[int, int], Dict], output_path: str
    ):
        """
        Export heatmap to GeoJSON format for visualization

        Args:
            heatmap: Output from generate_delivery_heatmap
            output_path: Path to save GeoJSON file
        """
        features = []

        for cell, data in heatmap.items():
            center = self._get_cell_center(cell)

            # Create square polygon for the cell
            half_size = self.grid_size_deg / 2
            coords = [
                [center[1] - half_size, center[0] - half_size],  # SW [lon, lat]
                [center[1] + half_size, center[0] - half_size],  # SE
                [center[1] + half_size, center[0] + half_size],  # NE
                [center[1] - half_size, center[0] + half_size],  # NW
                [center[1] - half_size, center[0] - half_size],  # Close polygon
            ]

            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "cell": f"{cell[0]},{cell[1]}",
                    "center_lat": center[0],
                    "center_lon": center[1],
                    "count": data["count"],
                    "avg_time": float(data["avg_time"]),
                    "success_rate": float(data["success_rate"]),
                    "density_score": float(data["density_score"]),
                    "efficiency_score": float(data["efficiency_score"]),
                },
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}

        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"Exported heatmap with {len(features)} cells to {output_path}")

    def generate_plotly_heatmap(
        self, heatmap: Dict[Tuple[int, int], Dict], metric: str = "efficiency_score"
    ):
        """
        Generate interactive Plotly heatmap visualization

        Args:
            heatmap: Output from generate_delivery_heatmap
            metric: Which metric to visualize

        Returns:
            Plotly figure object (requires plotly to be installed)
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error("Plotly not installed. Install with: pip install plotly")
            return None

        # Extract data for plotting
        lats = []
        lons = []
        values = []
        hover_texts = []

        for cell, data in heatmap.items():
            center = self._get_cell_center(cell)
            lats.append(center[0])
            lons.append(center[1])
            values.append(data.get(metric, 0))

            hover_text = f"""
            <b>Cell: {cell}</b><br>
            Deliveries: {data["count"]}<br>
            Avg Time: {data["avg_time"]:.1f}s<br>
            Success Rate: {data["success_rate"]:.2%}<br>
            Density: {data["density_score"]:.2f}/km²<br>
            Efficiency: {data["efficiency_score"]:.2f}
            """
            hover_texts.append(hover_text)

        # Create density mapbox
        fig = go.Figure(
            data=go.Densitymapbox(
                lat=lats,
                lon=lons,
                z=values,
                radius=20,
                colorscale="Viridis",
                hovertext=hover_texts,
                hoverinfo="text",
                showscale=True,
                colorbar=dict(title=metric.replace("_", " ").title()),
            )
        )

        # Calculate map center
        center_lat = np.mean(lats) if lats else 0
        center_lon = np.mean(lons) if lons else 0

        fig.update_layout(
            title=f"Delivery Heatmap - {metric.replace('_', ' ').title()}",
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11,
            ),
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
        )

        return fig

    def generate_matplotlib_heatmap(
        self, heatmap: Dict[Tuple[int, int], Dict], metric: str = "efficiency_score"
    ):
        """
        Generate static matplotlib heatmap

        Args:
            heatmap: Output from generate_delivery_heatmap
            metric: Which metric to visualize

        Returns:
            matplotlib figure
        """
        try:
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib not installed")
            return None

        fig, ax = plt.subplots(figsize=(12, 10))

        # Collect data
        cells = []
        values = []

        for cell, data in heatmap.items():
            cells.append(cell)
            values.append(data.get(metric, 0))

        if not cells:
            logger.warning("No data to plot")
            return fig

        # Normalize values for coloring
        values = np.array(values)
        norm_values = (values - values.min()) / (values.max() - values.min() + 1e-9)

        # Get colormap
        cmap = plt.cm.YlOrRd

        # Draw cells
        for cell, norm_val in zip(cells, norm_values):
            center = self._get_cell_center(cell)
            half_size = self.grid_size_deg / 2

            rect = patches.Rectangle(
                (center[1] - half_size, center[0] - half_size),
                self.grid_size_deg,
                self.grid_size_deg,
                facecolor=cmap(norm_val),
                edgecolor="gray",
                alpha=0.7,
                linewidth=0.5,
            )
            ax.add_patch(rect)

        # Set axis properties
        lats = [self._get_cell_center(c)[0] for c in cells]
        lons = [self._get_cell_center(c)[1] for c in cells]

        ax.set_xlim(min(lons) - self.grid_size_deg, max(lons) + self.grid_size_deg)
        ax.set_ylim(min(lats) - self.grid_size_deg, max(lats) + self.grid_size_deg)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Delivery Heatmap - {metric.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)

        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=values.min(), vmax=values.max())
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(metric.replace("_", " ").title())

        plt.tight_layout()
        return fig

    def get_zone_recommendations(
        self, heatmap: Dict[Tuple[int, int], Dict], target_zones: int = 5
    ) -> List[Dict]:
        """
        Get recommendations for optimal delivery zones

        Args:
            heatmap: Output from generate_delivery_heatmap
            target_zones: Number of zones to recommend

        Returns:
            List of zone recommendations with reasoning
        """
        zones = self.identify_high_efficiency_zones(
            heatmap, min_deliveries=5, top_n=target_zones
        )

        recommendations = []
        for i, zone in enumerate(zones, 1):
            recommendation = {
                "rank": i,
                "center": {"lat": zone.center[0], "lon": zone.center[1]},
                "metrics": {
                    "efficiency_score": round(zone.efficiency_score, 3),
                    "num_deliveries": zone.num_deliveries,
                    "success_rate": round(zone.success_rate, 3),
                    "density": round(zone.density_score, 2),
                },
                "reasoning": self._generate_zone_reasoning(zone),
            }
            recommendations.append(recommendation)

        return recommendations

    def _generate_zone_reasoning(self, zone: DeliveryZone) -> str:
        """Generate human-readable reasoning for zone recommendation"""
        reasons = []

        if zone.efficiency_score > 0.8:
            reasons.append("Exceptionally high efficiency")
        elif zone.efficiency_score > 0.6:
            reasons.append("Good efficiency rating")

        if zone.density_score > 10:
            reasons.append("High delivery density")
        elif zone.density_score > 5:
            reasons.append("Moderate delivery density")

        if zone.success_rate > 0.95:
            reasons.append("Excellent success rate")
        elif zone.success_rate > 0.85:
            reasons.append("Good success rate")

        if zone.avg_delivery_time < 900:  # 15 minutes
            reasons.append("Fast delivery times")

        return ", ".join(reasons) if reasons else "Stable performance zone"


# Example usage
if __name__ == "__main__":
    # Create sample data
    import random
    from datetime import datetime, timedelta

    deliveries = []
    base_time = datetime.now()

    for i in range(200):
        deliveries.append(
            {
                "lat": 40.7128 + random.uniform(-0.05, 0.05),
                "lon": -74.0060 + random.uniform(-0.05, 0.05),
                "delivery_time": random.uniform(600, 1800),
                "distance": random.uniform(1000, 5000),
                "status": "success" if random.random() > 0.1 else "failed",
                "timestamp": (
                    base_time + timedelta(hours=random.randint(0, 23))
                ).timestamp(),
                "hour": random.randint(0, 23),
            }
        )

    # Generate heatmap
    generator = HeatmapGenerator(grid_size_km=1.0)
    heatmap = generator.generate_delivery_heatmap(deliveries)

    print(f"Generated heatmap with {len(heatmap)} cells")

    # Identify top zones
    zones = generator.identify_high_efficiency_zones(heatmap, top_n=5)
    print(f"\nTop {len(zones)} zones:")
    for i, zone in enumerate(zones, 1):
        print(
            f"{i}. Efficiency: {zone.efficiency_score:.3f}, Deliveries: {zone.num_deliveries}"
        )

    # Get recommendations
    recommendations = generator.get_zone_recommendations(heatmap, target_zones=3)
    print("\nZone Recommendations:")
    for rec in recommendations:
        print(f"Rank {rec['rank']}: {rec['reasoning']}")

    # Export to GeoJSON
    generator.export_heatmap_geojson(heatmap, "delivery_heatmap.geojson")
    print("\n✅ Exported heatmap to delivery_heatmap.geojson")
