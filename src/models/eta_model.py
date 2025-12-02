"""
ETA Prediction Model - Complete implementation
Supports multiple model types: RandomForest, GradientBoosting, LightGBM, Neural Network

Place this file in: src/models/eta_model.py
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ETAPredictor:
    """
    Complete ETA prediction system with feature engineering and multiple model support
    """

    def __init__(self, model_type: str = "gradient_boosting"):
        """
        Args:
            model_type: One of ['random_forest', 'gradient_boosting', 'lightgbm', 'neural_net']
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.training_stats = {}

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate haversine distance in km"""
        R = 6371  # Earth radius in km

        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw delivery data

        Expected columns in input data:
        - origin_lat, origin_lon: Starting point
        - dest_lat, dest_lon: Destination point
        - timestamp: Unix timestamp or datetime
        - num_stops: Number of stops (optional)
        - traffic_level: 0-1 scale (optional)
        - weather: categorical (optional)
        - preparation_time: Restaurant prep time in seconds (optional)
        """
        features = pd.DataFrame()

        # Distance features
        features["haversine_distance"] = data.apply(
            lambda row: self._haversine_distance(
                row["origin_lat"], row["origin_lon"], row["dest_lat"], row["dest_lon"]
            ),
            axis=1,
        )

        # Coordinate deltas
        features["lat_delta"] = np.abs(data["dest_lat"] - data["origin_lat"])
        features["lon_delta"] = np.abs(data["dest_lon"] - data["origin_lon"])

        # Coordinate statistics
        features["lat_mean"] = (data["origin_lat"] + data["dest_lat"]) / 2
        features["lon_mean"] = (data["origin_lon"] + data["dest_lon"]) / 2

        # Time features (if timestamp available)
        if "timestamp" in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                data["timestamp"] = pd.to_datetime(data["timestamp"])

            features["hour"] = data["timestamp"].dt.hour
            features["day_of_week"] = data["timestamp"].dt.dayofweek
            features["is_weekend"] = (data["timestamp"].dt.dayofweek >= 5).astype(int)
            features["is_rush_hour"] = (
                (data["timestamp"].dt.hour.between(7, 9))
                | (data["timestamp"].dt.hour.between(17, 19))
            ).astype(int)

            # Cyclical encoding for hour and day
            features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
            features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
            features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
            features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

        # Number of stops (if available)
        if "num_stops" in data.columns:
            features["num_stops"] = data["num_stops"]
            features["distance_per_stop"] = features["haversine_distance"] / (
                data["num_stops"] + 1
            )

        # Traffic features (if available)
        if "traffic_level" in data.columns:
            features["traffic_level"] = data["traffic_level"]
            features["traffic_distance_interaction"] = (
                features["haversine_distance"] * data["traffic_level"]
            )

        # Weather features (if available)
        if "weather" in data.columns:
            weather_dummies = pd.get_dummies(data["weather"], prefix="weather")
            features = pd.concat([features, weather_dummies], axis=1)

        # Preparation time (if available)
        if "preparation_time" in data.columns:
            features["preparation_time"] = data["preparation_time"]

        self.feature_names = features.columns.tolist()
        return features

    def create_model(self):
        """Create model based on model_type"""
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )

        elif self.model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor

            self.model = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            )

        elif self.model_type == "lightgbm":
            try:
                import lightgbm as lgb

                self.model = lgb.LGBMRegressor(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=7,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1,
                )
            except ImportError:
                logger.error(
                    "LightGBM not installed. Install with: pip install lightgbm"
                )
                raise

        elif self.model_type == "neural_net":
            from sklearn.neural_network import MLPRegressor

            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation="relu",
                solver="adam",
                learning_rate="adaptive",
                max_iter=500,
                random_state=42,
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        data: pd.DataFrame,
        target_column: str = "eta_seconds",
        test_size: float = 0.2,
        validate: bool = True,
    ) -> Dict[str, float]:
        """
        Train the ETA prediction model

        Args:
            data: Training data with features and target
            target_column: Name of the target column (ETA in seconds)
            test_size: Fraction of data to use for testing
            validate: Whether to perform validation

        Returns:
            Dictionary with training metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        logger.info(f"Training {self.model_type} model with {len(data)} samples")

        # Engineer features
        X = self.engineer_features(data)
        y = data[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create and train model
        self.create_model()
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        metrics = {}
        if validate:
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)

            metrics = {
                "train_mae": mean_absolute_error(y_train, y_pred_train),
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "train_r2": r2_score(y_train, y_pred_train),
                "test_mae": mean_absolute_error(y_test, y_pred_test),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "test_r2": r2_score(y_test, y_pred_test),
                "test_mape": np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100,
            }

            logger.info(f"Training complete:")
            logger.info(f"  Test MAE: {metrics['test_mae']:.2f} seconds")
            logger.info(f"  Test RMSE: {metrics['test_rmse']:.2f} seconds")
            logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
            logger.info(f"  Test MAPE: {metrics['test_mape']:.2f}%")

        self.training_stats = {
            "model_type": self.model_type,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "trained_at": datetime.now().isoformat(),
            "metrics": metrics,
        }

        return metrics

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict ETA for new deliveries

        Args:
            data: DataFrame with same columns as training data

        Returns:
            Array of predicted ETAs in seconds
        """
        if self.model is None:
            raise ValueError(
                "Model not trained. Call train() first or load a saved model."
            )

        # Engineer features
        X = self.engineer_features(data)

        # Ensure all expected features are present
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Filling with zeros.")
            for feat in missing_features:
                X[feat] = 0

        # Reorder columns to match training
        X = X[self.feature_names]

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        return predictions

    def predict_single(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        timestamp: Optional[datetime] = None,
        **kwargs,
    ) -> float:
        """
        Predict ETA for a single delivery

        Args:
            origin_lat, origin_lon: Starting coordinates
            dest_lat, dest_lon: Destination coordinates
            timestamp: Delivery timestamp (default: now)
            **kwargs: Additional features (num_stops, traffic_level, etc.)

        Returns:
            Predicted ETA in seconds
        """
        if timestamp is None:
            timestamp = datetime.now()

        data = pd.DataFrame(
            [
                {
                    "origin_lat": origin_lat,
                    "origin_lon": origin_lon,
                    "dest_lat": dest_lat,
                    "dest_lon": dest_lon,
                    "timestamp": timestamp,
                    **kwargs,
                }
            ]
        )

        prediction = self.predict(data)[0]
        return float(prediction)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained")

        if hasattr(self.model, "feature_importances_"):
            importance = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            )
            return importance.sort_values("importance", ascending=False)
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()

    def save(self, filepath: str):
        """Save model to disk"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "training_stats": self.training_stats,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        # Also save training stats as JSON
        stats_path = filepath.replace(".pkl", "_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.training_stats, f, indent=2)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ETAPredictor":
        """Load model from disk"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        predictor = cls(model_type=model_data["model_type"])
        predictor.model = model_data["model"]
        predictor.scaler = model_data["scaler"]
        predictor.feature_names = model_data["feature_names"]
        predictor.training_stats = model_data.get("training_stats", {})

        logger.info(f"Model loaded from {filepath}")
        return predictor


# Utility function for generating synthetic training data
def generate_synthetic_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic delivery data for testing
    """
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "origin_lat": np.random.uniform(40.0, 41.0, n_samples),
            "origin_lon": np.random.uniform(-74.5, -73.5, n_samples),
            "dest_lat": np.random.uniform(40.0, 41.0, n_samples),
            "dest_lon": np.random.uniform(-74.5, -73.5, n_samples),
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1H"),
            "num_stops": np.random.randint(1, 10, n_samples),
            "traffic_level": np.random.uniform(0.2, 0.9, n_samples),
            "preparation_time": np.random.uniform(300, 1800, n_samples),
        }
    )

    # Generate realistic ETA (base distance + traffic + prep time + noise)
    predictor = ETAPredictor()
    distances = data.apply(
        lambda row: predictor._haversine_distance(
            row["origin_lat"], row["origin_lon"], row["dest_lat"], row["dest_lon"]
        ),
        axis=1,
    )

    # ETA formula: distance/speed + prep + traffic penalty + noise
    base_speed = 30  # km/h
    data["eta_seconds"] = (
        (distances / base_speed) * 3600  # Travel time
        + data["preparation_time"]  # Prep time
        + data["num_stops"] * 120  # Stop time
        + distances * data["traffic_level"] * 100  # Traffic penalty
        + np.random.normal(0, 120, n_samples)  # Noise
    )

    data["eta_seconds"] = data["eta_seconds"].clip(lower=300)  # Minimum 5 min

    return data
