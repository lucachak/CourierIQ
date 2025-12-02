from typing import List, Dict
from src.ml.preprocessor import Preprocess
from src.ml.model import RouteModel
from src.ml.utils import haversine

class RouteEngine:

    def __init__(self):
        self.pre = Preprocess()
        self.model = RouteModel()

    def compute_distance(self, points: List[Dict]):
        """
        Soma todas as distâncias entre pontos consecutivos.
        """
        total = 0.0
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            total += haversine(p1["lat"], p1["lng"], p2["lat"], p2["lng"])
        return total

    def compute_route(self, points: List[Dict]):
        """
        Processa a rota:
        - Normaliza coordenadas
        - Usa ML para prever tempo/risco
        - Calcula distância
        - Retorna relatório
        """

        coords = self.pre.normalize_coordinates(points)

        predicted_time = self.model.predict_time(coords)
        predicted_risk = self.model.predict_risk(coords)
        distance_km = self.compute_distance(points)

        return {
            "distance_km": distance_km,
            "predicted_time_min": predicted_time,
            "predicted_risk": predicted_risk,
            "points": points
        }
