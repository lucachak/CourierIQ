
# src/ml/model.py

import numpy as np

class RouteModel:
    """
    Modelo placeholder.
    No futuro você troca por um modelo real.
    """

    def predict_time(self, coords: np.ndarray) -> float:
        """
        Retorna tempo estimado baseado no tamanho da rota
        (placeholder: quanto mais pontos, maior o tempo).
        """
        n = coords.shape[0]
        return n * 3.5  # simulação simples

    def predict_risk(self, coords: np.ndarray) -> float:
        """
        Avalia risco da rota (dummy logic).
        """
        return float(coords.mean())  # placeholder
