from typing import List, Dict
import numpy as np

class Preprocess:

    def normalize_coordinates(self, points: List[Dict]):
        """
        Recebe uma lista de pontos:
        [{"lat": ..., "lng": ...}, ...]
        
        Converte em matriz NumPy normalizada.
        """
        lat = np.array([p["lat"] for p in points])
        lng = np.array([p["lng"] for p in points])

        lat_norm = (lat - lat.min()) / (lat.max() - lat.min() + 1e-9)
        lng_norm = (lng - lng.min()) / (lng.max() - lng.min() + 1e-9)

        coords = np.vstack([lat_norm, lng_norm]).T
        return coords
