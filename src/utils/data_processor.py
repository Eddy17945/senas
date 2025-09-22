# src/utils/data_processor.py

import numpy as np
from typing import List, Tuple

class DataProcessor:
    def __init__(self):
        pass
    
    def normalize_landmarks(self, landmarks: List) -> np.ndarray:
        """Normalizar landmarks"""
        if not landmarks:
            return np.array([])
        
        return np.array(landmarks)
    
    def extract_features(self, landmarks: List) -> List:
        """Extraer características de landmarks"""
        if not landmarks:
            return []
        
        # Características básicas
        features = []
        for i in range(0, len(landmarks), 3):
            if i+2 < len(landmarks):
                features.extend([landmarks[i], landmarks[i+1], landmarks[i+2]])
        
        return features