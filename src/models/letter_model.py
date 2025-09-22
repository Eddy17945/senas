# src/models/letter_model.py

import numpy as np
from typing import List, Optional

class LetterModel:
    def __init__(self):
        self.trained = False
        self.letters = ['A', 'B', 'C']
    
    def predict(self, features: List) -> Optional[str]:
        """Predecir letra basada en características"""
        # Implementación simple
        if not features:
            return None
        
        # Por ahora retorna una letra aleatoria para pruebas
        return np.random.choice(self.letters)
    
    def train(self, data, labels):
        """Entrenar el modelo"""
        self.trained = True
        print("Modelo entrenado")