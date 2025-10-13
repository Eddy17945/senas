# src/detector/syllable_classifier.py
# VERSIÓN MEJORADA - USA GestureClassifier para ambas manos

import numpy as np
import cv2
from typing import List, Optional, Dict, Tuple
from .gesture_classifier import GestureClassifier

class SyllableClassifier:
    def __init__(self):
        # Crear clasificadores separados para cada mano
        self.left_classifier = GestureClassifier()
        self.right_classifier = GestureClassifier()
        
        self.is_trained = True
        self.detection_history = []
        self.stability_threshold = 6  # Reducido de 8 a 6 para más rapidez
        
        # Definir sílabas soportadas (TODAS LAS COMBINACIONES)
        self.supported_syllables = {
            # Con M
            'MA': {'consonant': 'M', 'vowel': 'A'},
            'ME': {'consonant': 'M', 'vowel': 'E'},
            'MI': {'consonant': 'M', 'vowel': 'I'},
            'MO': {'consonant': 'M', 'vowel': 'O'},
            'MU': {'consonant': 'M', 'vowel': 'U'},
            # Con P
            'PA': {'consonant': 'P', 'vowel': 'A'},
            'PE': {'consonant': 'P', 'vowel': 'E'},
            'PI': {'consonant': 'P', 'vowel': 'I'},
            'PO': {'consonant': 'P', 'vowel': 'O'},
            'PU': {'consonant': 'P', 'vowel': 'U'},
            # Con L
            'LA': {'consonant': 'L', 'vowel': 'A'},
            'LE': {'consonant': 'L', 'vowel': 'E'},
            'LI': {'consonant': 'L', 'vowel': 'I'},
            'LO': {'consonant': 'L', 'vowel': 'O'},
            'LU': {'consonant': 'L', 'vowel': 'U'},
            # Con T
            'TA': {'consonant': 'T', 'vowel': 'A'},
            'TE': {'consonant': 'T', 'vowel': 'E'},
            'TI': {'consonant': 'T', 'vowel': 'I'},
            'TO': {'consonant': 'T', 'vowel': 'O'},
            'TU': {'consonant': 'T', 'vowel': 'U'},
            # Con S
            'SA': {'consonant': 'S', 'vowel': 'A'},
            'SE': {'consonant': 'S', 'vowel': 'E'},
            'SI': {'consonant': 'S', 'vowel': 'I'},
            'SO': {'consonant': 'S', 'vowel': 'O'},
            'SU': {'consonant': 'S', 'vowel': 'U'},
            # Con N
            'NA': {'consonant': 'N', 'vowel': 'A'},
            'NE': {'consonant': 'N', 'vowel': 'E'},
            'NI': {'consonant': 'N', 'vowel': 'I'},
            'NO': {'consonant': 'N', 'vowel': 'O'},
            'NU': {'consonant': 'N', 'vowel': 'U'},
            # Con R
            'RA': {'consonant': 'R', 'vowel': 'A'},
            'RE': {'consonant': 'R', 'vowel': 'E'},
            'RI': {'consonant': 'R', 'vowel': 'I'},
            'RO': {'consonant': 'R', 'vowel': 'O'},
            'RU': {'consonant': 'R', 'vowel': 'U'},
            # Con D
            'DA': {'consonant': 'D', 'vowel': 'A'},
            'DE': {'consonant': 'D', 'vowel': 'E'},
            'DI': {'consonant': 'D', 'vowel': 'I'},
            'DO': {'consonant': 'D', 'vowel': 'O'},
            'DU': {'consonant': 'D', 'vowel': 'U'},
        }
        
        # Letras que son consonantes (mano izquierda)
        self.consonants = ['M', 'P', 'L', 'T', 'S', 'N', 'R', 'D', 'B', 'C', 
                          'F', 'G', 'H', 'J', 'K', 'Q', 'V', 'W', 'X', 'Y', 'Z']
        
        # Letras que son vocales (mano derecha)
        self.vowels = ['A', 'E', 'I', 'O', 'U']
        
        # Historial de detecciones por mano
        self.left_hand_history = []
        self.right_hand_history = []
        
    def predict_syllable(self, left_hand_landmarks: Optional[List], 
                        right_hand_landmarks: Optional[List]) -> Optional[str]:
        """
        Predice sílaba usando GestureClassifier en ambas manos
        """
        if not left_hand_landmarks or not right_hand_landmarks:
            return None
        
        # Detectar letra en mano IZQUIERDA (consonante)
        left_letter = self.left_classifier.predict_gesture(left_hand_landmarks)
        
        # Detectar letra en mano DERECHA (vocal)
        right_letter = self.right_classifier.predict_gesture(right_hand_landmarks)
        
        # Agregar a historial
        self.left_hand_history.append(left_letter)
        self.right_hand_history.append(right_letter)
        
        # Mantener historial limitado
        if len(self.left_hand_history) > 10:
            self.left_hand_history = self.left_hand_history[-10:]
        if len(self.right_hand_history) > 10:
            self.right_hand_history = self.right_hand_history[-10:]
        
        # Verificar que sean consonante y vocal
        consonant = None
        vowel = None
        
        if left_letter in self.consonants:
            consonant = left_letter
        if right_letter in self.vowels:
            vowel = right_letter
        
        # Si no hay consonante o vocal, intentar al revés
        if not consonant and right_letter in self.consonants:
            consonant = right_letter
        if not vowel and left_letter in self.vowels:
            vowel = left_letter
        
        # Intentar formar sílaba
        syllable = self._combine_letters_to_syllable(consonant, vowel)
        
        # Aplicar estabilización
        return self._stabilize_detection(syllable)
    
    def _combine_letters_to_syllable(self, consonant: Optional[str], 
                                     vowel: Optional[str]) -> Optional[str]:
        """
        Combina consonante y vocal para formar sílaba
        """
        if not consonant or not vowel:
            return None
        
        syllable = consonant + vowel
        
        # Verificar si la sílaba está soportada
        if syllable in self.supported_syllables:
            return syllable
        
        # Si no está en la lista pero es válida, retornarla igual
        if consonant in self.consonants and vowel in self.vowels:
            return syllable
        
        return None
    
    def _stabilize_detection(self, syllable: Optional[str]) -> Optional[str]:
        """
        Aplica estabilización a la detección de sílabas
        """
        self.detection_history.append(syllable)
        
        # Mantener solo las detecciones recientes
        if len(self.detection_history) > self.stability_threshold * 2:
            self.detection_history = self.detection_history[-self.stability_threshold:]
        
        # Verificar estabilidad
        if len(self.detection_history) >= self.stability_threshold:
            recent_detections = self.detection_history[-self.stability_threshold:]
            
            if syllable and syllable in recent_detections:
                # Contar ocurrencias de la sílaba actual
                syllable_count = sum(1 for detection in recent_detections 
                                   if detection == syllable)
                
                # Requerir 50% de consistencia (más permisivo que antes)
                if syllable_count >= self.stability_threshold * 0.5:
                    return syllable
        
        return None
    
    def get_current_letters(self) -> Dict[str, Optional[str]]:
        """
        Obtiene las últimas letras detectadas en cada mano
        """
        left = self.left_hand_history[-1] if self.left_hand_history else None
        right = self.right_hand_history[-1] if self.right_hand_history else None
        
        return {
            'left': left if left in self.consonants else None,
            'right': right if right in self.vowels else None
        }
    
    def get_supported_syllables(self) -> List[str]:
        """
        Retorna lista de sílabas soportadas
        """
        return list(self.supported_syllables.keys())
    
    def get_detection_confidence(self) -> float:
        """
        Calcula confianza de la detección actual
        """
        if len(self.detection_history) < self.stability_threshold:
            return 0.0
        
        recent = self.detection_history[-self.stability_threshold:]
        if not recent[-1]:
            return 0.0
        
        consistent_count = sum(1 for detection in recent if detection == recent[-1])
        confidence = consistent_count / len(recent)
        
        return confidence
    
    def get_hand_confidences(self) -> Dict[str, float]:
        """
        Obtiene confianza de detección por cada mano
        """
        return {
            'left': self.left_classifier.get_detection_confidence(),
            'right': self.right_classifier.get_detection_confidence()
        }
    
    def reset_detection_history(self):
        """
        Reinicia el historial de detecciones
        """
        self.detection_history = []
        self.left_hand_history = []
        self.right_hand_history = []
        self.left_classifier.reset_detection_history()
        self.right_classifier.reset_detection_history()
    
    def set_stability_threshold(self, threshold: int):
        """
        Cambia el umbral de estabilidad
        """
        self.stability_threshold = max(1, min(15, threshold))