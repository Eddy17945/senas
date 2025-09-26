# src/detector/syllable_classifier.py

import numpy as np
import mediapipe as mp
import cv2
from typing import List, Optional, Dict, Tuple, Union

class SyllableClassifier:
    def __init__(self):
        self.is_trained = False
        self.detection_history = []
        self.stability_threshold = 8
        
        # Definir sílabas soportadas con vocales
        self.supported_syllables = {
            'MA': {'consonant': 'M', 'vowel': 'A'},
            'ME': {'consonant': 'M', 'vowel': 'E'},
            'MI': {'consonant': 'M', 'vowel': 'I'},
            'MO': {'consonant': 'M', 'vowel': 'O'},
            'MU': {'consonant': 'M', 'vowel': 'U'},
            'PA': {'consonant': 'P', 'vowel': 'A'},
            'PE': {'consonant': 'P', 'vowel': 'E'},
            'PI': {'consonant': 'P', 'vowel': 'I'},
            'PO': {'consonant': 'P', 'vowel': 'O'},
            'PU': {'consonant': 'P', 'vowel': 'U'},
            'LA': {'consonant': 'L', 'vowel': 'A'},
            'LE': {'consonant': 'L', 'vowel': 'E'},
            'LI': {'consonant': 'L', 'vowel': 'I'},
            'LO': {'consonant': 'L', 'vowel': 'O'},
            'LU': {'consonant': 'L', 'vowel': 'U'},
        }
        
        # Mapeo de letras a manos
        self.hand_assignment = {
            # Consonantes (mano izquierda típicamente)
            'M': 'left',
            'P': 'left', 
            'L': 'left',
            'T': 'left',
            'S': 'left',
            # Vocales (mano derecha típicamente)
            'A': 'right',
            'E': 'right',
            'I': 'right',
            'O': 'right',
            'U': 'right'
        }
        
    def predict_syllable(self, left_hand_landmarks: Optional[List], right_hand_landmarks: Optional[List]) -> Optional[str]:
        """
        Predice sílaba basada en gestos de ambas manos
        """
        if not left_hand_landmarks or not right_hand_landmarks:
            return None
        
        # Detectar letra en cada mano
        left_letter = self._detect_letter_in_hand(left_hand_landmarks, 'left')
        right_letter = self._detect_letter_in_hand(right_hand_landmarks, 'right')
        
        # Intentar formar sílaba
        syllable = self._combine_letters_to_syllable(left_letter, right_letter)
        
        # Aplicar estabilización
        return self._stabilize_detection(syllable)
    
    def _detect_letter_in_hand(self, landmarks: List, hand_side: str) -> Optional[str]:
        """
        Detecta qué letra está haciendo una mano específica
        """
        if not landmarks or len(landmarks) < 63:
            return None
        
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        features = self._extract_hand_features(landmarks_array)
        
        if hand_side == 'left':
            # Buscar consonantes típicas de mano izquierda
            return self._detect_consonant(features)
        else:
            # Buscar vocales típicas de mano derecha
            return self._detect_vowel(features)
    
    def _extract_hand_features(self, landmarks_array: np.ndarray) -> Dict[str, Union[bool, int, float]]:
        """
        Extrae características de una mano para clasificación
        """
        # Puntos clave
        wrist = landmarks_array[0]
        thumb_tip = landmarks_array[4]
        index_tip = landmarks_array[8]
        middle_tip = landmarks_array[12]
        ring_tip = landmarks_array[16]
        pinky_tip = landmarks_array[20]
        
        thumb_ip = landmarks_array[3]
        index_pip = landmarks_array[6]
        middle_pip = landmarks_array[10]
        ring_pip = landmarks_array[14]
        pinky_pip = landmarks_array[18]
        
        features = {}
        
        # Estados de extensión
        features['thumb_extended'] = thumb_tip[1] < thumb_ip[1]
        features['index_extended'] = index_tip[1] < index_pip[1]
        features['middle_extended'] = middle_tip[1] < middle_pip[1]
        features['ring_extended'] = ring_tip[1] < ring_pip[1]
        features['pinky_extended'] = pinky_tip[1] < pinky_pip[1]
        
        # Distancias clave
        features['thumb_index_dist'] = np.linalg.norm(thumb_tip - index_tip)
        features['index_middle_dist'] = np.linalg.norm(index_tip - middle_tip)
        
        # Configuraciones de dedos
        features['fingers_count'] = sum([
            features['thumb_extended'],
            features['index_extended'],
            features['middle_extended'],
            features['ring_extended'],
            features['pinky_extended']
        ])
        
        features['fist_closed'] = features['fingers_count'] == 0
        
        return features
    
    def _detect_consonant(self, features: Dict[str, Union[bool, int, float]]) -> Optional[str]:
        """
        Detecta consonantes en mano izquierda
        """
        # M: Pulgar bajo tres dedos doblados
        if (not features['index_extended'] and 
            not features['middle_extended'] and 
            not features['ring_extended'] and 
            features['thumb_extended']):
            return 'M'
        
        # P: Similar a K pero con orientación específica
        elif (features['index_extended'] and 
              features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended']):
            return 'P'
        
        # L: Índice y pulgar en forma de L
        elif (features['index_extended'] and 
              not features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['thumb_extended']):
            return 'L'
        
        # T: Puño con pulgar entre dedos
        elif (features['fist_closed'] and 
              features['thumb_extended']):
            return 'T'
        
        # S: Puño cerrado
        elif features['fist_closed']:
            return 'S'
        
        return None
    
    def _detect_vowel(self, features: Dict[str, Union[bool, int, float]]) -> Optional[str]:
        """
        Detecta vocales en mano derecha
        """
        # A: Puño cerrado con pulgar al lado
        if (features['fist_closed'] and 
            not features['thumb_extended']):
            return 'A'
        
        # E: Dedos curvados
        elif (not features['index_extended'] and 
              not features['middle_extended'] and 
              features['thumb_extended']):
            return 'E'
        
        # I: Solo meñique extendido
        elif (not features['index_extended'] and 
              not features['middle_extended'] and 
              not features['ring_extended'] and 
              features['pinky_extended']):
            return 'I'
        
        # O: Forma de círculo
        elif (features['thumb_index_dist'] < 0.08 and 
              features['fingers_count'] <= 2):
            return 'O'
        
        # U: Índice y medio juntos
        elif (features['index_extended'] and 
              features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['index_middle_dist'] < 0.05):
            return 'U'
        
        return None
    
    def _combine_letters_to_syllable(self, consonant: Optional[str], vowel: Optional[str]) -> Optional[str]:
        """
        Combina consonante y vocal para formar sílaba
        """
        if not consonant or not vowel:
            return None
        
        syllable = consonant + vowel
        
        # Verificar si la sílaba está en nuestro conjunto soportado
        if syllable in self.supported_syllables:
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
                
                # Si la mayoría detecta la misma sílaba, es estable
                if syllable_count >= self.stability_threshold * 0.6:
                    return syllable
        
        return None
    
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
    
    def reset_detection_history(self):
        """
        Reinicia el historial de detecciones
        """
        self.detection_history = []
    
    def set_stability_threshold(self, threshold: int):
        """
        Cambia el umbral de estabilidad
        """
        self.stability_threshold = max(1, threshold)