# src/detector/gesture_classifier_improved.py
# OPTIMIZADO PARA PRECISIÓN Y VELOCIDAD

import numpy as np
import cv2
from typing import List, Optional, Dict, Any, Union

class GestureClassifier:
    def __init__(self, model_path: str = None):
        self.model = None
        self.label_encoder = None
        self.is_trained = False
        self.detection_history = []
        self.stability_threshold = 3  # REDUCIDO de 5 a 3 para velocidad
        self.confidence_threshold = 0.6  # REDUCIDO para aceptar más detecciones
        
        # Lista completa del alfabeto soportado
        self.supported_letters = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        
        self.is_trained = True  # Activar directamente
    
    def predict_gesture(self, landmarks: List) -> Optional[str]:
        """Predice el gesto con VELOCIDAD y PRECISIÓN mejorada"""
        if not self.is_trained or not landmarks or len(landmarks) < 63:
            return None
        
        current_letter = self._classify_complete_alphabet(landmarks)
        
        # Estabilización RÁPIDA
        self.detection_history.append(current_letter)
        
        if len(self.detection_history) > self.stability_threshold * 2:
            self.detection_history = self.detection_history[-self.stability_threshold:]
        
        # DETECCIÓN MÁS RÁPIDA - solo 60% de estabilidad necesaria
        if len(self.detection_history) >= self.stability_threshold:
            recent_detections = self.detection_history[-self.stability_threshold:]
            
            if current_letter:
                letter_count = sum(1 for detection in recent_detections 
                                 if detection == current_letter)
                
                # CAMBIO: 50% en lugar de 60% para ser más responsivo
                if letter_count >= self.stability_threshold * 0.5:
                    return current_letter
        
        return None
    
    def _classify_complete_alphabet(self, landmarks: List) -> Optional[str]:
        """Clasificación mejorada con mejor precisión para Y, U y todas las letras"""
        if not landmarks or len(landmarks) < 63:
            return None
        
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        features = self._extract_finger_features_enhanced(landmarks_array)
        
        return self._classify_all_letters_improved(features, landmarks_array)
    
    def _extract_finger_features_enhanced(self, landmarks_array) -> Dict:
        """Extrae características MEJORADAS con más precisión"""
        # Puntos clave de MediaPipe
        wrist = landmarks_array[0]
        
        # Pulgar
        thumb_cmc = landmarks_array[1]
        thumb_mcp = landmarks_array[2]
        thumb_ip = landmarks_array[3]
        thumb_tip = landmarks_array[4]
        
        # Índice
        index_mcp = landmarks_array[5]
        index_pip = landmarks_array[6]
        index_dip = landmarks_array[7]
        index_tip = landmarks_array[8]
        
        # Medio
        middle_mcp = landmarks_array[9]
        middle_pip = landmarks_array[10]
        middle_dip = landmarks_array[11]
        middle_tip = landmarks_array[12]
        
        # Anular
        ring_mcp = landmarks_array[13]
        ring_pip = landmarks_array[14]
        ring_dip = landmarks_array[15]
        ring_tip = landmarks_array[16]
        
        # Meñique
        pinky_mcp = landmarks_array[17]
        pinky_pip = landmarks_array[18]
        pinky_dip = landmarks_array[19]
        pinky_tip = landmarks_array[20]
        
        features = {}
        
        # Estados de extensión MEJORADOS (comparación en Y)
        features['thumb_extended'] = thumb_tip[1] < thumb_ip[1] - 0.02
        features['index_extended'] = index_tip[1] < index_pip[1] - 0.03
        features['middle_extended'] = middle_tip[1] < middle_pip[1] - 0.03
        features['ring_extended'] = ring_tip[1] < ring_pip[1] - 0.03
        features['pinky_extended'] = pinky_tip[1] < pinky_pip[1] - 0.03
        
        # Ángulos de flexión PRECISOS
        features['thumb_angle'] = self._calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
        features['index_angle'] = self._calculate_angle(index_mcp, index_pip, index_tip)
        features['middle_angle'] = self._calculate_angle(middle_mcp, middle_pip, middle_tip)
        features['ring_angle'] = self._calculate_angle(ring_mcp, ring_pip, ring_tip)
        features['pinky_angle'] = self._calculate_angle(pinky_mcp, pinky_pip, pinky_tip)
        
        # Distancias CRÍTICAS para Y y U
        features['thumb_index_dist'] = np.linalg.norm(thumb_tip - index_tip)
        features['thumb_middle_dist'] = np.linalg.norm(thumb_tip - middle_tip)
        features['thumb_ring_dist'] = np.linalg.norm(thumb_tip - ring_tip)
        features['thumb_pinky_dist'] = np.linalg.norm(thumb_tip - pinky_tip)
        features['index_middle_dist'] = np.linalg.norm(index_tip - middle_tip)
        features['middle_ring_dist'] = np.linalg.norm(middle_tip - ring_tip)
        features['ring_pinky_dist'] = np.linalg.norm(ring_tip - pinky_tip)
        features['index_pinky_dist'] = np.linalg.norm(index_tip - pinky_tip)
        
        # Posiciones relativas EN X (crítico para Y)
        features['thumb_left_of_fingers'] = thumb_tip[0] < index_mcp[0] - 0.05
        features['thumb_right_of_fingers'] = thumb_tip[0] > pinky_mcp[0] + 0.05
        features['thumb_between_fingers'] = (thumb_tip[0] > index_mcp[0] and 
                                            thumb_tip[0] < pinky_mcp[0])
        
        # Alineación de dedos (crítico para U)
        features['index_middle_aligned'] = abs(index_tip[1] - middle_tip[1]) < 0.04
        features['fingers_pointing_up'] = (index_tip[1] < index_mcp[1] and 
                                          middle_tip[1] < middle_mcp[1])
        
        # Apertura de mano
        fingertips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        x_coords = [tip[0] for tip in fingertips]
        y_coords = [tip[1] for tip in fingertips]
        
        features['hand_width'] = max(x_coords) - min(x_coords)
        features['hand_height'] = max(y_coords) - min(y_coords)
        features['hand_openness'] = features['hand_width'] / 0.3 if features['hand_width'] > 0 else 0
        
        # Número de dedos extendidos
        extended_fingers = sum([
            features['thumb_extended'],
            features['index_extended'],
            features['middle_extended'],
            features['ring_extended'],
            features['pinky_extended']
        ])
        features['fingers_count'] = extended_fingers
        
        # Configuraciones especiales
        features['fist_closed'] = features['fingers_count'] == 0
        features['all_extended'] = features['fingers_count'] == 5
        features['two_fingers_up'] = features['fingers_count'] == 2
        features['three_fingers_up'] = features['fingers_count'] == 3
        
        # Detección de separación entre dedos (V vs U)
        features['fingers_separated'] = features['index_middle_dist'] > 0.06
        features['fingers_together'] = features['index_middle_dist'] < 0.04
        
        return features
    
    def _classify_all_letters_improved(self, features: Dict, landmarks_array) -> Optional[str]:
        """Clasificación MEJORADA con reglas más precisas especialmente para Y y U"""
        
        # ========== LETRA Y - MEJORADA ==========
        # Y: Pulgar y meñique extendidos, otros doblados
        if (not features['index_extended'] and 
            not features['middle_extended'] and 
            not features['ring_extended'] and 
            features['pinky_extended'] and 
            features['thumb_extended']):
            
            # Verificar distancia característica de Y
            if features['thumb_pinky_dist'] > 0.12:  # Reducido de 0.15
                # Verificar que índice/medio/anular estén realmente doblados
                if (features['index_angle'] < 150 and 
                    features['middle_angle'] < 150 and 
                    features['ring_angle'] < 150):
                    return "Y"
        
        # ========== LETRA U - MEJORADA ==========
        # U: Índice y medio extendidos JUNTOS y PARALELOS
        if (features['index_extended'] and 
            features['middle_extended'] and 
            not features['ring_extended'] and 
            not features['pinky_extended']):
            
            # Criterios MEJORADOS para U
            fingers_close = features['index_middle_dist'] < 0.05  # Más permisivo
            fingers_aligned = features['index_middle_aligned']
            pointing_up = features['fingers_pointing_up']
            thumb_not_extended = not features['thumb_extended'] or features['thumb_between_fingers']
            
            if fingers_close and fingers_aligned and pointing_up:
                return "U"
        
        # ========== LETRA V - MEJORADA ==========
        # V: Índice y medio extendidos SEPARADOS
        elif (features['index_extended'] and 
              features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['fingers_separated']):  # Usando feature específica
            return "V"
        
        # ========== OTRAS LETRAS (mantenemos lógica mejorada) ==========
        
        # LETRA A
        if (features['fist_closed'] and 
            features['thumb_left_of_fingers']):
            return "A"
        
        # LETRA B
        elif (features['index_extended'] and 
              features['middle_extended'] and 
              features['ring_extended'] and 
              features['pinky_extended'] and
              not features['thumb_extended'] and 
              features['fingers_together']):
            return "B"
        
        # LETRA C
        elif (features['fingers_count'] >= 2 and 
              features['thumb_index_dist'] > 0.08 and 
              features['thumb_index_dist'] < 0.20 and 
              features['hand_openness'] > 0.25):
            return "C"
        
        # LETRA D
        elif (features['index_extended'] and 
              not features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['thumb_middle_dist'] < 0.10):
            return "D"
        
        # LETRA E
        elif (features['fist_closed'] and 
              features['thumb_extended'] and 
              features['index_angle'] > 80):
            return "E"
        
        # LETRA F
        elif (not features['index_extended'] and 
              features['middle_extended'] and 
              features['ring_extended'] and 
              features['pinky_extended'] and
              features['thumb_index_dist'] < 0.08):
            return "F"
        
        # LETRA G
        elif (features['index_extended'] and 
              not features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['thumb_extended'] and 
              features['thumb_index_dist'] > 0.12):
            return "G"
        
        # LETRA H
        elif (features['index_extended'] and 
              features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['index_middle_dist'] < 0.06):
            return "H"
        
        # LETRA I
        elif (not features['index_extended'] and 
              not features['middle_extended'] and 
              not features['ring_extended'] and 
              features['pinky_extended'] and
              not features['thumb_extended']):
            return "I"
        
        # LETRA L
        elif (features['index_extended'] and 
              not features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['thumb_extended'] and 
              features['thumb_left_of_fingers']):
            return "L"
        
        # LETRA M
        elif (not features['index_extended'] and 
              not features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['thumb_extended']):
            return "M"
        
        # LETRA O
        elif (features['fist_closed'] and 
              features['thumb_index_dist'] < 0.10 and
              features['hand_openness'] > 0.15):
            return "O"
        
        # LETRA S
        elif (features['fist_closed'] and 
              not features['thumb_left_of_fingers'] and
              not features['thumb_extended']):
            return "S"
        
        # LETRA W
        elif (features['three_fingers_up'] and 
              not features['thumb_extended']):
            return "W"
        
        # LETRA X
        elif (not features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['index_angle'] > 40 and 
              features['index_angle'] < 120):
            return "X"
        
        # Si no coincide, usar clasificación por conteo
        return self._classify_by_finger_count(features)
    
    def _classify_by_finger_count(self, features: Dict) -> Optional[str]:
        """Clasificación de respaldo mejorada"""
        count = features['fingers_count']
        
        if count == 0:
            return "A" if features['thumb_left_of_fingers'] else "S"
        elif count == 1:
            if features['index_extended']:
                return "D"
            elif features['pinky_extended']:
                if features['thumb_extended']:
                    return "Y"
                else:
                    return "I"
        elif count == 2:
            if features['fingers_separated']:
                return "V"
            elif features['fingers_together']:
                return "U"
            else:
                return "H"
        elif count == 3:
            return "W"
        elif count == 4:
            return "B"
        elif count == 5:
            return "5"
        
        return None
    
    def _calculate_angle(self, point1, point2, point3):
        """Calcula el ángulo entre tres puntos"""
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        cos_angle = np.dot(vector1, vector2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.arccos(cos_angle) * 180 / np.pi
        return angle
    
    def get_supported_letters(self) -> List[str]:
        """Retorna la lista de letras soportadas"""
        return self.supported_letters
    
    def get_detection_confidence(self) -> float:
        """Calcula la confianza de la detección actual"""
        if len(self.detection_history) < self.stability_threshold:
            return 0.0
        
        recent = self.detection_history[-self.stability_threshold:]
        if not recent[0]:
            return 0.0
        
        consistent_count = sum(1 for detection in recent if detection == recent[-1])
        confidence = consistent_count / len(recent)
        
        return confidence
    
    def reset_detection_history(self):
        """Reinicia el historial de detecciones"""
        self.detection_history = []
    
    def set_stability_threshold(self, threshold: int):
        """Cambia el umbral de estabilidad (1-10)"""
        self.stability_threshold = max(1, min(10, threshold))