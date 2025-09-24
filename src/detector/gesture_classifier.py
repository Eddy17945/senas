# src/detector/gesture_classifier.py (Alfabeto Completo)

import numpy as np
import cv2
from typing import List, Optional, Dict
try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn no disponible, usando clasificador simple")
import pickle
import os

class GestureClassifier:
    def __init__(self, model_path: str = None):
        self.model = None
        self.label_encoder = None
        self.is_trained = False
        self.detection_history = []
        self.stability_threshold = 5
        self.confidence_threshold = 0.7
        
        # Lista completa del alfabeto soportado
        self.supported_letters = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def create_simple_classifier(self):
        """Crea un clasificador simple mejorado"""
        if HAS_SKLEARN:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = None
        self.is_trained = True
    
    def predict_gesture(self, landmarks: List) -> Optional[str]:
        """Predice el gesto usando landmarks precisos de MediaPipe"""
        if not self.is_trained or not landmarks or len(landmarks) < 63:
            return None
        
        current_letter = self._classify_complete_alphabet(landmarks)
        
        # Estabilización
        self.detection_history.append(current_letter)
        
        if len(self.detection_history) > self.stability_threshold * 2:
            self.detection_history = self.detection_history[-self.stability_threshold:]
        
        if len(self.detection_history) >= self.stability_threshold:
            recent_detections = self.detection_history[-self.stability_threshold:]
            
            if current_letter:
                letter_count = sum(1 for detection in recent_detections 
                                 if detection == current_letter)
                
                if letter_count >= self.stability_threshold * 0.6:
                    return current_letter
        
        return None
    
    def _classify_complete_alphabet(self, landmarks: List) -> Optional[str]:
        """Clasificación completa del alfabeto A-Z"""
        if not landmarks or len(landmarks) < 63:
            return None
        
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        features = self._extract_finger_features(landmarks_array)
        
        # Clasificar cada letra del alfabeto
        return self._classify_all_letters(features)
    
    def _extract_finger_features(self, landmarks_array) -> Dict:
        """Extrae características específicas de cada dedo usando landmarks de MediaPipe"""
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
        
        # Estados de extensión
        features['thumb_extended'] = thumb_tip[1] < thumb_ip[1]
        features['index_extended'] = index_tip[1] < index_pip[1]
        features['middle_extended'] = middle_tip[1] < middle_pip[1]
        features['ring_extended'] = ring_tip[1] < ring_pip[1]
        features['pinky_extended'] = pinky_tip[1] < pinky_pip[1]
        
        # Ángulos de flexión
        features['thumb_angle'] = self._calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
        features['index_angle'] = self._calculate_angle(index_mcp, index_pip, index_tip)
        features['middle_angle'] = self._calculate_angle(middle_mcp, middle_pip, middle_tip)
        features['ring_angle'] = self._calculate_angle(ring_mcp, ring_pip, ring_tip)
        features['pinky_angle'] = self._calculate_angle(pinky_mcp, pinky_pip, pinky_tip)
        
        # Distancias entre puntas
        features['thumb_index_dist'] = np.linalg.norm(thumb_tip - index_tip)
        features['thumb_middle_dist'] = np.linalg.norm(thumb_tip - middle_tip)
        features['thumb_ring_dist'] = np.linalg.norm(thumb_tip - ring_tip)
        features['thumb_pinky_dist'] = np.linalg.norm(thumb_tip - pinky_tip)
        features['index_middle_dist'] = np.linalg.norm(index_tip - middle_tip)
        features['middle_ring_dist'] = np.linalg.norm(middle_tip - ring_tip)
        features['ring_pinky_dist'] = np.linalg.norm(ring_tip - pinky_tip)
        
        # Posiciones relativas
        features['thumb_behind_fingers'] = thumb_tip[0] < index_mcp[0]
        features['thumb_across_palm'] = abs(thumb_tip[0] - wrist[0]) > 0.1
        
        # Agrupaciones de dedos
        features['fingers_together'] = (features['index_middle_dist'] < 0.05 and
                                      features['middle_ring_dist'] < 0.05 and
                                      features['ring_pinky_dist'] < 0.05)
        
        features['two_fingers_up'] = (features['index_extended'] and features['middle_extended'] and
                                    not features['ring_extended'] and not features['pinky_extended'])
        
        features['three_fingers_up'] = (features['index_extended'] and features['middle_extended'] and
                                      features['ring_extended'] and not features['pinky_extended'])
        
        # Orientaciones y formas
        fingertips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        x_coords = [tip[0] for tip in fingertips]
        y_coords = [tip[1] for tip in fingertips]
        
        features['hand_width'] = max(x_coords) - min(x_coords)
        features['hand_height'] = max(y_coords) - min(y_coords)
        features['hand_openness'] = features['hand_width'] / 0.3
        
        # Número de dedos extendidos
        extended_fingers = sum([
            features['thumb_extended'],
            features['index_extended'],
            features['middle_extended'],
            features['ring_extended'],
            features['pinky_extended']
        ])
        features['fingers_count'] = extended_fingers
        
        # Características específicas para ciertas letras
        features['fist_closed'] = features['fingers_count'] == 0
        features['pointing_gesture'] = (features['index_extended'] and 
                                      features['fingers_count'] == 1)
        
        return features
    
    def _classify_all_letters(self, features: Dict) -> Optional[str]:
        """Clasificación completa del alfabeto A-Z basado en las imágenes de referencia"""
        
        # LETRA A: Puño cerrado con pulgar al lado
        if (not features['index_extended'] and not features['middle_extended'] and 
            not features['ring_extended'] and not features['pinky_extended'] and
            features['thumb_behind_fingers']):
            return "A"
        
        # LETRA B: Cuatro dedos extendidos juntos, pulgar doblado
        elif (features['index_extended'] and features['middle_extended'] and 
              features['ring_extended'] and features['pinky_extended'] and
              not features['thumb_extended'] and features['fingers_together']):
            return "B"
        
        # LETRA C: Forma curva - mano parcialmente abierta
        elif (features['fingers_count'] >= 2 and features['thumb_index_dist'] > 0.1 and 
              features['thumb_index_dist'] < 0.25 and features['hand_openness'] > 0.3):
            return "C"
        
        # LETRA D: Índice extendido, pulgar tocando otros dedos
        elif (features['index_extended'] and not features['middle_extended'] and 
              not features['ring_extended'] and not features['pinky_extended'] and
              features['thumb_middle_dist'] < 0.08):
            return "D"
        
        # LETRA E: Dedos curvados hacia adentro, pulgar visible
        elif (not features['index_extended'] and not features['middle_extended'] and 
              not features['ring_extended'] and not features['pinky_extended'] and
              features['thumb_extended'] and features['index_angle'] > 90):
            return "E"
        
        # LETRA F: Círculo con pulgar e índice, otros dedos extendidos
        elif (not features['index_extended'] and features['middle_extended'] and 
              features['ring_extended'] and features['pinky_extended'] and
              features['thumb_index_dist'] < 0.06):
            return "F"
        
        # LETRA G: Índice extendido horizontalmente, pulgar extendido
        elif (features['index_extended'] and not features['middle_extended'] and 
              not features['ring_extended'] and not features['pinky_extended'] and
              features['thumb_extended'] and features['thumb_index_dist'] > 0.15):
            return "G"
        
        # LETRA H: Índice y medio extendidos horizontalmente
        elif (features['index_extended'] and features['middle_extended'] and 
              not features['ring_extended'] and not features['pinky_extended'] and
              features['thumb_behind_fingers'] and features['index_middle_dist'] < 0.08):
            return "H"
        
        # LETRA I: Solo meñique extendido
        elif (not features['index_extended'] and not features['middle_extended'] and 
              not features['ring_extended'] and features['pinky_extended'] and
              features['thumb_behind_fingers']):
            return "I"
        
        # LETRA J: Meñique extendido con movimiento (similar a I pero con orientación)
        elif (not features['index_extended'] and not features['middle_extended'] and 
              not features['ring_extended'] and features['pinky_extended'] and
              not features['thumb_behind_fingers']):
            return "J"
        
        # LETRA K: Índice y medio extendidos en V, pulgar toca medio
        elif (features['index_extended'] and features['middle_extended'] and 
              not features['ring_extended'] and not features['pinky_extended'] and
              features['index_middle_dist'] > 0.1 and features['thumb_middle_dist'] < 0.08):
            return "K"
        
        # LETRA L: Índice y pulgar en forma de L
        elif (features['index_extended'] and not features['middle_extended'] and 
              not features['ring_extended'] and not features['pinky_extended'] and
              features['thumb_extended'] and features['thumb_index_dist'] > 0.12):
            return "L"
        
        # LETRA M: Pulgar bajo tres dedos doblados
        elif (not features['index_extended'] and not features['middle_extended'] and 
              not features['ring_extended'] and not features['pinky_extended'] and
              features['thumb_extended'] and features['thumb_across_palm']):
            return "M"
        
        # LETRA N: Pulgar bajo dos dedos doblados
        elif (not features['index_extended'] and not features['middle_extended'] and 
              features['ring_extended'] and features['pinky_extended'] and
              features['thumb_behind_fingers']):
            return "N"
        
        # LETRA O: Todos los dedos formando círculo
        elif (features['fist_closed'] and features['thumb_index_dist'] < 0.08 and
              features['hand_openness'] > 0.2):
            return "O"
        
        # LETRA P: Similar a K pero con orientación hacia abajo
        elif (features['index_extended'] and features['middle_extended'] and 
              not features['ring_extended'] and not features['pinky_extended'] and
              features['thumb_middle_dist'] < 0.1 and features['index_middle_dist'] > 0.08):
            return "P"
        
        # LETRA Q: Similar a G pero con orientación hacia abajo
        elif (features['index_extended'] and not features['middle_extended'] and 
              not features['ring_extended'] and not features['pinky_extended'] and
              features['thumb_extended'] and features['hand_height'] > features['hand_width']):
            return "Q"
        
        # LETRA R: Índice y medio cruzados
        elif (features['index_extended'] and features['middle_extended'] and 
              not features['ring_extended'] and not features['pinky_extended'] and
              features['index_middle_dist'] < 0.05 and not features['thumb_extended']):
            return "R"
        
        # LETRA S: Puño cerrado con pulgar sobre dedos
        elif (features['fist_closed'] and not features['thumb_behind_fingers'] and
              features['thumb_across_palm']):
            return "S"
        
        # LETRA T: Puño cerrado con pulgar entre índice y medio
        elif (not features['index_extended'] and not features['middle_extended'] and 
              not features['ring_extended'] and not features['pinky_extended'] and
              features['thumb_extended'] and features['thumb_index_dist'] < 0.05):
            return "T"
        
        # LETRA U: Índice y medio extendidos juntos hacia arriba
        elif (features['two_fingers_up'] and features['index_middle_dist'] < 0.05 and
              features['thumb_behind_fingers']):
            return "U"
        
        # LETRA V: Índice y medio extendidos separados
        elif (features['two_fingers_up'] and features['index_middle_dist'] > 0.08 and
              features['thumb_behind_fingers']):
            return "V"
        
        # LETRA W: Tres dedos extendidos (índice, medio, anular)
        elif (features['three_fingers_up'] and features['thumb_behind_fingers']):
            return "W"
        
        # LETRA X: Índice parcialmente doblado (gancho)
        elif (not features['middle_extended'] and not features['ring_extended'] and 
              not features['pinky_extended'] and features['thumb_behind_fingers'] and
              features['index_angle'] > 45 and features['index_angle'] < 120):
            return "X"
        
        # LETRA Y: Pulgar y meñique extendidos
        elif (not features['index_extended'] and not features['middle_extended'] and 
              not features['ring_extended'] and features['pinky_extended'] and
              features['thumb_extended'] and features['thumb_pinky_dist'] > 0.15):
            return "Y"
        
        # LETRA Z: Índice extendido haciendo zigzag (similar a D pero con movimiento)
        elif (features['pointing_gesture'] and features['thumb_index_dist'] > 0.1):
            return "Z"
        
        # Si no coincide con ningún patrón específico, usar clasificación por conteo
        return self._classify_by_finger_count(features)
    
    def _classify_by_finger_count(self, features: Dict) -> Optional[str]:
        """Clasificación de respaldo basada en número de dedos"""
        count = features['fingers_count']
        
        if count == 0:
            return "A" if features['thumb_behind_fingers'] else "S"
        elif count == 1:
            if features['index_extended']:
                return "D"
            elif features['pinky_extended']:
                return "I"
            elif features['thumb_extended']:
                return "T"
        elif count == 2:
            if features['two_fingers_up']:
                return "V" if features['index_middle_dist'] > 0.08 else "U"
            else:
                return "C"
        elif count == 3:
            return "W"
        elif count == 4:
            return "B"
        elif count == 5:
            return "5"  # Mano completamente abierta
        
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
        """Cambia el umbral de estabilidad"""
        self.stability_threshold = max(1, threshold)
    
    def save_model(self, path: str):
        """Guarda el modelo entrenado"""
        if self.model:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'label_encoder': self.label_encoder,
                    'is_trained': self.is_trained
                }, f)
    
    def load_model(self, path: str):
        """Carga un modelo previamente entrenado"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.label_encoder = data.get('label_encoder')
                self.is_trained = data.get('is_trained', True)
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.create_simple_classifier()