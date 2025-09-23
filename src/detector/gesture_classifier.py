# src/detector/gesture_classifier.py (Versión MediaPipe)

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
        self.stability_threshold = 5  # Necesita 5 detecciones consecutivas
        self.confidence_threshold = 0.7
        
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
        """
        Predice el gesto usando landmarks precisos de MediaPipe
        """
        if not self.is_trained or not landmarks or len(landmarks) < 63:
            return None
        
        # Detectar letra actual
        current_letter = self._mediapipe_letter_detection(landmarks)
        
        # Agregar a historial para estabilización
        self.detection_history.append(current_letter)
        
        # Mantener solo las últimas detecciones
        if len(self.detection_history) > self.stability_threshold * 2:
            self.detection_history = self.detection_history[-self.stability_threshold:]
        
        # Verificar estabilidad
        if len(self.detection_history) >= self.stability_threshold:
            recent_detections = self.detection_history[-self.stability_threshold:]
            
            # Contar la detección más común
            if current_letter:
                letter_count = sum(1 for detection in recent_detections 
                                 if detection == current_letter)
                
                # Si la mayoría detecta la misma letra, es estable
                if letter_count >= self.stability_threshold * 0.6:  # 60% de consenso
                    return current_letter
        
        return None
    
    def _mediapipe_letter_detection(self, landmarks: List) -> Optional[str]:
        """
        Detección de letras usando landmarks precisos de MediaPipe
        """
        if not landmarks or len(landmarks) < 63:
            return None
        
        # Convertir a array para facilitar el acceso
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        
        # Obtener características específicas de cada dedo
        features = self._extract_finger_features(landmarks_array)
        
        # Clasificar basado en patrones específicos
        letter = self._classify_by_finger_patterns(features)
        
        return letter
    
    def _extract_finger_features(self, landmarks_array) -> Dict:
        """
        Extrae características específicas de cada dedo usando landmarks de MediaPipe
        """
        # IDs específicos de MediaPipe para cada punto de la mano
        wrist = landmarks_array[0]
        
        # Pulgar
        thumb_cmc = landmarks_array[1]   # Base del pulgar
        thumb_mcp = landmarks_array[2]   # Articulación MCP
        thumb_ip = landmarks_array[3]    # Articulación IP
        thumb_tip = landmarks_array[4]   # Punta
        
        # Índice
        index_mcp = landmarks_array[5]   # Base
        index_pip = landmarks_array[6]   # Articulación PIP
        index_dip = landmarks_array[7]   # Articulación DIP
        index_tip = landmarks_array[8]   # Punta
        
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
        
        # Calcular características
        features = {}
        
        # 1. Estado de extensión de cada dedo
        features['thumb_extended'] = thumb_tip[1] < thumb_ip[1]
        features['index_extended'] = index_tip[1] < index_pip[1]
        features['middle_extended'] = middle_tip[1] < middle_pip[1]
        features['ring_extended'] = ring_tip[1] < ring_pip[1]
        features['pinky_extended'] = pinky_tip[1] < pinky_pip[1]
        
        # 2. Ángulos de flexión
        features['thumb_angle'] = self._calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
        features['index_angle'] = self._calculate_angle(index_mcp, index_pip, index_tip)
        features['middle_angle'] = self._calculate_angle(middle_mcp, middle_pip, middle_tip)
        features['ring_angle'] = self._calculate_angle(ring_mcp, ring_pip, ring_tip)
        features['pinky_angle'] = self._calculate_angle(pinky_mcp, pinky_pip, pinky_tip)
        
        # 3. Distancias entre puntas de dedos
        features['thumb_index_dist'] = np.linalg.norm(thumb_tip - index_tip)
        features['thumb_middle_dist'] = np.linalg.norm(thumb_tip - middle_tip)
        features['index_middle_dist'] = np.linalg.norm(index_tip - middle_tip)
        features['middle_ring_dist'] = np.linalg.norm(middle_tip - ring_tip)
        features['ring_pinky_dist'] = np.linalg.norm(ring_tip - pinky_tip)
        
        # 4. Posiciones relativas
        features['thumb_behind_fingers'] = thumb_tip[0] < index_mcp[0]
        features['fingers_together'] = (features['index_middle_dist'] < 0.05 and
                                      features['middle_ring_dist'] < 0.05 and
                                      features['ring_pinky_dist'] < 0.05)
        
        # 5. Curvatura general de la mano
        fingertips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        x_coords = [tip[0] for tip in fingertips]
        y_coords = [tip[1] for tip in fingertips]
        
        features['hand_width'] = max(x_coords) - min(x_coords)
        features['hand_openness'] = features['hand_width'] / 0.3  # Normalizado
        
        # 6. Número de dedos extendidos
        extended_fingers = sum([
            features['thumb_extended'],
            features['index_extended'],
            features['middle_extended'],
            features['ring_extended'],
            features['pinky_extended']
        ])
        features['fingers_count'] = extended_fingers
        
        return features
    
    def _calculate_angle(self, point1, point2, point3):
        """
        Calcula el ángulo entre tres puntos
        """
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Evitar división por cero
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        cos_angle = np.dot(vector1, vector2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Evitar errores de precisión
        
        angle = np.arccos(cos_angle) * 180 / np.pi
        return angle
    
    def _classify_by_finger_patterns(self, features: Dict) -> Optional[str]:
        """
        Clasifica la letra basándose en patrones específicos de dedos
        """
        # LETRA A: Puño cerrado con pulgar al lado
        if (not features['index_extended'] and 
            not features['middle_extended'] and 
            not features['ring_extended'] and 
            not features['pinky_extended'] and
            features['thumb_behind_fingers']):
            return "A"
        
        # LETRA B: Todos los dedos extendidos excepto el pulgar
        elif (features['index_extended'] and 
              features['middle_extended'] and 
              features['ring_extended'] and 
              features['pinky_extended'] and
              not features['thumb_extended'] and
              features['fingers_together']):
            return "B"
        
        # LETRA C: Forma curva - dedos parcialmente curvados
        elif (features['fingers_count'] >= 2 and 
              features['thumb_index_dist'] > 0.1 and 
              features['thumb_index_dist'] < 0.25 and
              features['hand_openness'] > 0.3 and 
              features['hand_openness'] < 0.8):
            return "C"
        
        # LETRA D: Solo índice extendido, pulgar tocando otros dedos
        elif (features['index_extended'] and 
              not features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['thumb_middle_dist'] < 0.08):
            return "D"
        
        # LETRA E: Dedos curvados hacia adentro
        elif (not features['index_extended'] and 
              not features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['thumb_extended'] and
              features['index_angle'] > 90 and 
              features['middle_angle'] > 90):
            return "E"
        
        # LETRA F: Índice y pulgar haciendo círculo, otros extendidos
        elif (not features['index_extended'] and 
              features['middle_extended'] and 
              features['ring_extended'] and 
              features['pinky_extended'] and
              features['thumb_index_dist'] < 0.06):
            return "F"
        
        # LETRA G: Índice extendido horizontalmente
        elif (features['index_extended'] and 
              not features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['thumb_extended'] and
              abs(features['index_angle'] - 90) < 30):
            return "G"
        
        # LETRA I: Solo meñique extendido
        elif (not features['index_extended'] and 
              not features['middle_extended'] and 
              not features['ring_extended'] and 
              features['pinky_extended'] and
              features['thumb_behind_fingers']):
            return "I"
        
        # LETRA L: Índice y pulgar en forma de L
        elif (features['index_extended'] and 
              not features['middle_extended'] and 
              not features['ring_extended'] and 
              not features['pinky_extended'] and
              features['thumb_extended'] and
              abs(features['thumb_angle'] - 90) < 45):
            return "L"
        
        # LETRA O: Todos los dedos formando círculo
        elif (features['fingers_count'] == 0 and 
              features['thumb_index_dist'] < 0.08 and
              features['hand_openness'] > 0.2 and 
              features['hand_openness'] < 0.5):
            return "O"
        
        # Clasificación por número de dedos como respaldo
        return self._classify_by_finger_count(features)
    
    def _classify_by_finger_count(self, features: Dict) -> Optional[str]:
        """
        Clasificación de respaldo basada en número de dedos
        """
        count = features['fingers_count']
        
        if count == 0:
            return "A" if features['thumb_behind_fingers'] else "E"
        elif count == 1:
            if features['index_extended']:
                return "D"
            elif features['pinky_extended']:
                return "I"
            else:
                return "G"
        elif count == 2:
            return "V" if features['index_extended'] and features['middle_extended'] else "C"
        elif count == 4:
            return "B"
        elif count == 5:
            return "5"  # Mano completamente abierta
        
        return None
    
    def get_detection_confidence(self) -> float:
        """
        Calcula la confianza de la detección actual
        """
        if len(self.detection_history) < self.stability_threshold:
            return 0.0
        
        recent = self.detection_history[-self.stability_threshold:]
        if not recent[0]:  # Si la detección más reciente es None
            return 0.0
        
        # Contar detecciones consistentes
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