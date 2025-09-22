# src/detector/gesture_classifier.py (Versión Mejorada)

import numpy as np
import cv2
from typing import List, Optional
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
        self.stability_threshold = 3  # Necesita 3 detecciones consecutivas
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_landmarks(self, landmarks: List) -> np.ndarray:
        """Preprocesa los landmarks para el modelo"""
        if not landmarks or len(landmarks) < 63:
            return None
        
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        
        # Características más robustas
        features = []
        
        # 1. Centro de masa normalizado
        center = landmarks_array[0]
        features.extend([center[0], center[1]])
        
        # 2. Contar puntos activos (dedos visibles)
        active_points = sum(1 for i in range(len(landmarks_array)) 
                          if landmarks_array[i][0] != 0 or landmarks_array[i][1] != 0)
        features.append(active_points / 21.0)  # Normalizar
        
        # 3. Calcular distancias relativas
        if active_points > 1:
            for i in range(1, min(6, len(landmarks_array))):  # Primeros 5 puntos después del centro
                if landmarks_array[i][0] != 0:
                    dist = np.sqrt((landmarks_array[i][0] - center[0])**2 + 
                                 (landmarks_array[i][1] - center[1])**2)
                    features.append(dist)
                else:
                    features.append(0)
        
        # 4. Calcular dispersión de puntos
        valid_points = [point for point in landmarks_array if point[0] != 0 or point[1] != 0]
        if len(valid_points) > 1:
            x_coords = [point[0] for point in valid_points]
            y_coords = [point[1] for point in valid_points]
            
            x_spread = max(x_coords) - min(x_coords)
            y_spread = max(y_coords) - min(y_coords)
            
            features.extend([x_spread, y_spread])
        else:
            features.extend([0, 0])
        
        return np.array(features)
    
    def create_simple_classifier(self):
        """Crea un clasificador simple mejorado"""
        if HAS_SKLEARN:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = None
        self.is_trained = True
    
    def predict_gesture(self, landmarks: List) -> Optional[str]:
        """Predice el gesto con estabilización"""
        if not self.is_trained or not landmarks:
            return None
        
        # Detectar letra actual
        current_letter = self._enhanced_letter_detection(landmarks)
        
        # Agregar a historial para estabilización
        self.detection_history.append(current_letter)
        
        # Mantener solo las últimas detecciones
        if len(self.detection_history) > self.stability_threshold * 2:
            self.detection_history = self.detection_history[-self.stability_threshold:]
        
        # Verificar estabilidad
        if len(self.detection_history) >= self.stability_threshold:
            recent_detections = self.detection_history[-self.stability_threshold:]
            
            # Si todas las detecciones recientes son iguales, es estable
            if all(detection == current_letter for detection in recent_detections):
                return current_letter
        
        return None  # No es estable aún
    
    def _enhanced_letter_detection(self, landmarks: List) -> Optional[str]:
        """Detección mejorada de letras basada en características específicas"""
        if not landmarks or len(landmarks) < 63:
            return None
        
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        
        # Contar puntos activos (dedos/características visibles)
        active_points = sum(1 for point in landmarks_array 
                          if point[0] != 0 or point[1] != 0)
        
        # Analizar distribución de puntos
        valid_points = [(point[0], point[1]) for point in landmarks_array 
                       if point[0] != 0 or point[1] != 0]
        
        if not valid_points:
            return None
        
        # Características para clasificación
        center = valid_points[0] if valid_points else (0, 0)
        
        # Separar puntos en dedos (arriba del centro) y base (abajo del centro)
        finger_points = [p for p in valid_points[1:] if p[1] < center[1]]  # Arriba
        base_points = [p for p in valid_points[1:] if p[1] >= center[1]]   # Abajo
        
        num_fingers = len(finger_points)
        num_base = len(base_points)
        
        # Calcular dispersión horizontal de los dedos
        if finger_points:
            x_coords = [p[0] for p in finger_points]
            finger_spread = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 0
        else:
            finger_spread = 0
        
        # Reglas mejoradas para detección de letras
        
        # LETRA A: Puño cerrado - pocos puntos activos, concentrados
        if active_points <= 4 and finger_spread < 0.1:
            return "A"
        
        # LETRA B: Mano abierta - muchos puntos, bien distribuidos
        elif active_points >= 8 and num_fingers >= 4 and finger_spread > 0.15:
            return "B"
        
        # LETRA C: Forma curva - puntos intermedios, distribución específica
        elif 5 <= active_points <= 7 and 0.05 < finger_spread < 0.2:
            # Verificar si los puntos forman una curva
            if self._is_curved_shape(valid_points):
                return "C"
        
        # LETRA D: Un dedo extendido prominente
        elif active_points >= 5 and num_fingers == 1 and finger_spread < 0.08:
            return "D"
        
        # LETRA E: Dedos curvados hacia adentro
        elif active_points >= 6 and num_fingers >= 2 and finger_spread < 0.12:
            return "E"
        
        # Clasificación adicional basada en patrones geométricos
        return self._geometric_classification(valid_points, active_points)
    
    def _is_curved_shape(self, points):
        """Detectar si los puntos forman una forma curva (como la letra C)"""
        if len(points) < 3:
            return False
        
        # Calcular el centro de masa
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        
        # Calcular ángulos desde el centro
        angles = []
        for x, y in points:
            angle = np.arctan2(y - center_y, x - center_x)
            angles.append(angle)
        
        # Ordenar ángulos
        angles.sort()
        
        # Verificar si los ángulos cubren un arco (no un círculo completo)
        if len(angles) > 2:
            angle_range = angles[-1] - angles[0]
            # Para una C, el rango de ángulos debería ser entre π/2 y 3π/2
            if np.pi/3 < angle_range < 4*np.pi/3:
                return True
        
        return False
    
    def _geometric_classification(self, points, active_points):
        """Clasificación adicional basada en geometría"""
        if active_points < 3:
            return "A"  # Por defecto, pocos puntos = puño
        elif active_points > 10:
            return "B"  # Muchos puntos = mano abierta
        else:
            # Análisis intermedio
            if len(points) >= 2:
                # Calcular ratio de aspecto de la forma
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                x_range = max(x_coords) - min(x_coords)
                y_range = max(y_coords) - min(y_coords)
                
                if x_range > 0 and y_range > 0:
                    aspect_ratio = x_range / y_range
                    
                    if aspect_ratio > 1.5:  # Forma ancha
                        return "E"
                    elif aspect_ratio < 0.7:  # Forma alta
                        return "D"
                    else:  # Forma equilibrada
                        return "C"
            
            return None
    
    def reset_detection_history(self):
        """Reiniciar el historial de detecciones"""
        self.detection_history = []
    
    def set_stability_threshold(self, threshold: int):
        """Cambiar el umbral de estabilidad"""
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