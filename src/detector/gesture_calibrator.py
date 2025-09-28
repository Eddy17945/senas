# src/detector/gesture_calibrator.py

import numpy as np
import json
import os
from typing import Dict, List, Optional
from collections import defaultdict
import time

class GestureCalibrator:
    def __init__(self, config_path: str = "assets/gesture_config.json"):
        self.config_path = config_path
        self.gesture_samples = defaultdict(list)
        self.calibrated_thresholds = {}
        self.user_patterns = {}
        
        # Cargar configuración existente
        self.load_calibration()
        
        # Parámetros por defecto
        self.default_thresholds = {
            'stability_frames': 15,
            'confidence_threshold': 0.7,
            'angle_tolerance': 15,
            'distance_tolerance': 0.05
        }
        
    def collect_sample(self, letter: str, landmarks: List, confidence: float):
        """Recolecta una muestra de gesto para calibración"""
        if confidence < 0.8:  # Solo muestras de alta confianza
            return
            
        sample = {
            'landmarks': landmarks.copy(),
            'timestamp': time.time(),
            'confidence': confidence,
            'features': self._extract_calibration_features(landmarks)
        }
        
        self.gesture_samples[letter].append(sample)
        
        # Mantener solo las últimas 50 muestras por letra
        if len(self.gesture_samples[letter]) > 50:
            self.gesture_samples[letter] = self.gesture_samples[letter][-50:]
    
    def _extract_calibration_features(self, landmarks: List) -> Dict:
        """Extrae características clave para calibración"""
        if not landmarks or len(landmarks) < 63:
            return {}
        
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        
        features = {}
        
        # Puntos clave
        wrist = landmarks_array[0]
        thumb_tip = landmarks_array[4]
        index_tip = landmarks_array[8]
        middle_tip = landmarks_array[12]
        ring_tip = landmarks_array[16]
        pinky_tip = landmarks_array[20]
        
        # Características geométricas
        features['hand_width'] = self._calculate_hand_width(landmarks_array)
        features['hand_height'] = self._calculate_hand_height(landmarks_array)
        features['finger_angles'] = self._calculate_finger_angles(landmarks_array)
        features['finger_distances'] = self._calculate_finger_distances(landmarks_array)
        features['palm_center'] = self._calculate_palm_center(landmarks_array)
        
        return features
    
    def _calculate_hand_width(self, landmarks_array):
        """Calcula el ancho de la mano"""
        x_coords = landmarks_array[:, 0]
        return np.max(x_coords) - np.min(x_coords)
    
    def _calculate_hand_height(self, landmarks_array):
        """Calcula la altura de la mano"""
        y_coords = landmarks_array[:, 1]
        return np.max(y_coords) - np.min(y_coords)
    
    def _calculate_finger_angles(self, landmarks_array):
        """Calcula ángulos de flexión de los dedos"""
        angles = {}
        
        # Definir articulaciones para cada dedo
        finger_joints = {
            'thumb': [(1, 2, 3), (2, 3, 4)],
            'index': [(5, 6, 7), (6, 7, 8)],
            'middle': [(9, 10, 11), (10, 11, 12)],
            'ring': [(13, 14, 15), (14, 15, 16)],
            'pinky': [(17, 18, 19), (18, 19, 20)]
        }
        
        for finger, joints in finger_joints.items():
            finger_angles = []
            for joint in joints:
                if all(j < len(landmarks_array) for j in joint):
                    angle = self._calculate_angle(
                        landmarks_array[joint[0]],
                        landmarks_array[joint[1]],
                        landmarks_array[joint[2]]
                    )
                    finger_angles.append(angle)
            angles[finger] = finger_angles
        
        return angles
    
    def _calculate_finger_distances(self, landmarks_array):
        """Calcula distancias entre puntas de dedos"""
        distances = {}
        fingertips = [4, 8, 12, 16, 20]  # Índices de puntas
        
        for i, tip1 in enumerate(fingertips):
            for j, tip2 in enumerate(fingertips):
                if i < j and tip1 < len(landmarks_array) and tip2 < len(landmarks_array):
                    dist = np.linalg.norm(landmarks_array[tip1] - landmarks_array[tip2])
                    distances[f'{tip1}_{tip2}'] = dist
        
        return distances
    
    def _calculate_palm_center(self, landmarks_array):
        """Calcula el centro de la palma"""
        # Usar puntos base de los dedos
        palm_points = [0, 5, 9, 13, 17]  # Muñeca y bases de dedos
        valid_points = [landmarks_array[i] for i in palm_points if i < len(landmarks_array)]
        
        if valid_points:
            return np.mean(valid_points, axis=0).tolist()
        return [0, 0, 0]
    
    def _calculate_angle(self, point1, point2, point3):
        """Calcula ángulo entre tres puntos"""
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        norm1, norm2 = np.linalg.norm(vector1), np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return 0
        
        cos_angle = np.dot(vector1, vector2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.arccos(cos_angle) * 180 / np.pi
    
    def calibrate_gesture(self, letter: str) -> bool:
        """Calibra un gesto específico basado en las muestras recolectadas"""
        if letter not in self.gesture_samples or len(self.gesture_samples[letter]) < 10:
            return False
        
        samples = self.gesture_samples[letter]
        
        # Calcular patrones promedio
        pattern = self._calculate_average_pattern(samples)
        
        # Calcular tolerancias dinámicas
        tolerances = self._calculate_dynamic_tolerances(samples)
        
        # Guardar calibración
        self.user_patterns[letter] = {
            'pattern': pattern,
            'tolerances': tolerances,
            'sample_count': len(samples),
            'calibration_date': time.time()
        }
        
        return True
    
    def _calculate_average_pattern(self, samples: List) -> Dict:
        """Calcula el patrón promedio de las muestras"""
        if not samples:
            return {}
        
        # Extraer todas las características
        all_features = [sample['features'] for sample in samples]
        
        pattern = {}
        
        # Promediar características numéricas
        pattern['hand_width'] = np.mean([f.get('hand_width', 0) for f in all_features])
        pattern['hand_height'] = np.mean([f.get('hand_height', 0) for f in all_features])
        
        # Promediar ángulos de dedos
        pattern['finger_angles'] = {}
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            finger_angles = []
            for f in all_features:
                if finger in f.get('finger_angles', {}):
                    finger_angles.extend(f['finger_angles'][finger])
            
            if finger_angles:
                pattern['finger_angles'][finger] = np.mean(finger_angles, axis=0).tolist()
        
        # Promediar distancias
        pattern['finger_distances'] = {}
        for f in all_features:
            for dist_key, dist_value in f.get('finger_distances', {}).items():
                if dist_key not in pattern['finger_distances']:
                    pattern['finger_distances'][dist_key] = []
                pattern['finger_distances'][dist_key].append(dist_value)
        
        # Calcular promedios de distancias
        for dist_key, dist_values in pattern['finger_distances'].items():
            pattern['finger_distances'][dist_key] = np.mean(dist_values)
        
        return pattern
    
    def _calculate_dynamic_tolerances(self, samples: List) -> Dict:
        """Calcula tolerancias dinámicas basadas en la variabilidad del usuario"""
        if not samples:
            return self.default_thresholds.copy()
        
        tolerances = self.default_thresholds.copy()
        
        # Calcular variabilidad en las muestras
        confidences = [s['confidence'] for s in samples]
        confidence_std = np.std(confidences)
        
        # Ajustar tolerancias basado en consistencia del usuario
        if confidence_std < 0.1:  # Usuario muy consistente
            tolerances['angle_tolerance'] = 10
            tolerances['distance_tolerance'] = 0.03
            tolerances['stability_frames'] = 10
        elif confidence_std > 0.2:  # Usuario menos consistente
            tolerances['angle_tolerance'] = 20
            tolerances['distance_tolerance'] = 0.08
            tolerances['stability_frames'] = 20
        
        return tolerances
    
    def get_personalized_threshold(self, letter: str, feature: str) -> float:
        """Obtiene umbral personalizado para una letra y característica específica"""
        if letter in self.user_patterns:
            return self.user_patterns[letter]['tolerances'].get(
                feature, 
                self.default_thresholds.get(feature, 0.5)
            )
        return self.default_thresholds.get(feature, 0.5)
    
    def is_gesture_match(self, letter: str, landmarks: List, confidence: float) -> tuple:
        """Verifica si un gesto coincide con el patrón calibrado"""
        if letter not in self.user_patterns:
            return False, 0.5  # Sin calibración, usar detección estándar
        
        pattern = self.user_patterns[letter]['pattern']
        tolerances = self.user_patterns[letter]['tolerances']
        
        # Extraer características del gesto actual
        current_features = self._extract_calibration_features(landmarks)
        
        # Calcular score de similitud
        similarity_score = self._calculate_similarity_score(current_features, pattern, tolerances)
        
        # Determinar si hay match
        match_threshold = tolerances.get('confidence_threshold', 0.7)
        is_match = similarity_score >= match_threshold
        
        return is_match, similarity_score
    
    def _calculate_similarity_score(self, current_features: Dict, pattern: Dict, tolerances: Dict) -> float:
        """Calcula score de similitud entre gesto actual y patrón"""
        if not current_features or not pattern:
            return 0.0
        
        scores = []
        
        # Comparar dimensiones de mano
        if 'hand_width' in current_features and 'hand_width' in pattern:
            width_diff = abs(current_features['hand_width'] - pattern['hand_width'])
            width_score = max(0, 1 - (width_diff / tolerances['distance_tolerance']))
            scores.append(width_score)
        
        # Comparar ángulos de dedos
        if 'finger_angles' in current_features and 'finger_angles' in pattern:
            angle_scores = []
            for finger in pattern['finger_angles']:
                if finger in current_features['finger_angles']:
                    for i, (current_angle, pattern_angle) in enumerate(
                        zip(current_features['finger_angles'][finger], 
                            pattern['finger_angles'][finger])
                    ):
                        angle_diff = abs(current_angle - pattern_angle)
                        angle_score = max(0, 1 - (angle_diff / tolerances['angle_tolerance']))
                        angle_scores.append(angle_score)
            
            if angle_scores:
                scores.append(np.mean(angle_scores))
        
        # Comparar distancias entre dedos
        if 'finger_distances' in current_features and 'finger_distances' in pattern:
            distance_scores = []
            for dist_key in pattern['finger_distances']:
                if dist_key in current_features['finger_distances']:
                    dist_diff = abs(current_features['finger_distances'][dist_key] - 
                                   pattern['finger_distances'][dist_key])
                    distance_score = max(0, 1 - (dist_diff / tolerances['distance_tolerance']))
                    distance_scores.append(distance_score)
            
            if distance_scores:
                scores.append(np.mean(distance_scores))
        
        # Retornar score promedio
        return np.mean(scores) if scores else 0.0
    
    def auto_calibrate_from_usage(self):
        """Calibración automática basada en el uso continuo"""
        calibrated_letters = []
        
        for letter in self.gesture_samples:
            if self.calibrate_gesture(letter):
                calibrated_letters.append(letter)
        
        if calibrated_letters:
            self.save_calibration()
        
        return calibrated_letters
    
    def save_calibration(self):
        """Guarda la calibración en archivo JSON"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            calibration_data = {
                'user_patterns': self.user_patterns,
                'calibrated_thresholds': self.calibrated_thresholds,
                'last_update': time.time()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
                
        except Exception as e:
            print(f"Error guardando calibración: {e}")
    
    def load_calibration(self):
        """Carga calibración desde archivo JSON"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    
                self.user_patterns = data.get('user_patterns', {})
                self.calibrated_thresholds = data.get('calibrated_thresholds', {})
                
        except Exception as e:
            print(f"Error cargando calibración: {e}")
            self.user_patterns = {}
            self.calibrated_thresholds = {}
    
    def get_calibration_status(self) -> Dict:
        """Obtiene el estado actual de la calibración"""
        status = {
            'total_letters': len(self.user_patterns),
            'sample_counts': {},
            'calibrated_letters': list(self.user_patterns.keys()),
            'needs_calibration': []
        }
        
        # Contar muestras por letra
        for letter, samples in self.gesture_samples.items():
            status['sample_counts'][letter] = len(samples)
            
            # Determinar si necesita más calibración
            if len(samples) < 10:
                status['needs_calibration'].append(letter)
        
        return status
    
    def reset_calibration(self, letter: Optional[str] = None):
        """Resetea calibración para una letra específica o todas"""
        if letter:
            if letter in self.user_patterns:
                del self.user_patterns[letter]
            if letter in self.gesture_samples:
                self.gesture_samples[letter].clear()
        else:
            self.user_patterns.clear()
            self.gesture_samples.clear()
            self.calibrated_thresholds.clear()
        
        self.save_calibration()