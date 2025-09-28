# src/detector/advanced_hand_detector.py

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque
import math

class AdvancedHandDetector:
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.8,  # Aumentado para mayor precisión
                 min_tracking_confidence: float = 0.7):   # Aumentado para mayor estabilidad
        
        # Inicializar MediaPipe con configuración optimizada
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configurar el detector con parámetros optimizados
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1  # Usar modelo más complejo para mejor precisión
        )
        
        # Sistema de filtrado temporal
        self.landmarks_history = {
            'left': deque(maxlen=10),   # Mantener últimos 10 frames
            'right': deque(maxlen=10)
        }
        
        # Filtro de Kalman simplificado
        self.kalman_filters = {
            'left': self._init_kalman_filter(),
            'right': self._init_kalman_filter()
        }
        
        # Sistema de validación de gestos
        self.gesture_validator = GestureValidator()
        
        # Configuración de iluminación adaptativa
        self.lighting_adapter = LightingAdapter()
        
    def _init_kalman_filter(self):
        """Inicializa filtro de Kalman para suavizar landmarks"""
        return {
            'positions': np.zeros((21, 3)),  # 21 landmarks con x,y,z
            'velocities': np.zeros((21, 3)),
            'alpha': 0.7,  # Factor de suavizado
            'initialized': False
        }
    
    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detección avanzada de manos con múltiples mejoras
        """
        # 1. Preprocesamiento de imagen para mejor detección
        enhanced_frame = self.lighting_adapter.enhance_frame(frame)
        
        # 2. Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # 3. Procesar con MediaPipe
        results = self.hands.process(rgb_frame)
        
        # 4. Convertir de vuelta a BGR
        rgb_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # 5. Procesar y filtrar resultados
        hands_data = self._process_detection_results(results, processed_frame)
        
        # 6. Aplicar filtrado temporal y validación
        filtered_hands = self._apply_temporal_filtering(hands_data)
        
        # 7. Validar gestos
        validated_hands = self._validate_gestures(filtered_hands)
        
        # 8. Dibujar landmarks mejorados
        self._draw_enhanced_landmarks(processed_frame, validated_hands)
        
        return processed_frame, validated_hands
    
    def _process_detection_results(self, results, frame):
        """Procesa los resultados brutos de MediaPipe"""
        hands_data = {
            'left': None,
            'right': None,
            'landmarks_list': [],
            'confidence': {'left': 0.0, 'right': 0.0},
            'quality_score': {'left': 0.0, 'right': 0.0}
        }
        
        if not results.multi_hand_landmarks:
            return hands_data
        
        for i, (hand_landmarks, handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            # Determinar mano (invertir para imagen espejo)
            hand_label = handedness.classification[0].label
            confidence = handedness.classification[0].score
            actual_hand = "right" if hand_label == "Left" else "left"
            
            # Extraer landmarks
            landmarks = self._extract_high_precision_landmarks(hand_landmarks)
            
            if landmarks:
                # Calcular score de calidad
                quality_score = self._calculate_quality_score(landmarks, hand_landmarks)
                
                hands_data[actual_hand] = landmarks
                hands_data['landmarks_list'].append(landmarks)
                hands_data['confidence'][actual_hand] = confidence
                hands_data['quality_score'][actual_hand] = quality_score
        
        return hands_data
    
    def _extract_high_precision_landmarks(self, hand_landmarks):
        """Extrae landmarks con mayor precisión"""
        landmarks = []
        
        # Extraer coordenadas con mayor precisión
        for landmark in hand_landmarks.landmark:
            # Usar mayor precisión decimal
            x = round(landmark.x, 6)
            y = round(landmark.y, 6)
            z = round(landmark.z, 6)
            landmarks.extend([x, y, z])
        
        return landmarks
    
    def _calculate_quality_score(self, landmarks, hand_landmarks):
        """Calcula un score de calidad para la detección"""
        if not landmarks:
            return 0.0
        
        score = 1.0
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        
        # 1. Penalizar por landmarks fuera del frame
        out_of_bounds = sum(1 for point in landmarks_array 
                           if point[0] < 0 or point[0] > 1 or point[1] < 0 or point[1] > 1)
        score -= (out_of_bounds * 0.1)
        
        # 2. Verificar consistencia anatómica
        anatomical_score = self._check_anatomical_consistency(landmarks_array)
        score *= anatomical_score
        
        # 3. Penalizar por movimiento brusco (si hay historial)
        stability_score = self._check_stability(landmarks_array)
        score *= stability_score
        
        return max(0.0, min(1.0, score))
    
    def _check_anatomical_consistency(self, landmarks_array):
        """Verifica que los landmarks sean anatómicamente consistentes"""
        try:
            # Verificar que los dedos están en orden correcto
            finger_tips = [4, 8, 12, 16, 20]
            finger_bases = [2, 5, 9, 13, 17]
            
            consistency_score = 1.0
            
            # Verificar longitudes de dedos razonables
            for tip, base in zip(finger_tips, finger_bases):
                if tip < len(landmarks_array) and base < len(landmarks_array):
                    finger_length = np.linalg.norm(landmarks_array[tip] - landmarks_array[base])
                    # Longitud de dedo debe estar en rango razonable
                    if finger_length < 0.05 or finger_length > 0.3:
                        consistency_score -= 0.1
            
            return max(0.0, consistency_score)
        
        except Exception:
            return 0.5  # Score neutro si falla la verificación
    
    def _check_stability(self, landmarks_array):
        """Verifica estabilidad temporal de los landmarks"""
        # Implementación simplificada
        return 1.0  # Por ahora retorna score máximo
    
    def _apply_temporal_filtering(self, hands_data):
        """Aplica filtrado temporal para suavizar detecciones"""
        filtered_data = hands_data.copy()
        
        for hand in ['left', 'right']:
            if hands_data[hand] is not None:
                # Agregar a historial
                self.landmarks_history[hand].append(hands_data[hand])
                
                # Aplicar filtro si tenemos suficiente historial
                if len(self.landmarks_history[hand]) >= 3:
                    filtered_landmarks = self._smooth_landmarks(
                        self.landmarks_history[hand], 
                        hand
                    )
                    filtered_data[hand] = filtered_landmarks
        
        return filtered_data
    
    def _smooth_landmarks(self, landmarks_history, hand):
        """Suaviza landmarks usando promedio ponderado"""
        if not landmarks_history:
            return None
        
        # Convertir historial a array numpy
        history_array = np.array(list(landmarks_history))
        
        # Aplicar pesos (más peso a frames recientes)
        weights = np.linspace(0.5, 1.0, len(history_array))
        weights = weights / weights.sum()
        
        # Calcular promedio ponderado
        smoothed = np.average(history_array, axis=0, weights=weights)
        
        return smoothed.tolist()
    
    def _validate_gestures(self, hands_data):
        """Valida que los gestos detectados sean plausibles"""
        return self.gesture_validator.validate(hands_data)
    
    def _draw_enhanced_landmarks(self, frame, hands_data):
        """Dibuja landmarks con información adicional de calidad"""
        height, width = frame.shape[:2]
        
        for hand in ['left', 'right']:
            if hands_data[hand] is not None:
                landmarks = hands_data[hand]
                confidence = hands_data['confidence'][hand]
                quality = hands_data['quality_score'][hand]
                
                # Dibujar landmarks básicos
                self._draw_hand_landmarks(frame, landmarks, hand, confidence, quality)
                
                # Dibujar información de calidad
                self._draw_quality_indicators(frame, hand, confidence, quality, width, height)
    
    def _draw_hand_landmarks(self, frame, landmarks, hand, confidence, quality):
        """Dibuja los landmarks de una mano específica"""
        if len(landmarks) < 63:
            return
        
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        height, width = frame.shape[:2]
        
        # Color según la calidad
        if quality > 0.8:
            color = (0, 255, 0)  # Verde para alta calidad
        elif quality > 0.6:
            color = (0, 255, 255)  # Amarillo para calidad media
        else:
            color = (0, 0, 255)  # Rojo para baja calidad
        
        # Dibujar puntos importantes
        important_points = {
            0: "Muñeca",
            4: "Pulgar", 
            8: "Índice",
            12: "Medio",
            16: "Anular", 
            20: "Meñique"
        }
        
        for idx, name in important_points.items():
            if idx < len(landmarks_array):
                point = landmarks_array[idx]
                x, y = int(point[0] * width), int(point[1] * height)
                
                # Dibujar punto
                cv2.circle(frame, (x, y), 6, color, -1)
                cv2.circle(frame, (x, y), 6, (0, 0, 0), 2)
                
                # Etiqueta
                cv2.putText(frame, name, (x + 8, y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Dibujar conexiones entre puntos
        self._draw_hand_connections(frame, landmarks_array, color, width, height)
    
    def _draw_hand_connections(self, frame, landmarks_array, color, width, height):
        """Dibuja las conexiones entre landmarks"""
        # Conexiones básicas de la mano
        connections = [
            # Pulgar
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Índice  
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Medio
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Anular
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Meñique
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks_array) and end_idx < len(landmarks_array):
                start_point = landmarks_array[start_idx]
                end_point = landmarks_array[end_idx]
                
                start_x, start_y = int(start_point[0] * width), int(start_point[1] * height)
                end_x, end_y = int(end_point[0] * width), int(end_point[1] * height)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
    
    def _draw_quality_indicators(self, frame, hand, confidence, quality, width, height):
        """Dibuja indicadores de calidad de detección"""
        # Posición para mostrar info
        if hand == 'left':
            pos_x, pos_y = 10, 30
        else:
            pos_x, pos_y = width - 200, 30
        
        # Texto de información
        info_text = f"{hand.upper()}: {confidence:.2f} | Q: {quality:.2f}"
        
        # Color del texto basado en calidad
        if quality > 0.8:
            text_color = (0, 255, 0)
        elif quality > 0.6:
            text_color = (0, 255, 255)
        else:
            text_color = (0, 0, 255)
        
        cv2.putText(frame, info_text, (pos_x, pos_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


class GestureValidator:
    """Clase para validar la plausibilidad de gestos detectados"""
    
    def validate(self, hands_data):
        """Valida y potencialmente corrige gestos detectados"""
        validated_data = hands_data.copy()
        
        for hand in ['left', 'right']:
            if hands_data[hand] is not None:
                quality = hands_data['quality_score'][hand]
                
                # Solo mantener detecciones de alta calidad
                if quality < 0.5:
                    validated_data[hand] = None
                    validated_data['quality_score'][hand] = 0.0
        
        return validated_data


class LightingAdapter:
    """Clase para adaptarse a diferentes condiciones de iluminación"""
    
    def enhance_frame(self, frame):
        """Mejora el frame para mejor detección bajo diferentes iluminaciones"""
        # 1. Ecualización de histograma adaptativo
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 2. Ajuste de gamma si es necesario
        gamma = self._calculate_optimal_gamma(enhanced)
        if gamma != 1.0:
            enhanced = self._adjust_gamma(enhanced, gamma)
        
        return enhanced
    
    def _calculate_optimal_gamma(self, frame):
        """Calcula el gamma óptimo basado en la iluminación"""
        # Calcular brillo promedio
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Ajustar gamma basado en brillo
        if mean_brightness < 80:  # Imagen oscura
            return 1.2
        elif mean_brightness > 180:  # Imagen clara
            return 0.8
        else:
            return 1.0  # No ajuste necesario
    
    def _adjust_gamma(self, frame, gamma):
        """Aplica corrección gamma"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(frame, table)