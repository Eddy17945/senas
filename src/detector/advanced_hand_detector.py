# src/detector/advanced_hand_detector_optimized.py
# OPTIMIZADO PARA MOVIMIENTOS RÁPIDOS

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque

class AdvancedHandDetector:
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.6,  # REDUCIDO para capturar más
                 min_tracking_confidence: float = 0.5):   # REDUCIDO para seguimiento rápido
        
        # Inicializar MediaPipe con configuración RÁPIDA
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configurar el detector OPTIMIZADO para velocidad
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=0  # CAMBIO CRÍTICO: 0 = más rápido, 1 = más lento
        )
        
        # Sistema de filtrado temporal REDUCIDO
        self.landmarks_history = {
            'left': deque(maxlen=3),   # REDUCIDO de 10 a 3 para velocidad
            'right': deque(maxlen=3)
        }
        
        # Configuración de suavizado MÍNIMO
        self.smoothing_enabled = True
        self.smoothing_factor = 0.3  # Menos suavizado = más responsivo
        
        # Sistema de validación simplificado
        self.gesture_validator = GestureValidator()
        
        # Configuración de iluminación adaptativa
        self.lighting_adapter = LightingAdapter()
        
    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detección RÁPIDA de manos optimizada para movimientos veloces
        """
        # 1. Preprocesamiento LIGERO (sin procesamiento pesado)
        enhanced_frame = self.lighting_adapter.enhance_frame_fast(frame)
        
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
        
        # 6. Aplicar filtrado temporal MÍNIMO solo si está habilitado
        if self.smoothing_enabled:
            filtered_hands = self._apply_minimal_filtering(hands_data)
        else:
            filtered_hands = hands_data
        
        # 7. Validar gestos (validación rápida)
        validated_hands = self._validate_gestures_fast(filtered_hands)
        
        # 8. Dibujar landmarks
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
            
            # Extraer landmarks SIN procesamiento extra
            landmarks = self._extract_landmarks_fast(hand_landmarks)
            
            if landmarks:
                # Calcular score de calidad SIMPLE
                quality_score = confidence  # Usar confianza de MediaPipe directamente
                
                hands_data[actual_hand] = landmarks
                hands_data['landmarks_list'].append(landmarks)
                hands_data['confidence'][actual_hand] = confidence
                hands_data['quality_score'][actual_hand] = quality_score
        
        return hands_data
    
    def _extract_landmarks_fast(self, hand_landmarks):
        """Extrae landmarks RÁPIDO sin procesamiento extra"""
        landmarks = []
        
        # Extraer coordenadas con precisión normal (no excesiva)
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return landmarks
    
    def _apply_minimal_filtering(self, hands_data):
        """Aplica filtrado MÍNIMO para suavizar sin retraso"""
        filtered_data = hands_data.copy()
        
        for hand in ['left', 'right']:
            if hands_data[hand] is not None:
                # Agregar a historial
                self.landmarks_history[hand].append(hands_data[hand])
                
                # Aplicar suavizado LIGERO solo si hay suficiente historial
                if len(self.landmarks_history[hand]) >= 2:
                    filtered_landmarks = self._smooth_landmarks_fast(
                        self.landmarks_history[hand]
                    )
                    filtered_data[hand] = filtered_landmarks
        
        return filtered_data
    
    def _smooth_landmarks_fast(self, landmarks_history):
        """Suaviza landmarks con MÍNIMO retraso"""
        if not landmarks_history:
            return None
        
        # Si solo hay una muestra, devolverla directamente
        if len(landmarks_history) == 1:
            return landmarks_history[0]
        
        # Promedio simple ponderado (más peso al frame actual)
        current = np.array(landmarks_history[-1])
        previous = np.array(landmarks_history[-2])
        
        # Suavizado ligero: 70% actual, 30% anterior
        smoothed = current * 0.7 + previous * 0.3
        
        return smoothed.tolist()
    
    def _validate_gestures_fast(self, hands_data):
        """Valida gestos con criterios PERMISIVOS para velocidad"""
        validated_data = hands_data.copy()
        
        for hand in ['left', 'right']:
            if hands_data[hand] is not None:
                quality = hands_data['quality_score'][hand]
                
                # UMBRAL MUY BAJO para aceptar más detecciones
                if quality < 0.3:  # Solo rechazar si es MUY baja calidad
                    validated_data[hand] = None
                    validated_data['quality_score'][hand] = 0.0
        
        return validated_data
    
    def _draw_enhanced_landmarks(self, frame, hands_data):
        """Dibuja landmarks con información adicional"""
        height, width = frame.shape[:2]
        
        for hand in ['left', 'right']:
            if hands_data[hand] is not None:
                landmarks = hands_data[hand]
                confidence = hands_data['confidence'][hand]
                quality = hands_data['quality_score'][hand]
                
                # Dibujar landmarks básicos
                self._draw_hand_landmarks(frame, landmarks, hand, confidence, quality)
    
    def _draw_hand_landmarks(self, frame, landmarks, hand, confidence, quality):
        """Dibuja los landmarks de una mano específica"""
        if len(landmarks) < 63:
            return
        
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        height, width = frame.shape[:2]
        
        # Color verde siempre (confianza visual alta)
        color = (0, 255, 0)
        
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
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)
        
        # Dibujar conexiones entre puntos
        self._draw_hand_connections(frame, landmarks_array, color, width, height)
    
    def _draw_hand_connections(self, frame, landmarks_array, color, width, height):
        """Dibuja las conexiones entre landmarks"""
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
    
    def set_smoothing(self, enabled: bool):
        """Activa/desactiva el suavizado"""
        self.smoothing_enabled = enabled
    
    def set_smoothing_factor(self, factor: float):
        """Ajusta el factor de suavizado (0.0 = sin suavizado, 1.0 = máximo)"""
        self.smoothing_factor = max(0.0, min(1.0, factor))


class GestureValidator:
    """Clase SIMPLE para validar gestos"""
    
    def validate(self, hands_data):
        """Validación MÍNIMA para velocidad"""
        # En modo rápido, aceptar casi todo
        return hands_data


class LightingAdapter:
    """Clase OPTIMIZADA para adaptarse a iluminación"""
    
    def enhance_frame_fast(self, frame):
        """Mejora el frame con procesamiento MÍNIMO"""
        # Procesamiento ligero - solo ajuste de brillo si es necesario
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Solo ajustar si está muy oscuro o muy claro
        if mean_brightness < 60:  # Muy oscuro
            return self._brighten_frame(frame, 1.3)
        elif mean_brightness > 200:  # Muy claro
            return self._brighten_frame(frame, 0.8)
        else:
            return frame  # Sin procesamiento
    
    def _brighten_frame(self, frame, factor):
        """Ajusta brillo rápidamente"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)