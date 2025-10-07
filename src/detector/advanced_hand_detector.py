# src/detector/advanced_hand_detector_optimized.py
# OPTIMIZADO PARA VELOCIDAD + PRECISIÓN MEJORADA

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque

class AdvancedHandDetector:
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.65,  # BALANCEADO: velocidad + precisión
                 min_tracking_confidence: float = 0.55):   # BALANCEADO
        
        # Inicializar MediaPipe con configuración BALANCEADA
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configurar el detector BALANCEADO
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=0  # 0 = rápido (mantener)
        )
        
        # Sistema de filtrado temporal OPTIMIZADO
        self.landmarks_history = {
            'left': deque(maxlen=4),   # 4 frames = buen balance
            'right': deque(maxlen=4)
        }
        
        # Configuración de suavizado BALANCEADO
        self.smoothing_enabled = True
        self.smoothing_factor = 0.4  # 40% suavizado = buen balance
        
        # Sistema de validación mejorado
        self.gesture_validator = GestureValidator()
        
        # Configuración de iluminación adaptativa
        self.lighting_adapter = LightingAdapter()
        
        # Historial de calidad para validación
        self.quality_history = {
            'left': deque(maxlen=10),
            'right': deque(maxlen=10)
        }
        
    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detección BALANCEADA: rápida y precisa
        """
        # 1. Preprocesamiento OPTIMIZADO
        enhanced_frame = self.lighting_adapter.enhance_frame_fast(frame)  # CORREGIDO
        
        # 2. Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # 3. Procesar con MediaPipe
        results = self.hands.process(rgb_frame)
        
        # 4. Convertir de vuelta a BGR
        rgb_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # 5. Procesar resultados con validación mejorada
        hands_data = self._process_detection_results_enhanced(results, processed_frame)
        
        # 6. Aplicar filtrado temporal BALANCEADO
        if self.smoothing_enabled:
            filtered_hands = self._apply_smart_filtering(hands_data)
        else:
            filtered_hands = hands_data
        
        # 7. Validar gestos con sistema mejorado
        validated_hands = self._validate_gestures_enhanced(filtered_hands)
        
        # 8. Dibujar landmarks
        self._draw_enhanced_landmarks(processed_frame, validated_hands)
        
        return processed_frame, validated_hands
    
    def _process_detection_results_enhanced(self, results, frame):
        """Procesa resultados con validación de calidad mejorada"""
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
            # Determinar mano
            hand_label = handedness.classification[0].label
            confidence = handedness.classification[0].score
            actual_hand = "right" if hand_label == "Left" else "left"
            
            # Extraer landmarks
            landmarks = self._extract_landmarks_precise(hand_landmarks)
            
            if landmarks:
                # Calcular score de calidad MEJORADO
                quality_score = self._calculate_enhanced_quality(
                    landmarks, hand_landmarks, confidence
                )
                
                # Agregar a historial de calidad
                self.quality_history[actual_hand].append(quality_score)
                
                # Calcular calidad promedio
                avg_quality = np.mean(list(self.quality_history[actual_hand])) if \
                             self.quality_history[actual_hand] else quality_score
                
                hands_data[actual_hand] = landmarks
                hands_data['landmarks_list'].append(landmarks)
                hands_data['confidence'][actual_hand] = confidence
                hands_data['quality_score'][actual_hand] = avg_quality
        
        return hands_data
    
    def _extract_landmarks_precise(self, hand_landmarks):
        """Extrae landmarks con precisión balanceada"""
        landmarks = []
        
        # Extraer coordenadas con buena precisión
        for landmark in hand_landmarks.landmark:
            # 4 decimales = buen balance precisión/velocidad
            x = round(landmark.x, 4)
            y = round(landmark.y, 4)
            z = round(landmark.z, 4)
            landmarks.extend([x, y, z])
        
        return landmarks
    
    def _calculate_enhanced_quality(self, landmarks, hand_landmarks, confidence):
        """Calcula quality score mejorado"""
        if not landmarks:
            return 0.0
        
        score = confidence  # Empezar con confianza de MediaPipe
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        
        # 1. Verificar que puntos estén dentro del frame
        out_of_bounds = sum(1 for point in landmarks_array 
                           if point[0] < 0 or point[0] > 1 or 
                              point[1] < 0 or point[1] > 1)
        score -= (out_of_bounds * 0.05)
        
        # 2. Verificar consistencia anatómica MEJORADA
        anatomical_score = self._check_anatomical_consistency_enhanced(landmarks_array)
        score *= anatomical_score
        
        # 3. Verificar estabilidad temporal
        stability_score = self._check_temporal_stability(landmarks_array)
        score *= stability_score
        
        # 4. Bonus por visibilidad completa
        if out_of_bounds == 0:
            score *= 1.05
        
        return max(0.0, min(1.0, score))
    
    def _check_anatomical_consistency_enhanced(self, landmarks_array):
        """Verifica consistencia anatómica mejorada"""
        try:
            consistency_score = 1.0
            
            # Verificar longitudes de dedos
            finger_pairs = [
                (4, 2),   # Pulgar
                (8, 5),   # Índice
                (12, 9),  # Medio
                (16, 13), # Anular
                (20, 17)  # Meñique
            ]
            
            for tip, base in finger_pairs:
                if tip < len(landmarks_array) and base < len(landmarks_array):
                    length = np.linalg.norm(landmarks_array[tip] - landmarks_array[base])
                    
                    # Longitudes razonables: 0.08 - 0.25
                    if length < 0.08 or length > 0.25:
                        consistency_score -= 0.08
            
            # Verificar que dedos estén en orden correcto (no cruzados imposiblemente)
            fingertips_x = [landmarks_array[i][0] for i in [4, 8, 12, 16, 20] 
                           if i < len(landmarks_array)]
            
            # Verificar que no haya saltos extremos
            if len(fingertips_x) >= 4:
                diffs = [abs(fingertips_x[i+1] - fingertips_x[i]) for i in range(len(fingertips_x)-1)]
                if any(diff > 0.3 for diff in diffs):  # Saltos muy grandes
                    consistency_score -= 0.1
            
            return max(0.3, consistency_score)
        
        except Exception:
            return 0.7
    
    def _check_temporal_stability(self, landmarks_array):
        """Verifica estabilidad temporal para reducir jitter"""
        # Si no hay historial suficiente, asumir estable
        if not hasattr(self, '_last_landmarks') or self._last_landmarks is None:
            self._last_landmarks = landmarks_array
            return 1.0
        
        try:
            # Calcular diferencia con frame anterior
            diff = np.linalg.norm(landmarks_array - self._last_landmarks)
            
            # Movimiento razonable: 0.0 - 0.5
            if diff > 0.5:  # Movimiento muy brusco
                stability = 0.8
            elif diff > 0.3:  # Movimiento moderado
                stability = 0.9
            else:  # Movimiento suave
                stability = 1.0
            
            self._last_landmarks = landmarks_array
            return stability
        
        except Exception:
            return 0.9
    
    def _apply_smart_filtering(self, hands_data):
        """Aplica filtrado inteligente según calidad"""
        filtered_data = hands_data.copy()
        
        for hand in ['left', 'right']:
            if hands_data[hand] is not None:
                quality = hands_data['quality_score'][hand]
                
                # Agregar a historial
                self.landmarks_history[hand].append(hands_data[hand])
                
                # Aplicar filtrado según calidad
                if quality > 0.8:
                    # Alta calidad: poco suavizado (más responsivo)
                    if len(self.landmarks_history[hand]) >= 2:
                        filtered_landmarks = self._smooth_minimal(
                            self.landmarks_history[hand]
                        )
                        filtered_data[hand] = filtered_landmarks
                elif quality > 0.6:
                    # Calidad media: suavizado moderado
                    if len(self.landmarks_history[hand]) >= 3:
                        filtered_landmarks = self._smooth_moderate(
                            self.landmarks_history[hand]
                        )
                        filtered_data[hand] = filtered_landmarks
                else:
                    # Baja calidad: más suavizado para estabilizar
                    if len(self.landmarks_history[hand]) >= 4:
                        filtered_landmarks = self._smooth_strong(
                            self.landmarks_history[hand]
                        )
                        filtered_data[hand] = filtered_landmarks
        
        return filtered_data
    
    def _smooth_minimal(self, history):
        """Suavizado mínimo: 80% actual, 20% anterior"""
        current = np.array(history[-1])
        previous = np.array(history[-2])
        smoothed = current * 0.8 + previous * 0.2
        return smoothed.tolist()
    
    def _smooth_moderate(self, history):
        """Suavizado moderado: promedio de 3 últimos con pesos"""
        weights = np.array([0.2, 0.3, 0.5])  # Más peso al reciente
        history_array = np.array(list(history)[-3:])
        smoothed = np.average(history_array, axis=0, weights=weights)
        return smoothed.tolist()
    
    def _smooth_strong(self, history):
        """Suavizado fuerte: promedio de 4 últimos con pesos"""
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        history_array = np.array(list(history)[-4:])
        smoothed = np.average(history_array, axis=0, weights=weights)
        return smoothed.tolist()
    
    def _validate_gestures_enhanced(self, hands_data):
        """Valida gestos con criterios mejorados"""
        validated_data = hands_data.copy()
        
        for hand in ['left', 'right']:
            if hands_data[hand] is not None:
                quality = hands_data['quality_score'][hand]
                confidence = hands_data['confidence'][hand]
                
                # Umbral adaptativo según historial
                min_quality = 0.4 if len(self.quality_history[hand]) < 5 else 0.5
                min_confidence = 0.5
                
                # Rechazar solo si calidad Y confianza son muy bajas
                if quality < min_quality and confidence < min_confidence:
                    validated_data[hand] = None
                    validated_data['quality_score'][hand] = 0.0
        
        return validated_data
    
    def _draw_enhanced_landmarks(self, frame, hands_data):
        """Dibuja landmarks con información de calidad"""
        height, width = frame.shape[:2]
        
        for hand in ['left', 'right']:
            if hands_data[hand] is not None:
                landmarks = hands_data[hand]
                confidence = hands_data['confidence'][hand]
                quality = hands_data['quality_score'][hand]
                
                self._draw_hand_landmarks(frame, landmarks, hand, confidence, quality)
    
    def _draw_hand_landmarks(self, frame, landmarks, hand, confidence, quality):
        """Dibuja landmarks con colores según calidad"""
        if len(landmarks) < 63:
            return
        
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        height, width = frame.shape[:2]
        
        # Color según calidad
        if quality > 0.8:
            color = (0, 255, 0)  # Verde = excelente
        elif quality > 0.6:
            color = (0, 255, 255)  # Amarillo = bueno
        elif quality > 0.4:
            color = (0, 165, 255)  # Naranja = regular
        else:
            color = (0, 0, 255)  # Rojo = bajo
        
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
                
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)
        
        # Dibujar conexiones
        self._draw_hand_connections(frame, landmarks_array, color, width, height)
    
    def _draw_hand_connections(self, frame, landmarks_array, color, width, height):
        """Dibuja conexiones entre landmarks"""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
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
        """Ajusta el factor de suavizado"""
        self.smoothing_factor = max(0.0, min(1.0, factor))


class GestureValidator:
    """Validación mejorada de gestos"""
    
    def validate(self, hands_data):
        """Validación con criterios adaptativos"""
        return hands_data


class LightingAdapter:
    """Adaptación BALANCEADA a iluminación"""
    
    def enhance_frame_balanced(self, frame):
        """Mejora frame con procesamiento balanceado"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Solo procesar si hay problemas significativos
        if mean_brightness < 70:  # Muy oscuro
            return self._brighten_frame(frame, 1.25)
        elif mean_brightness > 190:  # Muy claro
            return self._brighten_frame(frame, 0.85)
        else:
            return frame  # No procesar para velocidad
    
    def _brighten_frame(self, frame, factor):
        """Ajusta brillo eficientemente"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
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