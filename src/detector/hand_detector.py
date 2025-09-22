# src/detector/hand_detector.py (Versión Mejorada)

import cv2
import numpy as np
from typing import Optional, Tuple, List
import math

class HandDetector:
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        
        # Configuración mejorada para detección de contornos
        self.min_contour_area = 8000
        self.max_contour_area = 80000
        
        # Variables para tracking y filtrado
        self.hand_history = []
        self.hand_detected = False
        self.last_hand_center = None
        self.detection_stability = 0
        
        # Configuración de colores de piel más robusta
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        
    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detecta manos en el frame usando OpenCV mejorado
        """
        processed_frame = frame.copy()
        hand_landmarks_list = []
        
        # Preprocesamiento mejorado
        # 1. Reducir ruido
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 2. Convertir a HSV para mejor detección de piel
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        
        # 3. Crear máscara de piel más robusta
        mask = self._create_skin_mask(hsv)
        
        # 4. Limpiar la máscara con operaciones morfológicas
        mask = self._clean_mask(mask)
        
        # 5. Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 6. Procesar contornos válidos
        valid_contours = self._filter_contours(contours)
        
        for contour in valid_contours:
            # Analizar contorno para extraer características
            hand_features = self._analyze_hand_contour(contour, processed_frame)
            
            if hand_features:
                landmarks = self._contour_to_landmarks(contour, hand_features, frame.shape)
                if landmarks:
                    hand_landmarks_list.append(landmarks)
                    self._draw_enhanced_landmarks(processed_frame, contour, hand_features)
        
        return processed_frame, hand_landmarks_list
    
    def _create_skin_mask(self, hsv_frame):
        """Crear máscara de piel más robusta"""
        # Máscara principal
        mask1 = cv2.inRange(hsv_frame, self.skin_lower, self.skin_upper)
        
        # Máscara adicional para tonos de piel más claros
        lower_skin2 = np.array([0, 10, 60], dtype=np.uint8)
        upper_skin2 = np.array([20, 150, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv_frame, lower_skin2, upper_skin2)
        
        # Combinar máscaras
        mask = cv2.bitwise_or(mask1, mask2)
        
        return mask
    
    def _clean_mask(self, mask):
        """Limpiar la máscara con operaciones morfológicas"""
        # Kernel para operaciones morfológicas
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)
        
        # Eliminar ruido pequeño
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Rellenar huecos
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Suavizar bordes
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        return mask
    
    def _filter_contours(self, contours):
        """Filtrar contornos para mantener solo los más probables de ser manos"""
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtro por área
            if not (self.min_contour_area < area < self.max_contour_area):
                continue
            
            # Filtro por forma (aspecto ratio)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            if not (0.4 < aspect_ratio < 2.5):  # Manos no son demasiado alargadas
                continue
            
            # Filtro por solidez (qué tan lleno está el contorno)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity < 0.6:  # Manos tienen buena solidez
                    continue
            
            valid_contours.append(contour)
        
        # Ordenar por área (más grande primero)
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        
        # Tomar solo las 2 más grandes (máximo 2 manos)
        return valid_contours[:2]
    
    def _analyze_hand_contour(self, contour, frame):
        """Analizar contorno para extraer características de la mano"""
        # Calcular hull convexo
        hull = cv2.convexHull(contour, returnPoints=False)
        
        if len(hull) < 4:
            return None
        
        # Calcular defectos de convexidad (espacios entre dedos)
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is None:
            return None
        
        # Encontrar centro de masa
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None
        
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
        
        # Analizar defectos para encontrar dedos
        finger_tips = []
        finger_valleys = []
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Calcular distancias
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            
            # Calcular ángulo
            if b != 0 and c != 0:
                angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 180 / math.pi
                
                # Si el ángulo es agudo, probablemente es un valle entre dedos
                if angle <= 90 and d > 20:
                    finger_valleys.append(far)
                    
                    # Los puntos start y end podrían ser puntas de dedos
                    if start[1] < center_y:  # Por encima del centro
                        finger_tips.append(start)
                    if end[1] < center_y:
                        finger_tips.append(end)
        
        # Eliminar duplicados de puntas de dedos
        finger_tips = self._remove_duplicate_points(finger_tips, threshold=30)
        
        return {
            'center': (center_x, center_y),
            'finger_tips': finger_tips,
            'finger_valleys': finger_valleys,
            'contour_area': cv2.contourArea(contour),
            'hull_area': cv2.contourArea(cv2.convexHull(contour))
        }
    
    def _remove_duplicate_points(self, points, threshold=30):
        """Eliminar puntos duplicados que están muy cerca"""
        if not points:
            return []
        
        unique_points = [points[0]]
        
        for point in points[1:]:
            is_duplicate = False
            for unique_point in unique_points:
                distance = math.sqrt((point[0] - unique_point[0])**2 + 
                                   (point[1] - unique_point[1])**2)
                if distance < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_points.append(point)
        
        return unique_points
    
    def _contour_to_landmarks(self, contour, hand_features, frame_shape):
        """Convertir características del contorno a landmarks"""
        height, width = frame_shape[:2]
        landmarks = []
        
        # Normalizar centro de masa
        center = hand_features['center']
        landmarks.extend([center[0] / width, center[1] / height, 0.0])
        
        # Agregar puntas de dedos normalizadas
        finger_tips = hand_features['finger_tips']
        for tip in finger_tips[:5]:  # Máximo 5 dedos
            landmarks.extend([tip[0] / width, tip[1] / height, 0.0])
        
        # Agregar valles entre dedos
        valleys = hand_features['finger_valleys']
        for valley in valleys[:4]:  # Máximo 4 valles
            landmarks.extend([valley[0] / width, valley[1] / height, 0.0])
        
        # Rellenar hasta 63 elementos (21 puntos * 3 coordenadas)
        while len(landmarks) < 63:
            landmarks.extend([0.0, 0.0, 0.0])
        
        return landmarks[:63]
    
    def _draw_enhanced_landmarks(self, frame, contour, hand_features):
        """Dibujar landmarks y características mejoradas"""
        # Dibujar contorno
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        # Dibujar hull convexo
        hull = cv2.convexHull(contour)
        cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)
        
        # Dibujar centro
        center = hand_features['center']
        cv2.circle(frame, center, 8, (255, 0, 255), -1)
        cv2.putText(frame, 'CENTER', (center[0] + 10, center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Dibujar puntas de dedos
        for i, tip in enumerate(hand_features['finger_tips']):
            cv2.circle(frame, tip, 6, (0, 255, 255), -1)
            cv2.putText(frame, f'F{i+1}', (tip[0] + 10, tip[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Dibujar valles
        for i, valley in enumerate(hand_features['finger_valleys']):
            cv2.circle(frame, valley, 4, (255, 255, 0), -1)
            cv2.putText(frame, f'V{i+1}', (valley[0] + 10, valley[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Mostrar número de dedos detectados
        num_fingers = len(hand_features['finger_tips'])
        cv2.putText(frame, f'Dedos: {num_fingers}', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def get_hand_bbox(self, landmarks: List, frame_shape: Tuple) -> Optional[Tuple]:
        """Obtiene la caja delimitadora de la mano"""
        if not landmarks:
            return None
        
        height, width = frame_shape[:2]
        
        # Convertir landmarks normalizados a píxeles
        x_coords = [landmarks[i] * width for i in range(0, len(landmarks), 3) if landmarks[i] > 0]
        y_coords = [landmarks[i] * height for i in range(1, len(landmarks), 3) if landmarks[i] > 0]
        
        if not x_coords or not y_coords:
            return None
        
        # Obtener coordenadas mínimas y máximas
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Añadir padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def extract_hand_region(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """Extrae la región de la mano del frame"""
        if bbox is None:
            return None
        
        x_min, y_min, x_max, y_max = bbox
        hand_region = frame[y_min:y_max, x_min:x_max]
        
        return hand_region if hand_region.size > 0 else None