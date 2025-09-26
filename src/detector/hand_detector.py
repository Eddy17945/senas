# src/detector/hand_detector.py (Versión MediaPipe)

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List

class HandDetector:
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configurar el detector de manos
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Variables para tracking mejorado
        self.hand_history = []
        self.detection_stability = 0
        
    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detecta manos usando MediaPipe y las clasifica por izquierda/derecha
        """
        # Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Procesar el frame con MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Convertir de vuelta a BGR
        rgb_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Diccionario para almacenar landmarks por mano
        hands_data = {
            'left': None,
            'right': None,
            'landmarks_list': []  # Para compatibilidad con código existente
        }
        
        # Procesar resultados si se detectan manos
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Determinar si es mano izquierda o derecha
                hand_label = handedness.classification[0].label
                hand_score = handedness.classification[0].score
                
                # MediaPipe devuelve "Left"/"Right" desde la perspectiva de la persona
                # Pero en imagen espejo necesitamos invertir
                actual_hand = "right" if hand_label == "Left" else "left"
                
                # Dibujar landmarks con estilo mejorado
                self._draw_enhanced_landmarks(
                    processed_frame, hand_landmarks, actual_hand, hand_score
                )
                
                # Extraer coordenadas de landmarks
                landmarks = self._extract_landmarks(hand_landmarks)
                if landmarks:
                    hands_data[actual_hand] = landmarks
                    hands_data['landmarks_list'].append(landmarks)  # Para compatibilidad
        
        return processed_frame, hands_data
    
    def _extract_landmarks(self, hand_landmarks) -> List:
        """
        Extrae las coordenadas normalizadas de los 21 landmarks de la mano
        """
        landmarks = []
        
        # MediaPipe proporciona 21 landmarks específicos para cada mano
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return landmarks
    
    def _draw_enhanced_landmarks(self, frame, hand_landmarks, hand_label, confidence):
        """
        Dibuja landmarks con información adicional y mejor estilo
        """
        height, width, _ = frame.shape
        
        # Dibujar conexiones de la mano
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Agregar etiquetas a landmarks importantes
        landmark_names = {
            0: "Muñeca",
            4: "Pulgar",
            8: "Índice", 
            12: "Medio",
            16: "Anular",
            20: "Meñique"
        }
        
        for idx, name in landmark_names.items():
            if idx < len(hand_landmarks.landmark):
                landmark = hand_landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                # Dibujar punto más visible
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
                cv2.circle(frame, (x, y), 8, (0, 0, 0), 2)
                
                # Etiqueta del punto
                cv2.putText(frame, name, (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Información de la mano detectada
        wrist = hand_landmarks.landmark[0]
        wrist_x = int(wrist.x * width)
        wrist_y = int(wrist.y * height)
        
        # Mostrar etiqueta de mano y confianza
        label_text = f"{hand_label}: {confidence:.2f}"
        cv2.putText(frame, label_text, (wrist_x - 50, wrist_y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Análisis de dedos extendidos
        fingers_up = self._count_fingers(hand_landmarks)
        cv2.putText(frame, f"Dedos: {fingers_up}", (wrist_x - 50, wrist_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    def _count_fingers(self, hand_landmarks) -> int:
        """
        Cuenta cuántos dedos están extendidos usando landmarks de MediaPipe
        """
        # IDs de las puntas de los dedos
        tip_ids = [4, 8, 12, 16, 20]  # Pulgar, Índice, Medio, Anular, Meñique
        pip_ids = [3, 6, 10, 14, 18]  # Articulaciones PIP correspondientes
        
        fingers = []
        landmarks = hand_landmarks.landmark
        
        # Pulgar (comparación horizontal)
        if landmarks[tip_ids[0]].x > landmarks[pip_ids[0]].x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Otros cuatro dedos (comparación vertical)
        for i in range(1, 5):
            if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return sum(fingers)
    
    def get_finger_positions(self, landmarks: List) -> dict:
        """
        Obtiene las posiciones específicas de cada dedo
        """
        if not landmarks or len(landmarks) < 63:  # 21 landmarks * 3 coordenadas
            return {}
        
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        
        finger_positions = {
            'wrist': landmarks_array[0],
            'thumb_tip': landmarks_array[4],
            'thumb_ip': landmarks_array[3],
            'index_tip': landmarks_array[8],
            'index_pip': landmarks_array[6],
            'middle_tip': landmarks_array[12],
            'middle_pip': landmarks_array[10],
            'ring_tip': landmarks_array[16],
            'ring_pip': landmarks_array[14],
            'pinky_tip': landmarks_array[20],
            'pinky_pip': landmarks_array[18]
        }
        
        return finger_positions
    
    def get_hand_orientation(self, landmarks: List) -> str:
        """
        Determina la orientación general de la mano
        """
        positions = self.get_finger_positions(landmarks)
        if not positions:
            return "unknown"
        
        wrist = positions['wrist']
        middle_tip = positions['middle_tip']
        
        # Calcular el vector de la mano
        dx = middle_tip[0] - wrist[0]
        dy = middle_tip[1] - wrist[1]
        
        # Determinar orientación basada en el ángulo
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        if -45 < angle < 45:
            return "right"
        elif 45 < angle < 135:
            return "down"
        elif angle > 135 or angle < -135:
            return "left"
        else:
            return "up"
    
    def calculate_gesture_features(self, landmarks: List) -> dict:
        """
        Calcula características avanzadas del gesto para clasificación
        """
        positions = self.get_finger_positions(landmarks)
        if not positions:
            return {}
        
        features = {}
        
        # 1. Estado de cada dedo (extendido/doblado)
        features['thumb_extended'] = positions['thumb_tip'][1] < positions['thumb_ip'][1]
        features['index_extended'] = positions['index_tip'][1] < positions['index_pip'][1]
        features['middle_extended'] = positions['middle_tip'][1] < positions['middle_pip'][1]
        features['ring_extended'] = positions['ring_tip'][1] < positions['ring_pip'][1]
        features['pinky_extended'] = positions['pinky_tip'][1] < positions['pinky_pip'][1]
        
        # 2. Distancias entre dedos
        features['thumb_index_distance'] = np.linalg.norm(
            positions['thumb_tip'] - positions['index_tip']
        )
        features['index_middle_distance'] = np.linalg.norm(
            positions['index_tip'] - positions['middle_tip']
        )
        
        # 3. Apertura de la mano
        fingertips = [positions['thumb_tip'], positions['index_tip'], 
                     positions['middle_tip'], positions['ring_tip'], positions['pinky_tip']]
        
        x_coords = [tip[0] for tip in fingertips]
        y_coords = [tip[1] for tip in fingertips]
        
        features['hand_width'] = max(x_coords) - min(x_coords)
        features['hand_height'] = max(y_coords) - min(y_coords)
        
        # 4. Centro de masa de la mano
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        features['center_of_mass'] = (center_x, center_y)
        
        return features
    
    def get_hand_bbox(self, landmarks: List, frame_shape: Tuple) -> Optional[Tuple]:
        """
        Obtiene la caja delimitadora de la mano con mejor precisión
        """
        if not landmarks or len(landmarks) < 63:
            return None
        
        height, width = frame_shape[:2]
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        
        # Obtener coordenadas en píxeles
        x_coords = landmarks_array[:, 0] * width
        y_coords = landmarks_array[:, 1] * height
        
        # Calcular bounding box
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Agregar padding proporcional
        padding_x = int((x_max - x_min) * 0.2)
        padding_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(width, x_max + padding_x)
        y_max = min(height, y_max + padding_y)
        
        return (x_min, y_min, x_max, y_max)
    
    def extract_hand_region(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """
        Extrae la región de la mano del frame
        """
        if bbox is None:
            return None
        
        x_min, y_min, x_max, y_max = bbox
        hand_region = frame[y_min:y_max, x_min:x_max]
        
        return hand_region if hand_region.size > 0 else None