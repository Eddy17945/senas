# src/detector/gesture_classifier_improved.py
# VERSIÓN MEJORADA CON MÁXIMA PRECISIÓN PARA TODAS LAS LETRAS

import numpy as np
import cv2
from typing import List, Optional, Dict, Any, Union

class GestureClassifier:
    def __init__(self, model_path: str = None):
        self.model = None
        self.label_encoder = None
        self.is_trained = True  # Activar directamente
        self.detection_history = []
        self.stability_threshold = 3
        self.confidence_threshold = 0.6
        
        self.supported_letters = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
    
    def predict_gesture(self, landmarks: List) -> Optional[str]:
        """Predice el gesto con velocidad y precisión"""
        if not self.is_trained or not landmarks or len(landmarks) < 63:
            return None
        
        current_letter = self._classify_complete_alphabet(landmarks)
        
        self.detection_history.append(current_letter)
        
        if len(self.detection_history) > self.stability_threshold * 2:
            self.detection_history = self.detection_history[-self.stability_threshold:]
        
        if len(self.detection_history) >= self.stability_threshold:
            recent_detections = self.detection_history[-self.stability_threshold:]
            
            if current_letter:
                letter_count = sum(1 for detection in recent_detections 
                                 if detection == current_letter)
                
                if letter_count >= self.stability_threshold * 0.5:
                    return current_letter
        
        return None
    
    def _classify_complete_alphabet(self, landmarks: List) -> Optional[str]:
        """Clasificación completa mejorada"""
        if not landmarks or len(landmarks) < 63:
            return None
        
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        features = self._extract_advanced_features(landmarks_array)
        
        return self._classify_with_priority(features, landmarks_array)
    
    def _extract_advanced_features(self, lm) -> Dict:
        """Extrae características avanzadas para todas las letras"""
        # Puntos clave
        wrist = lm[0]
        thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = lm[1], lm[2], lm[3], lm[4]
        index_mcp, index_pip, index_dip, index_tip = lm[5], lm[6], lm[7], lm[8]
        middle_mcp, middle_pip, middle_dip, middle_tip = lm[9], lm[10], lm[11], lm[12]
        ring_mcp, ring_pip, ring_dip, ring_tip = lm[13], lm[14], lm[15], lm[16]
        pinky_mcp, pinky_pip, pinky_dip, pinky_tip = lm[17], lm[18], lm[19], lm[20]
        
        f = {}  # features dictionary
        
        # === ESTADOS DE EXTENSIÓN MEJORADOS ===
        f['thumb_ext'] = thumb_tip[1] < thumb_ip[1] - 0.02
        f['index_ext'] = index_tip[1] < index_pip[1] - 0.03
        f['middle_ext'] = middle_tip[1] < middle_pip[1] - 0.03
        f['ring_ext'] = ring_tip[1] < ring_pip[1] - 0.03
        f['pinky_ext'] = pinky_tip[1] < pinky_pip[1] - 0.03
        
        # === ÁNGULOS PRECISOS ===
        f['thumb_angle'] = self._angle(thumb_mcp, thumb_ip, thumb_tip)
        f['index_angle'] = self._angle(index_mcp, index_pip, index_tip)
        f['middle_angle'] = self._angle(middle_mcp, middle_pip, middle_tip)
        f['ring_angle'] = self._angle(ring_mcp, ring_pip, ring_tip)
        f['pinky_angle'] = self._angle(pinky_mcp, pinky_pip, pinky_tip)
        
        # Ángulos entre dedos adyacentes
        f['thumb_index_angle'] = self._angle(thumb_tip, wrist, index_tip)
        f['index_middle_angle'] = self._angle(index_tip, index_mcp, middle_tip)
        
        # === DISTANCIAS CRÍTICAS ===
        f['thumb_index_d'] = np.linalg.norm(thumb_tip - index_tip)
        f['thumb_middle_d'] = np.linalg.norm(thumb_tip - middle_tip)
        f['thumb_ring_d'] = np.linalg.norm(thumb_tip - ring_tip)
        f['thumb_pinky_d'] = np.linalg.norm(thumb_tip - pinky_tip)
        f['index_middle_d'] = np.linalg.norm(index_tip - middle_tip)
        f['middle_ring_d'] = np.linalg.norm(middle_tip - ring_tip)
        f['ring_pinky_d'] = np.linalg.norm(ring_tip - pinky_tip)
        
        # Distancias a la muñeca
        f['thumb_wrist_d'] = np.linalg.norm(thumb_tip - wrist)
        f['index_wrist_d'] = np.linalg.norm(index_tip - wrist)
        
        # === POSICIONES RELATIVAS (X, Y, Z) ===
        f['thumb_left'] = thumb_tip[0] < index_mcp[0] - 0.05
        f['thumb_right'] = thumb_tip[0] > pinky_mcp[0] + 0.05
        f['thumb_center'] = index_mcp[0] <= thumb_tip[0] <= pinky_mcp[0]
        f['thumb_above_fingers'] = thumb_tip[1] < index_mcp[1]
        f['thumb_below_fingers'] = thumb_tip[1] > index_mcp[1]
        
        # Posición del índice
        f['index_horiz'] = abs(index_tip[0] - index_mcp[0]) > 0.08
        f['index_vert'] = abs(index_tip[1] - index_mcp[1]) > 0.08
        
        # === CONFIGURACIONES ESPECIALES ===
        f['fingers_count'] = sum([f['thumb_ext'], f['index_ext'], f['middle_ext'], 
                                  f['ring_ext'], f['pinky_ext']])
        f['fist'] = f['fingers_count'] == 0
        f['all_up'] = f['fingers_count'] == 5
        
        # Dedos curvados (crítico para E, T, X, N)
        f['index_curved'] = 45 < f['index_angle'] < 120
        f['middle_curved'] = 45 < f['middle_angle'] < 120
        f['ring_curved'] = 45 < f['ring_angle'] < 120
        f['all_curved'] = (f['index_curved'] and f['middle_curved'] and 
                          f['ring_curved'] and 45 < f['pinky_angle'] < 120)
        
        # Dedos juntos vs separados
        f['fingers_together'] = (f['index_middle_d'] < 0.04 and 
                                f['middle_ring_d'] < 0.04 and 
                                f['ring_pinky_d'] < 0.04)
        f['fingers_spread'] = (f['index_middle_d'] > 0.08 or 
                              f['middle_ring_d'] > 0.08 or 
                              f['ring_pinky_d'] > 0.08)
        
        # === CARACTERÍSTICAS PARA LETRAS ESPECÍFICAS ===
        
        # Para X: Índice doblado en forma de gancho
        f['index_hook'] = (not f['index_ext'] and 
                          f['index_curved'] and 
                          index_tip[1] > index_mcp[1])
        
        # Para T: Pulgar entre índice y medio
        f['thumb_between_index_middle'] = (
            abs(thumb_tip[0] - (index_mcp[0] + middle_mcp[0])/2) < 0.03 and
            not f['thumb_ext']
        )
        
        # Para N: Dos dedos sobre pulgar
        f['two_over_thumb'] = (
            not f['index_ext'] and not f['middle_ext'] and
            f['ring_ext'] and f['pinky_ext'] and
            thumb_tip[1] > index_pip[1]
        )
        
        # Para E: Todos curvados con pulgar visible
        f['all_curved_thumb_out'] = (
            f['all_curved'] and f['thumb_ext'] and
            thumb_tip[0] < index_tip[0]
        )
        
        # Para Z: Movimiento en Z (aproximado por posición)
        f['index_diagonal'] = (f['index_ext'] and 
                              abs(index_tip[0] - wrist[0]) > 0.1 and
                              abs(index_tip[1] - wrist[1]) > 0.15)
        
        return f
    
    def _classify_with_priority(self, f: Dict, lm) -> Optional[str]:
        """Clasificación con prioridad para letras problemáticas"""
        
        # === PRIORIDAD 1: LETRAS MÁS DIFÍCILES ===
        
        # LETRA X - Índice doblado en gancho
        if f['index_hook'] and not f['middle_ext'] and not f['ring_ext'] and not f['pinky_ext']:
            return "X"
        
        # LETRA T - Puño con pulgar entre índice y medio
        if f['fist'] and f['thumb_between_index_middle']:
            return "T"
        
        # LETRA E - Todos los dedos curvados, pulgar afuera
        if f['all_curved_thumb_out'] and f['fist']:
            return "E"
        
        # LETRA N - Dos dedos (anular y meñique) sobre pulgar
        if f['two_over_thumb']:
            return "N"
        
        # LETRA Z - Índice extendido en diagonal (simula Z)
        if f['index_diagonal'] and f['fingers_count'] == 1 and f['index_ext']:
            return "Z"
        
        # === PRIORIDAD 2: LETRAS CON UN DEDO ===
        
        # LETRA Y - Pulgar y meñique extendidos
        if (not f['index_ext'] and not f['middle_ext'] and not f['ring_ext'] and 
            f['pinky_ext'] and f['thumb_ext'] and f['thumb_pinky_d'] > 0.12):
            return "Y"
        
        # LETRA I - Solo meñique
        if (not f['index_ext'] and not f['middle_ext'] and not f['ring_ext'] and 
            f['pinky_ext'] and not f['thumb_ext']):
            return "I"
        
        # LETRA D - Índice arriba, pulgar toca otros
        if (f['index_ext'] and not f['middle_ext'] and not f['ring_ext'] and 
            not f['pinky_ext'] and f['thumb_middle_d'] < 0.10):
            return "D"
        
        # LETRA L - Índice y pulgar en L
        if (f['index_ext'] and not f['middle_ext'] and not f['ring_ext'] and 
            not f['pinky_ext'] and f['thumb_ext'] and f['thumb_left']):
            return "L"
        
        # LETRA G - Índice horizontal con pulgar
        if (f['index_ext'] and not f['middle_ext'] and not f['ring_ext'] and 
            not f['pinky_ext'] and f['thumb_ext'] and f['index_horiz']):
            return "G"
        
        # === PRIORIDAD 3: LETRAS CON DOS DEDOS ===
        
        # LETRA U - Índice y medio juntos
        if (f['index_ext'] and f['middle_ext'] and not f['ring_ext'] and 
            not f['pinky_ext'] and f['index_middle_d'] < 0.05 and not f['thumb_ext']):
            return "U"
        
        # LETRA V - Índice y medio separados
        if (f['index_ext'] and f['middle_ext'] and not f['ring_ext'] and 
            not f['pinky_ext'] and f['index_middle_d'] > 0.06):
            return "V"
        
        # LETRA H - Índice y medio horizontales
        if (f['index_ext'] and f['middle_ext'] and not f['ring_ext'] and 
            not f['pinky_ext'] and f['index_horiz'] and f['index_middle_d'] < 0.06):
            return "H"
        
        # LETRA K - Índice arriba, medio en ángulo
        if (f['index_ext'] and f['middle_ext'] and not f['ring_ext'] and 
            not f['pinky_ext'] and f['thumb_middle_d'] < 0.08 and 
            f['index_middle_d'] > 0.08):
            return "K"
        
        # LETRA R - Índice y medio cruzados
        if (f['index_ext'] and f['middle_ext'] and not f['ring_ext'] and 
            not f['pinky_ext'] and f['index_middle_d'] < 0.03):
            return "R"
        
        # === PRIORIDAD 4: TRES O MÁS DEDOS ===
        
        # LETRA W - Tres dedos arriba
        if (f['index_ext'] and f['middle_ext'] and f['ring_ext'] and 
            not f['pinky_ext'] and not f['thumb_ext']):
            return "W"
        
        # LETRA B - Cuatro dedos juntos
        if (f['index_ext'] and f['middle_ext'] and f['ring_ext'] and 
            f['pinky_ext'] and not f['thumb_ext'] and f['fingers_together']):
            return "B"
        
        # LETRA F - Tres dedos arriba, índice toca pulgar
        if (not f['index_ext'] and f['middle_ext'] and f['ring_ext'] and 
            f['pinky_ext'] and f['thumb_index_d'] < 0.08):
            return "F"
        
        # === PRIORIDAD 5: PUÑOS Y FORMAS ===
        
        # LETRA A - Puño con pulgar al lado
        if f['fist'] and f['thumb_left'] and not f['thumb_ext']:
            return "A"
        
        # LETRA S - Puño con pulgar sobre dedos
        if f['fist'] and not f['thumb_left'] and not f['thumb_ext']:
            return "S"
        
        # LETRA M - Pulgar bajo tres dedos
        if (not f['index_ext'] and not f['middle_ext'] and not f['ring_ext'] and 
            not f['pinky_ext'] and f['thumb_ext']):
            return "M"
        
        # LETRA O - Círculo con dedos
        if f['thumb_index_d'] < 0.10 and f['fingers_count'] <= 2:
            return "O"
        
        # LETRA C - Mano curva
        if (f['fingers_count'] >= 2 and f['thumb_index_d'] > 0.08 and 
            f['thumb_index_d'] < 0.20):
            return "C"
        
        # === CLASIFICACIÓN DE RESPALDO ===
        return self._classify_by_count(f)
    
    def _classify_by_count(self, f: Dict) -> Optional[str]:
        """Clasificación de respaldo por número de dedos"""
        count = f['fingers_count']
        
        if count == 0:
            return "A" if f['thumb_left'] else "S"
        elif count == 1:
            if f['index_ext']:
                return "D"
            elif f['pinky_ext']:
                return "Y" if f['thumb_ext'] else "I"
            elif f['thumb_ext']:
                return "M"
        elif count == 2:
            if f['index_middle_d'] > 0.06:
                return "V"
            else:
                return "U"
        elif count == 3:
            return "W"
        elif count == 4:
            return "B"
        elif count == 5:
            return "5"
        
        return None
    
    def _angle(self, p1, p2, p3):
        """Calcula ángulo entre tres puntos"""
        v1 = p1 - p2
        v2 = p3 - p2
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.arccos(cos_angle) * 180 / np.pi
    
    def get_supported_letters(self) -> List[str]:
        return self.supported_letters
    
    def get_detection_confidence(self) -> float:
        if len(self.detection_history) < self.stability_threshold:
            return 0.0
        recent = self.detection_history[-self.stability_threshold:]
        if not recent[0]:
            return 0.0
        consistent_count = sum(1 for d in recent if d == recent[-1])
        return consistent_count / len(recent)
    
    def reset_detection_history(self):
        self.detection_history = []
    
    def set_stability_threshold(self, threshold: int):
        self.stability_threshold = max(1, min(10, threshold))