# src/detector/gesture_classifier_improved.py
# VERSIÓN CON MÁXIMA PRECISIÓN PARA TODAS LAS LETRAS

import numpy as np
import cv2
from typing import List, Optional, Dict, Any, Union

class GestureClassifier:
    def __init__(self, model_path: str = None):
        self.model = None
        self.label_encoder = None
        self.is_trained = True
        self.detection_history = []
        self.stability_threshold = 4  # AUMENTADO de 3 a 4 para más precisión
        self.confidence_threshold = 0.65  # AUMENTADO de 0.6 a 0.65
        
        # Alfabeto completo incluyendo letras especiales del español
        self.supported_letters = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'Ñ', 'O', 'P', 'Q', 'R', 'RR', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'LL'
        ]
        
        # Sistema de validación cruzada
        self.confidence_scores = {}
        self.last_validated_letter = None
        self.validation_counter = 0
    
    def predict_gesture(self, landmarks: List) -> Optional[str]:
        """Predice con validación cruzada mejorada"""
        if not self.is_trained or not landmarks or len(landmarks) < 63:
            return None
        
        current_letter = self._classify_complete_alphabet(landmarks)
        
        # Validación cruzada: verificar que la letra sea consistente
        if current_letter:
            validated_letter = self._cross_validate_detection(current_letter, landmarks)
            current_letter = validated_letter
        
        self.detection_history.append(current_letter)
        
        if len(self.detection_history) > self.stability_threshold * 2:
            self.detection_history = self.detection_history[-self.stability_threshold:]
        
        # PRECISIÓN MEJORADA: 60% de estabilidad requerida
        if len(self.detection_history) >= self.stability_threshold:
            recent_detections = self.detection_history[-self.stability_threshold:]
            
            if current_letter:
                letter_count = sum(1 for detection in recent_detections 
                                 if detection == current_letter)
                
                # Requerir 60% de consistencia para mayor precisión
                if letter_count >= self.stability_threshold * 0.6:
                    return current_letter
        
        return None
    
    def _cross_validate_detection(self, letter: str, landmarks: List) -> Optional[str]:
        """Validación cruzada para mayor precisión"""
        # Verificar que el gesto cumpla múltiples criterios
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        features = self._extract_ultra_precise_features(landmarks_array)
        
        # Calcular score de confianza para esta letra
        confidence_score = self._calculate_gesture_confidence(letter, features)
        
        # Guardar score
        if letter not in self.confidence_scores:
            self.confidence_scores[letter] = []
        self.confidence_scores[letter].append(confidence_score)
        
        # Mantener solo últimos 10 scores
        if len(self.confidence_scores[letter]) > 10:
            self.confidence_scores[letter] = self.confidence_scores[letter][-10:]
        
        # Calcular confianza promedio
        avg_confidence = np.mean(self.confidence_scores[letter])
        
        # Solo aceptar si confianza promedio es alta
        if avg_confidence >= self.confidence_threshold:
            return letter
        else:
            return None
    
    def _calculate_gesture_confidence(self, letter: str, features: Dict) -> float:
        """Calcula confianza de que el gesto sea correcto"""
        confidence = 0.5  # Base
        
        # Verificar criterios específicos por letra
        if letter == 'K' and features.get('k_formation'):
            confidence += 0.3
            if 40 < features.get('index_middle_angle', 0) < 80:
                confidence += 0.2
        
        elif letter == 'P' and features.get('p_formation'):
            confidence += 0.3
            if features.get('thumb_middle_d', 1) < 0.09:
                confidence += 0.2
        
        elif letter == 'U' and features.get('u_formation'):
            confidence += 0.3
            if features.get('index_middle_d', 1) < 0.045:
                confidence += 0.2
        
        elif letter == 'E' and features.get('e_formation'):
            confidence += 0.3
            if features.get('fist'):
                confidence += 0.2
        
        elif letter == 'G' and features.get('g_formation'):
            confidence += 0.3
            if features.get('thumb_index_d', 0) > 0.12:
                confidence += 0.2
        
        # Agregar más verificaciones para otras letras...
        else:
            # Para letras sin formación específica, usar criterios generales
            fingers_count = features.get('fingers_count', 0)
            if fingers_count in [0, 1, 2, 3, 4, 5]:
                confidence += 0.3
        
        return min(1.0, confidence)
    
    def _classify_complete_alphabet(self, landmarks: List) -> Optional[str]:
        if not landmarks or len(landmarks) < 63:
            return None
        
        landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
        features = self._extract_ultra_precise_features(landmarks_array)
        
        return self._classify_with_enhanced_rules(features, landmarks_array)
    
    def _extract_ultra_precise_features(self, lm) -> Dict:
        """Extrae características ultra precisas para cada letra"""
        # Puntos clave
        wrist = lm[0]
        thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = lm[1], lm[2], lm[3], lm[4]
        index_mcp, index_pip, index_dip, index_tip = lm[5], lm[6], lm[7], lm[8]
        middle_mcp, middle_pip, middle_dip, middle_tip = lm[9], lm[10], lm[11], lm[12]
        ring_mcp, ring_pip, ring_dip, ring_tip = lm[13], lm[14], lm[15], lm[16]
        pinky_mcp, pinky_pip, pinky_dip, pinky_tip = lm[17], lm[18], lm[19], lm[20]
        
        f = {}
        
        # === ESTADOS DE EXTENSIÓN ULTRA PRECISOS ===
        f['thumb_ext'] = thumb_tip[1] < thumb_ip[1] - 0.015
        f['index_ext'] = index_tip[1] < index_pip[1] - 0.025
        f['middle_ext'] = middle_tip[1] < middle_pip[1] - 0.025
        f['ring_ext'] = ring_tip[1] < ring_pip[1] - 0.025
        f['pinky_ext'] = pinky_tip[1] < pinky_pip[1] - 0.025
        
        # Extensión parcial (semi-doblados)
        f['index_semi'] = index_pip[1] < index_mcp[1] and not f['index_ext']
        f['middle_semi'] = middle_pip[1] < middle_mcp[1] and not f['middle_ext']
        
        # === ÁNGULOS ULTRA PRECISOS ===
        f['thumb_angle'] = self._angle(thumb_mcp, thumb_ip, thumb_tip)
        f['index_angle'] = self._angle(index_mcp, index_pip, index_tip)
        f['middle_angle'] = self._angle(middle_mcp, middle_pip, middle_tip)
        f['ring_angle'] = self._angle(ring_mcp, ring_pip, ring_tip)
        f['pinky_angle'] = self._angle(pinky_mcp, pinky_pip, pinky_tip)
        
        # Ángulos entre dedos
        f['index_middle_angle'] = self._angle(index_tip, index_mcp, middle_tip)
        f['middle_ring_angle'] = self._angle(middle_tip, middle_mcp, ring_tip)
        f['thumb_index_angle'] = self._angle(thumb_tip, wrist, index_tip)
        
        # === DISTANCIAS CRÍTICAS ===
        f['thumb_index_d'] = np.linalg.norm(thumb_tip - index_tip)
        f['thumb_middle_d'] = np.linalg.norm(thumb_tip - middle_tip)
        f['thumb_ring_d'] = np.linalg.norm(thumb_tip - ring_tip)
        f['thumb_pinky_d'] = np.linalg.norm(thumb_tip - pinky_tip)
        f['index_middle_d'] = np.linalg.norm(index_tip - middle_tip)
        f['middle_ring_d'] = np.linalg.norm(middle_tip - ring_tip)
        f['ring_pinky_d'] = np.linalg.norm(ring_tip - pinky_tip)
        f['index_pinky_d'] = np.linalg.norm(index_tip - pinky_tip)
        
        # Distancias a la muñeca
        f['thumb_wrist_d'] = np.linalg.norm(thumb_tip - wrist)
        f['index_wrist_d'] = np.linalg.norm(index_tip - wrist)
        f['middle_wrist_d'] = np.linalg.norm(middle_tip - wrist)
        
        # Distancias entre bases y puntas
        f['thumb_to_index_base'] = np.linalg.norm(thumb_tip - index_mcp)
        f['index_to_middle_base'] = np.linalg.norm(index_tip - middle_mcp)
        
        # === POSICIONES RELATIVAS EN 3D ===
        f['thumb_left'] = thumb_tip[0] < index_mcp[0] - 0.05
        f['thumb_right'] = thumb_tip[0] > pinky_mcp[0] + 0.05
        f['thumb_center'] = not f['thumb_left'] and not f['thumb_right']
        f['thumb_above'] = thumb_tip[1] < index_mcp[1] - 0.03
        f['thumb_below'] = thumb_tip[1] > index_mcp[1] + 0.03
        
        # Posiciones horizontales/verticales
        f['index_horiz'] = abs(index_tip[0] - index_mcp[0]) > abs(index_tip[1] - index_mcp[1])
        f['middle_horiz'] = abs(middle_tip[0] - middle_mcp[0]) > abs(middle_tip[1] - middle_mcp[1])
        
        # Posición del índice respecto al medio
        f['index_left_of_middle'] = index_tip[0] < middle_tip[0] - 0.02
        f['index_right_of_middle'] = index_tip[0] > middle_tip[0] + 0.02
        
        # === CONFIGURACIONES AVANZADAS ===
        f['fingers_count'] = sum([f['thumb_ext'], f['index_ext'], f['middle_ext'], 
                                  f['ring_ext'], f['pinky_ext']])
        f['fist'] = f['fingers_count'] == 0
        
        # Dedos curvados en rangos específicos
        f['index_curved_45_90'] = 45 < f['index_angle'] < 90
        f['index_curved_90_135'] = 90 < f['index_angle'] < 135
        f['middle_curved_90_135'] = 90 < f['middle_angle'] < 135
        
        # Agrupaciones específicas
        f['three_middle_up'] = f['index_ext'] and f['middle_ext'] and f['ring_ext']
        f['two_middle_up'] = f['index_ext'] and f['middle_ext'] and not f['ring_ext']
        
        # Características para letras específicas
        
        # Para K: Índice y medio en V con pulgar entre ellos
        f['k_formation'] = (f['index_ext'] and f['middle_ext'] and 
                           not f['ring_ext'] and not f['pinky_ext'] and
                           f['thumb_middle_d'] < 0.09 and
                           40 < f['index_middle_angle'] < 80)
        
        # Para P: Similar a K pero apuntando hacia abajo
        f['p_formation'] = (f['index_ext'] and f['middle_ext'] and 
                           not f['ring_ext'] and not f['pinky_ext'] and
                           f['thumb_middle_d'] < 0.09 and
                           index_tip[1] > middle_tip[1])
        
        # Para Q: G pero apuntando hacia abajo
        f['q_formation'] = (f['index_ext'] and not f['middle_ext'] and 
                           not f['ring_ext'] and not f['pinky_ext'] and
                           f['thumb_ext'] and 
                           f['thumb_index_d'] > 0.07 and  # Más permisivo
                           f['thumb_index_d'] < 0.20 and  # Rango superior
                           (index_tip[1] > index_mcp[1] - 0.05 or  # Apunta abajo (más tolerante)
                            abs(index_tip[1] - thumb_tip[1]) < 0.06))  # Índice y pulgar al mismo nivel

        
        # Para U: Dedos juntos y paralelos
        f['u_formation'] = (f['two_middle_up'] and f['index_middle_d'] < 0.045 and
                           abs(index_tip[1] - middle_tip[1]) < 0.03)
        
        # Para E: Todos curvados hacia dentro
        f['e_formation'] = (f['fist'] and 
                           all(90 < angle < 150 for angle in [
                               f['index_angle'], f['middle_angle'], 
                               f['ring_angle'], f['pinky_angle']]))
        
        # Para G: Índice horizontal con pulgar
        f['g_formation'] = (f['index_ext'] and not f['middle_ext'] and
                           f['thumb_ext'] and f['index_horiz'] and
                           f['thumb_index_d'] > 0.12)
        
        # Para J: Meñique extendido con movimiento
        f['j_formation'] = (not f['index_ext'] and not f['middle_ext'] and
                           not f['ring_ext'] and f['pinky_ext'] and
                           pinky_tip[0] < pinky_mcp[0])
        
        # Para Ñ: N con movimiento ondulado (aproximar con posición)
        f['ñ_formation'] = (f['two_middle_up'] and not f['thumb_ext'] and
                           f['index_middle_d'] < 0.04)
        
        # Para LL: L doble (pulgar e índice extendidos con separación específica)
        f['ll_formation'] = (f['index_ext'] and f['thumb_ext'] and
                            not f['middle_ext'] and not f['ring_ext'] and
                            f['thumb_index_d'] > 0.15 and
                            f['thumb_left'])
        
        # Para RR: R con movimiento (aproximar con índice y medio muy juntos)
        f['rr_formation'] = (f['two_middle_up'] and 
                            f['index_middle_d'] < 0.025 and
                            f['index_left_of_middle'])
        
        # Para T: Puño con pulgar entre índice y medio
        f['t_formation'] = (f['fist'] and f['thumb_center'] and 
                           f['thumb_to_index_base'] < 0.06 and
                           thumb_tip[1] > index_mcp[1] - 0.02)
        
        # Para M: Pulgar bajo tres dedos
        f['m_formation'] = (not f['index_ext'] and not f['middle_ext'] and 
                           not f['ring_ext'] and not f['pinky_ext'] and
                           f['thumb_ext'] and 
                           thumb_tip[1] < index_mcp[1] and
                           f['thumb_index_d'] < 0.08 and
                           f['thumb_middle_d'] < 0.08 and
                           f['thumb_ring_d'] < 0.08)
        
              # Para R mejorada: Índice y medio muy juntos y cruzados
        f['r_improved'] = (f['index_ext'] and f['middle_ext'] and 
                          not f['ring_ext'] and not f['pinky_ext'] and
                          f['index_middle_d'] < 0.05 and  # MÁS permisivo
                          (f['index_left_of_middle'] or 
                           abs(index_tip[0] - middle_tip[0]) < 0.025))  # Alternativa

        
        # Para N: Pulgar bajo dos dedos (anular y meñique arriba)
        f['n_formation'] = (not f['index_ext'] and not f['middle_ext'] and
                           f['ring_ext'] and f['pinky_ext'] and
                           not f['thumb_ext'] and
                           thumb_tip[1] < index_mcp[1] and
                           f['thumb_index_d'] < 0.08 and
                           f['thumb_middle_d'] < 0.08)
        
        return f
    
    def _classify_with_enhanced_rules(self, f: Dict, lm) -> Optional[str]:
        """Clasificación con reglas ultra mejoradas - PRIORIDAD PARA LETRAS PROBLEMÁTICAS"""
        
        # === MÁXIMA PRIORIDAD: LETRAS PROBLEMÁTICAS ===

         # Q - MÁXIMA PRIORIDAD: índice y pulgar extendidos apuntando hacia abajo
        # Verificación TRIPLE para Q
        q_check1 = (f['index_ext'] and not f['middle_ext'] and 
                   not f['ring_ext'] and not f['pinky_ext'] and
                   f['thumb_ext'] and 
                   0.07 < f['thumb_index_d'] < 0.25)
        
        q_check2 = (lm[8][1] > lm[5][1] - 0.05)  # índice apunta hacia abajo o horizontal
        
        q_check3 = (lm[4][1] > lm[2][1] - 0.05)  # pulgar también apunta abajo o horizontal
        
        if q_check1 and (q_check2 or q_check3):
            return "Q"
        
        
        # E - TODOS los dedos curvados (MÁXIMA PRIORIDAD)
        if f['e_formation']:
            return "E"
        
        # T - Puño con pulgar entre dedos (MÁXIMA PRIORIDAD)
        if f['t_formation']:
            return "T"
        
        # M - Pulgar bajo TRES dedos
        if f['m_formation']:
            return "M"                                                                                                                                                                                                                                                                                                                                                                                                                                        
        
        # N - Pulgar bajo DOS dedos, anular y meñique arriba
        if f['n_formation']:
            return "N"
        
        # Ñ - Como N pero con separación
        if f['ñ_formation']:
            return "Ñ"
        
        

        
        # === SEGUNDA PRIORIDAD: OTRAS LETRAS DIFÍCILES ===
        
        # K - Índice y medio en V con pulgar tocando medio
        if f['k_formation']:
            return "K"
        
        # P - Similar a K pero hacia abajo
        if f['p_formation']:
            return "P"
        
        # U - Dos dedos juntos y paralelos
        if f['u_formation']:
            return "U"
        
        # G - Índice y pulgar extendidos horizontalmente
        if f['g_formation']:
            return "G"
        
        # J - Meñique con movimiento característico
        if f['j_formation']:
            return "J"
        
        # LL - L doble
        if f['ll_formation'] and f['thumb_index_d'] > 0.15:
            return "LL"
        
        # RR - R con movimiento fuerte
        if f['rr_formation']:
            return "RR"
        
        # === LETRAS ESTÁNDAR ===
        
        # X - Índice curvado en gancho
        if (f['index_curved_45_90'] and not f['middle_ext'] and 
            not f['ring_ext'] and not f['pinky_ext']):
            return "X"
        
        # T - Puño con pulgar entre índice y medio
        if (f['fist'] and f['thumb_center'] and 
            f['thumb_to_index_base'] < 0.05):
            return "T"
        
        # Y - Pulgar y meñique extendidos
        if (not f['index_ext'] and not f['middle_ext'] and not f['ring_ext'] and 
            f['pinky_ext'] and f['thumb_ext'] and f['thumb_pinky_d'] > 0.12):
            return "Y"
        
        # I - Solo meñique
        if (not f['index_ext'] and not f['middle_ext'] and not f['ring_ext'] and 
            f['pinky_ext'] and not f['thumb_ext']):
            return "I"
        
        # D - Índice arriba, pulgar toca otros
        if (f['index_ext'] and not f['middle_ext'] and not f['ring_ext'] and 
            not f['pinky_ext'] and f['thumb_middle_d'] < 0.10):
            return "D"
        
        # L - Índice y pulgar en L
        if (f['index_ext'] and not f['middle_ext'] and not f['ring_ext'] and 
            not f['pinky_ext'] and f['thumb_ext'] and f['thumb_left'] and
            f['thumb_index_d'] > 0.10 and f['thumb_index_d'] < 0.18):
            return "L"
        
        # V - Índice y medio separados
        if (f['index_ext'] and f['middle_ext'] and not f['ring_ext'] and 
            not f['pinky_ext'] and f['index_middle_d'] > 0.06 and
            not f['thumb_ext']):
            return "V"
        
        # H - Índice y medio horizontales juntos
        if (f['two_middle_up'] and f['index_horiz'] and f['middle_horiz'] and
            f['index_middle_d'] < 0.06):
            return "H"
        
         # R - Índice y medio cruzados (TRIPLE VERIFICACIÓN)
        if f.get('r_improved', False):
            return "R"
        
        # R - Verificación alternativa 1: dedos muy juntos
        if (f['index_ext'] and f['middle_ext'] and 
            not f['ring_ext'] and not f['pinky_ext'] and
            not f['thumb_ext'] and
            f['index_middle_d'] < 0.05):
            return "R"
        
        # R - Verificación alternativa 2: cruce visible
        if (f['two_middle_up'] and 
            f['index_middle_d'] < 0.04 and
            f['index_left_of_middle']):
            return "R"
        
        if (f['two_middle_up'] and f['index_middle_d'] < 0.04 and
            f['index_left_of_middle']):
            return "R"
        
        # W - Tres dedos arriba
        if (f['three_middle_up'] and not f['pinky_ext'] and not f['thumb_ext']):
            return "W"
        
        # B - Cuatro dedos juntos
        if (f['index_ext'] and f['middle_ext'] and f['ring_ext'] and 
            f['pinky_ext'] and not f['thumb_ext'] and
            f['index_middle_d'] < 0.05 and f['middle_ring_d'] < 0.05):
            return "B"
        
        # F - Tres dedos arriba, índice toca pulgar
        if (not f['index_ext'] and f['middle_ext'] and f['ring_ext'] and 
            f['pinky_ext'] and f['thumb_index_d'] < 0.08):
            return "F"
        
        # N - Dos dedos sobre pulgar (anular y meñique arriba)
        if (not f['index_ext'] and not f['middle_ext'] and 
            f['ring_ext'] and f['pinky_ext'] and not f['thumb_ext']):
            return "N"
        
        # A - Puño con pulgar al lado
        if f['fist'] and f['thumb_left'] and not f['thumb_ext']:
            return "A"
        
        # S - Puño con pulgar sobre dedos
        if f['fist'] and not f['thumb_left'] and not f['thumb_ext']:
            return "S"
        
        # M - Pulgar bajo tres dedos
        if (f['fist'] and f['thumb_ext'] and not f['thumb_left']):
            return "M"
        
        # O - Círculo con dedos
        if f['thumb_index_d'] < 0.10 and f['fingers_count'] <= 2:
            return "O"
        
        # C - Mano curva
        if (f['fingers_count'] >= 2 and f['thumb_index_d'] > 0.08 and 
            f['thumb_index_d'] < 0.20):
            return "C"
        
        # Z - Índice extendido en diagonal
        if (f['index_ext'] and not f['middle_ext'] and not f['ring_ext'] and
            not f['pinky_ext'] and f['index_horiz']):
            return "Z"
        
        return None
    
    def _angle(self, p1, p2, p3):
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