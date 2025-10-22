# src/detector/complete_word_detector.py
"""
Detector de palabras completas por gestos únicos
Aumenta 10x la velocidad de comunicación
"""

import numpy as np
from typing import Optional, Dict, List, Tuple

class CompleteWordDetector:
    def __init__(self):
        # Mapeo de gestos únicos a palabras completas
        self.word_gestures = {
            # Saludos (los más usados)
            'THUMBS_UP': 'HOLA',
            'WAVE': 'ADIOS',
            'PEACE': 'BUENOS',
            'OK_SIGN': 'GRACIAS',
            
            # Cortesía
            'PRAY_HANDS': 'POR FAVOR',
            'SORRY_GESTURE': 'DISCULPA',
            'THANK_YOU': 'GRACIAS',
            
            # Necesidades básicas
            'POINTING_UP': 'NECESITO',
            'DRINK_GESTURE': 'AGUA',
            'EAT_GESTURE': 'COMIDA',
            'BATHROOM_SIGN': 'BAÑO',
            
            # Familia
            'HEART_HANDS': 'TE AMO',
            'MAMA_SIGN': 'MAMA',
            'PAPA_SIGN': 'PAPA',
            
            # Respuestas rápidas
            'THUMBS_DOWN': 'NO',
            'SHAKA': 'OK',
            'CALL_ME': 'AYUDA',
            
            # Emociones
            'HAPPY_SIGN': 'FELIZ',
            'SAD_SIGN': 'TRISTE',
        }
        
        # Historial de detección
        self.detection_history = []
        self.stability_threshold = 8  # Frames necesarios para confirmar
        self.last_detected_word = None
        self.cooldown_frames = 0
        self.cooldown_threshold = 30  # Evitar repetición inmediata
        
        # Estadísticas de uso
        self.usage_stats = {}
        
    def detect_complete_word(self, landmarks, confidence: float) -> Optional[str]:
        """
        Detecta si el gesto corresponde a una palabra completa
        
        Args:
            landmarks: Lista de landmarks (puede venir como lista plana o array)
            confidence: Confianza de la detección
            
        Returns:
            Palabra detectada o None
        """
        if landmarks is None:
            return None
        
        # Convertir landmarks a array numpy si es necesario
        try:
            if isinstance(landmarks, list):
                # Si es una lista plana de 63 elementos (21 landmarks * 3 coords)
                if len(landmarks) >= 63:
                    landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
                # Si es lista de listas [[x,y,z], [x,y,z], ...]
                elif len(landmarks) >= 21 and isinstance(landmarks[0], (list, tuple)):
                    landmarks_array = np.array(landmarks[:21])
                else:
                    return None
            elif isinstance(landmarks, np.ndarray):
                # Si ya es array, asegurar que tiene la forma correcta
                if landmarks.size >= 63:
                    landmarks_array = landmarks.flatten()[:63].reshape(-1, 3)
                elif landmarks.shape[0] >= 21 and landmarks.shape[1] == 3:
                    landmarks_array = landmarks[:21]
                else:
                    return None
            else:
                return None
        except Exception as e:
            print(f"Error procesando landmarks: {e}")
            return None
        
        # Enfriar después de detección
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return None
        
        # Detectar gesto único
        gesture_type = self._classify_word_gesture(landmarks_array)
        
        if gesture_type:
            # Agregar a historial
            self.detection_history.append(gesture_type)
            
            # Mantener solo últimos frames
            if len(self.detection_history) > self.stability_threshold:
                self.detection_history = self.detection_history[-self.stability_threshold:]
            
            # Verificar estabilidad
            if len(self.detection_history) >= self.stability_threshold:
                recent = self.detection_history[-self.stability_threshold:]
                
                # Contar ocurrencias del gesto actual
                gesture_count = sum(1 for g in recent if g == gesture_type)
                
                # Si es consistente (70% de frames)
                if gesture_count >= self.stability_threshold * 0.7:
                    word = self.word_gestures.get(gesture_type)
                    
                    if word and word != self.last_detected_word:
                        # Registrar uso
                        self._register_usage(word)
                        
                        # Activar cooldown
                        self.last_detected_word = word
                        self.cooldown_frames = self.cooldown_threshold
                        
                        # Limpiar historial
                        self.detection_history.clear()
                        
                        return word
        
        return None
    
    def _classify_word_gesture(self, lm) -> Optional[str]:
        """
        Clasifica gestos únicos para palabras completas
        """
        try:
            # Verificar que tenemos suficientes landmarks
            if lm.shape[0] < 21:
                return None
            
            # Extraer puntos clave
            wrist = lm[0]
            thumb_tip = lm[4]
            thumb_ip = lm[3]
            thumb_mcp = lm[2]
            index_tip = lm[8]
            index_pip = lm[6]
            index_mcp = lm[5]
            middle_tip = lm[12]
            middle_pip = lm[10]
            ring_tip = lm[16]
            ring_pip = lm[14]
            pinky_tip = lm[20]
            pinky_pip = lm[18]
            
            # Calcular estados (dedos arriba/abajo)
            thumb_up = thumb_tip[1] < thumb_ip[1] - 0.03
            index_up = index_tip[1] < index_pip[1] - 0.03
            middle_up = middle_tip[1] < middle_pip[1] - 0.03
            ring_up = ring_tip[1] < ring_pip[1] - 0.03
            pinky_up = pinky_tip[1] < pinky_pip[1] - 0.03
            
            # Distancias importantes
            thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
            thumb_middle_dist = np.linalg.norm(thumb_tip - middle_tip)
            index_middle_dist = np.linalg.norm(index_tip - middle_tip)
            
            # ===== GESTOS DE PALABRAS =====
            
            # THUMBS_UP → "HOLA"
            # Pulgar arriba, otros dedos cerrados
            if (thumb_up and not index_up and not middle_up and 
                not ring_up and not pinky_up and
                thumb_tip[1] < wrist[1] - 0.1):
                return 'THUMBS_UP'
            
            # PEACE → "BUENOS"
            # Índice y medio en V
            if (index_up and middle_up and not ring_up and not pinky_up and
                not thumb_up and index_middle_dist > 0.06):
                return 'PEACE'
            
            # OK_SIGN → "GRACIAS"
            # Círculo con pulgar e índice
            if (thumb_index_dist < 0.06 and middle_up and ring_up and pinky_up):
                return 'OK_SIGN'
            
            # PRAY_HANDS → "POR FAVOR"
            # Manos juntas (aproximar con todos los dedos juntos hacia arriba)
            if (index_up and middle_up and ring_up and pinky_up and
                thumb_up and index_middle_dist < 0.04 and
                abs(index_tip[0] - middle_tip[0]) < 0.03):
                return 'PRAY_HANDS'
            
            # POINTING_UP → "NECESITO"
            # Solo índice arriba
            if (index_up and not middle_up and not ring_up and 
                not pinky_up and not thumb_up and
                index_tip[1] < index_mcp[1] - 0.08):
                return 'POINTING_UP'
            
            # SHAKA → "OK"
            # Pulgar y meñique extendidos
            if (thumb_up and not index_up and not middle_up and 
                not ring_up and pinky_up and
                np.linalg.norm(thumb_tip - pinky_tip) > 0.15):
                return 'SHAKA'
            
            # HEART_HANDS → "TE AMO"
            # Aproximar: pulgar e índice formando corazón
            if (thumb_up and index_up and not middle_up and
                thumb_index_dist < 0.10 and
                thumb_tip[1] < index_mcp[1] and
                index_tip[1] < index_mcp[1]):
                return 'HEART_HANDS'
            
            # CALL_ME → "AYUDA"
            # Pulgar y meñique extendidos cerca de la cara
            if (thumb_up and pinky_up and not index_up and 
                not middle_up and not ring_up and
                thumb_tip[0] < wrist[0] - 0.05):
                return 'CALL_ME'
            
            # THUMBS_DOWN → "NO"
            # Pulgar hacia abajo
            if (thumb_tip[1] > thumb_mcp[1] + 0.05 and
                not index_up and not middle_up and not ring_up and not pinky_up):
                return 'THUMBS_DOWN'
            
            return None
            
        except Exception as e:
            print(f"Error clasificando gesto: {e}")
            return None
    
    def _register_usage(self, word: str):
        """Registra uso de palabra para estadísticas"""
        if word not in self.usage_stats:
            self.usage_stats[word] = 0
        self.usage_stats[word] += 1
    
    def get_most_used_words(self, count: int = 10) -> List[Tuple[str, int]]:
        """Retorna palabras más usadas"""
        sorted_words = sorted(
            self.usage_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_words[:count]
    
    def get_available_word_gestures(self) -> Dict[str, str]:
        """Retorna diccionario de gestos → palabras disponibles"""
        return self.word_gestures.copy()
    
    def add_custom_word_gesture(self, gesture_type: str, word: str) -> bool:
        """Permite agregar gestos personalizados"""
        if gesture_type and word:
            self.word_gestures[gesture_type] = word
            return True
        return False
    
    def reset_detection(self):
        """Resetea el estado de detección"""
        self.detection_history.clear()
        self.last_detected_word = None
        self.cooldown_frames = 0
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas de uso"""
        return {
            'total_words_available': len(self.word_gestures),
            'total_detections': sum(self.usage_stats.values()),
            'unique_words_used': len(self.usage_stats),
            'most_used': self.get_most_used_words(5)
        }