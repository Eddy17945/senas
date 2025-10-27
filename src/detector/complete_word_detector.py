# src/detector/complete_word_detector.py
"""
Detector de palabras completas por gestos únicos
CON DESCRIPCIONES DE GESTOS REALES
"""

import numpy as np
from typing import Optional, Dict, List, Tuple

class CompleteWordDetector:
    def __init__(self):
        # Mapeo de gestos únicos a palabras completas
        self.word_gestures = {
            # ===== SALUDOS Y CORTESÍA =====
            'THUMBS_UP': 'HOLA',
            'WAVE': 'ADIOS', 
            'PEACE': 'BUENOS',
            'OK_SIGN': 'GRACIAS',
            'PRAY_HANDS': 'POR FAVOR',
            'BOW': 'DISCULPA',
            
            # ===== RESPUESTAS BÁSICAS =====
            'THUMBS_DOWN': 'NO',
            'NOD_YES': 'SI',
            'SHAKA': 'OK',
            'FIST_UP': 'BIEN',
            
            # ===== NECESIDADES =====
            'POINTING_UP': 'NECESITO',
            'DRINK_GESTURE': 'AGUA',
            'EAT_GESTURE': 'COMIDA',
            'BATHROOM_SIGN': 'BAÑO',
            'SLEEP_GESTURE': 'DORMIR',
            
            # ===== FAMILIA =====
            'HEART_HANDS': 'TE AMO',
            'MAMA_SIGN': 'MAMA',
            'PAPA_SIGN': 'PAPA',
            'BABY_ROCK': 'BEBE',
            'FAMILY_SIGN': 'FAMILIA',
            
            # ===== EMOCIONES =====
            'HAPPY_SIGN': 'FELIZ',
            'SAD_SIGN': 'TRISTE',
            'ANGRY_FIST': 'ENOJADO',
            'SCARED_HANDS': 'MIEDO',
            'LOVE_HEART': 'AMOR',
            
            # ===== ACCIONES =====
            'CALL_ME': 'AYUDA',
            'COME_HERE': 'VEN',
            'GO_AWAY': 'VETE',
            'WAIT_HAND': 'ESPERA',
            'STOP_HAND': 'ALTO',
            
            # ===== LUGARES =====
            'HOME_SIGN': 'CASA',
            'SCHOOL_SIGN': 'ESCUELA',
            'WORK_SIGN': 'TRABAJO',
            'HOSPITAL_CROSS': 'HOSPITAL',
            
            # ===== TIEMPO =====
            'NOW_SIGN': 'AHORA',
            'LATER_SIGN': 'DESPUES',
            'TODAY_SIGN': 'HOY',
            'TOMORROW_POINT': 'MAÑANA',
            
            # ===== COMUNICACIÓN =====
            'PHONE_CALL': 'TELEFONO',
            'WRITE_SIGN': 'ESCRIBIR',
            'READ_SIGN': 'LEER',
            'LISTEN_EAR': 'ESCUCHAR',
            
            # ===== ÚTILES =====
            'MONEY_RUB': 'DINERO',
            'FRIEND_LINK': 'AMIGO',
        }
        
        # DESCRIPCIONES DE GESTOS REALES (reemplazando emojis)
        self.gesture_descriptions = {
            # ===== SALUDOS Y CORTESÍA =====
            'THUMBS_UP': ('👍', 'Pulgar hacia arriba, mano cerrada'),
            'WAVE': ('👋', 'Mano abierta moviéndose de lado a lado'),
            'PEACE': ('✌️', 'Dedos índice y medio extendidos en V'),
            'OK_SIGN': ('👌', 'Pulgar e índice formando círculo'),
            'PRAY_HANDS': ('🙏', 'Ambas manos juntas como en oración'),
            'BOW': ('🙇', 'Inclinación de cabeza con manos juntas'),
            
            # ===== RESPUESTAS BÁSICAS =====
            'THUMBS_DOWN': ('👎', 'Pulgar hacia abajo, mano cerrada'),
            'NOD_YES': ('✅', 'Cabeza moviéndose arriba y abajo'),
            'SHAKA': ('🤙', 'Pulgar y meñique extendidos, otros dedos cerrados'),
            'FIST_UP': ('✊', 'Puño cerrado levantado'),
            
            # ===== NECESIDADES =====
            'POINTING_UP': ('☝️', 'Dedo índice señalando hacia arriba'),
            'DRINK_GESTURE': ('🤲', 'Mano como sosteniendo vaso, llevando a boca'),
            'EAT_GESTURE': ('🍴', 'Mano llevando comida imaginaria a la boca'),
            'BATHROOM_SIGN': ('🚽', 'Mano haciendo movimiento de "T"'),
            'SLEEP_GESTURE': ('😴', 'Manos juntas en lado de cabeza, ojos cerrados'),
            
            # ===== FAMILIA =====
            'HEART_HANDS': ('❤️', 'Ambas manos formando corazón en el pecho'),
            'MAMA_SIGN': ('👩', 'Dedo pulgar en barbilla, mano abierta'),
            'PAPA_SIGN': ('👨', 'Dedo pulgar en frente, mano abierta'),
            'BABY_ROCK': ('👶', 'Brazos como meciendo bebé'),
            'FAMILY_SIGN': ('👨‍👩‍👧', 'Manos formando círculo familiar'),
            
            # ===== EMOCIONES =====
            'HAPPY_SIGN': ('😊', 'Manos abiertas moviéndose cerca del pecho'),
            'SAD_SIGN': ('😢', 'Dedos trazando lágrimas en mejillas'),
            'ANGRY_FIST': ('😠', 'Puños cerrados temblando'),
            'SCARED_HANDS': ('😨', 'Manos abiertas a los lados de la cara'),
            'LOVE_HEART': ('💕', 'Manos cruzadas sobre el corazón'),
            
            # ===== ACCIONES =====
            'CALL_ME': ('📞', 'Mano en forma de teléfono en oreja'),
            'COME_HERE': ('👈', 'Mano con palma hacia arriba, dedos moviéndose'),
            'GO_AWAY': ('👉', 'Mano empujando hacia adelante'),
            'WAIT_HAND': ('✋', 'Mano abierta con palma hacia afuera'),
            'STOP_HAND': ('🛑', 'Mano abierta, palma hacia adelante'),
            
            # ===== LUGARES =====
            'HOME_SIGN': ('🏠', 'Manos formando techo sobre cabeza'),
            'SCHOOL_SIGN': ('🏫', 'Manos aplaudiendo como pizarra'),
            'WORK_SIGN': ('💼', 'Manos como sosteniendo maletín'),
            'HOSPITAL_CROSS': ('🏥', 'Dedos formando cruz en brazo'),
            
            # ===== TIEMPO =====
            'NOW_SIGN': ('⏰', 'Dedo índice tocando muñeca'),
            'LATER_SIGN': ('🕐', 'Mano moviéndose sobre hombro'),
            'TODAY_SIGN': ('📅', 'Dedo índice tocando nariz'),
            'TOMORROW_POINT': ('➡️', 'Mano señalando hacia adelante'),
            
            # ===== COMUNICACIÓN =====
            'PHONE_CALL': ('📱', 'Pulgar y meñique como teléfono'),
            'WRITE_SIGN': ('✍️', 'Mano como escribiendo en palma'),
            'READ_SIGN': ('📖', 'Manos como sosteniendo libro'),
            'LISTEN_EAR': ('👂', 'Mano en oreja inclinándose'),
            
            # ===== ÚTILES =====
            'MONEY_RUB': ('💰', 'Dedos pulgar e índice frotándose'),
            'FRIEND_LINK': ('🤝', 'Manos estrechándose'),
        }
        
        # Historial de detección
        self.detection_history = []
        self.stability_threshold = 8
        self.last_detected_word = None
        self.cooldown_frames = 0
        self.cooldown_threshold = 30
        
        # Estadísticas de uso
        self.usage_stats = {}
    
    def get_gesture_description(self, gesture_type: str) -> Tuple[str, str]:
        """Obtiene la descripción del gesto para mostrar en la interfaz"""
        return self.gesture_descriptions.get(gesture_type, ('❓', 'Gesto no descrito'))
    
    def get_all_gesture_descriptions(self) -> Dict[str, Tuple[str, str]]:
        """Obtiene todas las descripciones de gestos"""
        return self.gesture_descriptions.copy()

    # ... (el resto del código se mantiene igual)
    def detect_complete_word(self, landmarks, confidence: float) -> Optional[str]:
        """Detecta si el gesto corresponde a una palabra completa"""
        if landmarks is None:
            return None
        
        # Convertir landmarks a array numpy
        try:
            if isinstance(landmarks, list):
                if len(landmarks) >= 63:
                    landmarks_array = np.array(landmarks[:63]).reshape(-1, 3)
                elif len(landmarks) >= 21 and isinstance(landmarks[0], (list, tuple)):
                    landmarks_array = np.array(landmarks[:21])
                else:
                    return None
            elif isinstance(landmarks, np.ndarray):
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
                        
                        print(f"[PALABRA COMPLETA] ⚡ Detectada: {word} (gesto: {gesture_type})")
                        return word
        
        return None
    
    def _classify_word_gesture(self, lm) -> Optional[str]:
        """Clasifica gestos únicos para palabras completas"""
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
            middle_mcp = lm[9]
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
            thumb_pinky_dist = np.linalg.norm(thumb_tip - pinky_tip)
            
            # ===== GESTOS ORIGINALES =====
            
            # THUMBS_UP → "HOLA"
            if (thumb_up and not index_up and not middle_up and 
                not ring_up and not pinky_up and
                thumb_tip[1] < wrist[1] - 0.1):
                return 'THUMBS_UP'
            
            # PEACE → "BUENOS"
            if (index_up and middle_up and not ring_up and not pinky_up and
                not thumb_up and index_middle_dist > 0.06):
                return 'PEACE'
            
            # OK_SIGN → "GRACIAS"
            if (thumb_index_dist < 0.06 and middle_up and ring_up and pinky_up):
                return 'OK_SIGN'
            
            # PRAY_HANDS → "POR FAVOR"
            if (index_up and middle_up and ring_up and pinky_up and
                thumb_up and index_middle_dist < 0.04 and
                abs(index_tip[0] - middle_tip[0]) < 0.03):
                return 'PRAY_HANDS'
            
            # POINTING_UP → "NECESITO"
            if (index_up and not middle_up and not ring_up and 
                not pinky_up and not thumb_up and
                index_tip[1] < index_mcp[1] - 0.08):
                return 'POINTING_UP'
            
            # SHAKA → "OK"
            if (thumb_up and not index_up and not middle_up and 
                not ring_up and pinky_up and
                thumb_pinky_dist > 0.15):
                return 'SHAKA'
            
            # HEART_HANDS → "TE AMO"
            if (thumb_up and index_up and not middle_up and
                thumb_index_dist < 0.10 and
                thumb_tip[1] < index_mcp[1] and
                index_tip[1] < index_mcp[1]):
                return 'HEART_HANDS'
            
            # CALL_ME → "AYUDA"
            if (thumb_up and pinky_up and not index_up and 
                not middle_up and not ring_up and
                thumb_tip[0] < wrist[0] - 0.05):
                return 'CALL_ME'
            
            # THUMBS_DOWN → "NO"
            if (thumb_tip[1] > thumb_mcp[1] + 0.05 and
                not index_up and not middle_up and not ring_up and not pinky_up):
                return 'THUMBS_DOWN'
            
            # ===== NUEVOS GESTOS AMPLIADOS =====
            
            # FIST_UP → "BIEN" (puño cerrado hacia arriba)
            if (not index_up and not middle_up and not ring_up and 
                not pinky_up and not thumb_up and
                wrist[1] > index_mcp[1]):
                return 'FIST_UP'
            
            # STOP_HAND → "ALTO" (mano abierta hacia adelante)
            if (index_up and middle_up and ring_up and pinky_up and
                not thumb_up and
                index_middle_dist < 0.05 and
                abs(index_tip[1] - middle_tip[1]) < 0.04):
                return 'STOP_HAND'
            
            # COME_HERE → "VEN" (mano con dedos curvándose)
            if (index_up and middle_up and ring_up and pinky_up and
                not thumb_up and
                index_pip[1] < index_mcp[1] and
                middle_pip[1] < middle_mcp[1]):
                return 'COME_HERE'
            
            # WAIT_HAND → "ESPERA" (mano abierta horizontal)
            if (index_up and middle_up and ring_up and pinky_up and
                thumb_up and
                abs(index_tip[0] - middle_tip[0]) > 0.08):
                return 'WAIT_HAND'
            
            # LOVE_HEART → "AMOR" (forma de corazón más amplia)
            if (thumb_up and index_up and middle_up and
                not ring_up and not pinky_up and
                thumb_index_dist < 0.12 and
                thumb_middle_dist < 0.12):
                return 'LOVE_HEART'
            
            # PHONE_CALL → "TELEFONO" (pulgar y meñique como teléfono)
            if (thumb_up and pinky_up and not index_up and 
                not middle_up and not ring_up and
                thumb_tip[0] > wrist[0] and  # Pulgar a la derecha
                thumb_pinky_dist > 0.12):
                return 'PHONE_CALL'
            
            # NOW_SIGN → "AHORA" (ambas manos hacia abajo - aproximar con una)
            if (index_up and middle_up and not ring_up and 
                not pinky_up and not thumb_up and
                index_tip[1] > index_mcp[1] - 0.05):
                return 'NOW_SIGN'
            
            # FRIEND_LINK → "AMIGO" (índices enganchados - aproximar)
            if (index_up and not middle_up and not ring_up and 
                not pinky_up and thumb_up and
                abs(thumb_tip[0] - index_tip[0]) < 0.06):
                return 'FRIEND_LINK'
            
            # MONEY_RUB → "DINERO" (frotar dedos - aproximar con pulgar tocando dedos)
            if (thumb_index_dist < 0.04 and thumb_middle_dist < 0.04 and
                not ring_up and not pinky_up):
                return 'MONEY_RUB'
            
            # HOME_SIGN → "CASA" (techo con manos - aproximar)
            if (index_up and middle_up and ring_up and pinky_up and
                thumb_up and
                index_tip[1] < wrist[1] - 0.12):
                return 'HOME_SIGN'
            
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