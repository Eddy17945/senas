# src/detector/gesture_controls.py
"""
Gestos de Control para el Traductor de Señas
CONFIGURACIÓN PERSONALIZADA: Espacio con 2 manos abiertas
"""

import numpy as np
from typing import Optional, Dict, List
from collections import deque

class GestureControls:
    """
    Gestor de gestos de control especiales
    """
    
    def __init__(self):
        self.control_history = deque(maxlen=8)  # Más frames para más estabilidad
        self.last_control = None
        self.control_cooldown = 0
        self.cooldown_frames = 15  # REDUCIDO de 25 a 15 para repetir más rápido
        
        # NUEVO: Contador de frames sin detección
        self.frames_without_gesture = 0
        self.reset_threshold = 10  # Resetear después de 10 frames sin gesto
        
        # Configuración
        self.controls_enabled = True
        
        # Nombres de controles
        self.CONTROLS = {
            'DELETE': '⌫ Borrar letra',
            'SPACE': '␣ Espacio (2 manos abiertas)', 
            'CLEAR': '🗑️ Limpiar todo'
        }
    
    def process_control(self, control_gesture: Optional[str], 
                       both_hands_data: Optional[Dict] = None) -> Optional[str]:
        """
        Procesa un gesto de control y determina si ejecutarlo
        
        Args:
            control_gesture: Gesto detectado ('DELETE', 'SPACE', etc.)
            both_hands_data: Datos de ambas manos para gestos que requieren 2 manos
            
        Returns:
            Control confirmado o None
        """
        # Reducir cooldown
        if self.control_cooldown > 0:
            self.control_cooldown -= 1
            return None
        
        if not self.controls_enabled or not control_gesture:
            return None
        
        # CASO ESPECIAL 1: CLEAR requiere ambas manos (puños)
        if control_gesture == "CLEAR_CANDIDATE":
            if self._detect_clear_gesture(both_hands_data):
                control_gesture = "CLEAR"
            else:
                return None
        
        # CASO ESPECIAL 2: SPACE requiere ambas manos abiertas
        if control_gesture == "SPACE_CANDIDATE":
            if self._detect_space_both_hands(both_hands_data):
                control_gesture = "SPACE"
            else:
                return None
        
        # Agregar a historial
        self.control_history.append(control_gesture)
        
        # Verificar estabilidad (5 de 8 frames deben ser el mismo gesto)
        if len(self.control_history) >= 5:
            recent = list(self.control_history)[-5:]
            count = sum(1 for c in recent if c == control_gesture)
            
            if count >= 4:  # Al menos 4 de 5 (más estricto)
                # Evitar repetición inmediata del mismo control
                if control_gesture != self.last_control:
                    self.last_control = control_gesture
                    self.control_cooldown = self.cooldown_frames
                    
                    print(f"[CONTROL] ✅ Gesto confirmado: {control_gesture}")
                    return control_gesture
        
        return None
    
    def _detect_space_both_hands(self, both_hands_data: Optional[Dict]) -> bool:
        """
        ✋✋ ESPACIO: Detecta ambas manos ABIERTAS (5 dedos extendidos)
        """
        if not both_hands_data:
            return False
        
        left = both_hands_data.get('left')
        right = both_hands_data.get('right')
        
        # REQUERIR ambas manos
        if not left or not right:
            return False
        
        try:
            # Convertir a arrays
            left_array = np.array(left[:63]).reshape(-1, 3)
            right_array = np.array(right[:63]).reshape(-1, 3)
            
            # Verificar que ambas manos tengan 5 dedos extendidos
            left_open = self._is_hand_fully_open(left_array)
            right_open = self._is_hand_fully_open(right_array)
            
            if left_open and right_open:
                print(f"[DEBUG SPACE] ✋✋ Ambas manos abiertas detectadas!")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"[ERROR] Error detectando espacio: {e}")
            return False
    
    def _is_hand_fully_open(self, lm) -> bool:
        """
        Verifica si una mano está completamente abierta (5 dedos extendidos)
        """
        # Índices de las puntas y bases de los dedos
        finger_tips = [4, 8, 12, 16, 20]  # Pulgar, índice, medio, anular, meñique
        finger_bases = [2, 5, 9, 13, 17]
        
        fingers_extended = 0
        
        for tip_idx, base_idx in zip(finger_tips, finger_bases):
            tip = lm[tip_idx]
            base = lm[base_idx]
            
            # Verificar si el dedo está extendido (punta más arriba que la base)
            # Umbral más permisivo: 0.02 en lugar de 0.03
            if tip[1] < base[1] - 0.02:
                fingers_extended += 1
        
        # Considerar mano abierta si al menos 4 dedos están extendidos
        # (permite que un dedo esté ligeramente doblado)
        return fingers_extended >= 4
    
    def _detect_clear_gesture(self, both_hands_data: Optional[Dict]) -> bool:
        """
        🗑️ LIMPIAR TODO: Detecta ambos puños cerrados
        """
        if not both_hands_data:
            return False
        
        left = both_hands_data.get('left')
        right = both_hands_data.get('right')
        
        if not left or not right:
            return False
        
        try:
            # Verificar que ambas manos estén en puño
            left_array = np.array(left[:63]).reshape(-1, 3)
            right_array = np.array(right[:63]).reshape(-1, 3)
            
            left_fist = self._is_fist(left_array)
            right_fist = self._is_fist(right_array)
            
            if left_fist and right_fist:
                print(f"[DEBUG CLEAR] 👊👊 Ambos puños detectados!")
                return True
            return False
            
        except Exception as e:
            print(f"[ERROR] Error detectando clear: {e}")
            return False
    
    def _is_fist(self, lm) -> bool:
        """
        Verifica si la mano está en puño (todos los dedos cerrados)
        """
        fingers_extended = 0
        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [2, 5, 9, 13, 17]
        
        for tip, base in zip(finger_tips, finger_bases):
            # Si la punta está ARRIBA de la base, el dedo está extendido
            if lm[tip][1] < lm[base][1] - 0.03:
                fingers_extended += 1
        
        # Puño = 0 dedos extendidos
        return fingers_extended == 0
    
    def get_control_name(self, control: str) -> str:
        """Obtiene el nombre amigable del control"""
        return self.CONTROLS.get(control, control)
    
    def enable_controls(self, enabled: bool):
        """Activa/desactiva los controles"""
        self.controls_enabled = enabled
    
    def reset(self):
        """Reinicia el estado de los controles"""
        self.control_history.clear()
        self.last_control = None
        self.control_cooldown = 0
    
    def set_cooldown(self, frames: int):
        """Configura el tiempo de cooldown entre controles"""
        self.cooldown_frames = max(5, min(60, frames))