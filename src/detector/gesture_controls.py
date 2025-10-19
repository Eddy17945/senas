"""
Gestos de Control para el Traductor de Señas
Maneja gestos especiales como borrar, espacio, limpiar, etc.
"""

import numpy as np
from typing import Optional, Dict, List
from collections import deque

class GestureControls:
    """
    Gestor de gestos de control especiales
    """
    
    def __init__(self):
        self.control_history = deque(maxlen=5)  # Historial corto
        self.last_control = None
        self.control_cooldown = 0
        self.cooldown_frames = 20  # Esperar 20 frames entre controles
        
        # Configuración
        self.controls_enabled = True
        
        # Nombres de controles
        self.CONTROLS = {
            'DELETE': '⌫ Borrar',
            'SPACE': '␣ Espacio', 
            'CLEAR': '🗑️ Limpiar',
            'PAUSE': '⏸️ Pausa'
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
        
        # Caso especial: CLEAR requiere ambas manos
        if control_gesture == "CLEAR_CANDIDATE":
            if self._detect_clear_gesture(both_hands_data):
                control_gesture = "CLEAR"
            else:
                return None
        
        # Agregar a historial
        self.control_history.append(control_gesture)
        
        # Verificar estabilidad (3 de 5 frames deben ser el mismo gesto)
        if len(self.control_history) >= 3:
            recent = list(self.control_history)[-3:]
            count = sum(1 for c in recent if c == control_gesture)
            
            if count >= 2:  # Al menos 2 de 3
                # Evitar repetición inmediata del mismo control
                if control_gesture != self.last_control:
                    self.last_control = control_gesture
                    self.control_cooldown = self.cooldown_frames
                    return control_gesture
        
        return None
    
    def _detect_clear_gesture(self, both_hands_data: Optional[Dict]) -> bool:
        """
        Detecta gesto de LIMPIAR TODO (ambos puños)
        """
        if not both_hands_data:
            return False
        
        left = both_hands_data.get('left')
        right = both_hands_data.get('right')
        
        if not left or not right:
            return False
        
        # Verificar que ambas manos estén en puño
        left_array = np.array(left[:63]).reshape(-1, 3)
        right_array = np.array(right[:63]).reshape(-1, 3)
        
        # Verificar todos los dedos cerrados en ambas manos
        def is_fist(lm):
            fingers_extended = 0
            finger_tips = [4, 8, 12, 16, 20]
            finger_bases = [2, 5, 9, 13, 17]
            
            for tip, base in zip(finger_tips, finger_bases):
                if lm[tip][1] < lm[base][1] - 0.03:
                    fingers_extended += 1
            
            return fingers_extended == 0
        
        return is_fist(left_array) and is_fist(right_array)
    
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