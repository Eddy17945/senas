# src/utils/audio_manager.py

import pyttsx3
import threading
from typing import Optional

class AudioManager:
    def __init__(self, rate: int = 200, volume: float = 0.9):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Configurar voz en español si está disponible
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        
        self.is_speaking = False
    
    def speak(self, text: str, async_mode: bool = True):
        """
        Convierte texto a voz
        """
        if not text or self.is_speaking:
            return
        
        if async_mode:
            thread = threading.Thread(target=self._speak_sync, args=(text,))
            thread.daemon = True
            thread.start()
        else:
            self._speak_sync(text)
    
    def _speak_sync(self, text: str):
        """
        Conversión síncrona de texto a voz
        """
        try:
            self.is_speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error en síntesis de voz: {e}")
        finally:
            self.is_speaking = False
    
    def stop(self):
        """
        Detiene la reproducción de audio
        """
        try:
            self.engine.stop()
            self.is_speaking = False
        except Exception as e:
            print(f"Error deteniendo audio: {e}")
    
    def set_rate(self, rate: int):
        """
        Cambia la velocidad de la voz
        """
        self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        """
        Cambia el volumen de la voz (0.0 a 1.0)
        """
        self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
    
    def get_available_voices(self) -> list:
        """
        Obtiene la lista de voces disponibles
        """
        voices = self.engine.getProperty('voices')
        return [(voice.id, voice.name) for voice in voices]