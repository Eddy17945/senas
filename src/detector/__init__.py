"""
Módulo de detección de manos y clasificación de gestos
"""

from .hand_detector import HandDetector
from .gesture_classifier import GestureClassifier

__all__ = ['HandDetector', 'GestureClassifier']