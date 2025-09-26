# src/detector/__init__.py
"""
Módulo de detección de manos y clasificación de gestos
"""

from .hand_detector import HandDetector
from .gesture_classifier import GestureClassifier
from .syllable_classifier import SyllableClassifier

__all__ = ['HandDetector', 'GestureClassifier', 'SyllableClassifier']