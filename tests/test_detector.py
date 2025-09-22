# tests/test_detector.py

import unittest
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector.hand_detector import HandDetector
from detector.gesture_classifier import GestureClassifier

class TestDetector(unittest.TestCase):
    def setUp(self):
        self.hand_detector = HandDetector()
        self.gesture_classifier = GestureClassifier()
    
    def test_hand_detector_init(self):
        """Test de inicialización del detector"""
        self.assertIsNotNone(self.hand_detector)
    
    def test_gesture_classifier_init(self):
        """Test de inicialización del clasificador"""
        self.assertIsNotNone(self.gesture_classifier)

if __name__ == '__main__':
    unittest.main()