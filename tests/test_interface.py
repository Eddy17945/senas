# tests/test_interface.py

import unittest
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestInterface(unittest.TestCase):
    def test_import(self):
        """Test básico de importación"""
        try:
            from interface.main_window import MainWindow
            self.assertTrue(True)
        except ImportError:
            self.fail("No se pudo importar MainWindow")

if __name__ == '__main__':
    unittest.main()