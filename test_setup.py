# test_setup.py
"""
Script para probar que todas las dependencias están instaladas correctamente
"""

import sys
import os

def test_imports():
    """
    Prueba que todos los módulos se puedan importar correctamente
    """
    print("=== PROBANDO INSTALACIÓN ===")
    
    modules_to_test = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("PIL/Pillow", "PIL"),
        ("TkInter", "tkinter"),
        ("pyttsx3", "pyttsx3"),
        ("scikit-learn", "sklearn"),
    ]
    
    all_good = True
    
    for name, module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError as e:
            print(f"❌ {name} - ERROR: {e}")
            all_good = False
    
    print("\n=== PROBANDO CÁMARA ===")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Cámara - OK")
            cap.release()
        else:
            print("❌ Cámara - No se pudo abrir")
            all_good = False
    except Exception as e:
        print(f"❌ Cámara - ERROR: {e}")
        all_good = False
    
    print("\n=== PROBANDO TEXT-TO-SPEECH ===")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        print("✅ Text-to-Speech - OK")
        engine.stop()
    except Exception as e:
        print(f"❌ Text-to-Speech - ERROR: {e}")
        all_good = False
    
    print("\n=== PROBANDO MÓDULOS PROPIOS ===")
    try:
        # Cambiar al directorio del script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        sys.path.insert(0, script_dir)
        
        from src.detector.hand_detector import HandDetector
        print("✅ HandDetector - OK")
        
        from src.detector.gesture_classifier import GestureClassifier  
        print("✅ GestureClassifier - OK")
        
        from src.utils.audio_manager import AudioManager
        print("✅ AudioManager - OK")
        
        from src.config.settings import Config
        print("✅ Config - OK")
        
    except Exception as e:
        print(f"❌ Módulos propios - ERROR: {e}")
        all_good = False
    
    print("\n" + "="*30)
    
    if all_good:
        print("🎉 ¡Todo está listo! Puedes ejecutar: python main.py")
    else:
        print("⚠️  Hay problemas con la instalación.")
        print("   Verifica que tengas todos los archivos __init__.py")
        print("   Y la estructura de carpetas correcta.")
    
    print("="*30)
    return all_good

def show_system_info():
    """
    Muestra información del sistema
    """
    print("\n=== INFORMACIÓN DEL SISTEMA ===")
    print(f"Python: {sys.version}")
    print(f"Plataforma: {sys.platform}")
    
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except:
        pass
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except:
        pass

if __name__ == "__main__":
    show_system_info()
    test_imports()
    input("\nPresiona Enter para salir...")