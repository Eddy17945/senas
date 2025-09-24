import sys
import os
import traceback

# Agregar el directorio actual al path de Python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def check_dependencies():
    """Verifica que todas las dependencias estén instaladas"""
    missing = []
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import mediapipe
    except ImportError:
        missing.append("mediapipe")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import PIL
    except ImportError:
        missing.append("pillow")
    
    try:
        import pyttsx3
    except ImportError:
        missing.append("pyttsx3")
    
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")
    
    if missing:
        print("ERROR: Faltan las siguientes dependencias:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstala las dependencias con:")
        print("pip install " + " ".join(missing))
        return False
    
    return True

def test_camera():
    """Prueba rápida de la cámara"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✓ Cámara funcionando correctamente")
                return True
            else:
                print("✗ Cámara no puede leer frames")
                return False
        else:
            print("✗ No se puede abrir la cámara")
            return False
    except Exception as e:
        print(f"✗ Error probando cámara: {e}")
        return False

def main():
    """Función principal de la aplicación"""
    print("=" * 50)
    print("    TRADUCTOR DE LENGUAJE DE SEÑAS")
    print("=" * 50)
    
    # Verificar dependencias
    print("1. Verificando dependencias...")
    if not check_dependencies():
        input("\nPresiona Enter para salir...")
        return
    
    # Verificar cámara
    print("2. Probando cámara...")
    if not test_camera():
        print("\nSUGERENCIAS:")
        print("- Verifica que ninguna otra aplicación use la cámara")
        print("- Verifica permisos de cámara en tu sistema")
        print("- Cambia CAMERA_INDEX en settings.py (prueba 0, 1, 2)")
        input("\nPresiona Enter para salir...")
        return
    
    # Iniciar aplicación
    print("3. Iniciando aplicación...")
    try:
        from src.interface.main_window import MainWindow
        import tkinter as tk
        from tkinter import ttk
        
        # Crear y configurar la aplicación
        app = MainWindow()
        
        print("✓ Aplicación iniciada correctamente")
        print("\nInstrucciones:")
        print("- Haz clic en 'Iniciar Detección' para comenzar")
        print("- Muestra gestos frente a la cámara")
        print("- Las letras detectadas aparecerán en pantalla")
        print("- Usa 'Agregar Letra' para formar palabras")
        print("- Usa 'Reproducir Audio' para escuchar el texto")
        
        # Ejecutar la aplicación
        app.run()
        
    except ImportError as e:
        print(f"✗ Error importando módulos: {e}")
        print("\nVerifica que tengas la estructura correcta:")
        print("SENAS/")
        print("├── main.py")
        print("├── requirements.txt")
        print("└── src/")
        print("    ├── __init__.py")
        print("    ├── config/")
        print("    │   ├── __init__.py")
        print("    │   └── settings.py")
        print("    ├── detector/")
        print("    │   ├── __init__.py")
        print("    │   ├── hand_detector.py")
        print("    │   └── gesture_classifier.py")
        print("    ├── interface/")
        print("    │   ├── __init__.py")
        print("    │   └── main_window.py")
        print("    └── utils/")
        print("        ├── __init__.py")
        print("        └── audio_manager.py")
        
    except Exception as e:
        print(f"✗ Error inesperado: {e}")
        print("\nDetalles del error:")
        traceback.print_exc()
        
    finally:
        input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    main()