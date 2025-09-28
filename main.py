# main.py

import sys
import os
import traceback
import logging
from pathlib import Path

# Agregar el directorio actual al path de Python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def setup_logging():
    """Configura el sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('traductor_senas.log'),
            logging.StreamHandler()
        ]
    )

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
        from PIL import Image
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

def verify_project_structure():
    """Verifica y crea la estructura del proyecto si es necesaria"""
    required_dirs = [
        'src/detector',
        'src/interface', 
        'src/utils',
        'src/config',
        'models'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Creado directorio: {dir_path}")

def start_main_application():
    """Inicia la aplicación principal (llamada desde welcome screen)"""
    try:
        print("🚀 Iniciando aplicación principal...")
        
        # Importar la ventana principal
        from src.interface.main_window import MainWindow
        
        # Crear y ejecutar la aplicación
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
        
    except Exception as e:
        print(f"❌ Error iniciando aplicación principal: {e}")
        logging.error(f"Error en aplicación principal: {e}")
        traceback.print_exc()
        input("Presiona Enter para salir...")

def main():
    """Función principal de la aplicación"""
    print("=" * 60)
    print("    TRADUCTOR DE LENGUAJE DE SEÑAS")
    print("=" * 60)
    print()
    
    # Configurar logging
    setup_logging()
    
    try:
        # 1. Verificar dependencias
        print("1. Verificando dependencias...")
        if not check_dependencies():
            input("\nPresiona Enter para salir...")
            return
        
        print("✓ Todas las dependencias están instaladas")
        print()
        
        # 2. Verificar estructura del proyecto
        print("2. Verificando estructura del proyecto...")
        verify_project_structure()
        print("✓ Estructura del proyecto verificada")
        print()
        
        # 3. Verificar cámara
        print("3. Probando cámara...")
        if not test_camera():
            print("\nSUGERENCIAS:")
            print("- Verifica que ninguna otra aplicación use la cámara")
            print("- Verifica permisos de cámara en tu sistema")
            print("- Cambia CAMERA_INDEX en settings.py (prueba 0, 1, 2)")
            input("\nPresiona Enter para salir...")
            return
        
        print()
        
        # 4. Mostrar pantalla de bienvenida
        print("4. Iniciando interfaz de bienvenida...")
        
        try:
            # Importar y mostrar pantalla de bienvenida
            from src.interface.welcome_screen import show_welcome_screen
            
            # Mostrar pantalla de bienvenida
            show_welcome_screen(start_main_application)
            
        except ImportError as e:
            print(f"⚠️  No se pudo cargar la pantalla de bienvenida: {e}")
            print("Iniciando aplicación directamente...")
            start_main_application()
        
    except ImportError as e:
        print(f"✗ Error importando módulos: {e}")
        print("\nVerifica que tengas la estructura correcta:")
        print("TRADUCTOR/")
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
        print("    │   ├── gesture_classifier.py")
        print("    │   └── syllable_classifier.py")
        print("    ├── interface/")
        print("    │   ├── __init__.py")
        print("    │   ├── main_window.py")
        print("    │   └── welcome_screen.py")
        print("    └── utils/")
        print("        ├── __init__.py")
        print("        └── audio_manager.py")
        
        input("\nPresiona Enter para salir...")
        
    except Exception as e:
        print(f"✗ Error inesperado: {e}")
        print("\nDetalles del error:")
        traceback.print_exc()
        logging.error(f"Error inesperado en main: {e}")
        input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    main()