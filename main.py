# main.py

import sys
import os

# Agregar el directorio src al path de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.interface.main_window import MainWindow
    import tkinter as tk
    
    def main():
        """
        Función principal de la aplicación
        """
        try:
            # Crear y configurar la aplicación
            app = MainWindow()
            
            # Configurar tema de la aplicación
            style = tk.ttk.Style()
            style.theme_use('clam')  # Tema moderno
            
            print("=== TRADUCTOR DE LENGUAJE DE SEÑAS ===")
            print("Iniciando aplicación...")
            print("Presiona 'Iniciar Detección' para comenzar")
            print("Letras soportadas: A, B, C")
            print("=====================================")
            
            # Ejecutar la aplicación
            app.run()
            
        except Exception as e:
            print(f"Error iniciando la aplicación: {e}")
            input("Presiona Enter para salir...")
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    print("Error importando módulos:")
    print(f"  {e}")
    print("\nAsegúrate de instalar las dependencias:")
    print("  pip install -r requirements.txt")
    print("\nY que tengas la estructura de carpetas correcta.")
    input("Presiona Enter para salir...")