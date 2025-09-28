# src/interface/welcome_screen.py

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from typing import Callable, Optional

class WelcomeScreen:
    def __init__(self, on_start_callback: Callable):
        """
        Pantalla de bienvenida para el traductor de señas
        
        Args:
            on_start_callback: Función a ejecutar cuando se presione "Comenzar"
        """
        self.on_start_callback = on_start_callback
        self.root = tk.Tk()
        self.root.title("Traductor de Señas - Bienvenida")
        self.root.geometry("800x600")
        self.root.configure(bg='#34495E')
        self.root.resizable(False, False)
        
        # Centrar ventana
        self.center_window()
        
        # Crear interfaz
        self.setup_ui()
        
    def center_window(self):
        """Centra la ventana en la pantalla"""
        self.root.update_idletasks()
        width = 800
        height = 600
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Frame principal con scrollable
        main_frame = tk.Frame(self.root, bg='#34495E')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título principal
        title_label = tk.Label(
            main_frame,
            text="TRADUCTOR DE SEÑAS",
            font=('Arial', 28, 'bold'),
            fg='#ECF0F1',
            bg='#34495E'
        )
        title_label.pack(pady=(10, 5))
        
        # Subtítulo
        subtitle_label = tk.Label(
            main_frame,
            text="Sistema de Reconocimiento de Lenguaje de Señas",
            font=('Arial', 14),
            fg='#BDC3C7',
            bg='#34495E'
        )
        subtitle_label.pack(pady=(0, 30))
        
        # Frame para el icono/imagen
        icon_frame = tk.Frame(main_frame, bg='#2C3E50', relief=tk.RAISED, borderwidth=2)
        icon_frame.pack(pady=(0, 20))
        
        # Crear un icono simple de mano
        hand_text = """
          👋
        
     LENGUAJE
        DE
      SEÑAS
        """
        
        hand_label = tk.Label(
            icon_frame,
            text=hand_text,
            font=('Arial', 16, 'bold'),
            fg='#F39C12',
            bg='#2C3E50',
            padx=20,
            pady=15
        )
        hand_label.pack()
        
        # Descripción más compacta
        description_text = """Bienvenido al Traductor de Señas

Este sistema utiliza inteligencia artificial para reconocer
gestos del lenguaje de señas y convertirlos en texto y audio.

Características:
• Reconocimiento en tiempo real    • Soporte para letras A-Z
• Modo de sílabas                 • Síntesis de voz
• Calibración automática"""
        
        description_label = tk.Label(
            main_frame,
            text=description_text,
            font=('Arial', 11),
            fg='#ECF0F1',
            bg='#34495E',
            justify=tk.CENTER,
            padx=10
        )
        description_label.pack(pady=(10, 30))
        
        # BOTÓN DE COMENZAR - MÁS VISIBLE
        button_frame = tk.Frame(main_frame, bg='#34495E')
        button_frame.pack(pady=20)
        
        start_button = tk.Button(
            button_frame,
            text="COMENZAR",
            font=('Arial', 18, 'bold'),
            bg='#3498DB',
            fg='white',
            activebackground='#2980B9',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=3,
            padx=40,
            pady=15,
            cursor='hand2',
            command=self.start_application
        )
        start_button.pack()
        
        # Efecto hover para el botón
        def on_enter(e):
            start_button.config(bg='#2980B9')
        
        def on_leave(e):
            start_button.config(bg='#3498DB')
        
        start_button.bind("<Enter>", on_enter)
        start_button.bind("<Leave>", on_leave)
        
        # Frame inferior
        info_frame = tk.Frame(main_frame, bg='#34495E')
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))
        
        # Instrucciones
        instructions_label = tk.Label(
            info_frame,
            text="Asegúrate de tener una cámara conectada y buena iluminación",
            font=('Arial', 10),
            fg='#7F8C8D',
            bg='#34495E'
        )
        instructions_label.pack(pady=(0, 5))
        
        # Versión
        version_label = tk.Label(
            info_frame,
            text="v1.0 - Desarrollado con MediaPipe y OpenCV",
            font=('Arial', 9),
            fg='#7F8C8D',
            bg='#34495E'
        )
        version_label.pack()
        
        # Bind para tecla Enter
        self.root.bind('<Return>', lambda e: self.start_application())
        self.root.bind('<space>', lambda e: self.start_application())
        self.root.focus_set()
        
        # También permitir click en cualquier parte para debug
        main_frame.bind('<Button-1>', lambda e: self.start_application())
    
    def start_application(self):
        """Inicia la aplicación principal"""
        print("Botón COMENZAR presionado - Cerrando ventana de bienvenida")
        
        # Cerrar ventana de bienvenida
        self.root.destroy()
        
        # Llamar callback para iniciar aplicación principal
        if self.on_start_callback:
            self.on_start_callback()
    
    def show(self):
        """Muestra la pantalla de bienvenida"""
        # Hacer la ventana modal
        self.root.transient()
        self.root.grab_set()
        self.root.mainloop()

# Función auxiliar para usar desde main.py
def show_welcome_screen(on_start_callback: Callable):
    """
    Muestra la pantalla de bienvenida
    
    Args:
        on_start_callback: Función a ejecutar cuando se presione comenzar
    """
    welcome = WelcomeScreen(on_start_callback)
    welcome.show()