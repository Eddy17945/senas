# src/interface/main_window.py (Versión Estable con Mejoras)

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
from typing import Optional

from ..detector.hand_detector import HandDetector
from ..detector.gesture_classifier import GestureClassifier
from ..utils.audio_manager import AudioManager
from ..config.settings import Config

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(Config.WINDOW_TITLE)
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        self.root.configure(bg='#2C3E50')
        
        # Componentes principales
        self.hand_detector = HandDetector(
            max_num_hands=Config.MAX_NUM_HANDS,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
        )
        self.gesture_classifier = GestureClassifier()
        self.gesture_classifier.create_simple_classifier()
        self.audio_manager = AudioManager(
            rate=Config.VOICE_RATE,
            volume=Config.VOICE_VOLUME
        )
        
        # Variables de estado
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.detected_letter = ""
        self.word_buffer = ""
        self.detection_count = 0
        
        # Configurar interfaz
        self.setup_ui()
        self.setup_camera()
        
        # Protocolo de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Configura la interfaz de usuario mejorada"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame superior - Controles
        control_frame = ttk.LabelFrame(main_frame, text="Controles", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Botones de control
        self.start_button = ttk.Button(
            control_frame, 
            text="Iniciar Detección", 
            command=self.toggle_detection,
            style='Accent.TButton'
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = ttk.Button(
            control_frame, 
            text="Limpiar Texto", 
            command=self.clear_text
        )
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.speak_button = ttk.Button(
            control_frame, 
            text="Reproducir Audio", 
            command=self.speak_text
        )
        self.speak_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Nuevo: Control de sensibilidad
        ttk.Label(control_frame, text="Sensibilidad:").pack(side=tk.LEFT, padx=(20, 5))
        self.sensitivity_var = tk.IntVar(value=5)
        sensitivity_scale = ttk.Scale(
            control_frame, 
            from_=1, to=10, 
            variable=self.sensitivity_var,
            orient=tk.HORIZONTAL,
            length=100,
            command=self.update_sensitivity
        )
        sensitivity_scale.pack(side=tk.LEFT)
        
        # Frame central - Video y resultados
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame de video
        video_frame = ttk.LabelFrame(content_frame, text="Cámara", padding="5")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.video_label = ttk.Label(video_frame, text="Cámara no iniciada")
        self.video_label.pack(expand=True)
        
        # Frame de resultados - MEJORADO
        result_frame = ttk.LabelFrame(content_frame, text="Resultados", padding="10")
        result_frame.pack(side=tk.RIGHT, fill=tk.Y, ipadx=20)
        
        # Letra detectada con confianza
        ttk.Label(result_frame, text="Letra Detectada:", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        detection_info_frame = ttk.Frame(result_frame)
        detection_info_frame.pack(pady=5)
        
        self.letter_var = tk.StringVar(value="-")
        letter_display = ttk.Label(
            detection_info_frame, 
            textvariable=self.letter_var,
            font=('Arial', 48, 'bold'),
            foreground='#E74C3C'
        )
        letter_display.pack()
        
        # Nuevo: Barra de confianza simple
        ttk.Label(result_frame, text="Confianza:", font=('Arial', 10)).pack(anchor=tk.W, pady=(10, 0))
        self.confidence_var = tk.DoubleVar()
        confidence_bar = ttk.Progressbar(
            result_frame,
            variable=self.confidence_var,
            maximum=100,
            length=200
        )
        confidence_bar.pack(pady=2)
        
        self.confidence_label = ttk.Label(result_frame, text="0%")
        self.confidence_label.pack()
        
        # Palabra formada - mantenemos igual
        ttk.Label(result_frame, text="Palabra:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(20, 0))
        
        self.word_text = tk.Text(
            result_frame, 
            height=8, 
            width=25,
            font=('Arial', 14),
            wrap=tk.WORD
        )
        self.word_text.pack(pady=5)
        
        # Scrollbar para el texto
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.word_text.yview)
        self.word_text.configure(yscrollcommand=scrollbar.set)
        
        # Botones de palabra
        word_buttons_frame = ttk.Frame(result_frame)
        word_buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            word_buttons_frame, 
            text="Agregar Letra", 
            command=self.add_letter_to_word
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            word_buttons_frame, 
            text="Espacio", 
            command=self.add_space
        ).pack(side=tk.RIGHT)
        
        # Frame inferior - Estado mejorado
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Listo para iniciar")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # Nuevo: Contador de detecciones
        self.detection_counter_var = tk.StringVar(value="Detecciones: 0")
        counter_label = ttk.Label(status_frame, textvariable=self.detection_counter_var)
        counter_label.pack(side=tk.RIGHT)
        
        # Nuevo: Botón para mostrar letras soportadas
        letters_button = ttk.Button(
            status_frame,
            text="Ver Letras Soportadas",
            command=self.show_supported_letters
        )
        letters_button.pack(side=tk.RIGHT, padx=(0, 20))
    
    def update_sensitivity(self, value):
        """Actualiza la sensibilidad del clasificador"""
        sensitivity = int(float(value))
        self.gesture_classifier.set_stability_threshold(sensitivity)
    
    def show_supported_letters(self):
        """Muestra ventana con letras soportadas"""
        letters_window = tk.Toplevel(self.root)
        letters_window.title("Letras Soportadas")
        letters_window.geometry("600x400")
        letters_window.configure(bg='white')
        
        # Título
        title_label = ttk.Label(letters_window, text="Alfabeto de Lenguaje de Señas Soportado", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Frame para las letras
        letters_frame = ttk.Frame(letters_window)
        letters_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Mostrar letras en grid
        row = 0
        col = 0
        for i, letter in enumerate(Config.SUPPORTED_LETTERS):
            letter_label = ttk.Label(
                letters_frame, 
                text=letter, 
                font=('Arial', 20, 'bold'),
                background='lightblue',
                foreground='darkblue',
                width=3,
                anchor='center'
            )
            letter_label.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            col += 1
            if col >= 8:  # 8 letras por fila
                col = 0
                row += 1
        
        # Configurar grid
        for i in range(8):
            letters_frame.columnconfigure(i, weight=1)
        
        # Botón cerrar
        close_button = ttk.Button(letters_window, text="Cerrar", command=letters_window.destroy)
        close_button.pack(pady=10)
    
    def setup_camera(self):
        """Configura la cámara (versión original que funciona)"""
        try:
            self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            
            if not self.cap.isOpened():
                raise Exception("No se pudo abrir la cámara")
            
            self.status_var.set("Cámara configurada correctamente")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error configurando cámara: {e}")
            self.status_var.set("Error en cámara")
    
    def toggle_detection(self):
        """Alterna entre iniciar y detener la detección"""
        if not self.is_running:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        """Inicia la detección de gestos"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Cámara no disponible")
            return
        
        self.is_running = True
        self.start_button.config(text="Detener Detección")
        self.status_var.set("Detectando gestos...")
        
        # Iniciar hilo de detección
        detection_thread = threading.Thread(target=self.detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
    
    def stop_detection(self):
        """Detiene la detección de gestos"""
        self.is_running = False
        self.start_button.config(text="Iniciar Detección")
        self.status_var.set("Detección detenida")
    
    def detection_loop(self):
        """Bucle principal de detección"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Voltear frame horizontalmente para efecto espejo
                frame = cv2.flip(frame, 1)
                
                # Detectar manos
                processed_frame, hand_landmarks_list = self.hand_detector.detect_hands(frame)
                
                # Clasificar gesto si hay manos detectadas
                detected_letter = None
                if hand_landmarks_list:
                    for landmarks in hand_landmarks_list:
                        letter = self.gesture_classifier.predict_gesture(landmarks)
                        if letter:
                            detected_letter = letter
                            break
                
                # Actualizar interfaz
                self.update_ui(processed_frame, detected_letter)
                
            except Exception as e:
                print(f"Error en detección: {e}")
                continue
    
    def update_ui(self, frame, detected_letter):
        """Actualiza la interfaz con el frame y la letra detectada"""
        try:
            # Convertir frame para tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Actualizar video
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk
            
            # Actualizar letra detectada y estadísticas
            if detected_letter and detected_letter != self.detected_letter:
                self.detected_letter = detected_letter
                self.letter_var.set(detected_letter)
                self.detection_count += 1
                self.detection_counter_var.set(f"Detecciones: {self.detection_count}")
            elif not detected_letter:
                self.letter_var.set("-")
            
            # Actualizar confianza
            confidence = self.gesture_classifier.get_detection_confidence()
            self.confidence_var.set(confidence * 100)
            self.confidence_label.config(text=f"{confidence*100:.1f}%")
        
        except Exception as e:
            print(f"Error actualizando UI: {e}")
    
    def add_letter_to_word(self):
        """Agrega la letra detectada a la palabra"""
        if self.detected_letter and self.detected_letter != "-":
            self.word_text.insert(tk.END, self.detected_letter)
            self.word_text.see(tk.END)
    
    def add_space(self):
        """Agrega un espacio a la palabra"""
        self.word_text.insert(tk.END, " ")
        self.word_text.see(tk.END)
    
    def clear_text(self):
        """Limpia el texto formado"""
        self.word_text.delete(1.0, tk.END)
        self.letter_var.set("-")
        self.detected_letter = ""
    
    def speak_text(self):
        """Reproduce el texto formado en audio"""
        text = self.word_text.get(1.0, tk.END).strip()
        if text:
            self.audio_manager.speak(text)
            self.status_var.set(f"Reproduciendo: {text}")
        else:
            messagebox.showinfo("Información", "No hay texto para reproducir")
    
    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        self.stop_detection()
        if self.cap:
            self.cap.release()
        self.audio_manager.stop()
        self.root.destroy()
    
    def run(self):
        """Inicia la aplicación"""
        self.root.mainloop()