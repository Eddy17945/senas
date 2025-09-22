# src/interface/main_window.py

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
        
        # Configurar interfaz
        self.setup_ui()
        self.setup_camera()
        
        # Protocolo de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """
        Configura la interfaz de usuario
        """
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
        self.speak_button.pack(side=tk.LEFT)
        
        # Frame central - Video y resultados
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame de video
        video_frame = ttk.LabelFrame(content_frame, text="Cámara", padding="5")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.video_label = ttk.Label(video_frame, text="Cámara no iniciada")
        self.video_label.pack(expand=True)
        
        # Frame de resultados
        result_frame = ttk.LabelFrame(content_frame, text="Resultados", padding="10")
        result_frame.pack(side=tk.RIGHT, fill=tk.Y, ipadx=20)
        
        # Letra detectada
        ttk.Label(result_frame, text="Letra Detectada:", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        self.letter_var = tk.StringVar(value="-")
        letter_display = ttk.Label(
            result_frame, 
            textvariable=self.letter_var,
            font=('Arial', 48, 'bold'),
            foreground='#E74C3C'
        )
        letter_display.pack(pady=10)
        
        # Palabra formada
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
        
        # Frame inferior - Estado
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Listo para iniciar")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # Información de letras soportadas
        supported_letters = ", ".join(Config.SUPPORTED_LETTERS)
        info_label = ttk.Label(
            status_frame, 
            text=f"Letras soportadas: {supported_letters}",
            foreground='#7F8C8D'
        )
        info_label.pack(side=tk.RIGHT)
    
    def setup_camera(self):
        """
        Configura la cámara
        """
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
        """
        Alterna entre iniciar y detener la detección
        """
        if not self.is_running:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        """
        Inicia la detección de gestos
        """
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
        """
        Detiene la detección de gestos
        """
        self.is_running = False
        self.start_button.config(text="Iniciar Detección")
        self.status_var.set("Detección detenida")
    
    def detection_loop(self):
        """
        Bucle principal de detección
        """
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
        """
        Actualiza la interfaz con el frame y la letra detectada
        """
        try:
            # Convertir frame para tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Actualizar video
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk
            
            # Actualizar letra detectada
            if detected_letter and detected_letter != self.detected_letter:
                self.detected_letter = detected_letter
                self.letter_var.set(detected_letter)
            elif not detected_letter:
                self.letter_var.set("-")
        
        except Exception as e:
            print(f"Error actualizando UI: {e}")
    
    def add_letter_to_word(self):
        """
        Agrega la letra detectada a la palabra
        """
        if self.detected_letter and self.detected_letter != "-":
            self.word_text.insert(tk.END, self.detected_letter)
            self.word_text.see(tk.END)
    
    def add_space(self):
        """
        Agrega un espacio a la palabra
        """
        self.word_text.insert(tk.END, " ")
        self.word_text.see(tk.END)
    
    def clear_text(self):
        """
        Limpia el texto formado
        """
        self.word_text.delete(1.0, tk.END)
        self.letter_var.set("-")
        self.detected_letter = ""
    
    def speak_text(self):
        """
        Reproduce el texto formado en audio
        """
        text = self.word_text.get(1.0, tk.END).strip()
        if text:
            self.audio_manager.speak(text)
            self.status_var.set(f"Reproduciendo: {text}")
        else:
            messagebox.showinfo("Información", "No hay texto para reproducir")
    
    def on_closing(self):
        """
        Maneja el cierre de la aplicación
        """
        self.stop_detection()
        if self.cap:
            self.cap.release()
        self.audio_manager.stop()
        self.root.destroy()
    
    def run(self):
        """
        Inicia la aplicación
        """
        self.root.mainloop()