# src/interface/main_window.py (Versión Estable con Mejoras)

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
from typing import Optional

from ..detector.hand_detector import HandDetector
from ..detector.gesture_classifier import GestureClassifier
from ..detector.syllable_classifier import SyllableClassifier
from ..detector.advanced_hand_detector import AdvancedHandDetector
from ..detector.gesture_calibrator import GestureCalibrator
from ..utils.audio_manager import AudioManager
from ..config.settings import Config

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(Config.WINDOW_TITLE)
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        self.root.configure(bg='#2C3E50')
        
        # Componentes principales (versión mejorada)
        self.use_advanced_detector = True  # Flag para usar detector avanzado
        
        if self.use_advanced_detector:
            self.hand_detector = AdvancedHandDetector(
                max_num_hands=Config.MAX_NUM_HANDS,
                min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
            )
        else:
            self.hand_detector = HandDetector(
                max_num_hands=Config.MAX_NUM_HANDS,
                min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
            )
            
        self.gesture_classifier = GestureClassifier()
        # self.gesture_classifier.create_simple_classifier()
        self.syllable_classifier = SyllableClassifier()
        self.gesture_calibrator = GestureCalibrator()  # Nuevo calibrador
        self.audio_manager = AudioManager(
            rate=Config.VOICE_RATE,
            volume=Config.VOICE_VOLUME
        )
        
        # Variables de estado
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.detected_letter = ""
        self.detected_syllable = ""  # Nueva variable para sílabas
        self.detection_mode = "letters"  # "letters" o "syllables"
        self.word_buffer = ""
        self.detection_count = 0
        
        # Variables para auto-agregado de letras
        self.auto_add_enabled = True
        self.last_stable_letter = ""
        self.stable_letter_count = 0
        self.auto_add_threshold = 15  # Frames necesarios para auto-agregar
        self.last_added_letter = ""
        self.cooldown_count = 0
        self.cooldown_threshold = 30  # Frames de espera entre letras
        self.auto_space_enabled = False
        self.no_detection_count = 0
        self.auto_space_threshold = 90  # Frames sin detección para auto-espacio
        
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
        
        # Nuevo: Selector de modo
        mode_frame = ttk.LabelFrame(control_frame, text="Modo de Detección", padding="5")
        mode_frame.pack(side=tk.LEFT, padx=(20, 10))
        
        self.mode_var = tk.StringVar(value="letters")
        letters_radio = ttk.Radiobutton(
            mode_frame,
            text="Letras",
            variable=self.mode_var,
            value="letters",
            command=self.change_detection_mode
        )
        letters_radio.pack()
        
        syllables_radio = ttk.Radiobutton(
            mode_frame,
            text="Sílabas",
            variable=self.mode_var,
            value="syllables",
            command=self.change_detection_mode
        )
        syllables_radio.pack()
        
        # Nuevo: Control de auto-agregado
        auto_add_frame = ttk.Frame(control_frame)
        auto_add_frame.pack(side=tk.LEFT, padx=(20, 10))
        
        self.auto_add_var = tk.BooleanVar(value=True)
        auto_add_check = ttk.Checkbutton(
            auto_add_frame,
            text="Auto-agregar letras",
            variable=self.auto_add_var,
            command=self.toggle_auto_add
        )
        auto_add_check.pack()
        
        # Checkbox para auto-espacio
        self.auto_space_var = tk.BooleanVar(value=False)
        auto_space_check = ttk.Checkbutton(
            auto_add_frame,
            text="Auto-espacio",
            variable=self.auto_space_var,
            command=self.toggle_auto_space
        )
        auto_space_check.pack()
        
        # Control de velocidad de auto-agregado
        ttk.Label(auto_add_frame, text="Velocidad:", font=('Arial', 8)).pack()
        self.speed_var = tk.IntVar(value=5)  # 5 = velocidad media
        speed_scale = ttk.Scale(
            auto_add_frame,
            from_=1, to=10,
            variable=self.speed_var,
            orient=tk.HORIZONTAL,
            length=80,
            command=self.update_auto_add_speed
        )
        speed_scale.pack()
        
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
        
        # Nuevo: Botón para gestión de precisión
        precision_button = ttk.Button(
            status_frame,
            text="Gestión de Precisión",
            command=self.show_precision_manager
        )
        precision_button.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Nuevo: Botón para mostrar galería de referencias
        gallery_button = ttk.Button(
            status_frame,
            text="Galería de Referencias",
            command=self.show_reference_gallery
        )
        gallery_button.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Nuevo: Botón para mostrar letras soportadas
        letters_button = ttk.Button(
            status_frame,
            text="Ver Letras Soportadas",
            command=self.show_supported_letters
        )
        letters_button.pack(side=tk.RIGHT, padx=(0, 20))
    
    def change_detection_mode(self):
        """Cambia entre modo de letras y sílabas"""
        self.detection_mode = self.mode_var.get()
        
        # Resetear historial de detecciones al cambiar modo
        if self.detection_mode == "syllables":
            self.syllable_classifier.reset_detection_history()
            self.status_var.set("Modo sílabas activado - Use ambas manos")
        else:
            self.gesture_classifier.reset_detection_history()
            self.status_var.set("Modo letras activado - Use una mano")
        
        # Limpiar detección actual
        self.detected_letter = ""
        self.detected_syllable = ""
        self.letter_var.set("-")
    
    def toggle_auto_space(self):
        """Activa/desactiva el auto-espacio"""
        self.auto_space_enabled = self.auto_space_var.get()
        status = "activado" if self.auto_space_enabled else "desactivado"
        self.status_var.set(f"Auto-espacio {status}")
    
    def toggle_auto_add(self):
        """Activa/desactiva el auto-agregado de letras"""
        self.auto_add_enabled = self.auto_add_var.get()
        status = "activado" if self.auto_add_enabled else "desactivado"
        self.status_var.set(f"Auto-agregado {status}")
    
    def update_auto_add_speed(self, value):
        """Actualiza la velocidad del auto-agregado"""
        speed = int(float(value))
        # Convertir escala 1-10 a frames necesarios (más velocidad = menos frames)
        self.auto_add_threshold = max(5, 25 - (speed * 2))
        self.cooldown_threshold = max(10, 50 - (speed * 4))
    
    def update_sensitivity(self, value):
        """Actualiza la sensibilidad del clasificador"""
        sensitivity = int(float(value))
        self.gesture_classifier.set_stability_threshold(sensitivity)
    
    def show_precision_manager(self):
        """Muestra la ventana de gestión de precisión"""
        try:
            from .precision_manager import PrecisionManager
            manager = PrecisionManager(self, self.gesture_calibrator)
            manager.show_precision_window()
        except ImportError as e:
            messagebox.showerror("Error", f"No se pudo cargar el gestor de precisión: {e}")
    
    def show_reference_gallery(self):
        """Muestra la galería de referencias"""
        try:
            from .reference_gallery import ReferenceGallery
            gallery = ReferenceGallery(self)
            gallery.show_gallery()
        except ImportError as e:
            messagebox.showerror("Error", f"No se pudo cargar la galería: {e}")
    
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
                
                # Detectar manos (ahora retorna diccionario con left/right)
                processed_frame, hands_data = self.hand_detector.detect_hands(frame)
                
                detected_result = None
                
                if self.detection_mode == "syllables":
                    # Modo sílabas - necesita ambas manos
                    if hands_data['left'] and hands_data['right']:
                        detected_result = self.syllable_classifier.predict_syllable(
                            hands_data['left'], 
                            hands_data['right']
                        )
                elif self.detection_mode == "letters":
                    # Modo letras - usa cualquier mano disponible
                    if hands_data['landmarks_list']:
                        for landmarks in hands_data['landmarks_list']:
                            letter = self.gesture_classifier.predict_gesture(landmarks)
                            if letter:
                                detected_result = letter
                                
                                # Recolectar muestra para calibración automática
                                confidence = hands_data.get('confidence', {}).get('left', 0) or \
                                           hands_data.get('confidence', {}).get('right', 0)
                                self.gesture_calibrator.collect_sample(letter, landmarks, confidence)
                                break
                
                # Actualizar interfaz
                self.update_ui(processed_frame, detected_result, hands_data)
                
            except Exception as e:
                print(f"Error en detección: {e}")
                continue
    
    def update_ui(self, frame, detected_result, hands_data):
        """Actualiza la interfaz con el frame y el resultado detectado (letra o sílaba)"""
        try:
            # Convertir frame para tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Actualizar video
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk
            
            # Lógica de auto-agregado según el modo
            if self.auto_add_enabled and detected_result:
                self.handle_auto_add_logic(detected_result)
            
            # Actualizar display según el modo
            if self.detection_mode == "syllables":
                self.update_syllable_display(detected_result, hands_data)
            else:
                self.update_letter_display(detected_result)
            
        except Exception as e:
            print(f"Error actualizando UI: {e}")
    
    def update_letter_display(self, detected_letter):
        """Actualiza display para modo letras"""
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
    
    def update_syllable_display(self, detected_syllable, hands_data):
        """Actualiza display para modo sílabas"""
        if detected_syllable and detected_syllable != self.detected_syllable:
            self.detected_syllable = detected_syllable
            self.letter_var.set(detected_syllable)  # Usar el mismo display
            self.detection_count += 1
            self.detection_counter_var.set(f"Sílabas: {self.detection_count}")
        elif not detected_syllable:
            # Mostrar estado de las manos
            left_status = "✓" if hands_data['left'] else "✗"
            right_status = "✓" if hands_data['right'] else "✗"
            self.letter_var.set(f"L:{left_status} R:{right_status}")
        
        # Actualizar confianza
        confidence = self.syllable_classifier.get_detection_confidence()
        self.confidence_var.set(confidence * 100)
        self.confidence_label.config(text=f"{confidence*100:.1f}%")
    
    def handle_auto_add_logic(self, detected_letter):
        """Maneja la lógica de auto-agregado de letras"""
        # Reducir cooldown si está activo
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            return
        
        if detected_letter and detected_letter != "-":
            # Resetear contador de no detección
            self.no_detection_count = 0
            
            # Si es la misma letra que la anterior, incrementar contador
            if detected_letter == self.last_stable_letter:
                self.stable_letter_count += 1
            else:
                # Nueva letra detectada, reiniciar contador
                self.last_stable_letter = detected_letter
                self.stable_letter_count = 1
            
            # Si la letra ha sido estable por suficiente tiempo
            if (self.stable_letter_count >= self.auto_add_threshold and 
                detected_letter != self.last_added_letter):
                
                # Agregar la letra automáticamente
                self.auto_add_letter(detected_letter)
                
                # Resetear contadores y activar cooldown
                self.last_added_letter = detected_letter
                self.stable_letter_count = 0
                self.cooldown_count = self.cooldown_threshold
                
        else:
            # No hay letra detectada
            self.stable_letter_count = 0
            
            # Contar frames sin detección para auto-espacio
            if self.auto_space_enabled:
                self.no_detection_count += 1
                
                # Si ha pasado suficiente tiempo sin detección, agregar espacio
                if self.no_detection_count >= self.auto_space_threshold:
                    self.auto_add_space()
                    self.no_detection_count = 0  # Resetear contador
    
    def auto_add_space(self):
        """Agrega automáticamente un espacio"""
        # Verificar que el último caracter no sea ya un espacio
        current_text = self.word_text.get(1.0, tk.END)
        if current_text and current_text[-2] != " ":  # -2 porque el último es \n
            self.word_text.insert(tk.END, " ")
            self.word_text.see(tk.END)
            self.status_var.set("Auto-espacio agregado")
    
    def auto_add_letter(self, letter):
        """Agrega automáticamente una letra al texto"""
        if letter and letter != "-":
            self.word_text.insert(tk.END, letter)
            self.word_text.see(tk.END)
            
            # Actualizar status con indicación visual
            self.status_var.set(f"Auto-agregado: {letter}")
            
            # Opcional: efecto visual breve
            self.root.after(1000, lambda: self.status_var.set("Detectando gestos...") 
                           if self.is_running else None)
    
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