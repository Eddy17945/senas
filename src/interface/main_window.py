# src/interface/main_window.py
# VERSIÓN CON DISEÑO MODERNO Y COLORES

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
from ..detector.gesture_controls import GestureControls
from ..utils.word_dictionary import WordDictionary
from ..utils.word_suggester import WordSuggester
from ..utils.audio_manager import AudioManager
from ..config.settings import Config

class MainWindow:
    # PALETA DE COLORES MODERNA
    COLORS = {
        'primary': '#2563EB',      # Azul principal
        'secondary': '#3B82F6',    # Azul claro
        'accent': '#10B981',       # Verde éxito
        'danger': '#EF4444',       # Rojo
        'warning': '#F59E0B',      # Amarillo/Naranja
        'bg_dark': '#1E293B',      # Fondo oscuro
        'bg_light': '#F8FAFC',     # Fondo claro
        'text_dark': '#0F172A',    # Texto oscuro
        'text_light': '#F8FAFC',   # Texto claro
        'border': '#CBD5E1',       # Bordes
        'hover': '#1E40AF',        # Hover azul oscuro
    }
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤟 " + Config.WINDOW_TITLE)
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        
        # Permitir redimensionar pero con tamaño mínimo
        self.root.minsize(1000, 600)  # Tamaño mínimo para ver todo
        self.root.resizable(True, True)  # Permitir redimensionar
        
        # Configurar tema de colores
        self.setup_styles()
        self.root.configure(bg=self.COLORS['bg_light'])
        
        # Componentes principales
        self.use_advanced_detector = True
        
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
        self.syllable_classifier = SyllableClassifier()
        self.gesture_calibrator = GestureCalibrator()
        self.gesture_controls = GestureControls()
        self.word_dictionary = WordDictionary()
        self.word_suggester = WordSuggester()
        self.audio_manager = AudioManager(
            rate=Config.VOICE_RATE,
            volume=Config.VOICE_VOLUME
        )
        
        # Variables de estado
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.detected_letter = ""
        self.detected_syllable = ""
        self.detection_mode = "letters"
        self.word_buffer = ""
        self.detection_count = 0
        
        # Variables para auto-agregado
        self.auto_add_enabled = True
        self.last_stable_letter = ""
        self.stable_letter_count = 0
        self.auto_add_threshold = 15
        self.last_added_letter = ""
        self.cooldown_count = 0
        self.cooldown_threshold = 30
        self.auto_space_enabled = False
        self.no_detection_count = 0
        self.auto_space_threshold = 90

        # Variables para controles
        self.control_gesture_detected = None
        self.show_control_feedback = False
        self.control_feedback_text = ""

        # Variables para sugerencias
        self.current_suggestions = []
        self.suggestions_enabled = True
        
        # Configurar interfaz
        self.setup_ui()
        self.setup_camera()
        
        # Protocolo de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Configura los estilos personalizados de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Estilo para botones principales
        style.configure(
            'Primary.TButton',
            background=self.COLORS['primary'],
            foreground='white',
            borderwidth=0,
            focuscolor='none',
            font=('Segoe UI', 10, 'bold'),
            padding=10
        )
        style.map('Primary.TButton',
                 background=[('active', self.COLORS['hover'])],
                 foreground=[('active', 'white')])
        
        # Estilo para botones de éxito
        style.configure(
            'Success.TButton',
            background=self.COLORS['accent'],
            foreground='white',
            borderwidth=0,
            font=('Segoe UI', 10),
            padding=8
        )
        style.map('Success.TButton',
                 background=[('active', '#059669')])
        
        # Estilo para botones de peligro
        style.configure(
            'Danger.TButton',
            background=self.COLORS['danger'],
            foreground='white',
            borderwidth=0,
            font=('Segoe UI', 10),
            padding=8
        )
        style.map('Danger.TButton',
                 background=[('active', '#DC2626')])
        
        # Estilo para frames
        style.configure(
            'Card.TFrame',
            background='white',
            relief='flat'
        )
        
        style.configure(
            'TLabelframe',
            background='white',
            borderwidth=2,
            relief='groove'
        )
        style.configure(
            'TLabelframe.Label',
            background='white',
            foreground=self.COLORS['text_dark'],
            font=('Segoe UI', 11, 'bold')
        )
        
        # Estilo para labels
        style.configure(
            'Title.TLabel',
            background='white',
            foreground=self.COLORS['primary'],
            font=('Segoe UI', 14, 'bold')
        )
        
        style.configure(
            'TLabel',
            background='white',
            foreground=self.COLORS['text_dark'],
            font=('Segoe UI', 10)
        )
        
        # Estilo para radio buttons
        style.configure(
            'TRadiobutton',
            background='white',
            foreground=self.COLORS['text_dark'],
            font=('Segoe UI', 9)
        )
        
        # Estilo para checkbuttons
        style.configure(
            'TCheckbutton',
            background='white',
            foreground=self.COLORS['text_dark'],
            font=('Segoe UI', 9)
        )
    
    def setup_ui(self):
        """Configura la interfaz de usuario moderna"""
        # Frame principal con padding
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg_light'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # ========== HEADER - TÍTULO Y CONTROLES PRINCIPALES ==========
        header_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Título principal
        title_label = tk.Label(
            header_frame,
            text="🤟 Traductor de Lenguaje de Señas",
            font=('Segoe UI', 20, 'bold'),
            bg='white',
            fg=self.COLORS['primary']
        )
        title_label.pack(pady=15)
        
        # Frame de controles principales
        control_frame = tk.Frame(header_frame, bg='white')
        control_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        # Botones principales con colores
        buttons_frame = tk.Frame(control_frame, bg='white')
        buttons_frame.pack(side=tk.LEFT)
        
        self.start_button = tk.Button(
            buttons_frame,
            text="▶ Iniciar Detección",
            command=self.toggle_detection,
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.start_button.bind('<Enter>', lambda e: self.start_button.config(bg=self.COLORS['hover']))
        self.start_button.bind('<Leave>', lambda e: self.start_button.config(bg=self.COLORS['primary']))
        
        clear_button = tk.Button(
            buttons_frame,
            text="🗑 Limpiar",
            command=self.clear_text,
            bg=self.COLORS['danger'],
            fg='white',
            font=('Segoe UI', 10),
            relief='flat',
            padx=15,
            pady=10,
            cursor='hand2'
        )
        clear_button.pack(side=tk.LEFT, padx=5)
        clear_button.bind('<Enter>', lambda e: clear_button.config(bg='#DC2626'))
        clear_button.bind('<Leave>', lambda e: clear_button.config(bg=self.COLORS['danger']))
        
        speak_button = tk.Button(
            buttons_frame,
            text="🔊 Reproducir",
            command=self.speak_text,
            bg=self.COLORS['accent'],
            fg='white',
            font=('Segoe UI', 10),
            relief='flat',
            padx=15,
            pady=10,
            cursor='hand2'
        )
        speak_button.pack(side=tk.LEFT, padx=5)
        speak_button.bind('<Enter>', lambda e: speak_button.config(bg='#059669'))
        speak_button.bind('<Leave>', lambda e: speak_button.config(bg=self.COLORS['accent']))
        
        # Modo de detección
        mode_frame = tk.LabelFrame(
            control_frame,
            text="Modo",
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 10, 'bold'),
            relief='groove',
            bd=2
        )
        mode_frame.pack(side=tk.LEFT, padx=20)
        
        self.mode_var = tk.StringVar(value="letters")
        
        letters_radio = tk.Radiobutton(
            mode_frame,
            text="📝 Letras",
            variable=self.mode_var,
            value="letters",
            command=self.change_detection_mode,
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 9),
            selectcolor=self.COLORS['secondary']
        )
        letters_radio.pack(anchor=tk.W, padx=10, pady=2)
        
        syllables_radio = tk.Radiobutton(
            mode_frame,
            text="🔤 Sílabas",
            variable=self.mode_var,
            value="syllables",
            command=self.change_detection_mode,
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 9),
            selectcolor=self.COLORS['secondary']
        )
        syllables_radio.pack(anchor=tk.W, padx=10, pady=2)
        
        # Configuración rápida
        config_frame = tk.LabelFrame(
            control_frame,
            text="⚙ Configuración",
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 10, 'bold'),
            relief='groove',
            bd=2
        )
        config_frame.pack(side=tk.LEFT, padx=10)
        
        self.auto_add_var = tk.BooleanVar(value=True)
        auto_add_check = tk.Checkbutton(
            config_frame,
            text="Auto-agregar",
            variable=self.auto_add_var,
            command=self.toggle_auto_add,
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 9),
            selectcolor=self.COLORS['accent']
        )
        auto_add_check.pack(anchor=tk.W, padx=10, pady=2)
        
        self.auto_space_var = tk.BooleanVar(value=False)
        auto_space_check = tk.Checkbutton(
            config_frame,
            text="Auto-espacio",
            variable=self.auto_space_var,
            command=self.toggle_auto_space,
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 9),
            selectcolor=self.COLORS['accent']
        )
        auto_space_check.pack(anchor=tk.W, padx=10, pady=2)
        
        # ========== CONTENIDO PRINCIPAL ==========
        content_frame = tk.Frame(main_frame, bg=self.COLORS['bg_light'])
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame izquierdo - VIDEO
        left_frame = tk.Frame(content_frame, bg='white', relief='raised', bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        video_header = tk.Frame(left_frame, bg=self.COLORS['primary'], height=40)
        video_header.pack(fill=tk.X)
        
        tk.Label(
            video_header,
            text="📹 Cámara en Vivo",
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=8)
        
        self.video_label = tk.Label(
            left_frame,
            text="Presiona 'Iniciar' para comenzar",
            bg='#F1F5F9',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 12),
            width=80,   # FIJO: ancho en caracteres
            height=30   # FIJO: alto en líneas
        )
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Controles de ajuste
        controls_frame = tk.Frame(left_frame, bg='white')
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Velocidad
        vel_frame = tk.Frame(controls_frame, bg='white')
        vel_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            vel_frame,
            text="⚡ Velocidad:",
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 9, 'bold')
        ).pack()
        
        self.speed_var = tk.IntVar(value=5)
        speed_scale = tk.Scale(
            vel_frame,
            from_=1, to=10,
            variable=self.speed_var,
            orient=tk.HORIZONTAL,
            length=120,
            command=self.update_auto_add_speed,
            bg='white',
            fg=self.COLORS['primary'],
            troughcolor=self.COLORS['border'],
            highlightthickness=0
        )
        speed_scale.pack()
        
        # Sensibilidad
        sens_frame = tk.Frame(controls_frame, bg='white')
        sens_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            sens_frame,
            text="🎯 Sensibilidad:",
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 9, 'bold')
        ).pack()
        
        self.sensitivity_var = tk.IntVar(value=5)
        sensitivity_scale = tk.Scale(
            sens_frame,
            from_=1, to=10,
            variable=self.sensitivity_var,
            orient=tk.HORIZONTAL,
            length=120,
            command=self.update_sensitivity,
            bg='white',
            fg=self.COLORS['primary'],
            troughcolor=self.COLORS['border'],
            highlightthickness=0
        )
        sensitivity_scale.pack()
        
        # Frame derecho - RESULTADOS
        right_frame = tk.Frame(content_frame, bg='white', width=320, relief='raised', bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        result_header = tk.Frame(right_frame, bg=self.COLORS['accent'], height=40)
        result_header.pack(fill=tk.X)
        
        tk.Label(
            result_header,
            text="✨ Resultados",
            bg=self.COLORS['accent'],
            fg='white',
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=8)
        
        # Letra detectada
        detection_frame = tk.Frame(right_frame, bg=self.COLORS['bg_light'], relief='groove', bd=2)
        detection_frame.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(
            detection_frame,
            text="Letra Detectada",
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 11, 'bold')
        ).pack(pady=5)
        
        self.letter_var = tk.StringVar(value="-")
        letter_display = tk.Label(
            detection_frame,
            textvariable=self.letter_var,
            bg='white',
            fg=self.COLORS['primary'],
            font=('Arial', 56, 'bold'),
            relief='flat',
            width=3,
            height=1
        )
        letter_display.pack(pady=10)
        
        # Barra de confianza
        confidence_frame = tk.Frame(detection_frame, bg=self.COLORS['bg_light'])
        confidence_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            confidence_frame,
            text="Confianza:",
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 9)
        ).pack()
        
        self.confidence_var = tk.DoubleVar()
        
        # Canvas para barra de progreso personalizada
        self.confidence_canvas = tk.Canvas(
            confidence_frame,
            height=25,
            bg='white',
            highlightthickness=1,
            highlightbackground=self.COLORS['border']
        )
        self.confidence_canvas.pack(fill=tk.X, pady=5)
        
        self.confidence_label = tk.Label(
            confidence_frame,
            text="0%",
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 10, 'bold')
        )
        self.confidence_label.pack()
        
        # Área de texto
        text_frame = tk.Frame(right_frame, bg='white')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        tk.Label(
            text_frame,
            text="📝 Texto Formado",
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor=tk.W, pady=5)
        
        text_container = tk.Frame(text_frame, bg=self.COLORS['border'], relief='flat', bd=1)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.word_text = tk.Text(
            text_container,
            height=8,
            width=28,
            font=('Consolas', 13),
            wrap=tk.WORD,
            bg='white',
            fg=self.COLORS['text_dark'],
            relief='flat',
            padx=8,
            pady=8
        )
        self.word_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Botones de texto
        text_buttons_frame = tk.Frame(text_frame, bg='white')
        text_buttons_frame.pack(fill=tk.X, pady=10)
        
        add_letter_btn = tk.Button(
            text_buttons_frame,
            text="+ Letra",
            command=self.add_letter_to_word,
            bg=self.COLORS['secondary'],
            fg='white',
            font=('Segoe UI', 9),
            relief='flat',
            padx=10,
            pady=5,
            cursor='hand2'
        )
        add_letter_btn.pack(side=tk.LEFT, padx=2)
        
        space_btn = tk.Button(
            text_buttons_frame,
            text="Espacio",
            command=self.add_space,
            bg=self.COLORS['border'],
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 9),
            relief='flat',
            padx=10,
            pady=5,
            cursor='hand2'
        )
        space_btn.pack(side=tk.RIGHT, padx=2)


          # ========== SECCIÓN DE SUGERENCIAS ==========
        suggestions_container = tk.Frame(text_frame, bg='white')
        suggestions_container.pack(fill=tk.X, pady=10)
        
        # Frame para sugerencias dinámicas
        self.suggestions_frame = tk.Frame(
            suggestions_container, 
            bg='white',
            height=35
        )
        self.suggestions_frame.pack(fill=tk.X)
        
        # Separador
        tk.Frame(suggestions_container, bg=self.COLORS['border'], height=1).pack(fill=tk.X, pady=5)
        
        # Palabras rápidas (siempre visibles)
        quick_words_label = tk.Label(
            suggestions_container,
            text="⚡ Palabras Rápidas:",
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 9, 'bold')
        )
        quick_words_label.pack(anchor=tk.W, pady=(5, 5))
        
        quick_words_frame = tk.Frame(suggestions_container, bg='white')
        quick_words_frame.pack(fill=tk.X)
        
        # Crear botones de palabras rápidas (2 filas)
        quick_words = self.word_dictionary.get_quick_words()
        
        for i, word in enumerate(quick_words[:10]):  # Máximo 10
            row = i // 5
            col = i % 5
            
            btn = tk.Button(
                quick_words_frame,
                text=word,
                command=lambda w=word: self.apply_quick_word(w),
                bg=self.COLORS['accent'],
                fg='white',
                font=('Segoe UI', 8),
                relief='flat',
                padx=8,
                pady=3,
                cursor='hand2',
                width=8
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky='ew')
            btn.bind('<Enter>', lambda e, b=btn: b.config(bg='#059669'))
            btn.bind('<Leave>', lambda e, b=btn: b.config(bg=self.COLORS['accent']))
        
        # Configurar columnas para que se expandan
        for i in range(5):
            quick_words_frame.columnconfigure(i, weight=1)
        
        # Separador
        tk.Frame(suggestions_container, bg=self.COLORS['border'], height=1).pack(fill=tk.X, pady=5)
        
        # Botón de frases completas
        phrases_btn = tk.Button(
            suggestions_container,
            text="📝 Ver Frases Completas",
            command=self.show_phrases_window,
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 9),
            relief='flat',
            padx=10,
            pady=5,
            cursor='hand2'
        )
        phrases_btn.pack(fill=tk.X, pady=5)
        phrases_btn.bind('<Enter>', lambda e: phrases_btn.config(bg=self.COLORS['hover']))
        phrases_btn.bind('<Leave>', lambda e: phrases_btn.config(bg=self.COLORS['primary']))

        
        # ========== BARRA DE ESTADO ==========
        status_bar = tk.Frame(main_frame, bg=self.COLORS['bg_dark'], height=35)
        status_bar.pack(fill=tk.X, pady=(15, 0))
        
        self.status_var = tk.StringVar(value="✓ Listo para iniciar")
        status_label = tk.Label(
            status_bar,
            textvariable=self.status_var,
            bg=self.COLORS['bg_dark'],
            fg=self.COLORS['text_light'],
            font=('Segoe UI', 9)
        )
        status_label.pack(side=tk.LEFT, padx=15, pady=5)
        
        self.detection_counter_var = tk.StringVar(value="Detecciones: 0")
        counter_label = tk.Label(
            status_bar,
            textvariable=self.detection_counter_var,
            bg=self.COLORS['bg_dark'],
            fg=self.COLORS['accent'],
            font=('Segoe UI', 9, 'bold')
        )
        counter_label.pack(side=tk.RIGHT, padx=15, pady=5)
        
        # Botones de ayuda
        help_buttons_frame = tk.Frame(status_bar, bg=self.COLORS['bg_dark'])
        help_buttons_frame.pack(side=tk.RIGHT, padx=10)
        
        # Botón Galería de Referencias
        gallery_btn = tk.Button(
            help_buttons_frame,
            text="🖼 Referencias",
            command=self.show_reference_gallery,
            bg=self.COLORS['warning'],
            fg='white',
            font=('Segoe UI', 8),
            relief='flat',
            padx=8,
            pady=3,
            cursor='hand2'
        )
        gallery_btn.pack(side=tk.LEFT, padx=2)
        gallery_btn.bind('<Enter>', lambda e: gallery_btn.config(bg='#D97706'))
        gallery_btn.bind('<Leave>', lambda e: gallery_btn.config(bg=self.COLORS['warning']))
        
        # Botón Gestión de Precisión
        precision_btn = tk.Button(
            help_buttons_frame,
            text="🎯 Calibrar",
            command=self.show_precision_manager,
            bg=self.COLORS['accent'],
            fg='white',
            font=('Segoe UI', 8),
            relief='flat',
            padx=8,
            pady=3,
            cursor='hand2'
        )
        precision_btn.pack(side=tk.LEFT, padx=2)
        precision_btn.bind('<Enter>', lambda e: precision_btn.config(bg='#059669'))
        precision_btn.bind('<Leave>', lambda e: precision_btn.config(bg=self.COLORS['accent']))
        
        # Botón Letras Soportadas
        letters_btn = tk.Button(
            help_buttons_frame,
            text="📚 Letras",
            command=self.show_supported_letters,
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 8),
            relief='flat',
            padx=8,
            pady=3,
            cursor='hand2'
        )
        letters_btn.pack(side=tk.LEFT, padx=2)
        letters_btn.bind('<Enter>', lambda e: letters_btn.config(bg=self.COLORS['hover']))
        letters_btn.bind('<Leave>', lambda e: letters_btn.config(bg=self.COLORS['primary']))



    def show_phrases_window(self):
        """Muestra ventana con frases completas comunes"""
        phrases_window = tk.Toplevel(self.root)
        phrases_window.title("📝 Frases Comunes")
        phrases_window.geometry("500x600")
        phrases_window.configure(bg='white')
        
        # Header
        header = tk.Frame(phrases_window, bg=self.COLORS['primary'], height=60)
        header.pack(fill=tk.X)
        
        tk.Label(
            header,
            text="📝 Frases Completas Comunes",
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 16, 'bold')
        ).pack(pady=15)
        
        # Contenedor con scroll
        container = tk.Frame(phrases_window, bg='white')
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Canvas y scrollbar
        canvas = tk.Canvas(container, bg='white', highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Obtener frases
        phrases = self.word_dictionary.get_common_phrases()
        
        # Crear botones de frases
        for phrase in phrases:
            phrase_btn = tk.Button(
                scrollable_frame,
                text=phrase,
                command=lambda p=phrase: self.select_phrase(p, phrases_window),
                bg=self.COLORS['bg_light'],
                fg=self.COLORS['text_dark'],
                font=('Segoe UI', 10),
                relief='raised',
                bd=2,
                padx=15,
                pady=10,
                cursor='hand2',
                anchor='w',
                justify='left'
            )
            phrase_btn.pack(fill=tk.X, pady=3)
            phrase_btn.bind('<Enter>', lambda e, b=phrase_btn: b.config(bg=self.COLORS['secondary'], fg='white'))
            phrase_btn.bind('<Leave>', lambda e, b=phrase_btn: b.config(bg=self.COLORS['bg_light'], fg=self.COLORS['text_dark']))
        
        # Botón cerrar
        close_btn = tk.Button(
            phrases_window,
            text="Cerrar",
            command=phrases_window.destroy,
            bg=self.COLORS['danger'],
            fg='white',
            font=('Segoe UI', 10),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2'
        )
        close_btn.pack(pady=10)
    
    def select_phrase(self, phrase: str, window):
        """Selecciona una frase y cierra la ventana"""
        self.apply_phrase(phrase)
        window.destroy()   
    
    def update_confidence_bar(self, confidence):
        """Actualiza la barra de confianza personalizada"""
        self.confidence_canvas.delete("all")
        width = self.confidence_canvas.winfo_width()
        height = self.confidence_canvas.winfo_height()
        
        if width > 1:
            # Dibujar fondo
            self.confidence_canvas.create_rectangle(
                0, 0, width, height,
                fill='#E2E8F0',
                outline=''
            )
            
            # Dibujar barra de progreso
            bar_width = int(width * (confidence / 100))
            
            # Color según confianza
            if confidence > 75:
                color = self.COLORS['accent']
            elif confidence > 50:
                color = self.COLORS['warning']
            else:
                color = self.COLORS['danger']
            
            self.confidence_canvas.create_rectangle(
                0, 0, bar_width, height,
                fill=color,
                outline=''
            )
    
    # [RESTO DE MÉTODOS SE MANTIENEN IGUALES]
    def change_detection_mode(self):
        self.detection_mode = self.mode_var.get()
        if self.detection_mode == "syllables":
            self.syllable_classifier.reset_detection_history()
            self.status_var.set("✓ Modo sílabas - Use ambas manos")
        else:
            self.gesture_classifier.reset_detection_history()
            self.status_var.set("✓ Modo letras - Use una mano")
        self.detected_letter = ""
        self.detected_syllable = ""
        self.letter_var.set("-")
    
    def toggle_auto_space(self):
        self.auto_space_enabled = self.auto_space_var.get()
        status = "activado" if self.auto_space_enabled else "desactivado"
        self.status_var.set(f"✓ Auto-espacio {status}")
    
    def toggle_auto_add(self):
        self.auto_add_enabled = self.auto_add_var.get()
        status = "activado" if self.auto_add_enabled else "desactivado"
        self.status_var.set(f"✓ Auto-agregado {status}")
    
    def update_auto_add_speed(self, value):
        speed = int(float(value))
        self.auto_add_threshold = max(5, 25 - (speed * 2))
        self.cooldown_threshold = max(10, 50 - (speed * 4))
    
    def update_sensitivity(self, value):
        sensitivity = int(float(value))
        self.gesture_classifier.set_stability_threshold(sensitivity)
    
    def show_precision_manager(self):
        try:
            from .precision_manager import PrecisionManager
            manager = PrecisionManager(self, self.gesture_calibrator)
            manager.show_precision_window()
        except ImportError as e:
            messagebox.showerror("Error", f"No se pudo cargar: {e}")
    
    def show_reference_gallery(self):
        try:
            from .reference_gallery import ReferenceGallery
            gallery = ReferenceGallery(self)
            gallery.show_gallery()
        except ImportError as e:
            messagebox.showerror("Error", f"No se pudo cargar: {e}")
    
    def show_supported_letters(self):
        letters_window = tk.Toplevel(self.root)
        letters_window.title("📚 Letras Soportadas")
        letters_window.geometry("700x500")
        letters_window.configure(bg='white')
        
        # Header
        header = tk.Frame(letters_window, bg=self.COLORS['primary'], height=60)
        header.pack(fill=tk.X)
        
        tk.Label(
            header,
            text="📚 Alfabeto de Lenguaje de Señas",
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 18, 'bold')
        ).pack(pady=15)
        
        # Grid de letras
        letters_container = tk.Frame(letters_window, bg='white')
        letters_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        row, col = 0, 0
        for i, letter in enumerate(Config.SUPPORTED_LETTERS):
            letter_frame = tk.Frame(
                letters_container,
                bg=self.COLORS['secondary'],
                relief='raised',
                bd=2
            )
            letter_frame.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')
            
            tk.Label(
                letter_frame,
                text=letter,
                font=('Arial', 24, 'bold'),
                bg=self.COLORS['secondary'],
                fg='white',
                width=2,
                height=1
            ).pack(padx=10, pady=10)
            
            col += 1
            if col >= 7:
                col = 0
                row += 1
        
        for i in range(7):
            letters_container.columnconfigure(i, weight=1)
        
        # Botón cerrar
        tk.Button(
            letters_window,
            text="Cerrar",
            command=letters_window.destroy,
            bg=self.COLORS['danger'],
            fg='white',
            font=('Segoe UI', 11),
            relief='flat',
            padx=30,
            pady=10,
            cursor='hand2'
        ).pack(pady=20)
    
    def setup_camera(self):
        try:
            self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            
            if not self.cap.isOpened():
                raise Exception("No se pudo abrir la cámara")
            
            self.status_var.set("✓ Cámara configurada")
        except Exception as e:
            messagebox.showerror("Error", f"Error cámara: {e}")
            self.status_var.set("✗ Error en cámara")
    
    def toggle_detection(self):
        if not self.is_running:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Cámara no disponible")
            return
        
        self.is_running = True
        self.start_button.config(text="⏸ Detener", bg=self.COLORS['danger'])
        self.status_var.set("🔴 Detectando gestos...")
        
        detection_thread = threading.Thread(target=self.detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
    
    def stop_detection(self):
        self.is_running = False
        self.start_button.config(text="▶ Iniciar Detección", bg=self.COLORS['primary'])
        self.status_var.set("✓ Detección detenida")
    
    def detection_loop(self):
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                processed_frame, hands_data = self.hand_detector.detect_hands(frame)
                
                detected_result = None
                control_result = None
                
                # ===== DETECTAR GESTOS DE CONTROL PRIMERO =====
                if hands_data['landmarks_list']:
                    for landmarks in hands_data['landmarks_list']:
                        # Detectar gesto de control
                        control_gesture = self.gesture_classifier.detect_control_gesture(landmarks)
                        
                        if control_gesture:
                            # Procesar el control
                            control_result = self.gesture_controls.process_control(
                                control_gesture, 
                                both_hands_data=hands_data
                            )
                            
                            if control_result:
                                self.execute_control_gesture(control_result)
                                break
                
                # ===== DETECTAR LETRAS/SÍLABAS (solo si no hay control activo) =====
                if not control_result:
                    if self.detection_mode == "syllables":
                        if hands_data['left'] and hands_data['right']:
                            detected_result = self.syllable_classifier.predict_syllable(
                                hands_data['left'], 
                                hands_data['right']
                            )
                    elif self.detection_mode == "letters":
                        if hands_data['landmarks_list']:
                            for landmarks in hands_data['landmarks_list']:
                                letter = self.gesture_classifier.predict_gesture(landmarks)
                                if letter:
                                    detected_result = letter
                                    confidence = hands_data.get('confidence', {}).get('left', 0) or \
                                               hands_data.get('confidence', {}).get('right', 0)
                                    self.gesture_calibrator.collect_sample(letter, landmarks, confidence)
                                    break
                
                self.update_ui(processed_frame, detected_result, hands_data, control_result)
                
            except Exception as e:
                print(f"Error en detección: {e}")
                continue
    
    def update_ui(self, frame, detected_result, hands_data, control_result=None):
        try:
            # Obtener tamaño actual del label
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            if label_width <= 1:
                label_width = 640
                label_height = 480
            
            # Calcular proporción del frame
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            if label_width / label_height > aspect_ratio:
                new_height = label_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = label_width
                new_height = int(new_width / aspect_ratio)
            
            # ===== DIBUJAR INDICADOR DE CONTROL SI HAY UNO ACTIVO =====
            if control_result and self.show_control_feedback:
                # Dibujar overlay con el control detectado
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (300, 80), (0, 100, 255), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                control_name = self.gesture_controls.get_control_name(control_result)
                cv2.putText(frame, control_name, (20, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            frame_resized = cv2.resize(frame, (new_width, new_height), 
                                      interpolation=cv2.INTER_AREA)
            
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk
            
            # Solo procesar auto-add si NO hay control activo
            if self.auto_add_enabled and detected_result and not control_result:
                self.handle_auto_add_logic(detected_result)
            
            if self.detection_mode == "syllables":
                self.update_syllable_display(detected_result, hands_data)
            else:
                self.update_letter_display(detected_result)
            
        except Exception as e:
            print(f"Error actualizando UI: {e}")
    
    def update_letter_display(self, detected_letter):
        if detected_letter and detected_letter != self.detected_letter:
            self.detected_letter = detected_letter
            self.letter_var.set(detected_letter)
            self.detection_count += 1
            self.detection_counter_var.set(f"Detecciones: {self.detection_count}")
        elif not detected_letter:
            self.letter_var.set("-")
        
        confidence = self.gesture_classifier.get_detection_confidence()
        self.confidence_var.set(confidence * 100)
        self.confidence_label.config(text=f"{confidence*100:.1f}%")
        self.update_confidence_bar(confidence * 100)
    
    def update_syllable_display(self, detected_syllable, hands_data):
        if detected_syllable and detected_syllable != self.detected_syllable:
            self.detected_syllable = detected_syllable
            self.letter_var.set(detected_syllable)
            self.detection_count += 1
            self.detection_counter_var.set(f"Sílabas: {self.detection_count}")
        elif not detected_syllable:
            left_status = "✓" if hands_data['left'] else "✗"
            right_status = "✓" if hands_data['right'] else "✗"
            self.letter_var.set(f"L:{left_status} R:{right_status}")
        
        confidence = self.syllable_classifier.get_detection_confidence()
        self.confidence_var.set(confidence * 100)
        self.confidence_label.config(text=f"{confidence*100:.1f}%")
        self.update_confidence_bar(confidence * 100)
    
    def handle_auto_add_logic(self, detected_letter):
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            return
        
        if detected_letter and detected_letter != "-":
            self.no_detection_count = 0
            
            if detected_letter == self.last_stable_letter:
                self.stable_letter_count += 1
            else:
                self.last_stable_letter = detected_letter
                self.stable_letter_count = 1
            
            if (self.stable_letter_count >= self.auto_add_threshold and 
                detected_letter != self.last_added_letter):
                
                self.auto_add_letter(detected_letter)
                self.last_added_letter = detected_letter
                self.stable_letter_count = 0
                self.cooldown_count = self.cooldown_threshold
                
        else:
            self.stable_letter_count = 0
            
            if self.auto_space_enabled:
                self.no_detection_count += 1
                
                if self.no_detection_count >= self.auto_space_threshold:
                    self.auto_add_space()
                    self.no_detection_count = 0

    # 4. AGREGAR NUEVO MÉTODO execute_control_gesture:
    def execute_control_gesture(self, control: str):
        """
        Ejecuta un gesto de control
        """
        if control == "DELETE":
            self.delete_last_letter()
            self.show_control_feedback_message("⌫ Letra borrada")
            
        elif control == "SPACE":
            self.add_space()
            self.show_control_feedback_message("␣ Espacio agregado")
            
        elif control == "CLEAR":
            self.clear_text()
            self.show_control_feedback_message("🗑️ Texto limpiado")
            
       # elif control == "PAUSE":
        #    self.toggle_pause_detection()
         #   status = "pausada" if not self.is_running else "reanudada"
          #  self.show_control_feedback_message(f"⏸️ Detección {status}")
    
    def delete_last_letter(self):
        """Borra la última letra del texto"""
        current_text = self.word_text.get(1.0, tk.END)
        if len(current_text) > 1:  # Más que solo el salto de línea
            self.word_text.delete("end-2c", "end-1c")
            self.word_text.see(tk.END)
    
    def toggle_pause_detection(self):
        """Pausa/reanuda la detección temporalmente"""
        if self.is_running:
            self.stop_detection()
        else:
            self.start_detection()


    def update_suggestions(self):
        """Actualiza las sugerencias basadas en el texto actual"""
        current_text = self.word_text.get(1.0, tk.END).strip()
        
        if current_text:
            # Obtener sugerencias
            self.current_suggestions = self.word_suggester.update_current_word(current_text)
            
            # Actualizar botones de sugerencias si existen
            if hasattr(self, 'suggestion_buttons'):
                self.update_suggestion_buttons()
    
    def apply_suggestion(self, suggestion: str):
        """Aplica una sugerencia seleccionada"""
        current_text = self.word_text.get(1.0, tk.END).strip()
        
        # Completar palabra actual
        new_text = self.word_suggester.complete_word(suggestion, current_text)
        
        # Actualizar texto
        self.word_text.delete(1.0, tk.END)
        self.word_text.insert(1.0, new_text)
        self.word_text.see(tk.END)
        
        # Limpiar sugerencias
        self.current_suggestions = []
        if hasattr(self, 'suggestion_buttons'):
            self.update_suggestion_buttons()
        
        self.status_var.set(f"✓ Palabra completada: {suggestion}")
    
    def apply_quick_word(self, word: str):
        """Aplica una palabra rápida"""
        current_text = self.word_text.get(1.0, tk.END).strip()
        
        if current_text and not current_text.endswith(' '):
            new_text = current_text + ' ' + word + ' '
        else:
            new_text = current_text + word + ' '
        
        self.word_text.delete(1.0, tk.END)
        self.word_text.insert(1.0, new_text)
        self.word_text.see(tk.END)
        
        self.status_var.set(f"✓ Palabra agregada: {word}")
    
    def apply_phrase(self, phrase: str):
        """Aplica una frase completa"""
        current_text = self.word_text.get(1.0, tk.END).strip()
        new_text = self.word_suggester.add_phrase(phrase, current_text)
        
        self.word_text.delete(1.0, tk.END)
        self.word_text.insert(1.0, new_text)
        self.word_text.see(tk.END)
        
        self.status_var.set(f"✓ Frase agregada: {phrase}")
    
    def update_suggestion_buttons(self):
        """Actualiza los botones de sugerencias"""
        # Limpiar botones existentes
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()
        
        # Crear nuevos botones con sugerencias
        if self.current_suggestions:
            tk.Label(
                self.suggestions_frame,
                text="💡 Sugerencias:",
                bg='white',
                fg=self.COLORS['text_dark'],
                font=('Segoe UI', 9, 'bold')
            ).pack(side=tk.LEFT, padx=5)
            
            for suggestion in self.current_suggestions[:3]:  # Máximo 3
                btn = tk.Button(
                    self.suggestions_frame,
                    text=suggestion,
                    command=lambda s=suggestion: self.apply_suggestion(s),
                    bg=self.COLORS['secondary'],
                    fg='white',
                    font=('Segoe UI', 9),
                    relief='flat',
                    padx=10,
                    pady=3,
                    cursor='hand2'
                )
                btn.pack(side=tk.LEFT, padx=2)
                btn.bind('<Enter>', lambda e, b=btn: b.config(bg=self.COLORS['hover']))
                btn.bind('<Leave>', lambda e, b=btn: b.config(bg=self.COLORS['secondary']))

    
    def show_control_feedback_message(self, message: str):
        """Muestra mensaje de feedback visual del control ejecutado"""
        self.control_feedback_text = message
        self.show_control_feedback = True
        
        # Actualizar status
        self.status_var.set(message)
        
        # Ocultar mensaje después de 2 segundos
        self.root.after(2000, self.hide_control_feedback)
    
    def hide_control_feedback(self):
        """Oculta el mensaje de feedback"""
        self.show_control_feedback = False
        if self.is_running:
            self.status_var.set("🔴 Detectando gestos...")
    
    def auto_add_space(self):
        current_text = self.word_text.get(1.0, tk.END)
        if current_text and current_text[-2] != " ":
            self.word_text.insert(tk.END, " ")
            self.word_text.see(tk.END)
            self.status_var.set("✓ Auto-espacio agregado")
    
    def auto_add_letter(self, letter):
        """Auto-agrega letra y actualiza sugerencias"""
        if letter and letter != "-":
            self.word_text.insert(tk.END, letter)
            self.word_text.see(tk.END)
            self.status_var.set(f"✓ Auto-agregado: {letter}")
            
            # Actualizar sugerencias
            if self.suggestions_enabled:
                self.update_suggestions()
            
            self.root.after(1000, lambda: self.status_var.set("🔴 Detectando gestos...") 
                           if self.is_running else None)
    
    def add_letter_to_word(self):
        """Agrega letra y actualiza sugerencias"""
        if self.detected_letter and self.detected_letter != "-":
            self.word_text.insert(tk.END, self.detected_letter)
            self.word_text.see(tk.END)
            
            # Actualizar sugerencias
            if self.suggestions_enabled:
                self.update_suggestions()
    
    def add_space(self):
        self.word_text.insert(tk.END, " ")
        self.word_text.see(tk.END)
    
    def clear_text(self):
        self.word_text.delete(1.0, tk.END)
        self.letter_var.set("-")
        self.detected_letter = ""
    
    def speak_text(self):
        text = self.word_text.get(1.0, tk.END).strip()
        if text:
            self.audio_manager.speak(text)
            self.status_var.set(f"🔊 Reproduciendo: {text}")
        else:
            messagebox.showinfo("Información", "No hay texto para reproducir")
    
    def on_closing(self):
        self.stop_detection()
        if self.cap:
            self.cap.release()
        self.audio_manager.stop()
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()