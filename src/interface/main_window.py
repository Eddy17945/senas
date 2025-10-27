# src/interface/main_window.py
# VERSIÓN COMPLETA: Palabras Completas + Sílabas + Letras

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
from ..detector.complete_word_detector import CompleteWordDetector
from ..utils.word_dictionary import WordDictionary
from ..utils.word_suggester import WordSuggester
from ..utils.sentence_bank import SentenceBank
from ..utils.word_sentence_manager import WordSentenceManager
from ..utils.audio_manager import AudioManager
from ..config.settings import Config

class MainWindow:
    # PALETA DE COLORES MODERNA
    COLORS = {
        'primary': '#2563EB',
        'secondary': '#3B82F6',
        'accent': '#10B981',
        'danger': '#EF4444',
        'warning': '#F59E0B',
        'bg_dark': '#1E293B',
        'bg_light': '#F8FAFC',
        'text_dark': '#0F172A',
        'text_light': '#F8FAFC',
        'border': '#CBD5E1',
        'hover': '#1E40AF',
    }
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤟 " + Config.WINDOW_TITLE)
        
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)
        self.root.resizable(True, True)
        
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
        self.complete_word_detector = CompleteWordDetector()
        self.word_dictionary = WordDictionary()
        self.word_suggester = WordSuggester()
        self.sentence_bank = SentenceBank()
        self.word_sentence_manager = WordSentenceManager()
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
        
        # Variables para palabras completas
        self.complete_word_mode_enabled = True
        self.last_complete_word = ""
        
        # Configurar interfaz
        self.setup_ui()
        self.setup_camera()
        
        # Protocolo de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Configura los estilos personalizados de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        
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
        
        style.configure(
            'TRadiobutton',
            background='white',
            foreground=self.COLORS['text_dark'],
            font=('Segoe UI', 9)
        )
        
        style.configure(
            'TCheckbutton',
            background='white',
            foreground=self.COLORS['text_dark'],
            font=('Segoe UI', 9)
        )
    
    def setup_ui(self):
        """Configura la interfaz de usuario moderna"""
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg_light'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # HEADER
        header_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(
            header_frame,
            text="🤟 Traductor de Lenguaje de Señas",
            font=('Segoe UI', 20, 'bold'),
            bg='white',
            fg=self.COLORS['primary']
        )
        title_label.pack(pady=15)
        
        control_frame = tk.Frame(header_frame, bg='white')
        control_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
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
        
        # Configuración
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
        
        self.complete_word_var = tk.BooleanVar(value=True)
        complete_word_check = tk.Checkbutton(
            config_frame,
            text="⚡ Palabras completas",
            variable=self.complete_word_var,
            command=self.toggle_complete_word_mode,
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 9, 'bold'),
            selectcolor=self.COLORS['primary']
        )
        complete_word_check.pack(anchor=tk.W, padx=10, pady=2)
        
        # CONTENIDO PRINCIPAL
        content_frame = tk.Frame(main_frame, bg=self.COLORS['bg_light'])
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame izquierdo - VIDEO
        left_frame = tk.Frame(content_frame, bg='white', relief='raised', bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        video_header = tk.Frame(left_frame, bg=self.COLORS['primary'], height=40)
        video_header.pack(fill=tk.X)
        video_header.pack_propagate(False)
        
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
            font=('Segoe UI', 12)
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controles de ajuste
        controls_frame = tk.Frame(left_frame, bg='white')
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
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
        right_frame = tk.Frame(content_frame, bg='white', width=420, relief='raised', bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
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
        detection_frame.pack(fill=tk.X, padx=15, pady=8)
        
        tk.Label(
            detection_frame,
            text="Letra Detectada",
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 9, 'bold')
        ).pack(pady=2)
        
        self.letter_var = tk.StringVar(value="-")
        letter_display = tk.Label(
            detection_frame,
            textvariable=self.letter_var,
            bg='white',
            fg=self.COLORS['primary'],
            font=('Arial', 32, 'bold'),
            relief='flat',
            width=3,
            height=1
        )
        letter_display.pack(pady=3)
        
        # Barra de confianza
        confidence_frame = tk.Frame(detection_frame, bg=self.COLORS['bg_light'])
        confidence_frame.pack(fill=tk.X, padx=10, pady=2)
        
        self.confidence_var = tk.DoubleVar()
        
        self.confidence_canvas = tk.Canvas(
            confidence_frame,
            height=15,
            bg='white',
            highlightthickness=1,
            highlightbackground=self.COLORS['border']
        )
        self.confidence_canvas.pack(fill=tk.X, pady=2)
        
        self.confidence_label = tk.Label(
            confidence_frame,
            text="0%",
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 8, 'bold')
        )
        self.confidence_label.pack()
        
        # Área de texto
        text_frame = tk.Frame(right_frame, bg='white')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=3)
        
        # PALABRA ACTUAL
        current_word_section = tk.Frame(text_frame, bg='white')
        current_word_section.pack(fill=tk.X, pady=(0, 3))
        
        tk.Label(
            current_word_section,
            text="✍️ Palabra Actual",
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 8, 'bold')
        ).pack(anchor=tk.W, pady=1)
        
        current_word_container = tk.Frame(current_word_section, bg=self.COLORS['secondary'], relief='flat', bd=2)
        current_word_container.pack(fill=tk.X)
        
        self.current_word_text = tk.Text(
            current_word_container,
            height=1,
            font=('Consolas', 12, 'bold'),
            wrap=tk.WORD,
            bg='white',
            fg=self.COLORS['primary'],
            relief='flat',
            padx=8,
            pady=3
        )
        self.current_word_text.pack(fill=tk.X, padx=2, pady=2)
        
        # ORACIÓN COMPLETA
        sentence_section = tk.Frame(text_frame, bg='white')
        sentence_section.pack(fill=tk.X, pady=(3, 0))
        
        sentence_header = tk.Frame(sentence_section, bg='white')
        sentence_header.pack(fill=tk.X)
        
        tk.Label(
            sentence_header,
            text="📝 Oración Completa",
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 8, 'bold')
        ).pack(side=tk.LEFT, pady=1)
        
        self.word_count_label = tk.Label(
            sentence_header,
            text="0 palabras",
            bg='white',
            fg=self.COLORS['border'],
            font=('Segoe UI', 7, 'italic')
        )
        self.word_count_label.pack(side=tk.RIGHT, pady=1)
        
        sentence_container = tk.Frame(sentence_section, bg=self.COLORS['border'], relief='flat', bd=1)
        sentence_container.pack(fill=tk.X)
        
        self.sentence_text = tk.Text(
            sentence_container,
            height=2,
            font=('Consolas', 10),
            wrap=tk.WORD,
            bg='white',
            fg=self.COLORS['text_dark'],
            relief='flat',
            padx=6,
            pady=3
        )
        self.sentence_text.pack(fill=tk.X, padx=2, pady=2)
        
        self.word_text = self.sentence_text
        
        self.current_word_text.bind('<KeyRelease>', self.on_text_change)
        self.sentence_text.bind('<KeyRelease>', self.on_text_change)
        
        # Botones de texto
        text_buttons_frame = tk.Frame(text_frame, bg='white')
        text_buttons_frame.pack(fill=tk.X, pady=3)
        
        add_letter_btn = tk.Button(
            text_buttons_frame,
            text="+ Letra",
            command=self.add_letter_to_word,
            bg=self.COLORS['secondary'],
            fg='white',
            font=('Segoe UI', 7),
            relief='flat',
            padx=6,
            pady=3,
            cursor='hand2'
        )
        add_letter_btn.pack(side=tk.LEFT, padx=2)
        
        space_btn = tk.Button(
            text_buttons_frame,
            text="⎵ Espacio",
            command=self.confirm_word,
            bg=self.COLORS['accent'],
            fg='white',
            font=('Segoe UI', 7, 'bold'),
            relief='flat',
            padx=8,
            pady=3,
            cursor='hand2'
        )
        space_btn.pack(side=tk.LEFT, padx=2)
        
        undo_btn = tk.Button(
            text_buttons_frame,
            text="↶ Deshacer",
            command=self.undo_last_word,
            bg=self.COLORS['warning'],
            fg='white',
            font=('Segoe UI', 7),
            relief='flat',
            padx=6,
            pady=3,
            cursor='hand2'
        )
        undo_btn.pack(side=tk.RIGHT, padx=2)

        # SUGERENCIAS
        suggestions_container = tk.Frame(text_frame, bg='white')
        suggestions_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        tk.Label(
            suggestions_container,
            text="💡 Sugerencias:",
            bg='white',
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 8, 'bold')
        ).pack(anchor=tk.W)
        
        self.suggestions_frame = tk.Frame(suggestions_container, bg='white', height=30)
        self.suggestions_frame.pack(fill=tk.X, pady=2)
        self.suggestions_frame.pack_propagate(False)
        
        tk.Label(
            self.suggestions_frame,
            text="Escribe para sugerencias...",
            bg='white',
            fg=self.COLORS['border'],
            font=('Segoe UI', 7, 'italic')
        ).pack()
        
        tk.Frame(suggestions_container, bg=self.COLORS['border'], height=1).pack(fill=tk.X, pady=5)
        
        phrases_btn = tk.Button(
            suggestions_container,
            text="📝 Ver Frases Completas",
            command=self.show_phrases_window,
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 8, 'bold'),
            relief='flat',
            padx=8,
            pady=5,
            cursor='hand2'
        )
        phrases_btn.pack(fill=tk.X)
        phrases_btn.bind('<Enter>', lambda e: phrases_btn.config(bg=self.COLORS['hover']))
        phrases_btn.bind('<Leave>', lambda e: phrases_btn.config(bg=self.COLORS['primary']))
        
        # BARRA DE ESTADO
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
        
        tools_buttons_frame = tk.Frame(status_bar, bg=self.COLORS['bg_dark'])
        tools_buttons_frame.pack(side=tk.RIGHT, padx=10)
        
        calibrate_btn = tk.Button(
            tools_buttons_frame,
            text="🎯 Calibrar",
            command=self.show_precision_manager,
            bg=self.COLORS['accent'],
            fg='white',
            font=('Segoe UI', 9, 'bold'),
            relief='flat',
            padx=15,
            pady=5,
            cursor='hand2'
        )
        calibrate_btn.pack(side=tk.LEFT, padx=3)
        calibrate_btn.bind('<Enter>', lambda e: calibrate_btn.config(bg='#059669'))
        calibrate_btn.bind('<Leave>', lambda e: calibrate_btn.config(bg=self.COLORS['accent']))
        
        gallery_btn = tk.Button(
            tools_buttons_frame,
            text="🖼 Referencias",
            command=self.show_reference_gallery,
            bg=self.COLORS['warning'],
            fg='white',
            font=('Segoe UI', 9),
            relief='flat',
            padx=12,
            pady=5,
            cursor='hand2'
        )
        gallery_btn.pack(side=tk.LEFT, padx=3)
        gallery_btn.bind('<Enter>', lambda e: gallery_btn.config(bg='#D97706'))
        gallery_btn.bind('<Leave>', lambda e: gallery_btn.config(bg=self.COLORS['warning']))
        
        letters_btn = tk.Button(
            tools_buttons_frame,
            text="📚 Letras",
            command=self.show_supported_letters,
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 9),
            relief='flat',
            padx=12,
            pady=5,
            cursor='hand2'
        )
        letters_btn.pack(side=tk.LEFT, padx=3)
        letters_btn.bind('<Enter>', lambda e: letters_btn.config(bg=self.COLORS['hover']))
        letters_btn.bind('<Leave>', lambda e: letters_btn.config(bg=self.COLORS['primary']))

    def show_phrases_window(self):
        """Muestra ventana del banco de oraciones"""
        phrases_window = tk.Toplevel(self.root)
        phrases_window.title("📚 Banco de Oraciones")
        phrases_window.geometry("800x700")
        phrases_window.configure(bg='white')
        
        header = tk.Frame(phrases_window, bg=self.COLORS['primary'], height=70)
        header.pack(fill=tk.X)
        
        tk.Label(
            header,
            text="📚 Banco de Oraciones Completas",
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 18, 'bold')
        ).pack(pady=10)
        
        tk.Label(
            header,
            text="Selecciona una categoría y haz clic en la oración para agregarla",
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 10)
        ).pack()
        
        main_container = tk.Frame(phrases_window, bg='white')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        style = ttk.Style()
        style.configure('TNotebook', background='white')
        style.configure('TNotebook.Tab', padding=[20, 10], font=('Segoe UI', 10))
        
        categories = self.sentence_bank.get_categories()
        
        for category_key in categories:
            category_info = self.sentence_bank.get_category_info(category_key)
            sentences = self.sentence_bank.get_sentences(category_key)
            
            tab_frame = tk.Frame(notebook, bg='white')
            notebook.add(tab_frame, text=category_info['name'])
            
            canvas = tk.Canvas(tab_frame, bg='white', highlightthickness=0)
            scrollbar = tk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg='white')
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e, c=canvas: c.configure(scrollregion=c.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=750)
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
            scrollbar.pack(side="right", fill="y")
            
            for sentence in sentences:
                sentence_btn = tk.Button(
                    scrollable_frame,
                    text=sentence,
                    command=lambda s=sentence, w=phrases_window: self.select_sentence_from_bank(s, w),
                    bg=self.COLORS['bg_light'],
                    fg=self.COLORS['text_dark'],
                    font=('Segoe UI', 11),
                    relief='raised',
                    bd=2,
                    padx=20,
                    pady=12,
                    cursor='hand2',
                    anchor='w',
                    justify='left'
                )
                sentence_btn.pack(fill=tk.X, pady=3, padx=10)
                
                sentence_btn.bind(
                    '<Enter>',
                    lambda e, b=sentence_btn: b.config(
                        bg=self.COLORS['secondary'],
                        fg='white',
                        font=('Segoe UI', 11, 'bold')
                    )
                )
                sentence_btn.bind(
                    '<Leave>',
                    lambda e, b=sentence_btn: b.config(
                        bg=self.COLORS['bg_light'],
                        fg=self.COLORS['text_dark'],
                        font=('Segoe UI', 11)
                    )
                )
        
        footer = tk.Frame(phrases_window, bg=self.COLORS['bg_light'], height=60)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        
        stats = self.sentence_bank.get_statistics()
        
        stats_text = f"📊 {stats['total_sentences']} oraciones en {stats['total_categories']} categorías"
        tk.Label(
            footer,
            text=stats_text,
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text_dark'],
            font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=20, pady=15)
        
        close_btn = tk.Button(
            footer,
            text="Cerrar",
            command=phrases_window.destroy,
            bg=self.COLORS['danger'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            padx=30,
            pady=10,
            cursor='hand2'
        )
        close_btn.pack(side=tk.RIGHT, padx=20, pady=10)
        close_btn.bind('<Enter>', lambda e: close_btn.config(bg='#DC2626'))
        close_btn.bind('<Leave>', lambda e: close_btn.config(bg=self.COLORS['danger']))
    
    def select_sentence_from_bank(self, sentence: str, window):
        """Selecciona una oración del banco y cierra la ventana"""
        self.apply_phrase(sentence)
        
        if hasattr(self, 'sentence_bank'):
            self.sentence_bank.register_usage(sentence)
        
        window.destroy()
        self.status_var.set(f"✓ Oración agregada: {sentence}")
    
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
            self.confidence_canvas.create_rectangle(
                0, 0, width, height,
                fill='#E2E8F0',
                outline=''
            )
            
            bar_width = int(width * (confidence / 100))
            
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
    
    def toggle_complete_word_mode(self):
        """Activa/desactiva el modo de palabras completas"""
        self.complete_word_mode_enabled = self.complete_word_var.get()
        status = "activado" if self.complete_word_mode_enabled else "desactivado"
        self.status_var.set(f"⚡ Modo palabras completas {status}")
        
        if self.complete_word_mode_enabled:
            self.show_word_gestures_info()
    
    def process_complete_word(self, word: str):
        """Procesa una palabra completa detectada"""
        if not word or word == self.last_complete_word:
            return
        
        try:
            print(f"[DEBUG] Palabra completa detectada: {word}")
            
            if self.word_sentence_manager.add_word_by_gesture(word):
                self.update_text_display()
                self.last_complete_word = word
                self.status_var.set(f"⚡ PALABRA: {word}")
                
                self.root.after(2000, lambda: setattr(self, 'last_complete_word', ''))
                self.root.after(
                    3000, 
                    lambda: self.status_var.set("🔴 Detectando gestos...") 
                           if self.is_running else None
                )
                
                print(f"[DEBUG] Palabra '{word}' agregada exitosamente")
            else:
                print(f"[DEBUG] Error al agregar palabra '{word}'")
                
        except Exception as e:
            print(f"[ERROR] Error procesando palabra completa: {e}")
            import traceback
            traceback.print_exc()
    
    def draw_word_overlay(self, frame, word: str):
        """Dibuja overlay verde cuando detecta palabra completa"""
        try:
            height, width = frame.shape[:2]
            overlay = frame.copy()
            
            cv2.rectangle(
                overlay, 
                (10, 10), 
                (width - 10, 120), 
                (0, 200, 100),
                -1
            )
            
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.rectangle(
                frame,
                (10, 10),
                (width - 10, 120),
                (0, 255, 0),
                3
            )
            
            cv2.putText(
                frame, 
                "PALABRA:", 
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8,
                (255, 255, 255),
                2
            )
            
            cv2.putText(
                frame, 
                word, 
                (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5,
                (255, 255, 255),
                3
            )
            
            cv2.putText(
                frame, 
                "OK", 
                (width - 80, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 
                2.0,
                (0, 255, 0),
                4
            )
            
        except Exception as e:
            print(f"Error dibujando overlay: {e}")
    
    def show_word_gestures_info(self):
        """Muestra ventana con TODAS las palabras completas disponibles"""
        info_window = tk.Toplevel(self.root)
        info_window.title("⚡ Palabras Completas por Gesto")
        info_window.geometry("900x700")
        info_window.configure(bg='white')
        
        # Header
        header = tk.Frame(info_window, bg=self.COLORS['primary'], height=70)
        header.pack(fill=tk.X)
        
        tk.Label(
            header,
            text="⚡ Gestos para Palabras Completas",
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 18, 'bold')
        ).pack(pady=10)
        
        tk.Label(
            header,
            text=f"Total: {len(self.complete_word_detector.get_available_word_gestures())} palabras disponibles",
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 11)
        ).pack()
        
        # Contenido con scroll
        main_container = tk.Frame(info_window, bg='white')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Canvas con scrollbar
        canvas = tk.Canvas(main_container, bg='white', highlightthickness=0)
        scrollbar = tk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=850)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Obtener gestos disponibles
        gestures = self.complete_word_detector.get_available_word_gestures()
        
        # Descripciones organizadas por categorías
        gesture_categories = {
            '🤝 SALUDOS Y CORTESÍA': {
                'THUMBS_UP': ('👍', 'Pulgar arriba'),
                'WAVE': ('👋', 'Mano oscilando'),
                'PEACE': ('✌️', 'Dedos en V'),
                'OK_SIGN': ('👌', 'Círculo OK'),
                'PRAY_HANDS': ('🙏', 'Manos juntas'),
                'BOW': ('🙇', 'Reverencia'),
            },
            '💬 RESPUESTAS': {
                'THUMBS_DOWN': ('👎', 'Pulgar abajo'),
                'NOD_YES': ('✅', 'Asentir'),
                'SHAKA': ('🤙', 'Shaka'),
                'FIST_UP': ('✊', 'Puño arriba'),
            },
            '🍽️ NECESIDADES': {
                'POINTING_UP': ('☝️', 'Índice arriba'),
                'DRINK_GESTURE': ('🥤', 'Beber'),
                'EAT_GESTURE': ('🍴', 'Comer'),
                'BATHROOM_SIGN': ('🚽', 'Baño'),
                'SLEEP_GESTURE': ('😴', 'Dormir'),
            },
            '👨‍👩‍👧 FAMILIA': {
                'HEART_HANDS': ('❤️', 'Corazón con manos'),
                'MAMA_SIGN': ('👩', 'Mamá'),
                'PAPA_SIGN': ('👨', 'Papá'),
                'BABY_ROCK': ('👶', 'Mecer bebé'),
                'FAMILY_SIGN': ('👨‍👩‍👧', 'Familia'),
            },
            '😊 EMOCIONES': {
                'HAPPY_SIGN': ('😊', 'Sonrisa'),
                'SAD_SIGN': ('😢', 'Triste'),
                'ANGRY_FIST': ('😠', 'Puño enojado'),
                'SCARED_HANDS': ('😨', 'Manos asustadas'),
                'LOVE_HEART': ('💕', 'Amor'),
            },
            '🎯 ACCIONES': {
                'CALL_ME': ('📞', 'Llámame'),
                'COME_HERE': ('👈', 'Ven aquí'),
                'GO_AWAY': ('👉', 'Vete'),
                'WAIT_HAND': ('✋', 'Espera'),
                'STOP_HAND': ('🛑', 'Alto'),
            },
            '📍 LUGARES': {
                'HOME_SIGN': ('🏠', 'Casa'),
                'SCHOOL_SIGN': ('🏫', 'Escuela'),
                'WORK_SIGN': ('💼', 'Trabajo'),
                'HOSPITAL_CROSS': ('🏥', 'Hospital'),
            },
            '⏰ TIEMPO': {
                'NOW_SIGN': ('⏰', 'Ahora'),
                'LATER_SIGN': ('🕐', 'Después'),
                'TODAY_SIGN': ('📅', 'Hoy'),
                'TOMORROW_POINT': ('➡️', 'Mañana'),
            },
            '💼 ÚTILES': {
                'PHONE_CALL': ('📱', 'Teléfono'),
                'WRITE_SIGN': ('✍️', 'Escribir'),
                'READ_SIGN': ('📖', 'Leer'),
                'LISTEN_EAR': ('👂', 'Escuchar'),
                'MONEY_RUB': ('💰', 'Dinero'),
                'FRIEND_LINK': ('🤝', 'Amigo'),
            },
        }
        
        # Mostrar por categorías
        for category_name, category_gestures in gesture_categories.items():
            # Header de categoría
            category_header = tk.Frame(scrollable_frame, bg=self.COLORS['primary'], height=40)
            category_header.pack(fill=tk.X, pady=(10, 5))
            
            tk.Label(
                category_header,
                text=category_name,
                bg=self.COLORS['primary'],
                fg='white',
                font=('Segoe UI', 12, 'bold')
            ).pack(pady=8)
            
            # Gestos de la categoría
            for gesture_type, (emoji, description) in  category_gestures.items():
                word = gestures.get(gesture_type)
                
                if word:  # Solo mostrar si el gesto está disponible
                    gesture_frame = tk.Frame(scrollable_frame, bg=self.COLORS['bg_light'], 
                                            relief='raised', bd=1)
                    gesture_frame.pack(fill=tk.X, pady=2, padx=10)
                    
                    tk.Label(
                        gesture_frame,
                        text=f"{emoji}  {description}",
                        bg=self.COLORS['bg_light'],
                        fg=self.COLORS['text_dark'],
                        font=('Segoe UI', 10),
                        width=25,
                        anchor='w'
                    ).pack(side=tk.LEFT, padx=10, pady=6)
                    
                    tk.Label(
                        gesture_frame,
                        text="→",
                        bg=self.COLORS['bg_light'],
                        fg=self.COLORS['primary'],
                        font=('Segoe UI', 12, 'bold')
                    ).pack(side=tk.LEFT, padx=5)
                    
                    tk.Label(
                        gesture_frame,
                        text=word,
                        bg=self.COLORS['bg_light'],
                        fg=self.COLORS['primary'],
                        font=('Segoe UI', 11, 'bold')
                    ).pack(side=tk.LEFT, padx=10)
        
        # Footer con botón cerrar
        footer = tk.Frame(info_window, bg='white', height=60)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        
        tk.Button(
            footer,
            text="Entendido",
            command=info_window.destroy,
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            padx=30,
            pady=10,
            cursor='hand2'
        ).pack(pady=15)
    
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
        
        header = tk.Frame(letters_window, bg=self.COLORS['primary'], height=60)
        header.pack(fill=tk.X)
        
        tk.Label(
            header,
            text="📚 Alfabeto de Lenguaje de Señas",
            bg=self.COLORS['primary'],
            fg='white',
            font=('Segoe UI', 18, 'bold')
        ).pack(pady=15)
        
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
        """BUCLE PRINCIPAL - Palabras Completas + Sílabas + Letras"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                processed_frame, hands_data = self.hand_detector.detect_hands(frame)
                
                detected_result = None
                control_result = None
                complete_word_result = None
                
                # PRIORIDAD 1: PALABRAS COMPLETAS
                if (self.complete_word_mode_enabled and 
                    hands_data['landmarks_list'] and 
                    len(hands_data['landmarks_list']) > 0):
                    
                    landmarks = hands_data['landmarks_list'][0]
                    confidence = hands_data.get('confidence', {}).get('left', 0) or \
                               hands_data.get('confidence', {}).get('right', 0)
                    
                    complete_word_result = self.complete_word_detector.detect_complete_word(
                        landmarks, 
                        confidence
                    )
                    
                    if complete_word_result:
                        self.process_complete_word(complete_word_result)
                        self.draw_word_overlay(processed_frame, complete_word_result)
                        self.update_ui(processed_frame, None, hands_data, None)
                        continue
                
                # PRIORIDAD 2: GESTOS DE CONTROL
                if hands_data['landmarks_list']:
                    for landmarks in hands_data['landmarks_list']:
                        control_gesture = self.gesture_classifier.detect_control_gesture(landmarks)
                        
                        if control_gesture:
                            control_result = self.gesture_controls.process_control(
                                control_gesture, 
                                both_hands_data=hands_data
                            )
                            
                            if control_result:
                                self.execute_control_gesture(control_result)
                                break
                
                # PRIORIDAD 3: LETRAS/SÍLABAS
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
                import traceback
                traceback.print_exc()
                continue
    
    def update_ui(self, frame, detected_result, hands_data, control_result=None):
        try:
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            if label_width <= 1:
                label_width = 640
                label_height = 480
            
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            if label_width / label_height > aspect_ratio:
                new_height = label_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = label_width
                new_height = int(new_width / aspect_ratio)
            
            if control_result and self.show_control_feedback:
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
        """Actualiza el display cuando se detecta una sílaba - CORREGIDO"""
        if detected_syllable and detected_syllable != self.detected_syllable:
            # IMPORTANTE: Actualizar la variable de instancia
            self.detected_syllable = detected_syllable
            
            # Actualizar el label con la sílaba
            self.letter_var.set(detected_syllable)
            
            # Incrementar contador
            self.detection_count += 1
            self.detection_counter_var.set(f"Sílabas: {self.detection_count}")
            
            print(f"[DEBUG] Sílaba detectada y mostrada: {detected_syllable}")
            
        elif not detected_syllable:
            # Si no hay detección, mostrar estado de las manos
            left_status = "✓" if hands_data.get('left') else "✗"
            right_status = "✓" if hands_data.get('right') else "✗"
            self.letter_var.set(f"L:{left_status} R:{right_status}")
            
            # Limpiar la variable si no hay detección
            self.detected_syllable = ""
        
        # Actualizar barra de confianza
        confidence = self.syllable_classifier.get_detection_confidence()
        self.confidence_var.set(confidence * 100)
        self.confidence_label.config(text=f"{confidence*100:.1f}%")
        self.update_confidence_bar(confidence * 100)
    
    def handle_auto_add_logic(self, detected_result):
        """Maneja auto-agregado para LETRAS Y SÍLABAS - CORREGIDO"""
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            return
        
        if detected_result and detected_result != "-":
            self.no_detection_count = 0
            
            if detected_result == self.last_stable_letter:
                self.stable_letter_count += 1
            else:
                self.last_stable_letter = detected_result
                self.stable_letter_count = 1
            
            # Ajustar umbral según si es letra o sílaba
            threshold = self.auto_add_threshold
            if len(detected_result) > 1:  # Es una sílaba
                threshold = self.auto_add_threshold + 5
            
            if (self.stable_letter_count >= threshold and 
                detected_result != self.last_added_letter):
                
                self.auto_add_letter(detected_result)
                self.last_added_letter = detected_result
                self.stable_letter_count = 0
                self.cooldown_count = self.cooldown_threshold
                
        else:
            self.stable_letter_count = 0
            
            if self.auto_space_enabled:
                self.no_detection_count += 1
                
                if self.no_detection_count >= self.auto_space_threshold:
                    self.auto_add_space()
                    self.no_detection_count = 0

    def execute_control_gesture(self, control: str):
        """Ejecuta un gesto de control"""
        if control == "DELETE":
            self.delete_last_letter()
            self.show_control_feedback_message("⌫ Letra borrada")
            
        elif control == "SPACE":
            self.add_space()
            self.show_control_feedback_message("␣ Espacio agregado")
            
        elif control == "CLEAR":
            self.clear_text()
            self.show_control_feedback_message("🗑️ Texto limpiado")
    
    def delete_last_letter(self):
        """Borra la última letra usando el gestor"""
        if self.word_sentence_manager.delete_last_letter():
            self.update_text_display()
            self.status_var.set("⌫ Letra borrada")
    
    def toggle_pause_detection(self):
        """Pausa/reanuda la detección temporalmente"""
        if self.is_running:
            self.stop_detection()
        else:
            self.start_detection()

    def on_text_change(self, event=None):
        """Callback cuando cambia el texto"""
        if self.suggestions_enabled:
            if hasattr(self, '_suggestion_timer'):
                self.root.after_cancel(self._suggestion_timer)
            
            self._suggestion_timer = self.root.after(500, self.update_suggestions)
    
    def on_text_modified(self, event=None):
        """Callback cuando se modifica el texto"""
        self.word_text.edit_modified(False)
    
    def update_suggestions(self):
        """Actualiza las sugerencias basadas en la PALABRA ACTUAL"""
        try:
            current_word = self.word_sentence_manager.get_current_word()
            
            if current_word and len(current_word) >= 2:
                self.current_suggestions = self.word_suggester.update_current_word(current_word)
            else:
                self.current_suggestions = []
            
            self.update_suggestion_buttons()
            
        except Exception as e:
            print(f"Error actualizando sugerencias: {e}")
    
    def apply_suggestion(self, suggestion: str):
        """Aplica una sugerencia seleccionada"""
        self.word_sentence_manager.clear_current_word()
        
        for letter in suggestion:
            self.word_sentence_manager.add_letter(letter)
        
        self.update_text_display()
        self.status_var.set(f"✓ Sugerencia aplicada: {suggestion}")
    
    def apply_quick_word(self, word: str):
        """Aplica una palabra rápida"""
        self.word_sentence_manager.clear_current_word()
        
        for letter in word:
            self.word_sentence_manager.add_letter(letter)
        
        self.word_sentence_manager.add_space()
        self.update_text_display()
        self.status_var.set(f"✓ Palabra rápida agregada: {word}")
    
    def apply_phrase(self, phrase: str):
        """Aplica una frase completa"""
        if self.word_sentence_manager.add_complete_sentence(phrase):
            self.update_text_display()
            self.status_var.set(f"✓ Frase agregada: {phrase}")
    
    def update_suggestion_buttons(self):
        """Actualiza los botones de sugerencias"""
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()
        
        if self.current_suggestions and len(self.current_suggestions) > 0:
            for i, suggestion in enumerate(self.current_suggestions[:5]):
                btn = tk.Button(
                    self.suggestions_frame,
                    text=suggestion,
                    command=lambda s=suggestion: self.apply_suggestion(s),
                    bg=self.COLORS['secondary'],
                    fg='white',
                    font=('Segoe UI', 9),
                    relief='flat',
                    padx=12,
                    pady=5,
                    cursor='hand2'
                )
                btn.pack(side=tk.LEFT, padx=3, pady=2)
                btn.bind('<Enter>', lambda e, b=btn: b.config(bg=self.COLORS['hover']))
                btn.bind('<Leave>', lambda e, b=btn: b.config(bg=self.COLORS['secondary']))
        else:
            tk.Label(
                self.suggestions_frame,
                text="Escribe para ver sugerencias...",
                bg='white',
                fg=self.COLORS['border'],
                font=('Segoe UI', 9, 'italic')
            ).pack(pady=5)
    
    def show_control_feedback_message(self, message: str):
        """Muestra mensaje de feedback"""
        self.control_feedback_text = message
        self.show_control_feedback = True
        self.status_var.set(message)
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
        """Auto-agrega letra O SÍLABA - CORREGIDO"""
        if letter and letter != "-":
            # Verificar si es una sílaba (más de 1 carácter) o una letra
            if len(letter) > 1:
                # Es una sílaba - agregar cada letra
                print(f"[DEBUG] Auto-agregando SÍLABA: {letter}")
                for char in letter:
                    if self.word_sentence_manager.add_letter(char):
                        print(f"[DEBUG] Letra '{char}' agregada de sílaba '{letter}'")
            else:
                # Es una letra individual
                print(f"[DEBUG] Auto-agregando LETRA: {letter}")
                self.word_sentence_manager.add_letter(letter)
            
            # Actualizar display
            self.update_text_display()
            self.status_var.set(f"✓ Auto-agregado: {letter}")
            
            # Actualizar sugerencias
            if self.suggestions_enabled:
                self.root.after(100, self.update_suggestions)
            
            self.root.after(1000, lambda: self.status_var.set("🔴 Detectando gestos...") 
                           if self.is_running else None)
    
    def add_letter_to_word(self):
        """Agrega letra O SÍLABA manualmente - CORREGIDO"""
        detected = None
        
        # Verificar qué modo estamos usando
        if self.detection_mode == "syllables":
            detected = self.detected_syllable
        else:
            detected = self.detected_letter
        
        if detected and detected != "-":
            # Verificar si es sílaba o letra
            if len(detected) > 1:
                # Es una sílaba - agregar cada letra
                print(f"[DEBUG] Agregando manualmente SÍLABA: {detected}")
                for char in detected:
                    self.word_sentence_manager.add_letter(char)
            else:
                # Es una letra individual
                print(f"[DEBUG] Agregando manualmente LETRA: {detected}")
                self.word_sentence_manager.add_letter(detected)
            
            # Actualizar display
            self.update_text_display()
            
            # Actualizar sugerencias
            if self.suggestions_enabled:
                self.root.after(100, self.update_suggestions)
    
    def add_space(self):
        """Agrega espacio (confirma palabra)"""
        if self.word_sentence_manager.add_space():
            self.update_text_display()
            self.current_suggestions = []
            self.update_suggestion_buttons()
    
    def update_text_display(self):
        """Actualiza la visualización de palabra actual y oración"""
        try:
            current_word = self.word_sentence_manager.get_current_word()
            self.current_word_text.delete(1.0, tk.END)
            if current_word:
                self.current_word_text.insert(1.0, current_word)
            
            sentence = self.word_sentence_manager.get_complete_sentence()
            self.sentence_text.delete(1.0, tk.END)
            if sentence:
                self.sentence_text.insert(1.0, sentence)
            
            stats = self.word_sentence_manager.get_statistics()
            word_count = stats['sentence_word_count']
            self.word_count_label.config(text=f"{word_count} palabras")
            
            if current_word and len(current_word) >= 2:
                self.current_suggestions = self.word_suggester.update_current_word(current_word)
                self.update_suggestion_buttons()
            elif not current_word:
                self.current_suggestions = []
                self.update_suggestion_buttons()
            
        except Exception as e:
            print(f"Error actualizando display: {e}")
    
    def confirm_word(self):
        """Confirma la palabra actual"""
        if self.word_sentence_manager.add_space():
            self.update_text_display()
            self.status_var.set("✓ Palabra agregada a la oración")
            self.current_suggestions = []
            self.update_suggestion_buttons()
        else:
            self.status_var.set("⚠ No hay palabra para agregar")
    
    def undo_last_word(self):
        """Deshace la última palabra agregada"""
        if self.word_sentence_manager.undo_last_word():
            self.update_text_display()
            self.status_var.set("↶ Última palabra deshecha")
        else:
            self.status_var.set("⚠ No hay nada para deshacer")
    
    def clear_text(self):
        """Limpia todo el texto"""
        self.word_sentence_manager.clear_all()
        self.update_text_display()
        self.letter_var.set("-")
        self.detected_letter = ""
        self.detected_syllable = ""
        self.current_suggestions = []
        self.update_suggestion_buttons()
        self.status_var.set("✓ Texto limpiado")
    
    def speak_text(self):
        """Reproduce el texto completo usando TTS"""
        text = self.word_sentence_manager.finalize_sentence()
        if text:
            self.audio_manager.speak(text)
            self.status_var.set(f"🔊 Reproduciendo: {text}")
            self.update_text_display()
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