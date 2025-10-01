# src/config/settings_optimized.py
# CONFIGURACIÓN OPTIMIZADA PARA VELOCIDAD Y PRECISIÓN

class Config:
    # ========== CONFIGURACIÓN DE CÁMARA ==========
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_INDEX = 0
    CAMERA_FPS = 30  # NUEVO: Limitar FPS para mejor procesamiento
    
    # ========== CONFIGURACIÓN DE DETECCIÓN (OPTIMIZADA) ==========
    MAX_NUM_HANDS = 2
    
    # CRÍTICO: Valores más bajos para capturar movimientos rápidos
    MIN_DETECTION_CONFIDENCE = 0.6  # Reducido de 0.7 a 0.6
    MIN_TRACKING_CONFIDENCE = 0.5   # Mantener en 0.5
    
    # NUEVO: Parámetros de velocidad
    FAST_MODE = True  # Activar modo rápido por defecto
    SMOOTHING_ENABLED = True  # Suavizado mínimo
    SMOOTHING_FACTOR = 0.3  # Factor bajo para respuesta rápida
    
    # ========== CONFIGURACIÓN DE CLASIFICACIÓN ==========
    # CRÍTICO: Reducir para detección más rápida
    STABILITY_THRESHOLD = 3  # Reducido de 5 a 3 frames
    CONFIDENCE_THRESHOLD = 0.6  # Reducido de 0.7 a 0.6
    
    # NUEVO: Umbrales específicos para letras problemáticas
    LETTER_THRESHOLDS = {
        'Y': {'min_distance': 0.12, 'max_angle': 150},
        'U': {'max_distance': 0.05, 'alignment_tolerance': 0.04},
        'V': {'min_separation': 0.06},
        'I': {'strict_mode': True},
        'L': {'angle_tolerance': 15}
    }
    
    # ========== CONFIGURACIÓN DE INTERFAZ ==========
    WINDOW_TITLE = "Traductor LSE - Modo Rápido"
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 600
    
    # NUEVO: Configuración de UI
    UI_UPDATE_INTERVAL = 30  # ms entre actualizaciones de UI
    SHOW_FPS = True  # Mostrar FPS en pantalla
    SHOW_DEBUG_INFO = False  # Desactivar info de debug para velocidad
    
    # ========== CONFIGURACIÓN DEL MODELO ==========
    MODEL_INPUT_SIZE = (64, 64)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    MODEL_COMPLEXITY = 0  # CRÍTICO: 0 = rápido, 1 = lento pero preciso
    
    # ========== ALFABETO SOPORTADO ==========
    SUPPORTED_LETTERS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]
    
    # NUEVO: Letras con detección mejorada
    IMPROVED_LETTERS = ['Y', 'U', 'V', 'I', 'L', 'H', 'K', 'P']
    
    # ========== CONFIGURACIÓN DE AUDIO ==========
    VOICE_RATE = 200
    VOICE_VOLUME = 0.9
    
    # ========== CONFIGURACIÓN DE AUTO-AGREGADO ==========
    AUTO_ADD_ENABLED = True
    AUTO_ADD_THRESHOLD = 12  # Reducido de 15 a 12 frames
    AUTO_ADD_COOLDOWN = 25   # Reducido de 30 a 25 frames
    AUTO_SPACE_ENABLED = False
    AUTO_SPACE_THRESHOLD = 60  # Reducido de 90 a 60 frames
    
    # ========== RUTAS ==========
    MODEL_PATH = "src/models/trained_models/"
    DATA_PATH = "data/"
    ASSETS_PATH = "assets/"
    CALIBRATION_PATH = "assets/gesture_config.json"
    
    # ========== CONFIGURACIÓN DE CALIBRACIÓN ==========
    CALIBRATION_ENABLED = True
    MIN_CALIBRATION_SAMPLES = 10
    AUTO_CALIBRATION = True  # Calibración automática durante uso
    
    # ========== CONFIGURACIÓN DE RENDIMIENTO ==========
    # NUEVO: Optimizaciones de rendimiento
    USE_MULTIPROCESSING = False  # Puede causar problemas en algunos sistemas
    BUFFER_SIZE = 1  # Sin buffer para mínima latencia
    SKIP_FRAMES = 0  # No saltar frames
    
    # NUEVO: Configuración de procesamiento de imagen
    PREPROCESSING_ENABLED = True
    ADAPTIVE_BRIGHTNESS = True
    CONTRAST_ENHANCEMENT = False  # Desactivar para velocidad
    
    # ========== MODOS DE OPERACIÓN ==========
    MODES = {
        'ULTRA_FAST': {
            'model_complexity': 0,
            'stability_threshold': 2,
            'smoothing_factor': 0.2,
            'skip_frames': 1
        },
        'BALANCED': {  # Modo recomendado
            'model_complexity': 0,
            'stability_threshold': 3,
            'smoothing_factor': 0.3,
            'skip_frames': 0
        },
        'HIGH_PRECISION': {
            'model_complexity': 1,
            'stability_threshold': 5,
            'smoothing_factor': 0.5,
            'skip_frames': 0
        }
    }
    
    # Modo activo por defecto
    ACTIVE_MODE = 'BALANCED'
    
    @classmethod
    def set_mode(cls, mode_name: str):
        """Cambia el modo de operación"""
        if mode_name in cls.MODES:
            mode_config = cls.MODES[mode_name]
            cls.MODEL_COMPLEXITY = mode_config['model_complexity']
            cls.STABILITY_THRESHOLD = mode_config['stability_threshold']
            cls.SMOOTHING_FACTOR = mode_config['smoothing_factor']
            cls.SKIP_FRAMES = mode_config['skip_frames']
            cls.ACTIVE_MODE = mode_name
            return True
        return False
    
    @classmethod
    def get_current_mode(cls) -> str:
        """Obtiene el modo activo actual"""
        return cls.ACTIVE_MODE
    
    @classmethod
    def optimize_for_speed(cls):
        """Optimiza configuración para máxima velocidad"""
        cls.MODEL_COMPLEXITY = 0
        cls.MIN_DETECTION_CONFIDENCE = 0.5
        cls.STABILITY_THRESHOLD = 2
        cls.SMOOTHING_ENABLED = False
        cls.PREPROCESSING_ENABLED = False
        cls.SHOW_DEBUG_INFO = False
    
    @classmethod
    def optimize_for_precision(cls):
        """Optimiza configuración para máxima precisión"""
        cls.MODEL_COMPLEXITY = 1
        cls.MIN_DETECTION_CONFIDENCE = 0.7
        cls.STABILITY_THRESHOLD = 5
        cls.SMOOTHING_ENABLED = True
        cls.SMOOTHING_FACTOR = 0.5
        cls.PREPROCESSING_ENABLED = True