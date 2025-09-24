# src/config/settings.py

class Config:
    # Configuración de la cámara
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_INDEX = 0
    
    # Configuración de detección
    MAX_NUM_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Configuración de la interfaz
    WINDOW_TITLE = "Traductor de Lenguaje de Señas"
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 600
    
    # Configuración del modelo
    MODEL_INPUT_SIZE = (64, 64)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Letras que vamos a detectar (alfabeto completo)
    SUPPORTED_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    # Configuración de audio
    VOICE_RATE = 200
    VOICE_VOLUME = 0.9
    
    # Rutas
    MODEL_PATH = "src/models/trained_models/"
    DATA_PATH = "data/"
    ASSETS_PATH = "assets/"