"""
API REST para Traductor de Señas
Conecta tu código Python existente con Flutter
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importaciones
sys.path.insert(0, str(Path(__file__).parent))

# Importa tus módulos existentes según tu estructura
try:
    from src.detector import HandDetector, GestureClassifier, SyllableClassifier
    from src.utils import WordSuggester, WordDictionary, SentenceBank
    MODELS_LOADED = True
    print("✅ Módulos importados correctamente")
except ImportError as e:
    print(f"⚠️ Error importando módulos: {e}")
    print("⚠️ La API funcionará en modo básico")
    MODELS_LOADED = False

app = FastAPI(
    title="Traductor de Señas API",
    description="API para detección y clasificación de lenguaje de señas",
    version="1.0.0"
)

# Configurar CORS para permitir peticiones desde Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para los modelos
hand_detector = None
gesture_classifier = None
syllable_classifier = None
word_suggester = None
word_dictionary = None
sentence_bank = None

@app.on_event("startup")
async def startup_event():
    """Inicializar modelos al arrancar el servidor"""
    global hand_detector, gesture_classifier, syllable_classifier
    global word_suggester, word_dictionary, sentence_bank
    
    if not MODELS_LOADED:
        print("⚠️ Modelos no disponibles - modo básico")
        return
    
    try:
        print("🔄 Cargando modelos ML...")
        
        # Inicializar detector de manos
        hand_detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        print("✓ HandDetector cargado")
        
        # Inicializar clasificadores
        try:
            gesture_classifier = GestureClassifier()
            print("✓ GestureClassifier cargado")
        except Exception as e:
            print(f"⚠️ GestureClassifier no disponible: {e}")
        
        try:
            syllable_classifier = SyllableClassifier()
            print("✓ SyllableClassifier cargado")
        except Exception as e:
            print(f"⚠️ SyllableClassifier no disponible: {e}")
        
        # Inicializar utilidades
        try:
            word_suggester = WordSuggester()
            print("✓ WordSuggester cargado")
        except Exception as e:
            print(f"⚠️ WordSuggester no disponible: {e}")
        
        try:
            word_dictionary = WordDictionary()
            print("✓ WordDictionary cargado")
        except Exception as e:
            print(f"⚠️ WordDictionary no disponible: {e}")
        
        try:
            sentence_bank = SentenceBank()
            print("✓ SentenceBank cargado")
        except Exception as e:
            print(f"⚠️ SentenceBank no disponible: {e}")
        
        print("✅ Sistema inicializado correctamente")
        
    except Exception as e:
        print(f"❌ Error durante la inicialización: {e}")


@app.get("/")
async def root():
    """Endpoint de verificación"""
    return {
        "status": "online",
        "message": "API Traductor de Señas funcionando",
        "version": "1.0.0",
        "models_loaded": MODELS_LOADED
    }


@app.get("/health")
async def health_check():
    """Verifica estado de los modelos"""
    components_status = {
        "hand_detector": hand_detector is not None,
        "gesture_classifier": gesture_classifier is not None,
        "syllable_classifier": syllable_classifier is not None,
        "word_suggester": word_suggester is not None,
        "word_dictionary": word_dictionary is not None,
        "sentence_bank": sentence_bank is not None,
    }
    
    all_healthy = hand_detector is not None
    
    return {
        "status": "healthy" if all_healthy else "partial",
        "models_loaded": MODELS_LOADED,
        "components": components_status
    }


@app.post("/detect-hand")
async def detect_hand(file: UploadFile = File(...)):
    """
    Detecta manos en una imagen usando MediaPipe
    
    Parámetros:
    - file: Imagen desde Flutter (JPEG/PNG)
    
    Retorna:
    - detected: bool (si se detectó mano)
    - hands_data: información de manos detectadas
    - frame_shape: dimensiones del frame
    """
    if not hand_detector:
        raise HTTPException(status_code=503, detail="HandDetector no disponible")
    
    try:
        # Leer imagen desde Flutter
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {
                "success": False,
                "error": "No se pudo decodificar la imagen"
            }
        
        # Detectar manos usando tu HandDetector
        processed_frame, hands_data = hand_detector.detect_hands(image)
        
        # Verificar si se detectaron manos
        detected = (hands_data.get('left') is not None or 
                   hands_data.get('right') is not None)
        
        return {
            "success": True,
            "detected": detected,
            "left_hand": hands_data.get('left') is not None,
            "right_hand": hands_data.get('right') is not None,
            "landmarks_count": len(hands_data.get('landmarks_list', [])),
            "frame_shape": image.shape[:2]
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/classify-gesture")
async def classify_gesture(file: UploadFile = File(...)):
    """
    Clasifica el gesto/seña en la imagen
    
    Retorna:
    - gesture: letra o palabra detectada
    - confidence: confianza de la predicción
    - alternatives: otras posibles interpretaciones
    """
    if not hand_detector:
        raise HTTPException(status_code=503, detail="HandDetector no disponible")
    
    if not gesture_classifier:
        raise HTTPException(status_code=503, detail="GestureClassifier no disponible")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {
                "success": False,
                "error": "No se pudo decodificar la imagen"
            }
        
        # Detectar mano primero
        processed_frame, hands_data = hand_detector.detect_hands(image)
        
        if not hands_data['landmarks_list']:
            return {
                "success": False,
                "message": "No se detectó ninguna mano"
            }
        
        # Clasificar gesto - ajusta según tu implementación
        # Aquí asumo que tu GestureClassifier tiene un método classify
        try:
            classification = gesture_classifier.classify(hands_data['landmarks_list'][0])
            
            return {
                "success": True,
                "gesture": classification.get("gesture", ""),
                "confidence": classification.get("confidence", 0.0),
                "alternatives": classification.get("alternatives", [])
            }
        except AttributeError:
            # Si tu classifier usa otro método, ajústalo aquí
            return {
                "success": False,
                "error": "Método classify no disponible - revisa GestureClassifier"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/detect-realtime")
async def detect_realtime(file: UploadFile = File(...)):
    """
    Detección completa en tiempo real
    Combina detección + clasificación + análisis de características
    """
    if not hand_detector:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {
                "success": False,
                "error": "Imagen inválida"
            }
        
        # Detectar manos
        processed_frame, hands_data = hand_detector.detect_hands(image)
        
        if not hands_data['landmarks_list']:
            return {
                "success": True,
                "detected": False,
                "message": "Esperando gesto..."
            }
        
        # Obtener características del gesto
        landmarks = hands_data['landmarks_list'][0]
        features = hand_detector.calculate_gesture_features(landmarks)
        orientation = hand_detector.get_hand_orientation(landmarks)
        
        # Clasificar si el clasificador está disponible
        gesture = "unknown"
        confidence = 0.0
        
        if gesture_classifier:
            try:
                classification = gesture_classifier.classify(landmarks)
                gesture = classification.get("gesture", "unknown")
                confidence = classification.get("confidence", 0.0)
            except:
                pass
        
        # Sugerencias de palabras
        suggestions = []
        if word_suggester and gesture != "unknown":
            try:
                suggestions = word_suggester.suggest(gesture)
            except:
                pass
        
        return {
            "success": True,
            "detected": True,
            "gesture": gesture,
            "confidence": confidence,
            "orientation": orientation,
            "features": {
                "fingers_extended": {
                    "thumb": features.get("thumb_extended", False),
                    "index": features.get("index_extended", False),
                    "middle": features.get("middle_extended", False),
                    "ring": features.get("ring_extended", False),
                    "pinky": features.get("pinky_extended", False),
                },
                "hand_openness": features.get("hand_width", 0),
            },
            "word_suggestions": suggestions[:5] if suggestions else [],
            "hands_detected": {
                "left": hands_data.get('left') is not None,
                "right": hands_data.get('right') is not None
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/translate-sequence")
async def translate_sequence(data: dict):
    """
    Traduce una secuencia de gestos a texto
    
    Body JSON:
    {
        "gestures": ["A", "M", "O", "R"],
        "mode": "word" | "sentence"
    }
    """
    try:
        gestures = data.get("gestures", [])
        mode = data.get("mode", "word")
        
        if not gestures:
            return {
                "success": False,
                "error": "No se proporcionaron gestos"
            }
        
        if mode == "word":
            # Formar palabra
            word = "".join(gestures)
            suggestions = []
            
            if word_suggester:
                try:
                    suggestions = word_suggester.suggest_complete_word(word)
                except:
                    pass
            
            return {
                "success": True,
                "word": word,
                "suggestions": suggestions[:10] if suggestions else []
            }
        
        else:  # sentence
            # Formar oración
            sentence = " ".join(gestures)
            
            return {
                "success": True,
                "sentence": sentence
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/dictionary")
async def get_dictionary():
    """
    Retorna el diccionario de señas disponibles
    """
    try:
        dictionary_data = {
            "letters": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            "numbers": list(range(10)),
            "total_gestures": 26
        }
        
        # Si tienes WordDictionary disponible, úsalo
        if word_dictionary:
            try:
                # Ajusta según los métodos de tu WordDictionary
                dictionary_data["common_words"] = word_dictionary.get_common_words()
            except:
                dictionary_data["common_words"] = ["HOLA", "GRACIAS", "AMOR"]
        
        return {
            "success": True,
            "dictionary": dictionary_data
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/config/update")
async def update_config(data: dict):
    """
    Actualiza configuración del sistema
    """
    try:
        confidence_threshold = data.get("confidence_threshold", 0.7)
        max_hands = data.get("max_hands", 2)
        
        # Actualizar configuración del detector si es necesario
        # Esto depende de cómo implementaste tu HandDetector
        
        return {
            "success": True,
            "message": "Configuración actualizada",
            "config": {
                "confidence_threshold": confidence_threshold,
                "max_hands": max_hands
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Endpoint de prueba simple
@app.get("/test")
async def test_endpoint():
    """Endpoint simple para verificar conectividad"""
    return {
        "status": "ok",
        "message": "Servidor funcionando correctamente",
        "timestamp": "2024-11-11"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Iniciando servidor API Traductor de Señas")
    print("=" * 60)
    print("📡 La API estará disponible en: http://localhost:8000")
    print("📖 Documentación interactiva: http://localhost:8000/docs")
    print("📊 Esquema OpenAPI: http://localhost:8000/redoc")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Permite conexiones desde otros dispositivos
        port=8000,
        reload=True,     # Recarga automática en desarrollo
        log_level="info"
    )