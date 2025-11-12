"""
API REST para Traductor de Señas - CON DETECTOR AVANZADO
Usa AdvancedHandDetector para mayor precisión
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# IMPORTAR DETECTOR AVANZADO
try:
    from src.detector.advanced_hand_detector import AdvancedHandDetector
    from src.detector.gesture_classifier import GestureClassifier
    from src.detector.syllable_classifier import SyllableClassifier
    from src.detector.complete_word_detector import CompleteWordDetector
    from src.utils.word_suggester import WordSuggester
    from src.utils.word_dictionary import WordDictionary
    from src.utils.sentence_bank import SentenceBank
    MODELS_LOADED = True
    print("✅ Módulos AVANZADOS importados correctamente")
except ImportError as e:
    print(f"⚠️ Error importando módulos: {e}")
    MODELS_LOADED = False

app = FastAPI(
    title="Traductor de Señas API - Avanzado",
    description="API con detector avanzado optimizado",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
hand_detector = None
gesture_classifier = None
syllable_classifier = None
complete_word_detector = None
word_suggester = None
word_dictionary = None
sentence_bank = None


def convert_numpy_types(obj):
    """Convierte tipos NumPy a tipos Python nativos"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def initialize_models():
    """Inicializar DETECTOR AVANZADO"""
    global hand_detector, gesture_classifier, syllable_classifier
    global complete_word_detector, word_suggester, word_dictionary, sentence_bank
    
    if not MODELS_LOADED:
        print("⚠️ Modelos no disponibles")
        return
    
    try:
        print("🔄 Cargando modelos AVANZADOS...")
        
        # USAR DETECTOR AVANZADO
        hand_detector = AdvancedHandDetector(
            max_num_hands=2,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55
        )
        print("✓ AdvancedHandDetector cargado (OPTIMIZADO)")
        
        try:
            gesture_classifier = GestureClassifier()
            print("✓ GestureClassifier cargado")
        except Exception as e:
            print(f"⚠️ GestureClassifier: {e}")
        
        try:
            syllable_classifier = SyllableClassifier()
            print("✓ SyllableClassifier cargado")
        except Exception as e:
            print(f"⚠️ SyllableClassifier: {e}")
        
        try:
            complete_word_detector = CompleteWordDetector()
            print("✓ CompleteWordDetector cargado")
        except Exception as e:
            print(f"⚠️ CompleteWordDetector: {e}")
        
        try:
            word_suggester = WordSuggester()
            print("✓ WordSuggester cargado")
        except Exception as e:
            print(f"⚠️ WordSuggester: {e}")
        
        try:
            word_dictionary = WordDictionary()
            print("✓ WordDictionary cargado")
        except Exception as e:
            print(f"⚠️ WordDictionary: {e}")
        
        try:
            sentence_bank = SentenceBank()
            print("✓ SentenceBank cargado")
        except Exception as e:
            print(f"⚠️ SentenceBank: {e}")
        
        print("✅ Sistema AVANZADO inicializado")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


initialize_models()


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "API Traductor de Señas - DETECTOR AVANZADO",
        "version": "2.0.0",
        "models_loaded": MODELS_LOADED,
        "detector_type": "AdvancedHandDetector"
    }


@app.get("/health")
async def health_check():
    components = {
        "hand_detector": hand_detector is not None,
        "gesture_classifier": gesture_classifier is not None,
        "syllable_classifier": syllable_classifier is not None,
        "complete_word_detector": complete_word_detector is not None,
        "word_suggester": word_suggester is not None,
        "word_dictionary": word_dictionary is not None,
        "sentence_bank": sentence_bank is not None,
    }
    
    return {
        "status": "healthy" if hand_detector else "partial",
        "models_loaded": MODELS_LOADED,
        "detector_type": "AdvancedHandDetector" if hand_detector else "None",
        "components": components
    }


@app.post("/detect-realtime")
async def detect_realtime(file: UploadFile = File(...)):
    """
    Detección EN TIEMPO REAL con detector AVANZADO
    Incluye: palabras completas, gestos, letras
    """
    if not hand_detector:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")
    
    try:
        # Leer imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"success": False, "error": "Imagen inválida"}
        
        # DETECTAR con detector AVANZADO
        processed_frame, hands_data = hand_detector.detect_hands(image)
        
        # Sin detección
        if not hands_data['landmarks_list']:
            return {
                "success": True,
                "detected": False,
                "message": "Esperando gesto...",
                "quality_score": {"left": 0.0, "right": 0.0}
            }
        
        # Obtener landmarks y features
        landmarks = hands_data['landmarks_list'][0]
        
        # PRIORIDAD 1: PALABRAS COMPLETAS
        complete_word = None
        if complete_word_detector and len(hands_data['landmarks_list']) > 0:
            confidence = hands_data.get('confidence', {}).get('left', 0) or \
                        hands_data.get('confidence', {}).get('right', 0)
            
            try:
                complete_word = complete_word_detector.detect_complete_word(
                    landmarks, 
                    confidence
                )
            except Exception as e:
                print(f"Error palabra completa: {e}")
        
        if complete_word:
            return convert_numpy_types({
                "success": True,
                "detected": True,
                "type": "COMPLETE_WORD",
                "complete_word": str(complete_word),
                "gesture": str(complete_word),
                "confidence": float(hands_data['confidence'].get('left', 0) or 
                                   hands_data['confidence'].get('right', 0)),
                "quality_score": {
                    "left": float(hands_data['quality_score'].get('left', 0)),
                    "right": float(hands_data['quality_score'].get('right', 0))
                },
                "hands_detected": {
                    "left": bool(hands_data.get('left') is not None),
                    "right": bool(hands_data.get('right') is not None)
                }
            })
        
        # PRIORIDAD 2: LETRAS INDIVIDUALES
        gesture = "unknown"
        confidence = 0.0
        
        if gesture_classifier:
            try:
                detected_gesture = gesture_classifier.predict_gesture(landmarks)
                if detected_gesture:
                    gesture = str(detected_gesture)
                    confidence = float(gesture_classifier.get_detection_confidence())
            except Exception as e:
                print(f"Error clasificando gesto: {e}")
        
        # Calcular features
        try:
            features = hand_detector.calculate_gesture_features(landmarks)
            orientation = hand_detector.get_hand_orientation(landmarks)
        except:
            features = {}
            orientation = "unknown"
        
        # Sugerencias
        suggestions = []
        if word_suggester and gesture != "unknown":
            try:
                sugg = word_suggester.suggest(gesture)
                suggestions = [str(s) for s in sugg[:5]] if sugg else []
            except:
                pass
        
        # RESPUESTA COMPLETA
        result = {
            "success": True,
            "detected": True,
            "type": "LETTER",
            "gesture": str(gesture),
            "confidence": float(confidence),
            "orientation": str(orientation),
            "quality_score": {
                "left": float(hands_data['quality_score'].get('left', 0)),
                "right": float(hands_data['quality_score'].get('right', 0))
            },
            "features": {
                "fingers_extended": {
                    "thumb": bool(features.get("thumb_extended", False)),
                    "index": bool(features.get("index_extended", False)),
                    "middle": bool(features.get("middle_extended", False)),
                    "ring": bool(features.get("ring_extended", False)),
                    "pinky": bool(features.get("pinky_extended", False)),
                },
                "hand_openness": float(features.get("hand_width", 0)),
            },
            "word_suggestions": suggestions,
            "hands_detected": {
                "left": bool(hands_data.get('left') is not None),
                "right": bool(hands_data.get('right') is not None)
            }
        }
        
        return convert_numpy_types(result)
    
    except Exception as e:
        print(f"Error en detect_realtime: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/detect-hand")
async def detect_hand(file: UploadFile = File(...)):
    """Detecta manos con detector avanzado"""
    if not hand_detector:
        raise HTTPException(status_code=503, detail="HandDetector no disponible")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"success": False, "error": "Imagen inválida"}
        
        processed_frame, hands_data = hand_detector.detect_hands(image)
        
        detected = (hands_data.get('left') is not None or 
                   hands_data.get('right') is not None)
        
        result = {
            "success": True,
            "detected": bool(detected),
            "left_hand": bool(hands_data.get('left') is not None),
            "right_hand": bool(hands_data.get('right') is not None),
            "landmarks_count": int(len(hands_data.get('landmarks_list', []))),
            "frame_shape": [int(x) for x in image.shape[:2]],
            "quality_score": {
                "left": float(hands_data['quality_score'].get('left', 0)),
                "right": float(hands_data['quality_score'].get('right', 0))
            }
        }
        
        return convert_numpy_types(result)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/classify-gesture")
async def classify_gesture(file: UploadFile = File(...)):
    """Clasifica gestos con mayor precisión"""
    if not hand_detector or not gesture_classifier:
        raise HTTPException(status_code=503, detail="Sistema no disponible")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"success": False, "error": "Imagen inválida"}
        
        processed_frame, hands_data = hand_detector.detect_hands(image)
        
        if not hands_data['landmarks_list']:
            return {
                "success": False,
                "message": "No se detectó ninguna mano"
            }
        
        landmarks = hands_data['landmarks_list'][0]
        gesture = gesture_classifier.predict_gesture(landmarks)
        
        if not gesture:
            return {
                "success": False,
                "message": "No se pudo clasificar el gesto"
            }
        
        result = {
            "success": True,
            "gesture": str(gesture),
            "confidence": float(gesture_classifier.get_detection_confidence()),
            "quality_score": float(hands_data['quality_score'].get('left', 0) or 
                                  hands_data['quality_score'].get('right', 0)),
            "alternatives": []
        }
        
        return convert_numpy_types(result)
    
    except Exception as e:
        print(f"Error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/translate-sequence")
async def translate_sequence(data: dict):
    """Traduce secuencia de gestos"""
    try:
        gestures = data.get("gestures", [])
        mode = data.get("mode", "word")
        
        if not gestures:
            return {"success": False, "error": "No hay gestos"}
        
        if mode == "word":
            word = "".join(gestures)
            suggestions = []
            
            if word_suggester:
                try:
                    sugg = word_suggester.suggest_complete_word(word)
                    suggestions = [str(s) for s in sugg[:10]] if sugg else []
                except:
                    pass
            
            return {
                "success": True,
                "word": str(word),
                "suggestions": suggestions
            }
        else:
            return {
                "success": True,
                "sentence": " ".join(gestures)
            }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/dictionary")
async def get_dictionary():
    """Retorna diccionario"""
    try:
        dictionary_data = {
            "letters": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            "numbers": list(range(10)),
            "total_gestures": 26
        }
        
        if word_dictionary:
            try:
                dictionary_data["common_words"] = word_dictionary.get_common_words()
            except:
                dictionary_data["common_words"] = ["HOLA", "GRACIAS", "AMOR"]
        
        return {
            "success": True,
            "dictionary": dictionary_data
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/test")
async def test():
    return {
        "status": "ok",
        "message": "API Avanzada funcionando",
        "detector": "AdvancedHandDetector",
        "timestamp": "2024-11-12"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🚀 SERVIDOR API CON DETECTOR AVANZADO")
    print("=" * 70)
    print("📡 URL: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("🎯 Detector: AdvancedHandDetector (OPTIMIZADO)")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )