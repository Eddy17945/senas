"""
API REST para Traductor de Señas - VERSIÓN SIMPLE
Solo detección con MediaPipe, clasificación en Flutter
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import os

# Puerto dinámico para Render
PORT = int(os.environ.get("PORT", 8000))

# Crear app
app = FastAPI(
    title="Traductor de Señas API - Simple",
    description="API con MediaPipe básico para detección de manos",
    version="3.0.0"
)

# CORS para Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("✅ MediaPipe Hands inicializado correctamente")


def convert_numpy_types(obj):
    """Convierte tipos NumPy a tipos Python nativos"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.flatten().tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "API Traductor de Señas - SIMPLE",
        "version": "3.0.0",
        "models_loaded": True,
        "detector_type": "MediaPipe Hands (Basic)"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "detector_type": "MediaPipe Hands",
        "components": {
            "mediapipe_hands": hands is not None
        }
    }


@app.post("/detect-realtime")
async def detect_realtime(file: UploadFile = File(...)):
    """
    Detección EN TIEMPO REAL - Solo envía landmarks
    La clasificación se hace en Flutter
    """
    try:
        # Leer imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"success": False, "error": "Imagen inválida"}
        
        # Redimensionar si es muy grande
        height, width = image.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convertir BGR a RGB para MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar manos
        results = hands.process(image_rgb)
        
        # Sin detección
        if not results.multi_hand_landmarks:
            return {
                "success": True,
                "detected": False,
                "message": "Esperando gesto...",
                "raw_hands_data": {"left": None, "right": None}
            }
        
        # Extraer landmarks de la primera mano detectada
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Convertir landmarks a lista plana [x, y, z, x, y, z, ...]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                float(landmark.x),
                float(landmark.y),
                float(landmark.z)
            ])
        
        # Determinar qué mano es (izquierda/derecha)
        handedness = "right"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label.lower()
        
        # Preparar datos de las manos
        raw_hands_data = {
            "left": landmarks if handedness == "left" else None,
            "right": landmarks if handedness == "right" else None
        }
        
        # Calcular confianza
        confidence = 0.0
        if results.multi_handedness:
            confidence = results.multi_handedness[0].classification[0].score
        
        # Respuesta simplificada
        result = {
            "success": True,
            "detected": True,
            "type": "LETTER",
            "gesture": "unknown",  # Flutter clasificará
            "confidence": float(confidence),
            "orientation": "neutral",
            "quality_score": {
                "left": float(confidence) if handedness == "left" else 0.0,
                "right": float(confidence) if handedness == "right" else 0.0
            },
            "features": {
                "fingers_extended": {
                    "thumb": False,
                    "index": False,
                    "middle": False,
                    "ring": False,
                    "pinky": False
                },
                "hand_openness": 0.0
            },
            "word_suggestions": [],
            "hands_detected": {
                "left": handedness == "left",
                "right": handedness == "right"
            },
            "raw_hands_data": raw_hands_data
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
    """Detecta si hay manos en la imagen"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"success": False, "error": "Imagen inválida"}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        detected = results.multi_hand_landmarks is not None
        
        result = {
            "success": True,
            "detected": bool(detected),
            "landmarks_count": len(results.multi_hand_landmarks) if detected else 0,
            "frame_shape": [int(x) for x in image.shape[:2]]
        }
        
        return convert_numpy_types(result)
    
    except Exception as e:
        print(f"Error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/detect-landmarks-fast")
async def detect_landmarks_fast(file: UploadFile = File(...)):
    """ULTRA RÁPIDO: Solo detecta landmarks"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"success": False, "error": "Imagen inválida"}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return {
                "success": True,
                "detected": False,
                "landmarks": None
            }
        
        # Extraer landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                float(landmark.x),
                float(landmark.y),
                float(landmark.z)
            ])
        
        confidence = 0.0
        if results.multi_handedness:
            confidence = results.multi_handedness[0].classification[0].score
        
        result = {
            "success": True,
            "detected": True,
            "landmarks": landmarks,
            "confidence": float(confidence)
        }
        
        return convert_numpy_types(result)
    
    except Exception as e:
        print(f"Error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/test")
async def test():
    return {
        "status": "ok",
        "message": "API Simple funcionando",
        "detector": "MediaPipe Hands (Basic)",
        "timestamp": "2024-11-23"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🚀 SERVIDOR API SIMPLE - SOLO MEDIAPIPE")
    print("=" * 70)
    print(f"📡 Puerto: {PORT}")
    print("📖 Docs: /docs")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info"
    )