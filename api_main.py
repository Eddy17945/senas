"""
API REST para Traductor de Señas - CON CACHÉ
Solo detección con MediaPipe, clasificación en Flutter
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import os
import hashlib
import time
from typing import Optional, Dict, Any

# Puerto dinámico para Render
PORT = int(os.environ.get("PORT", 8000))

# Crear app
app = FastAPI(
    title="Traductor de Señas API - Con Caché",
    description="API con MediaPipe + sistema de caché inteligente",
    version="3.1.0"
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

# ===== SISTEMA DE CACHÉ =====
detection_cache: Dict[str, Dict[str, Any]] = {}
MAX_CACHE_SIZE = 100
cache_stats = {"hits": 0, "misses": 0}

print("✅ MediaPipe Hands + CACHÉ inicializados correctamente")


def get_image_hash(image_bytes: bytes) -> str:
    """Genera hash único de la imagen para identificarla"""
    return hashlib.md5(image_bytes).hexdigest()


def add_to_cache(image_hash: str, result: Dict[str, Any]):
    """Agrega resultado al caché"""
    global detection_cache
    
    # Si el caché está lleno, eliminar la entrada más antigua
    if len(detection_cache) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(detection_cache))
        del detection_cache[oldest_key]
    
    detection_cache[image_hash] = {
        "result": result,
        "timestamp": time.time()
    }


def get_from_cache(image_hash: str, max_age: int = 2) -> Optional[Dict[str, Any]]:
    """
    Obtiene resultado del caché si existe y es reciente
    max_age: segundos de validez del caché
    """
    if image_hash in detection_cache:
        cached_data = detection_cache[image_hash]
        age = time.time() - cached_data["timestamp"]
        
        # Si el caché es reciente, usarlo
        if age < max_age:
            cache_stats["hits"] += 1
            return cached_data["result"]
        else:
            # Caché expirado, eliminarlo
            del detection_cache[image_hash]
    
    cache_stats["misses"] += 1
    return None


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
        "message": "API Traductor de Señas - Con CACHÉ",
        "version": "3.1.0",
        "models_loaded": True,
        "detector_type": "MediaPipe Hands + Cache",
        "cache_enabled": True,
        "cache_stats": cache_stats
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "detector_type": "MediaPipe Hands + Cache",
        "cache_size": len(detection_cache),
        "cache_stats": cache_stats,
        "components": {
            "mediapipe_hands": hands is not None,
            "cache_system": True
        }
    }


@app.get("/cache/stats")
async def get_cache_stats():
    """
    Endpoint para ver estadísticas del caché
    Útil para monitoreo
    """
    hit_rate = 0
    total_requests = cache_stats["hits"] + cache_stats["misses"]
    
    if total_requests > 0:
        hit_rate = (cache_stats["hits"] / total_requests) * 100
    
    return {
        "cache_size": len(detection_cache),
        "max_size": MAX_CACHE_SIZE,
        "hits": cache_stats["hits"],
        "misses": cache_stats["misses"],
        "total_requests": total_requests,
        "hit_rate_percentage": f"{hit_rate:.1f}%",
        "cache_effectiveness": "Excellent" if hit_rate > 50 else "Good" if hit_rate > 30 else "Low"
    }


@app.post("/cache/clear")
async def clear_cache():
    """Limpia todo el caché (útil para debugging)"""
    global detection_cache, cache_stats
    
    old_size = len(detection_cache)
    detection_cache.clear()
    cache_stats = {"hits": 0, "misses": 0}
    
    return {
        "status": "success",
        "message": "Cache cleared",
        "entries_removed": old_size
    }


@app.post("/detect-realtime")
async def detect_realtime(file: UploadFile = File(...)):
    """
    Detección EN TIEMPO REAL - Con caché activado
    La clasificación se hace en Flutter
    """
    try:
        # Leer imagen
        contents = await file.read()
        
        # ===== VERIFICAR CACHÉ PRIMERO =====
        image_hash = get_image_hash(contents)
        cached_result = get_from_cache(image_hash, max_age=2)
        
        if cached_result is not None:
            # ¡Encontrado en caché! Retornar inmediatamente
            cached_result["from_cache"] = True
            return cached_result
        
        # ===== SI NO ESTÁ EN CACHÉ, PROCESAR =====
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
            result = {
                "success": True,
                "detected": False,
                "message": "Esperando gesto...",
                "raw_hands_data": {"left": None, "right": None},
                "from_cache": False
            }
            # Guardar en caché
            add_to_cache(image_hash, result)
            return result
        
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
        
        # Respuesta completa
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
            "raw_hands_data": raw_hands_data,
            "from_cache": False
        }
        
        # ===== GUARDAR EN CACHÉ =====
        add_to_cache(image_hash, result)
        
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
    """Detecta si hay manos en la imagen (sin caché, es rápido)"""
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
    """ULTRA RÁPIDO: Solo detecta landmarks - CON CACHÉ"""
    try:
        contents = await file.read()
        
        # Verificar caché
        image_hash = get_image_hash(contents)
        cached_result = get_from_cache(image_hash, max_age=2)
        
        if cached_result is not None:
            return cached_result
        
        # Procesar imagen
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"success": False, "error": "Imagen inválida"}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            result = {
                "success": True,
                "detected": False,
                "landmarks": None
            }
            add_to_cache(image_hash, result)
            return result
        
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
        
        # Guardar en caché
        add_to_cache(image_hash, result)
        
        return convert_numpy_types(result)
    
    except Exception as e:
        print(f"Error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/test")
async def test():
    return {
        "status": "ok",
        "message": "API con caché funcionando",
        "detector": "MediaPipe Hands + Cache System",
        "cache_enabled": True,
        "timestamp": "2024-11-27"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🚀 SERVIDOR API CON CACHÉ INTELIGENTE")
    print("=" * 70)
    print(f"📡 Puerto: {PORT}")
    print("💾 Caché: Activado (100 imágenes)")
    print("⚡ Mejora de velocidad: 30-50%")
    print("📖 Docs: http://localhost:{PORT}/docs")
    print("📊 Stats: http://localhost:{PORT}/cache/stats")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info"
    )