# debug_imports.py
# Crea este archivo en la misma carpeta que test_setup.py y ejecútalo

import os
import sys

print("=== DIAGNÓSTICO DE IMPORTACIONES ===")
print(f"Directorio actual: {os.getcwd()}")
print(f"Python path: {sys.path}")
print()

# Verificar estructura de archivos
print("=== VERIFICANDO ESTRUCTURA ===")
required_files = [
    "src/__init__.py",
    "src/detector/__init__.py", 
    "src/detector/syllable_classifier.py",
    "src/detector/gesture_classifier.py",
    "src/detector/hand_detector.py",
    "src/interface/__init__.py",
    "src/interface/reference_gallery.py"
]

for file_path in required_files:
    exists = os.path.exists(file_path)
    print(f"{'✅' if exists else '❌'} {file_path}")
    
    if exists and file_path.endswith('.py'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = len(content.splitlines())
                print(f"    └─ {lines} líneas, {len(content)} caracteres")
        except Exception as e:
            print(f"    └─ Error leyendo archivo: {e}")

print()

# Probar importaciones paso a paso
print("=== PROBANDO IMPORTACIONES ===")

try:
    print("1. Importando 'src'...")
    import src
    print("   ✅ src - OK")
except Exception as e:
    print(f"   ❌ src - ERROR: {e}")
    sys.exit(1)

try:
    print("2. Importando 'src.detector'...")
    import src.detector
    print("   ✅ src.detector - OK")
except Exception as e:
    print(f"   ❌ src.detector - ERROR: {e}")
    sys.exit(1)

try:
    print("3. Importando 'src.detector.syllable_classifier'...")
    import src.detector.syllable_classifier
    print("   ✅ src.detector.syllable_classifier - OK")
except Exception as e:
    print(f"   ❌ src.detector.syllable_classifier - ERROR: {e}")
    print(f"   Detalles del error: {type(e).__name__}: {e}")
    
    # Información adicional para este error específico
    print("\n   === INFORMACIÓN ADICIONAL ===")
    syllable_file = "src/detector/syllable_classifier.py"
    if os.path.exists(syllable_file):
        try:
            with open(syllable_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"   Primeras 10 líneas del archivo:")
                for i, line in enumerate(lines[:10], 1):
                    print(f"   {i:2d}: {line.rstrip()}")
        except Exception as read_error:
            print(f"   Error leyendo syllable_classifier.py: {read_error}")

try:
    print("4. Importando 'src.detector.gesture_classifier'...")
    import src.detector.gesture_classifier
    print("   ✅ src.detector.gesture_classifier - OK")
except Exception as e:
    print(f"   ❌ src.detector.gesture_classifier - ERROR: {e}")

try:
    print("5. Importando 'src.detector.hand_detector'...")
    import src.detector.hand_detector
    print("   ✅ src.detector.hand_detector - OK")
except Exception as e:
    print(f"   ❌ src.detector.hand_detector - ERROR: {e}")

print()
print("=== DIAGNÓSTICO COMPLETO ===")
print("Si alguna importación falló, revisa el archivo específico mencionado.")
print("El problema probablemente está en errores de sintaxis o imports dentro del archivo.")