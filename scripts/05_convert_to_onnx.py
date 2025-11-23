# scripts/check_model_integrity.py

from pathlib import Path
import os

MODEL_PATH = Path(r"D:\Documentos\Universidad\Tec - 7mo Semestre\BigData\fakenews-model\models\bert_fakenews_v1")
checkpoint_path = MODEL_PATH / "pytorch_model.bin"

print(f"📂 Verificando: {checkpoint_path}")
print(f"   ¿Existe? {checkpoint_path.exists()}")

if checkpoint_path.exists():
    size_mb = checkpoint_path.stat().st_size / (1024**2)
    print(f"   Tamaño: {size_mb:.2f} MB")
    
    # Un modelo DistilBERT debería pesar ~250-270 MB
    if size_mb < 200:
        print(f"   ⚠️ ADVERTENCIA: Archivo muy pequeño (esperado: ~260 MB)")
        print(f"   El modelo probablemente no se guardó completamente")
    elif size_mb > 300:
        print(f"   ⚠️ ADVERTENCIA: Archivo muy grande (esperado: ~260 MB)")
    else:
        print(f"   ✅ Tamaño parece correcto")
    
    # Intentar leer los primeros bytes
    try:
        with open(checkpoint_path, 'rb') as f:
            header = f.read(10)
            print(f"   Header (primeros 10 bytes): {header.hex()}")
            
            # PyTorch pickle debería empezar con '80' (protocolo pickle)
            if header[0] == 0x80:
                print(f"   ✅ Header parece válido (protocolo pickle)")
            else:
                print(f"   ❌ Header INVÁLIDO (no es archivo pickle de PyTorch)")
    except Exception as e:
        print(f"   ❌ Error leyendo archivo: {e}")

# Listar TODOS los archivos del modelo
print(f"\n📁 Archivos en {MODEL_PATH}:")
for item in sorted(MODEL_PATH.rglob('*')):
    if item.is_file():
        size = item.stat().st_size / (1024**2)
        print(f"   - {item.name:<40} {size:>8.2f} MB")