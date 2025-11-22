# scripts/05_convert_to_onnx.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

MODEL_PATH = './models/bert_fakenews_v1'
ONNX_PATH = './models/bert_fakenews_v1_onnx'

print("🔄 Convirtiendo modelo a ONNX...")

# Cargar modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Crear input dummy
dummy_input = tokenizer(
    "This is a test sentence",
    return_tensors='pt',
    max_length=512,
    padding='max_length',
    truncation=True
)

# Exportar a ONNX
Path(ONNX_PATH).mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    f"{ONNX_PATH}/model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size'}
    },
    opset_version=14
)

# Guardar tokenizer
tokenizer.save_pretrained(ONNX_PATH)

print(f"✅ Modelo ONNX guardado en: {ONNX_PATH}")

# Probar velocidad
import time
import onnxruntime as ort

session = ort.InferenceSession(f"{ONNX_PATH}/model.onnx")

# Benchmark
texts = ["Test sentence"] * 100
start = time.time()

for text in texts:
    inputs = tokenizer(text, return_tensors='np', max_length=512, 
                      padding='max_length', truncation=True)
    outputs = session.run(None, {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    })

end = time.time()
print(f"\n⚡ ONNX Inference: {(end-start)/100*1000:.2f}ms por sample")
