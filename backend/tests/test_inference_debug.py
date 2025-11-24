# test_inference_debug.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import numpy as np

# Cargar modelo
MODEL_PATH = Path("models/bert_fakenews_v1")  # Ajusta tu path

print("🔄 Cargando modelo...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH), local_files_only=True)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"✅ Modelo cargado en: {device}\n")

# Casos de prueba
test_cases = [
    {
        "text": "The Federal Reserve announced new interest rate policies during today's meeting in Washington.",
        "expected": "real"
    },
    {
        "text": "CNN reports that the President signed a new bill into law after months of negotiations.",
        "expected": "real"
    },
    {
        "text": "Scientists at MIT published research findings in the journal Nature this week.",
        "expected": "real"
    },
    {
        "text": "BREAKING: Aliens confirmed to have landed in Area 51, government officials admit cover-up!",
        "expected": "fake"
    },
    {
        "text": "You won't believe what this celebrity did! Doctors hate this one weird trick!",
        "expected": "fake"
    },
]

def predict_with_details(text):
    """Predicción con detalles de debugging"""
    
    word_count = len(text.split())
    
    # Tokenizar
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    ).to(device)
    
    # Predicción
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0].cpu().numpy()
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    
    # Ajustar según longitud (tu código)
    adjusted = probs.copy()
    
    if word_count < 50:
        boost_factor = 1.5
        penalty_factor = 0.7
        adjusted[1] = probs[1] * boost_factor
        adjusted[0] = probs[0] * penalty_factor
    elif word_count < 100:
        boost_factor = 1.2
        penalty_factor = 0.85
        adjusted[1] = probs[1] * boost_factor
        adjusted[0] = probs[0] * penalty_factor
    
    # Re-normalizar
    total = adjusted.sum()
    adjusted = adjusted / total
    
    # Calcular score
    score = adjusted[1] * 100
    
    # Clasificación
    if score >= 60:
        classification = "real"
    elif score >= 40:
        classification = "uncertain"
    else:
        classification = "fake"
    
    return {
        'word_count': word_count,
        'logits': logits,
        'probs_original': probs,
        'probs_adjusted': adjusted,
        'score': score,
        'classification': classification
    }

# Probar todos los casos
print("="*80)
print("🧪 TESTS DE PREDICCIÓN")
print("="*80)

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. Text: {test['text'][:80]}...")
    print(f"   Expected: {test['expected'].upper()}")
    
    result = predict_with_details(test['text'])
    
    print(f"\n   📊 DETALLES:")
    print(f"   Word count: {result['word_count']}")
    print(f"   Logits: Fake={result['logits'][0]:.4f}, Real={result['logits'][1]:.4f}")
    print(f"   Probs (original): Fake={result['probs_original'][0]:.4f}, Real={result['probs_original'][1]:.4f}")
    print(f"   Probs (adjusted): Fake={result['probs_adjusted'][0]:.4f}, Real={result['probs_adjusted'][1]:.4f}")
    print(f"   Score: {result['score']:.2f}/100")
    print(f"   Classification: {result['classification'].upper()}")
    
    # Verificar si coincide
    match = "✅" if result['classification'] == test['expected'] else "❌"
    print(f"   {match} {'CORRECTO' if match == '✅' else 'INCORRECTO'}")
    print("-"*80)

print("\n" + "="*80)
print("ANÁLISIS COMPLETO")
print("="*80)