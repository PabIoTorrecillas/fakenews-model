
# ============================================
# backend/inference.py - Lógica de Predicción (PyTorch)
# ============================================

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelInference:
    """Clase para manejar inferencia del modelo (PyTorch)"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        
        # Si es carpeta principal, buscar checkpoint más reciente
        if not (self.model_path / "config.json").exists():
            checkpoints = sorted(self.model_path.glob("checkpoint-*"))
            if checkpoints:
                self.model_path = checkpoints[-1]
                logger.info(f"📂 Usando checkpoint: {self.model_path.name}")
            else:
                raise FileNotFoundError(
                    f"No se encontró config.json ni checkpoints en {model_path}"
                )
        
        logger.info(f"🔄 Cargando modelo desde: {self.model_path}")
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), 
            local_files_only=True
        )
        
        # Cargar modelo
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path), 
            local_files_only=True
        )
        
        # Configurar device (GPU si está disponible)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Modelo cargado en: {self.device}")
        
        # Warm-up (primera inferencia suele ser lenta)
        self._warmup()
    
    def _warmup(self):
        """Warm-up del modelo con inferencia dummy"""
        logger.info("🔥 Warming up model...")
        dummy_text = "This is a warm-up sentence for the model."
        try:
            self.predict(dummy_text)
            logger.info("✅ Warm-up completado")
        except Exception as e:
            logger.warning(f"⚠️ Warm-up falló: {e}")
    
    def predict(self, text: str) -> dict:
        """
        Hacer predicción
        
        Args:
            text: Texto a analizar
            
        Returns:
            dict con classification, score, confidence, probabilities
        """
        # Tokenizar
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            padding='max_length',
            truncation=True
        ).to(self.device)
        
        # Predicción
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            probs = probs[0].cpu().numpy()
        
        return self._format_result(probs)
    
    def _format_result(self, probs):
        """Formatear resultado"""
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        
        # Calcular score (0-100)
        # Score = 0 (muy fake) a 100 (muy real)
        if prediction == 1:  # Real
            score = 50 + (confidence * 50)
            classification = "real"
        else:  # Fake
            score = 50 - (confidence * 50)
            classification = "fake"
        
        return {
            'classification': classification,
            'score': round(score, 2),
            'confidence': round(confidence, 4),
            'probabilities': {
                'fake': round(float(probs[0]), 4),
                'real': round(float(probs[1]), 4)
            }
        }
