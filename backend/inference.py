# backend/inference.py - VERSIÓN MEJORADA COMPLETA

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
        
        # Buscar checkpoint si es necesario
        if not (self.model_path / "config.json").exists():
            checkpoints = sorted(self.model_path.glob("checkpoint-*"))
            if checkpoints:
                self.model_path = checkpoints[-1]
                logger.info(f"📂 Usando checkpoint: {self.model_path.name}")
        
        logger.info(f"🔄 Cargando modelo desde: {self.model_path}")
        
        # Cargar tokenizer y modelo
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), 
            local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path), 
            local_files_only=True
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Modelo cargado en: {self.device}")
        self._warmup()
    
    def _warmup(self):
        """Warm-up del modelo"""
        logger.info("🔥 Warming up model...")
        dummy_text = "This is a warm-up sentence for the model."
        try:
            self.predict(dummy_text)
            logger.info("✅ Warm-up completado")
        except Exception as e:
            logger.warning(f"⚠️ Warm-up falló: {e}")
    
    def predict(self, text: str) -> dict:
        """
        Hacer predicción con ajustes por longitud de texto
        """
        word_count = len(text.split())
        
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
        
        # Ajustar probabilidades según longitud del texto
        adjusted_probs = self._adjust_probabilities(probs, word_count)
        
        return self._format_result(adjusted_probs, word_count)
    
    def _adjust_probabilities(self, probs, word_count):
        """
        Ajusta probabilidades según características del texto
        """
        adjusted = probs.copy()
        
        # Textos muy cortos (<50 palabras) - probablemente titulares
        if word_count < 50:
            # El modelo tiende a clasificar titulares como fake
            # Aumentamos confianza en Real
            boost_factor = 1.5  # 50% más peso a Real
            penalty_factor = 0.7  # 30% menos peso a Fake
            
            adjusted[1] = probs[1] * boost_factor  # Real
            adjusted[0] = probs[0] * penalty_factor  # Fake
        
        # Textos cortos (50-100 palabras)
        elif word_count < 100:
            boost_factor = 1.2
            penalty_factor = 0.85
            
            adjusted[1] = probs[1] * boost_factor
            adjusted[0] = probs[0] * penalty_factor
        
        # Re-normalizar probabilidades
        total = adjusted.sum()
        adjusted = adjusted / total
        
        return adjusted
    
    def _format_result(self, probs, word_count):
        """
        Formatear resultado con clasificación mejorada
        """
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        
        # Calcular score base (0-100)
        prob_real = probs[1]
        prob_fake = probs[0]
        
        # Score: 0 (muy fake) a 100 (muy real)
        # Basado en la probabilidad de Real
        score = prob_real * 100
        
        # Clasificación con threshold ajustado
        if score >= 60:
            classification = "real"
        elif score >= 40:
            classification = "uncertain"
        else:
            classification = "fake"
        
        # Advertencia para textos muy cortos
        warning = None
        if word_count < 30:
            warning = "Text too short for reliable analysis"
        
        result = {
            'classification': classification,
            'score': round(score, 2),
            'confidence': round(confidence, 4),
            'probabilities': {
                'fake': round(float(probs[0]), 4),
                'real': round(float(probs[1]), 4)
            }
        }
        
        if warning:
            result['warning'] = warning
        
        return result