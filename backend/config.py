from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]  # En producción: dominios específicos
    
    # Model (PyTorch - Sin ONNX)
    MODEL_PATH: str = "./models/bert_fakenews_v1"
    # O usar checkpoint específico:
    # MODEL_PATH: str = "./models/bert_fakenews_v1/checkpoint-3000"
    
    # Rate Limiting (para implementar después)
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 3600  # 1 hora
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()