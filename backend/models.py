
# ============================================
# backend/models.py - Pydantic Models
# ============================================

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict

class AnalysisRequest(BaseModel):
    """Request model para análisis"""
    text: str = Field(..., min_length=10, max_length=10000)
    url: Optional[str] = None
    user_id: Optional[str] = None
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Breaking news: scientists discover cure for all diseases!",
                "url": "https://example.com/article",
                "user_id": "user123"
            }
        }

class AnalysisResponse(BaseModel):
    """Response model para análisis"""
    classification: str  # "fake" o "real"
    score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Dict[str, float]
    processing_time_ms: float
    cached: bool = False
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "classification": "fake",
                "score": 15.32,
                "confidence": 0.8936,
                "probabilities": {
                    "fake": 0.8936,
                    "real": 0.1064
                },
                "processing_time_ms": 234.56,
                "cached": False,
                "timestamp": "2024-01-15T10:30:00.000Z"
            }
        }

class HealthResponse(BaseModel):
    """Response model para health check"""
    status: str
    timestamp: str
    model_loaded: bool
    model_type: str
    version: str