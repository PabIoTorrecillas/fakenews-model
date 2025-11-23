# ============================================
# backend/app.py - API Principal (VERSIÓN PYTORCH)
# ============================================

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime
import logging

from models import AnalysisRequest, AnalysisResponse, HealthResponse
from inference import ModelInference
from config import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# INICIALIZAR APP
# ============================================

app = FastAPI(
    title="Fake News Detector API",
    description="ML-powered API for detecting fake news using BERT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo al iniciar
model_inference = None

@app.on_event("startup")
async def startup_event():
    """Cargar modelo al iniciar la API"""
    global model_inference
    logger.info("🚀 Starting Fake News Detector API...")
    
    try:
        model_inference = ModelInference(model_path=settings.MODEL_PATH)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup al cerrar"""
    logger.info("👋 Shutting down API...")

# ============================================
# MIDDLEWARE - Request Logging
# ============================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log todas las requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"completed in {process_time:.2f}ms"
    )
    
    return response

# ============================================
# ENDPOINTS
# ============================================

@app.get("/", tags=["Info"])
def root():
    """Root endpoint"""
    return {
        "service": "Fake News Detector API",
        "version": "1.0.0",
        "status": "operational",
        "model": "BERT (PyTorch)",
        "endpoints": {
            "analyze": "/analyze",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model_inference is not None,
        "model_type": "PyTorch",
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_text(request: AnalysisRequest):
    """
    Analizar texto para detectar fake news
    
    **Parameters:**
    - text: Texto a analizar (min 10 caracteres, max 10,000)
    - url: URL de origen (opcional)
    - user_id: ID del usuario (opcional)
    
    **Returns:**
    - classification: "fake" o "real"
    - score: Puntuación 0-100 (0=fake, 100=real)
    - confidence: Confianza del modelo (0-1)
    - probabilities: Probabilidades detalladas {fake, real}
    - processing_time_ms: Tiempo de procesamiento
    - timestamp: Timestamp ISO de la predicción
    
    **Example Request:**
    ```json
    {
      "text": "Breaking news: scientists discover cure for all diseases!",
      "url": "https://example.com/article"
    }
    ```
    
    **Example Response:**
    ```json
    {
      "classification": "fake",
      "score": 15.32,
      "confidence": 0.8936,
      "probabilities": {
        "fake": 0.8936,
        "real": 0.1064
      },
      "processing_time_ms": 234.56,
      "cached": false,
      "timestamp": "2024-01-15T10:30:00.000Z"
    }
    ```
    """
    start_time = time.time()
    
    # Validaciones
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Text must be at least 10 characters long"
        )
    
    if len(request.text) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Text too long. Maximum 10,000 characters"
        )
    
    try:
        # Hacer predicción
        result = model_inference.predict(request.text)
        
        # Calcular tiempo de procesamiento
        processing_time = (time.time() - start_time) * 1000
        
        # Preparar respuesta
        response = AnalysisResponse(
            classification=result['classification'],
            score=result['score'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time_ms=round(processing_time, 2),
            cached=False,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(
            f"Analysis completed: score={result['score']:.1f}, "
            f"classification={result['classification']}, "
            f"time={processing_time:.1f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/stats", tags=["Info"])
def get_stats():
    """
    Estadísticas de uso del API
    
    (Implementar según necesites con base de datos)
    """
    return {
        "message": "Stats endpoint - implement with database",
        "total_requests": 0,
        "uptime": "operational"
    }

@app.post("/batch-analyze", tags=["Analysis"])
async def batch_analyze(texts: list[str]):
    """
    Analizar múltiples textos en batch
    
    **Parameters:**
    - texts: Lista de textos a analizar
    
    **Returns:**
    - Lista de resultados
    """
    if not texts or len(texts) == 0:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if len(texts) > 10:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 10 texts per batch request"
        )
    
    results = []
    
    for i, text in enumerate(texts):
        try:
            if len(text.strip()) < 10:
                results.append({
                    "index": i,
                    "error": "Text too short (min 10 chars)"
                })
                continue
            
            result = model_inference.predict(text)
            results.append({
                "index": i,
                "text_preview": text[:100] + "...",
                "classification": result['classification'],
                "score": result['score'],
                "confidence": result['confidence']
            })
        except Exception as e:
            results.append({
                "index": i,
                "error": str(e)
            })
    
    return {"results": results, "total": len(texts)}

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found", 
            "path": request.url.path,
            "message": "Check /docs for available endpoints"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Please contact support if this persists"
        }
    )

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
