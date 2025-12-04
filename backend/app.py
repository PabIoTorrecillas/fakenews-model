# backend/app.py - ACTUALIZADO

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime
from urllib.parse import urlparse
import logging

from models import AnalysisRequest, AnalysisResponse, HealthResponse, DomainReputationResponse
from inference import ModelInference
from config import settings
from database import mongodb  # ← NUEVO

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
    description="ML-powered API with MongoDB integration",
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
    """Cargar modelo y conectar a MongoDB al iniciar"""
    global model_inference
    logger.info("🚀 Starting Fake News Detector API...")
    
    try:
        # Conectar a MongoDB
        await mongodb.connect()
        
        # Cargar modelo ML
        model_inference = ModelInference(model_path=settings.MODEL_PATH)
        logger.info("✅ Model loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cerrar conexiones al apagar"""
    logger.info("👋 Shutting down API...")
    await mongodb.close()

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
        "features": {
            "ml_model": "BERT (DistilBERT)",
            "database": "MongoDB",
            "caching": "Enabled"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint"""
    
    # Verificar conexión a MongoDB
    try:
        total_analyses = await mongodb.get_total_analyses_count()
        db_status = "connected"
    except:
        total_analyses = 0
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model_inference is not None,
        "model_type": "PyTorch + BERT",
        "database": db_status,
        "total_analyses": total_analyses,
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_text(request: AnalysisRequest):
    """
    Analizar texto con caché en MongoDB
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
        # ==========================================
        # 1. BUSCAR EN CACHÉ (MongoDB)
        # ==========================================
        
        cached_result = None
        if request.url:
            cached_result = await mongodb.get_analysis_by_url(request.url)
            
            if cached_result:
                # Convertir ObjectId a string y limpiar
                cached_result['_id'] = str(cached_result['_id'])
                
                logger.info(f"📦 Cache hit para URL: {request.url}")
                
                return AnalysisResponse(
                    classification=cached_result['classification'],
                    score=cached_result['score'],
                    confidence=cached_result['confidence'],
                    probabilities=cached_result['probabilities'],
                    processing_time_ms=5.0,  # Muy rápido desde caché
                    cached=True,
                    timestamp=cached_result['analyzed_at'].isoformat()
                )
        
        # ==========================================
        # 2. HACER PREDICCIÓN (no está en caché)
        # ==========================================
        
        result = model_inference.predict(request.text)
        
        processing_time = (time.time() - start_time) * 1000
        
        # ==========================================
        # 3. GUARDAR EN MONGODB
        # ==========================================
        
        # Extraer dominio de la URL
        domain = None
        if request.url:
            try:
                parsed_url = urlparse(request.url)
                domain = parsed_url.netloc.replace('www.', '')
            except:
                domain = None
        
        # Preparar documento para MongoDB
        analysis_doc = {
            'url': request.url,
            'domain': domain,
            'text_preview': request.text[:200],  # Primeros 200 chars
            'classification': result['classification'],
            'score': result['score'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'processing_time_ms': round(processing_time, 2),
            'user_id': request.user_id,
            'analyzed_at': datetime.utcnow()
        }
        
        # Guardar análisis
        await mongodb.save_analysis(analysis_doc)
        
        # Actualizar reputación del dominio
        if domain:
            is_fake = result['classification'] == 'fake'
            await mongodb.update_domain_reputation(domain, is_fake)
        
        # Actualizar estadísticas diarias
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
        await mongodb.update_daily_stats(date_str, result['classification'])
        
        # ==========================================
        # 4. PREPARAR RESPUESTA
        # ==========================================
        
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
            f"✅ Analysis completed: score={result['score']:.1f}, "
            f"classification={result['classification']}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/domain-reputation/{domain}", response_model=DomainReputationResponse, tags=["Domains"])
async def get_domain_reputation(domain: str):
    """
    Obtener reputación de un dominio
    
    Ejemplo: /domain-reputation/cnn.com
    """
    try:
        domain_data = await mongodb.get_domain_reputation(domain)
        
        if not domain_data:
            raise HTTPException(
                status_code=404,
                detail=f"Domain '{domain}' not found in database"
            )
        
        # Limpiar ObjectId
        domain_data['_id'] = str(domain_data['_id'])
        
        return DomainReputationResponse(
            domain=domain_data['domain'],
            reputation_score=domain_data['reputation_score'],
            total_analyses=domain_data['total_analyses'],
            fake_count=domain_data['fake_count'],
            real_count=domain_data['real_count'],
            fake_ratio=domain_data['fake_ratio'],
            first_seen=domain_data['first_seen'].isoformat(),
            last_analyzed=domain_data['last_analyzed'].isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting domain reputation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top-domains", tags=["Domains"])
async def get_top_domains(limit: int = 10, sort_by: str = "total"):
    """
    Obtener top dominios más analizados o mejor reputación
    
    Query params:
    - limit: número de resultados (default: 10)
    - sort_by: "total" o "reputation" (default: "total")
    """
    try:
        domains = await mongodb.get_top_domains(limit, sort_by)
        
        # Limpiar ObjectIds
        for domain in domains:
            domain['_id'] = str(domain['_id'])
        
        return {
            "total": len(domains),
            "sort_by": sort_by,
            "domains": domains
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting top domains: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics", tags=["Statistics"])
async def get_statistics(days: int = 7):
    """
    Obtener estadísticas de los últimos N días
    
    Query param:
    - days: número de días (default: 7)
    """
    try:
        stats = await mongodb.get_statistics(days)
        
        # Limpiar ObjectIds y calcular totales
        total_analyses = 0
        total_fake = 0
        total_real = 0
        total_uncertain = 0
        
        for stat in stats:
            stat['_id'] = str(stat['_id'])
            total_analyses += stat.get('total_analyses', 0)
            
            classifications = stat.get('classifications', {})
            total_fake += classifications.get('fake', 0)
            total_real += classifications.get('real', 0)
            total_uncertain += classifications.get('uncertain', 0)
        
        return {
            "period_days": days,
            "total_analyses": total_analyses,
            "summary": {
                "fake": total_fake,
                "real": total_real,
                "uncertain": total_uncertain
            },
            "daily_stats": stats
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-history/{user_id}", tags=["Users"])
async def get_user_history(user_id: str, limit: int = 50):
    """
    Obtener histórico de análisis de un usuario
    
    Path param:
    - user_id: ID del usuario
    
    Query param:
    - limit: número de resultados (default: 50, max: 100)
    """
    if limit > 100:
        limit = 100
    
    try:
        history = await mongodb.get_user_history(user_id, limit)
        
        # Limpiar ObjectIds
        for item in history:
            item['_id'] = str(item['_id'])
        
        return {
            "user_id": user_id,
            "total_results": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting user history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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