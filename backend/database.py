# backend/database.py 

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    """Gestor de conexión a MongoDB"""
    
    def __init__(self):
        self.client = None
        self.db = None
        
    async def connect(self):
        """Conectar a MongoDB"""
        # Obtener URL de conexión desde variable de entorno
        mongo_url = os.getenv(
            'MONGODB_URL', 
            'mongodb://localhost:27017'  # Default local
        )
        
        try:
            self.client = AsyncIOMotorClient(mongo_url)
            self.db = self.client.fakenews_db  # Nombre de la base de datos
            
            # Verificar conexión
            await self.client.admin.command('ping')
            logger.info(" Conectado a MongoDB exitosamente")
            
            # Crear índices para optimizar búsquedas
            await self.create_indexes()
            
        except Exception as e:
            logger.error(f"❌ Error conectando a MongoDB: {e}")
            raise
    
    async def create_indexes(self):
        """Crear índices en las colecciones"""
        
        # Colección: analyses (análisis realizados)
        await self.db.analyses.create_index([("url", ASCENDING)])
        await self.db.analyses.create_index([("analyzed_at", DESCENDING)])
        await self.db.analyses.create_index([("user_id", ASCENDING)])
        await self.db.analyses.create_index([("domain", ASCENDING)])
        
        # Colección: domains (reputación de dominios)
        await self.db.domains.create_index([("domain", ASCENDING)], unique=True)
        await self.db.domains.create_index([("reputation_score", DESCENDING)])
        
        # Colección: statistics (estadísticas del sistema)
        await self.db.statistics.create_index([("date", DESCENDING)])
        
        logger.info(" Índices de MongoDB creados")
    
    async def close(self):
        """Cerrar conexión"""
        if self.client:
            self.client.close()
            logger.info("👋 Conexión a MongoDB cerrada")
    
    # ==========================================
    # OPERACIONES: Análisis
    # ==========================================
    
    async def save_analysis(self, analysis_data: dict):
        """
        Guardar un análisis en la base de datos
        
        Args:
            analysis_data: {
                'url': str,
                'domain': str,
                'text_preview': str (primeros 200 chars),
                'classification': str,
                'score': float,
                'confidence': float,
                'probabilities': dict,
                'processing_time_ms': float,
                'user_id': str (opcional),
                'analyzed_at': datetime
            }
        """
        try:
            result = await self.db.analyses.insert_one(analysis_data)
            logger.info(f" Análisis guardado con ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f" Error guardando análisis: {e}")
            return None
    
    async def get_analysis_by_url(self, url: str):
        """Obtener análisis previo de una URL"""
        return await self.db.analyses.find_one(
            {"url": url},
            sort=[("analyzed_at", DESCENDING)]
        )
    
    async def get_user_history(self, user_id: str, limit: int = 50):
        """Obtener histórico de análisis de un usuario"""
        cursor = self.db.analyses.find(
            {"user_id": user_id}
        ).sort("analyzed_at", DESCENDING).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def get_total_analyses_count(self):
        """Obtener total de análisis realizados"""
        return await self.db.analyses.count_documents({})
    
    # ==========================================
    # OPERACIONES: Reputación de Dominios
    # ==========================================
    
    async def update_domain_reputation(self, domain: str, is_fake: bool):
        """
        Actualizar score de reputación de un dominio
        
        Args:
            domain: nombre del dominio (ej: "cnn.com")
            is_fake: True si el análisis determinó fake, False si real
        """
        # Buscar dominio existente
        existing = await self.db.domains.find_one({"domain": domain})
        
        if existing:
            # Actualizar contadores
            new_total = existing['total_analyses'] + 1
            new_fake_count = existing['fake_count'] + (1 if is_fake else 0)
            new_fake_ratio = new_fake_count / new_total
            
            # Calcular nuevo reputation_score (0-100)
            # Score alto = confiable, score bajo = sospechoso
            reputation_score = (1 - new_fake_ratio) * 100
            
            await self.db.domains.update_one(
                {"domain": domain},
                {
                    "$set": {
                        "total_analyses": new_total,
                        "fake_count": new_fake_count,
                        "real_count": new_total - new_fake_count,
                        "fake_ratio": round(new_fake_ratio, 4),
                        "reputation_score": round(reputation_score, 2),
                        "last_analyzed": datetime.utcnow()
                    }
                }
            )
        else:
            # Crear nuevo registro de dominio
            await self.db.domains.insert_one({
                "domain": domain,
                "total_analyses": 1,
                "fake_count": 1 if is_fake else 0,
                "real_count": 0 if is_fake else 1,
                "fake_ratio": 1.0 if is_fake else 0.0,
                "reputation_score": 0.0 if is_fake else 100.0,
                "first_seen": datetime.utcnow(),
                "last_analyzed": datetime.utcnow()
            })
        
        logger.info(f" Reputación actualizada para {domain}")
    
    async def get_domain_reputation(self, domain: str):
        """Obtener información de reputación de un dominio"""
        return await self.db.domains.find_one({"domain": domain})
    
    async def get_top_domains(self, limit: int = 10, sort_by: str = "total"):
        """
        Obtener top dominios
        
        Args:
            limit: número de resultados
            sort_by: "total" (más analizados), "reputation" (mejor reputación)
        """
        if sort_by == "reputation":
            sort_field = [("reputation_score", DESCENDING)]
        else:
            sort_field = [("total_analyses", DESCENDING)]
        
        cursor = self.db.domains.find().sort(sort_field).limit(limit)
        return await cursor.to_list(length=limit)
    
    # ==========================================
    # OPERACIONES: Estadísticas
    # ==========================================
    
    async def update_daily_stats(self, date_str: str, classification: str):
        """
        Actualizar estadísticas diarias
        
        Args:
            date_str: fecha en formato "YYYY-MM-DD"
            classification: "fake", "real" o "uncertain"
        """
        await self.db.statistics.update_one(
            {"date": date_str},
            {
                "$inc": {
                    f"classifications.{classification}": 1,
                    "total_analyses": 1
                },
                "$set": {
                    "last_updated": datetime.utcnow()
                }
            },
            upsert=True  # Crear si no existe
        )
    
    async def get_statistics(self, days: int = 7):
        """Obtener estadísticas de los últimos N días"""
        cursor = self.db.statistics.find().sort(
            "date", DESCENDING
        ).limit(days)
        
        return await cursor.to_list(length=days)

# Instancia global
mongodb = MongoDB()