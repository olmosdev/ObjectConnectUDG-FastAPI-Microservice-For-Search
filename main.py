import json
import asyncio
from typing import List
from contextlib import asynccontextmanager
import numpy as np

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from models import (
    PublicationCreate, SearchQuery, 
    MatchResult, SearchResponse, 
    StatusResponse
)
from supabase_service import db_manager
from ml_service import ml_service
from config import settings, get_logger

logger = get_logger(__name__)

async def sync_background_task():
    """Tarea de fondo para sincronización semántica periódica."""
    while True:
        await ml_service.sync_all()
        await asyncio.sleep(settings.SYNC_INTERVAL_SECONDS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Maneja el inicio y cierre del servicio semántico."""
    logger.info("Iniciando microservicio de búsqueda SEMÁNTICA (SBERT)...")
    ml_service.load_models()
    task = asyncio.create_task(sync_background_task())
    yield
    logger.info("Cerrando microservicio semántico...")
    task.cancel()

app = FastAPI(
    title="Pure Search Engine v2 (Semantic)", 
    description="Motor de búsqueda semántica usando Sentence Transformers y K-Means",
    lifespan=lifespan
)

@app.get("/status", tags=["Health"])
async def get_status():
    """Informa el estado de los modelos semánticos."""
    return {
        "semantic_model": settings.SBERT_MODEL_NAME,
        "models_loaded": ml_service.model is not None,
        "vector_dimension": settings.FIXED_DIM,
        "training_locked": ml_service.training_lock.locked()
    }

bearer_scheme = HTTPBearer()

@app.post("/buscar", response_model=SearchResponse)
async def search(query: SearchQuery, token: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """ Realiza una búsqueda semántica (basada en significado). """

    # 0. Autenticación JWT de Supabase
    try:
        user_response = db_manager.client.auth.get_user(token.credentials)
        user = user_response.user
        logger.info(f"Búsqueda semántica solicitada por: {user.email}")
    except Exception:
        logger.warning("Intento de búsqueda con token inválido.")
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    # 1. Verificar modelos
    if ml_service.model is None or ml_service.kmeans is None:
        logger.error("Error de búsqueda: Modelos semánticos no cargados.")
        raise HTTPException(status_code=500, detail="Semantic models not loaded. Please wait for initial sync.")
    
    # 2. Generar vector de consulta semántico (Embedding)
    title_q = query.title if query.title else ""
    text_q = ml_service.clean_text(f"{title_q} {query.description}")
    
    try:
        # Generar embedding de 384 dimensiones
        query_embedding = ml_service.model.encode(text_q)
        
        # 3. Predecir clúster semántico
        query_cluster_id = int(ml_service.kmeans.predict(query_embedding.reshape(1, -1))[0])

        # 4. Búsqueda por similitud de coseno en Supabase
        raw_matches = db_manager.match_posts_by_cluster(
            query_embedding_text=json.dumps(query_embedding.tolist()),
            p_cluster_id=query_cluster_id,
            p_match_threshold=query.similarity_threshold,
            p_match_count=20
        )
        
        matches = [MatchResult(**m) for m in (raw_matches or [])]
        logger.info(f"Búsqueda semántica exitosa. Encontrados: {len(matches)} matches en clúster {query_cluster_id}")
        return SearchResponse(total_found=len(matches), matches=matches)
        
    except Exception as e:
        logger.exception("Error durante el procesamiento de la búsqueda semántica.")
        raise HTTPException(status_code=500, detail="Error interno durante la búsqueda.")

@app.post("/publicaciones", response_model=List[PublicationCreate], tags=["Supabase"])
async def get_publications():
    """Retorna todas las publicaciones de Supabase."""
    try:
        posts = db_manager.get_all_posts()
        results = []
        for p in posts:
            categories = p.get('categories')
            category_name = categories.get('name') if isinstance(categories, dict) else "Sin categoría"
            results.append({
                "id": p["id"],
                "title": p["title"],
                "description": p["description"],
                "category_name": category_name
            })
        return results
    except Exception as e:
        logger.error(f"Error al obtener publicaciones: {e}")
        raise HTTPException(status_code=500, detail="Error al conectar con la base de datos.")

@app.post("/train-models", response_model=StatusResponse, tags=["ML Models"], include_in_schema=False)
async def train_models_endpoint():
    """Sincronización semántica manual."""
    logger.info("Sincronización semántica manual solicitada.")
    success = await ml_service.sync_all()
    if success:
        return StatusResponse(status="ok", message="Sincronización semántica completada.")
    raise HTTPException(status_code=500, detail="La sincronización falló.")
