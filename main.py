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
    """
    Background task that periodically synchronizes ML models and vectors.
    """
    while True:
        await ml_service.sync_all()
        await asyncio.sleep(settings.SYNC_INTERVAL_SECONDS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown lifecycle.
    """
    logger.info("Iniciando microservicio de búsqueda...")
    # Load initial models from disk
    ml_service.load_models()
    # Start the background synchronization task
    task = asyncio.create_task(sync_background_task())
    yield
    logger.info("Cerrando microservicio...")
    task.cancel()

app = FastAPI(
    title="Pure Search Engine", 
    description="Vectorized similarity search engine using K-Means and TF-IDF",
    lifespan=lifespan
)

@app.get("/status", tags=["Health"])
async def get_status():
    """
    Returns engine readiness and current training status.
    """
    return {
        "models_loaded": ml_service.vectorizer is not None,
        "vector_dimension": len(ml_service.vectorizer.get_feature_names_out()) if ml_service.vectorizer else 0,
        "training_locked": ml_service.training_lock.locked(),
        "sync_interval": settings.SYNC_INTERVAL_SECONDS
    }

bearer_scheme = HTTPBearer()

@app.post("/buscar", response_model=SearchResponse)
async def search(query: SearchQuery, token: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    Performs optimized similarity search using cluster-based vector matching.
    """

    # 0. Supabase JWT Authentication
    try:
        user_response = db_manager.client.auth.get_user(token.credentials)
        user = user_response.user
        logger.info(f"Búsqueda solicitada por: {user.email}")
    except Exception:
        logger.warning("Intento de búsqueda con token inválido.")
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    # 1. Verify that ML models are loaded
    if ml_service.vectorizer is None or ml_service.kmeans is None:
        logger.error("Error de búsqueda: Modelos no cargados.")
        raise HTTPException(status_code=500, detail="ML models not loaded. Please train the model first.")
    
    # 2. Generate query vector (with text normalization)
    title_q = query.title if query.title else ""
    text_q = ml_service.clean_text(f"{title_q} {query.description}")
    
    try:
        query_embedding = ml_service.vectorizer.transform([text_q]).toarray()[0]
        
        # Pad to fixed dimension if necessary
        if len(query_embedding) < settings.FIXED_DIM:
            query_embedding = np.pad(query_embedding, (0, settings.FIXED_DIM - len(query_embedding)), mode='constant', constant_values=0)

        # Handle cases where the query produces no known terms
        if all(val == 0 for val in query_embedding):
            logger.info(f"Consulta vacía después de limpieza: '{text_q}'")
            return SearchResponse(total_found=0, matches=[], message="La consulta no produjo términos conocidos por el modelo.")

        # 3. Predict query cluster
        query_cluster_id = int(ml_service.kmeans.predict(query_embedding.reshape(1, -1))[0])

        # 4. Execute vector search in Supabase
        raw_matches = db_manager.match_posts_by_cluster(
            query_embedding_text=json.dumps(query_embedding.tolist()),
            p_cluster_id=query_cluster_id,
            p_match_threshold=query.similarity_threshold,
            p_match_count=20
        )
        
        matches = [MatchResult(**m) for m in (raw_matches or [])]
        logger.info(f"Búsqueda exitosa. Encontrados: {len(matches)} matches en clúster {query_cluster_id}")
        return SearchResponse(total_found=len(matches), matches=matches)
        
    except Exception as e:
        logger.exception("Error durante el procesamiento de la búsqueda.")
        raise HTTPException(status_code=500, detail="Error interno durante la búsqueda.")

@app.post("/publicaciones", response_model=List[PublicationCreate], tags=["Supabase"])
async def get_publications():
    """
    Retrieves all publications from Supabase with their associated category names.
    """
    try:
        posts = db_manager.get_all_posts()
        results = []
        for p in posts:
            # Secure handling of category relationship
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
        raise HTTPException(status_code=500, detail="Error interno al procesar las publicaciones.")

@app.post("/train-models", response_model=StatusResponse, tags=["ML Models"], include_in_schema=False)
async def train_models_endpoint():
    """
    Manually triggers ML model training and vectorization.
    """
    logger.info("Entrenamiento manual solicitado vía API.")
    success = await ml_service.sync_all()
    if success:
        return StatusResponse(status="ok", message="Modelos entrenados y cargados exitosamente.")
    raise HTTPException(status_code=500, detail="El entrenamiento falló o ya hay uno en progreso.")
