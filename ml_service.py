import os
import joblib
import asyncio
import time
import numpy as np
import re
import unicodedata
from datetime import datetime
from sklearn.cluster import KMeans

from supabase_service import db_manager
from config import settings, get_logger

logger = get_logger(__name__)

# --- Resource Setup ---
os.makedirs(settings.MODEL_FOLDER, exist_ok=True)
MODEL_KMEANS = f"{settings.MODEL_FOLDER}/kmeans.joblib"

class MLService:
    def __init__(self):
        # The SBERT model is pre-trained and does not need local training
        self.model = None 
        self.kmeans = None
        self.training_lock = asyncio.Lock()

    def clean_text(self, text: str) -> str:
        """
        Normalize text for better semantic understanding.
        """
        if not text:
            return ""
        text = text.lower()
        # Remove accents
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        # Keep only alphanumeric and basic punctuation for context
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        return text.strip()

    def load_models(self):
        """
        Load SBERT model and KMeans from disk/cache.
        """
        try:
            # Lazy loading of sentence-transformers
            from sentence_transformers import SentenceTransformer
            import torch

            # Force CPU usage to avoid CUDA errors in Fly.io
            device = "cpu"
            logger.info(f"Usando dispositivo: {device}")

            # 1. Load Sentence Transformer (from cache or internet)
            logger.info(f"Cargando modelo semántico: {settings.SBERT_MODEL_NAME}...")
            self.model = SentenceTransformer(settings.SBERT_MODEL_NAME, device=device)
            
            # 2. Load KMeans if exists
            if os.path.exists(MODEL_KMEANS):
                self.kmeans = joblib.load(MODEL_KMEANS)
                logger.info("Modelo KMeans cargado exitosamente.")
            else:
                logger.warning("No se encontró el modelo KMeans. Se requiere sincronización inicial.")
            
            return True
        except Exception as e:
            logger.error(f"Error crítico al cargar modelos de ML: {e}")
            return False

    def _train_logic(self, all_posts):
        """
        Internal logic: Generate Semantic Embeddings and Train KMeans.
        """
        # 1. Prepare Semantic Corpus
        corpus = []
        for p in all_posts:
            categories = p.get('categories')
            category_name = categories.get('name') if isinstance(categories, dict) else ""
            raw_text = f"{p.get('title', '')}. {p.get('description', '')}. Categoría: {category_name}"
            corpus.append(self.clean_text(raw_text))
        
        # 2. Generate Semantic Embeddings (Deep Learning)
        logger.info(f"Generando vectores semánticos para {len(corpus)} documentos...")
        embeddings = self.model.encode(corpus, show_progress_bar=False)
        
        # 3. Dynamic K Calculation
        unique_categories = {p.get('product_category_id') for p in all_posts if p.get('product_category_id') is not None}
        num_categories = len(unique_categories)
        k = max(2, min(num_categories, len(all_posts)))
        
        logger.info(f"SINCRO ML: Detectadas {num_categories} categorías. Entrenando KMeans con K={k}.")
        
        # 4. Train KMeans on Semantic Vectors
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(embeddings)

        # 5. Save KMeans to disk
        joblib.dump(km, MODEL_KMEANS)
        
        return embeddings, clusters

    async def sync_all(self):
        """
        Perform Semantic Sync: Fetch -> Embed -> Cluster -> Upsert.
        """
        if self.training_lock.locked():
            logger.warning("Sincronización semántica en progreso. Omitiendo ciclo.")
            return False

        async with self.training_lock:
            try:
                logger.info("Iniciando ciclo de sincronización SEMÁNTICA...")
                start_time = time.time()
                
                all_posts = db_manager.get_all_posts()
                if len(all_posts) < 2:
                    logger.warning("Posts insuficientes para sincronización.")
                    return False

                # Semantic Encoding and Clustering (Threaded)
                loop = asyncio.get_event_loop()
                embeddings, clusters = await loop.run_in_executor(None, self._train_logic, all_posts)

                # Prepare 384-dim Vectors for Supabase
                new_vectors = []
                valid_post_ids = [post["id"] for post in all_posts]
                
                for i, post in enumerate(all_posts):
                    vector_val = embeddings[i].tolist()
                    new_vectors.append({
                        "post_id": post["id"],
                        "vector_embedding": vector_val,
                        "cluster_id": int(clusters[i])
                    })

                if new_vectors:
                    db_manager.upsert_post_vectors(new_vectors)
                    db_manager.delete_orphaned_vectors(valid_post_ids)
                
                # Refresh KMeans in memory
                if os.path.exists(MODEL_KMEANS):
                    self.kmeans = joblib.load(MODEL_KMEANS)
                
                duration = time.time() - start_time
                logger.info(f"Sincronización semántica (SBERT) completada en {duration:.2f}s.")
                return True

            except Exception as e:
                logger.exception(f"Error en sincronización semántica: {e}")
                return False

# Single instance of the Semantic Service
ml_service = MLService()
