import os
import json
import joblib
import asyncio
import time
import numpy as np
import re
import unicodedata
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords

from supabase_service import db_manager
from config import settings, get_logger

logger = get_logger(__name__)

# --- Resource Setup ---
os.makedirs(settings.MODEL_FOLDER, exist_ok=True)
MODEL_VEC = f"{settings.MODEL_FOLDER}/vectorizer.joblib"
MODEL_KMEANS = f"{settings.MODEL_FOLDER}/kmeans.joblib"

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Descargando stopwords de NLTK...")
    nltk.download('stopwords')
stop_words_es = stopwords.words('spanish')

class MLService:
    def __init__(self):
        self.vectorizer = None
        self.kmeans = None
        self.training_lock = asyncio.Lock()

    def clean_text(self, text: str) -> str:
        """
        Normalize text: lowercase, remove accents, and strip special characters.
        """
        if not text:
            return ""
        # To lowercase
        text = text.lower()
        # Remove accents
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    def load_models(self):
        """
        Load ML models from disk into memory.
        """
        if os.path.exists(MODEL_VEC) and os.path.exists(MODEL_KMEANS):
            try:
                self.vectorizer = joblib.load(MODEL_VEC)
                self.kmeans = joblib.load(MODEL_KMEANS)
                logger.info("Modelos de ML cargados exitosamente desde disco.")
                return True
            except Exception as e:
                logger.error(f"Error al cargar modelos de ML: {e}")
        else:
            logger.warning("No se encontraron archivos de modelo. Se requiere entrenamiento inicial.")
        return False

    def _train_logic(self, all_posts):
        """
        Internal training logic (Synchronous for CPU-bound tasks).
        """
        corpus = []
        for p in all_posts:
            # Safe handling of categories relationship (can be None)
            categories = p.get('categories')
            category_name = categories.get('name') if isinstance(categories, dict) else None
            
            raw_text = f"{p.get('title', '')} {p.get('description', '')} {category_name if category_name else ''}"
            corpus.append(self.clean_text(raw_text))
        
        # Calculate K dynamically based on unique categories
        unique_categories = {p.get('product_category_id') for p in all_posts if p.get('product_category_id') is not None}
        num_categories = len(unique_categories)
        
        # K will be the number of categories, minimum 2 and maximum the number of posts
        k = max(2, min(num_categories, len(all_posts)))
        logger.info(f"SINCRO ML: Se detectaron {num_categories} categorías únicas en Supabase. Configurando K={k} clústeres para el modelo KMeans.")
        logger.info(f"Entrenando modelos con un corpus de {len(corpus)} documentos.")
        
        # 1. Train Models
        vec = TfidfVectorizer(stop_words=stop_words_es, max_features=settings.FIXED_DIM, ngram_range=(1, 2))
        matrix = vec.fit_transform(corpus)
        
        n_features = matrix.shape[1]
        matrix_array = matrix.toarray()
        if n_features < settings.FIXED_DIM:
            matrix_array = np.pad(matrix_array, ((0, 0), (0, settings.FIXED_DIM - n_features)), mode='constant', constant_values=0)
        
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(matrix_array)

        # 2. Save models to disk
        joblib.dump(vec, MODEL_VEC)
        joblib.dump(km, MODEL_KMEANS)
        
        return matrix_array, clusters

    async def sync_all(self):
        """
        Perform the full sync cycle: Fetch -> Train -> Upsert -> Reload.
        """
        if self.training_lock.locked():
            logger.warning("Ya hay una sincronización en progreso. Omitiendo este ciclo.")
            return False

        async with self.training_lock:
            try:
                logger.info("Iniciando ciclo de sincronización completa...")
                start_time = time.time()
                
                # Fetch all posts from Supabase
                all_posts = db_manager.get_all_posts()
                if len(all_posts) < 2:
                    logger.warning("No hay suficientes posts para entrenar (mínimo 2).")
                    return False

                # Execute heavy training in an executor to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                matrix, clusters = await loop.run_in_executor(None, self._train_logic, all_posts)

                # Prepare vector data for Supabase
                new_vectors = []
                valid_post_ids = [post["id"] for post in all_posts]
                
                for i, post in enumerate(all_posts):
                    vector_val = matrix[i].tolist()
                    if any(val != 0 for val in vector_val):
                        new_vectors.append({
                            "post_id": post["id"],
                            "vector_embedding": vector_val,
                            "cluster_id": int(clusters[i])
                        })

                # Perform Atomic Upsert and cleanup
                if new_vectors:
                    db_manager.upsert_post_vectors(new_vectors)
                    db_manager.delete_orphaned_vectors(valid_post_ids)
                
                # Reload global models in memory
                self.load_models()
                duration = time.time() - start_time
                logger.info(f"Sincronización completada exitosamente en {duration:.2f}s.")
                return True

            except Exception as e:
                logger.exception(f"Error crítico durante la sincronización de ML: {e}")
                return False

# Single instance of the service
ml_service = MLService()
