import os
import json
import joblib
import asyncio
import time
import logging
from typing import List
from datetime import datetime
from contextlib import asynccontextmanager
import numpy as np

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords

from models import (
    PublicationCreate, SearchQuery, 
    MatchResult, SearchResponse, 
    StatusResponse, BatchStatusResponse,
    VectorRecord,
    PostFromSupabase,
    PostVectorResponse,
)
from supabase_service import db_manager

# --- Configuration and Constants ---
FOLDER = "data"
MODEL_VEC = f"{FOLDER}/vectorizer.joblib"
MODEL_KMEANS = f"{FOLDER}/kmeans.joblib"
FIXED_DIM = 1000

os.makedirs(FOLDER, exist_ok=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words_es = stopwords.words('spanish')

# --- Global ML Model Loading (Global State and Security) ---
global_vectorizer = None
global_kmeans = None
training_lock = asyncio.Lock() 

def load_ml_models():
    global global_vectorizer, global_kmeans
    if os.path.exists(MODEL_VEC) and os.path.exists(MODEL_KMEANS):
        try:
            global_vectorizer = joblib.load(MODEL_VEC)
            global_kmeans = joblib.load(MODEL_KMEANS)
            print("ML models loaded successfully at startup.")
        except Exception as e:
            print(f"ERROR: Failed to load ML models at startup: {e}")
            global_vectorizer = None
            global_kmeans = None
    else:
        print("WARNING: ML model files (vectorizer.joblib or kmeans.joblib) not found at startup. Please train the model first.")

async def sync_models_and_vectors():
    """
    Background task: Every 60 seconds it checks Supabase, trains and vectorizes everything.
    """
    while True:
        start_total = time.time()
        # We try to get the block. If it's already being trained, we wait for the next cycle.
        if not training_lock.locked():
            async with training_lock:
                try:
                    print(f"[{datetime.now()}] Starting sync cycle...")
                    start_fetch = time.time()
                    all_posts = db_manager.get_all_posts()
                    fetch_time = time.time() - start_fetch
                    print(f"[{datetime.now()}] Fetched {len(all_posts)} posts in {fetch_time:.2f}s")
                    
                    if len(all_posts) >= 2:
                        # 1. Train Models
                        start_train = time.time()
                        # Obtener nombres de categorías para cada post
                        corpus = []
                        for p in all_posts:
                            category_name = db_manager.get_category_name_by_id(p['product_category_id'])
                            if category_name:
                                text = f"{p['title']} {p['description']} {category_name}"
                            else:
                                text = f"{p['title']} {p['description']}"
                            corpus.append(text)
                        vec = TfidfVectorizer(stop_words=stop_words_es, max_features=FIXED_DIM, ngram_range=(1, 2))
                        matrix = vec.fit_transform(corpus)
                        
                        # Pad to fixed dimension
                        n_features = matrix.shape[1]
                        if n_features < FIXED_DIM:
                            matrix = np.pad(matrix.toarray(), ((0, 0), (0, FIXED_DIM - n_features)), mode='constant', constant_values=0)
                        else:
                            matrix = matrix.toarray()
                        
                        k = min(4, len(all_posts))
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        clusters = km.fit_predict(matrix)
                        train_time = time.time() - start_train
                        print(f"[{datetime.now()}] Trained models in {train_time:.2f}s")

                        # 2. Save to disk
                        start_save = time.time()
                        joblib.dump(vec, MODEL_VEC)
                        joblib.dump(km, MODEL_KMEANS)
                        save_time = time.time() - start_save
                        print(f"[{datetime.now()}] Saved models in {save_time:.2f}s")

                        # 3. Update vectors in Supabase (Batch)
                        start_vectorize = time.time()
                        new_vectors = []
                        matrix_array = matrix
                        for i, post in enumerate(all_posts):
                            vector_val = matrix_array[i].tolist()
                            if any(val != 0 for val in vector_val):
                                new_vectors.append({
                                    "post_id": post["id"],
                                    "vector_embedding": vector_val,
                                    "cluster_id": int(clusters[i])
                                })

                        if new_vectors:
                            # Bulk cleaning and upload
                            db_manager.client.table("post_vectors").delete().gt("post_id", 0).execute()
                            db_manager.client.table("post_vectors").insert(new_vectors).execute()
                        vectorize_time = time.time() - start_vectorize
                        print(f"[{datetime.now()}] Vectorized and updated DB in {vectorize_time:.2f}s")
                        
                        # 4. Refresh models in memory
                        load_ml_models()
                        total_time = time.time() - start_total
                        print(f"[{datetime.now()}] Sync complete: {len(new_vectors)} posts updated in {total_time:.2f}s total.")
                    else:
                        print("Not enough posts to train (min 2).")

                except Exception as e:
                    print(f"Error during background sync: {e}")
        
        await asyncio.sleep(300) # Wait for 5 minutes

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the start and close of the App."""
    load_ml_models() # Carga inicial
    task = asyncio.create_task(sync_models_and_vectors()) # Start the clock
    yield
    task.cancel() # Stops the clock when you turn it off

app = FastAPI(title="Pure Search Engine", lifespan=lifespan)

@app.get("/status", tags=["Health"])
async def get_status():
    """It reports if the engine is ready and when the last training session was.."""
    return {
        "models_loaded": global_vectorizer is not None,
        "vector_dimension": len(global_vectorizer.get_feature_names_out()) if global_vectorizer else 0,
        "training_locked": training_lock.locked()
    }

bearer_scheme = HTTPBearer()

@app.post("/buscar", response_model=SearchResponse)
async def search(query: SearchQuery, token: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Performs a similarity search in Supabase using optimized vector search by cluster."""

    # 0. Autenticar al usuario usando el token JWT de Supabase.
    try:
        # db_manager.client.auth.get_user() valida el token.
        # Si es inválido o expirado, lanzará una excepción. También devuelve los datos del usuario.
        user_response = db_manager.client.auth.get_user(token.credentials)
        user = user_response.user
        
        # Extraer nombre completo de los metadatos del usuario.
        # El .get() previene errores si 'full_name' no estuviera presente.
        full_name = user.user_metadata.get('full_name', 'Name not available')
        
        print(f"INFO: Search performed by: Name='{full_name}', Email='{user.email}'")
    except Exception:
        raise HTTPException(
            status_code=401
        )

    # 1. Verify that the models are loaded globally
    global global_vectorizer, global_kmeans
    if global_vectorizer is None or global_kmeans is None:
        raise HTTPException(status_code=500, detail="ML models not loaded. Please train the model first or restart the application if the files exist..")
    
    vectorizer = global_vectorizer
    kmeans = global_kmeans


    # 2. Generate query vector
    title_q = query.title if query.title else ""
    text_q = f"{title_q} {query.description}"
    query_embedding = vectorizer.transform([text_q]).toarray()[0]

    # Pad to fixed dimension
    if len(query_embedding) < FIXED_DIM:
        query_embedding = np.pad(query_embedding, (0, FIXED_DIM - len(query_embedding)), mode='constant', constant_values=0)

    # Verify if query_embedding has the expected dimension (FIXED_DIM)
    expected_dimension = FIXED_DIM
    if len(query_embedding) != expected_dimension:
        if len(query_embedding) == 0:
            return SearchResponse(total_found=0, matches=[], message="The query did not produce a vector with dimensions (the vectorizer did not generate dimensions). Check the model training..")
        else:
            raise HTTPException(status_code=500, detail=f"The generated query vector has an unexpected dimension ({len(query_embedding)}), it was expected {expected_dimension}. Review the model.")
    
    # Check if the query vector is a vector of zeros (and has the correct dimension)
    if all(val == 0 for val in query_embedding):
        return SearchResponse(total_found=0, matches=[], message="The query did not produce a meaningful vector (possibly outside the model's vocabulary or only stopwords).")

    # 3. Predict query cluster
    query_cluster_id = int(kmeans.predict(query_embedding.reshape(1, -1))[0])

    # 4. Call the optimized search in Supabase
    try:
        # We define a limit on the number of matches we want to search for
        match_count = 20 # You can adjust this number as needed.
        
        raw_matches = db_manager.match_posts_by_cluster(
            query_embedding_text=json.dumps(query_embedding.tolist()), # We convert the list to a JSON string
            p_cluster_id=query_cluster_id,
            p_match_threshold=query.similarity_threshold,
            p_match_count=match_count
        )
        
        if not raw_matches:
            return SearchResponse(total_found=0, matches=[])
        
        # FastAPI and Pydantic will handle the validation when returning List[MatchResult]
        matches = [MatchResult(**m) for m in raw_matches]
        
        return SearchResponse(total_found=len(matches), matches=matches)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing search in Supabase: {str(e)}")


@app.post("/publicaciones", response_model=List[PublicationCreate], tags=["Supabase"], summary="View all Supabase posts")
async def get_publications():
    """Returns the complete list of publications from Supabase. Only ID, Title, and Description are retrieved."""

    try:
        posts = db_manager.get_all_posts()
        # Mapear para incluir category_name
        publications = []
        for p in posts:
            category_name = db_manager.get_category_name_by_id(p['product_category_id']) or "Sin categoría"
            publications.append({
                "id": p["id"],
                "title": p["title"],
                "description": p["description"],
                "category_name": category_name
            })
        return publications
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving publications from Supabase: {str(e)}")

@app.get("/vectores", response_model=List[PostVectorResponse], tags=["Supabase"], summary="View all vectorized data from Supabase")
async def get_vectors():
    """Returns the numerical representations (vectors) and clusters of the publications from Supabase."""
    try:
        vectors = db_manager.get_all_post_vectors()
        if not vectors:
            return []
        
        processed_vectors = []
        for v in vectors:
            # pgvector returns the vector as a string, we need to convert it to a list
            v['vector_embedding'] = json.loads(v['vector_embedding'])
            processed_vectors.append(PostVectorResponse(**v))
        return processed_vectors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving vectors from Supabase: {str(e)}")

@app.post("/train-models", response_model=StatusResponse, tags=["ML Models"], summary="Train and save Vectorization and K-Means models",  include_in_schema=False)
async def train_models_endpoint():
    """
    Train the TfidfVectorizer and KMeans model with current Supabase publications and save them in joblib files.
    """
    try:
        all_posts = db_manager.get_all_posts()
        if not all_posts:
            raise HTTPException(status_code=400, detail="There are no publications in Supabase to train the models.")
        
        # Ensure there's enough data for KMeans
        if len(all_posts) < 2:
            raise HTTPException(status_code=400, detail="At least 2 publications are needed to train the KMeans model.")
        corpus = []
        for p in all_posts:
            category_name = db_manager.get_category_name_by_id(p['product_category_id'])
            if category_name:
                text = f"{p['title']} {p['description']} {category_name}"
            else:
                text = f"{p['title']} {p['description']}"
            corpus.append(text)

        # Initialize and train TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words=stop_words_es, max_features=FIXED_DIM, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus)

        # Pad to fixed dimension
        n_features = matrix.shape[1]
        if n_features < FIXED_DIM:
            matrix = np.pad(matrix.toarray(), ((0, 0), (0, FIXED_DIM - n_features)), mode='constant', constant_values=0)
        else:
            matrix = matrix.toarray()

        # Initialize and train KMeans
        k = min(4, len(all_posts)) # Adjust K based on data size
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(matrix)

        # Save models
        joblib.dump(vectorizer, MODEL_VEC)
        joblib.dump(kmeans, MODEL_KMEANS)
        
        # Reload global models in the running app
        load_ml_models()

        return StatusResponse(status="ok", message="ML models successfully trained and loaded.")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training or saving ML models: {str(e)}")

@app.post("/vectorize-posts", response_model=BatchStatusResponse, tags=["Supabase"], summary="Vectorize (or revectorize) all Supabase publications.",  include_in_schema=False)
async def vectorize_posts():
    """
    Vectorize all posts in Supabase and save them in the 'post_vectors' table.
    If vectors for posts already exist, replace them with the new ones.
    """
    global global_vectorizer, global_kmeans
    if global_vectorizer is None or global_kmeans is None:
        raise HTTPException(status_code=500, detail="ML models not loaded. Train the model first (POST /train-models) or restart the application if the files exist.")
    
    vectorizer = global_vectorizer
    kmeans = global_kmeans

    # Determine if it is the first vectorization for the final message
    try:
        existing_vectors_response = db_manager.client.table("post_vectors").select("count").execute()
        existing_vectors_count = existing_vectors_response.data[0]['count'] if existing_vectors_response.data else 0
        is_first_vectorization = existing_vectors_count == 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verifying existing vectors in Supabase: {str(e)}")

    try:
        # 1. Remove all existing vectors to perform a full revectorization
        # This removes everything with post_id > 0 (which should be all valid entries)
        db_manager.client.table("post_vectors").delete().gt("post_id", 0).execute() 
        
        # 2. Get all Supabase publications
        all_posts = db_manager.get_all_posts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Supabase or deleting existing vectors: {str(e)}")

    if not all_posts:
        return BatchStatusResponse(status="ok", items_processed=0, message="There are no publications in the database to vectorize.")

    # 3. Vectorize and predict clusters for all publications
    new_vectors_to_insert = []
    
    # Pre-process corpus for vectorization
    corpus = []
    for p in all_posts:
        category_name = db_manager.get_category_name_by_id(p['product_category_id'])
        if category_name:
            text = f"{p['title']} {p['description']} {category_name}"
        else:
            text = f"{p['title']} {p['description']}"
        corpus.append(text)
    
    # Handling cases where the corpus may not produce a vector (e.g., empty vocabulary)
    matrix = vectorizer.transform(corpus)
    
    # Pad to fixed dimension
    n_features = matrix.shape[1]
    if n_features < FIXED_DIM:
        matrix = np.pad(matrix.toarray(), ((0, 0), (0, FIXED_DIM - n_features)), mode='constant', constant_values=0)
    else:
        matrix = matrix.toarray()
    clusters = kmeans.predict(matrix)

    matrix_array = matrix
    for i, post in enumerate(all_posts):
        vector_embedding = matrix_array[i].tolist()
        
        # Filter out empty or zero vectors to avoid errors in pgvector and NaN
        if not vector_embedding or all(val == 0 for val in vector_embedding):
            continue # Skip this post if its vector is not valid

        new_vectors_to_insert.append({
            "post_id": post["id"],
            "vector_embedding": vector_embedding,
            "cluster_id": int(clusters[i])
        })
    
    if not new_vectors_to_insert:
         return BatchStatusResponse(status="error", items_processed=0, message="No valid vectors could be generated for any publication. Ensure there is sufficient data and the model is properly trained.")

    try:
        # 4. Batch insertion of data
        db_manager.client.table("post_vectors").insert(new_vectors_to_insert).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inserting vectors into Supabase: {str(e)}")

    message = "First vectorization successfully completed." if is_first_vectorization else "Revectorization successfully completed."
    return BatchStatusResponse(status="ok", items_processed=len(new_vectors_to_insert), message=message)
