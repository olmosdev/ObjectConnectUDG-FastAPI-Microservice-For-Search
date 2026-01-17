import os
import json
import joblib
from typing import List
from fastapi import FastAPI, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

from models import (
    PublicationCreate, SearchQuery, 
    MatchResult, SearchResponse, 
    StatusResponse, BatchStatusResponse,
    VectorRecord 
)

app = FastAPI(title="Matchmaking Search Engine - Local Dev")

FOLDER = "data"
DB_PUBLICATIONS = f"{FOLDER}/publications.json"
DB_VECTORS = f"{FOLDER}/vectors.json"
MODEL_VEC = f"{FOLDER}/vectorizer.joblib"
MODEL_KMEANS = f"{FOLDER}/kmeans.joblib"

os.makedirs(FOLDER, exist_ok=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words_es = stopwords.words('spanish')

# --- HELPERS ---
def load_json(path):
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def train_and_vectorize():
    db = load_json(DB_PUBLICATIONS)
    if len(db) < 2: return
    corpus = [f"[{d['category'].upper()}] {d['title']} {d['description']}" for d in db]
    vectorizer = TfidfVectorizer(stop_words=stop_words_es, max_features=500, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    k = min(4, len(db))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(matrix)
    joblib.dump(vectorizer, MODEL_VEC)
    joblib.dump(kmeans, MODEL_KMEANS)
    vectors_list = []
    matrix_array = matrix.toarray()
    for i, doc in enumerate(db):
        vectors_list.append({
            "id": doc["id"],
            "vector": matrix_array[i].tolist(),
            "cluster": int(kmeans.labels_[i])
        })
    save_json(DB_VECTORS, vectors_list)

@app.post("/publicar", response_model=StatusResponse)
async def publish(item: PublicationCreate):
    db = load_json(DB_PUBLICATIONS)
    if any(d['id'] == item.id for d in db):
        raise HTTPException(status_code=400, detail="ID ya existente")
    db.append(item.model_dump())
    save_json(DB_PUBLICATIONS, db)
    train_and_vectorize()
    return StatusResponse(status="ok", message="Publicación integrada", processed_id=item.id)

@app.post("/publicar/lote", response_model=BatchStatusResponse)
async def publish_batch(items: List[PublicationCreate]):
    db = load_json(DB_PUBLICATIONS)
    existing_ids = {d['id'] for d in db}
    new_items = []
    for item in items:
        if item.id not in existing_ids:
            new_items.append(item.model_dump())
            existing_ids.add(item.id)
    if not new_items:
        return BatchStatusResponse(status="error", items_processed=0, message="IDs duplicados")
    db.extend(new_items)
    save_json(DB_PUBLICATIONS, db)
    train_and_vectorize()
    return BatchStatusResponse(status="ok", items_processed=len(new_items), message="Carga masiva completa")

@app.post("/buscar", response_model=SearchResponse)
async def search(query: SearchQuery):
    if not os.path.exists(MODEL_VEC):
        raise HTTPException(status_code=400, detail="Modelo no entrenado")
    vec = joblib.load(MODEL_VEC)
    db_orig = load_json(DB_PUBLICATIONS)
    db_vect = load_json(DB_VECTORS)
    title_q = query.title if query.title else ""
    text_q = f"[{query.category.upper()}] {title_q} {query.description}"
    v_query = vec.transform([text_q]).toarray()
    matches = []
    for record in db_vect:
        sim = cosine_similarity(v_query, [record["vector"]])[0][0]
        if sim >= query.similarity_threshold:
            orig = next(d for d in db_orig if d["id"] == record["id"])
            matches.append(MatchResult(**orig, similarity=round(float(sim), 4)))
    matches.sort(key=lambda x: x.similarity, reverse=True)
    return SearchResponse(total_found=len(matches), matches=matches)

@app.get("/publicaciones", response_model=List[PublicationCreate], summary="Ver todos los datos originales")
async def get_publications():
    """Retorna la lista completa de objetos perdidos en formato legible."""
    return load_json(DB_PUBLICATIONS)

@app.get("/vectores", response_model=List[VectorRecord], summary="Ver todos los datos vectorizados")
async def get_vectors():
    """Retorna la representación numérica (vectores) y clústeres de los datos."""
    return load_json(DB_VECTORS)