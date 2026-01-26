import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseManager:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_APIKEY_SERVICE_ROLE")
        if not url or not key:
            raise ValueError("Faltan las credenciales en el archivo .env")
        self.client: Client = create_client(url, key)

    def get_all_posts(self):
        """Trae todos los registros de la tabla 'posts' necesarios para la vectorización."""
        response = self.client.table("posts").select("id, title, description").execute()
        return response.data

    def get_all_post_vector_ids(self):
        """Trae todos los IDs de los vectores ya procesados en 'post_vectors'."""
        response = self.client.table("post_vectors").select("post_id").execute()
        # Devuelve un set de IDs para una búsqueda rápida O(1)
        return {item['post_id'] for item in response.data}

    def insert_post_vector(self, post_id: int, vector: list, cluster_id: int):
        """Inserta un nuevo registro de vector en la tabla 'post_vectors'."""
        response = self.client.table("post_vectors").insert({
            "post_id": post_id,
            "vector_embedding": vector,
            "cluster_id": cluster_id
        }).execute()
        return response.data

    def get_all_post_vectors(self):
        """Trae todos los registros de la tabla 'post_vectors'."""
        response = self.client.table("post_vectors").select("*").execute()
        return response.data

    def insert_post(self, post_data: dict):
        """Inserta una nueva publicación en la tabla 'posts'."""
        response = self.client.table("posts").insert(post_data).execute()
        return response.data

    def match_posts_by_cluster(self, query_embedding_text: str, p_cluster_id: int, p_match_threshold: float, p_match_count: int):
        """
        Llama a la función RPC de Supabase 'match_posts_by_cluster' para realizar una búsqueda
        de similitud dentro de un clúster específico.
        """
        response = self.client.rpc(
            'match_posts_by_cluster',
            {
                'query_embedding_text': query_embedding_text, # Changed key here
                'p_cluster_id': p_cluster_id,
                'p_match_threshold': p_match_threshold,
                'p_match_count': p_match_count
            }
        ).execute()
        return response.data

# Instancia única para usar en todo el proyecto
db_manager = SupabaseManager()