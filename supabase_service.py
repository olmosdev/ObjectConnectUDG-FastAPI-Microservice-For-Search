from supabase import create_client, Client
from config import settings, get_logger

logger = get_logger(__name__)

class SupabaseManager:
    """
    Manages all direct interactions with Supabase database.
    """
    def __init__(self):
        url = settings.SUPABASE_URL
        key = settings.SUPABASE_KEY
        if not url or not key:
            logger.error("Faltan las credenciales de Supabase en el archivo .env")
            raise ValueError("Faltan las credenciales en el archivo .env")
        self.client: Client = create_client(url, key)
        logger.info("Conexión con Supabase establecida.")

    def get_all_posts(self):
        """
        Fetches all records from 'posts' table with their category name in a single join.
        """
        response = self.client.table("posts").select("id, title, description, product_category_id, categories(name)").execute()
        return response.data

    def get_all_post_vector_ids(self):
        """
        Fetches all post IDs already processed in the 'post_vectors' table.
        """
        response = self.client.table("post_vectors").select("post_id").execute()
        # Returns a set of IDs for O(1) fast lookup
        return {item['post_id'] for item in response.data}

    def upsert_post_vectors(self, vectors_data: list):
        """
        Perform a bulk upsert of vectors. 
        Updates if post_id exists, otherwise inserts a new record.
        """
        if not vectors_data:
            return None
        # Requires a unique constraint (usually post_id) on the table
        response = self.client.table("post_vectors").upsert(
            vectors_data, 
            on_conflict="post_id"
        ).execute()
        return response.data

    def delete_orphaned_vectors(self, valid_post_ids: list):
        """
        Removes vectors for posts that no longer exist in the main 'posts' table.
        """
        if not valid_post_ids:
            return None
        response = self.client.table("post_vectors").delete().not_.in_("post_id", valid_post_ids).execute()
        return response.data

    def get_all_post_vectors(self):
        """
        Fetches all records from the 'post_vectors' table.
        """
        response = self.client.table("post_vectors").select("*").execute()
        return response.data

    def insert_post(self, post_data: dict):
        """
        Inserts a new post record into the 'posts' table.
        """
        response = self.client.table("posts").insert(post_data).execute()
        return response.data

    def get_category_name_by_id(self, category_id: int):
        """
        Retrieves a category name by its ID.
        """
        if category_id is None:
            return None
        response = self.client.table("categories").select("name").eq("id", category_id).execute()
        return response.data[0]['name'] if response.data else None

    def match_posts_by_cluster(self, query_embedding_text: str, p_cluster_id: int, p_match_threshold: float, p_match_count: int):
        """
        Calls the Supabase RPC function 'match_posts_by_cluster' for similarity search
        within a specific cluster.
        """
        response = self.client.rpc(
            'match_posts_by_cluster',
            {
                'query_embedding_text': query_embedding_text,
                'p_cluster_id': p_cluster_id,
                'p_match_threshold': p_match_threshold,
                'p_match_count': p_match_count
            }
        ).execute()
        return response.data

# Single shared instance for the entire project
db_manager = SupabaseManager()
