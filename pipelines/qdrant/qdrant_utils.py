from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import os
from dotenv import load_dotenv
load_dotenv()

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60.0
)

def create_collection_if_not_exists(collection_name: str, vector_size: int = 1024):
    collections = qdrant.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if not exists:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' dibuat")
    else:
        print(f"Collection '{collection_name}' sudah ada")

    return qdrant
