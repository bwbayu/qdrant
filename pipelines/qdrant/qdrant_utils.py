from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

QDRANT_URL = "https://caf7b79d-b105-4640-ad40-97a0ab12faf3.us-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.5Oia1gdKMwHBu-OL0QJkYZ9lZGrsUa6urzva2KU4PUw"

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
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
