import os
from dotenv import load_dotenv
load_dotenv()
import qdrant_client
from twelvelabs import AsyncTwelveLabs, TwelveLabs
import asyncio

qdrant_client_async = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60.0,
)

twelve_labs_client = TwelveLabs(api_key=os.getenv("TWELVE_LABS_API_KEY"))

def query_twelve_labs(query_text: str, collection_name: str, top_k: int = 3):
    """Embed text query di TwelveLabs lalu cari di Qdrant"""
    try:
        query_embedding = twelve_labs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            text=query_text,
        )

        if not query_embedding.text_embedding.segments:
            raise ValueError("No embedding segments returned from TwelveLabs")

        vector = query_embedding.text_embedding.segments[0].float_

        response = qdrant_client_async.query_points(
            collection_name=collection_name,
            query=vector,
            limit=top_k,
        )

        points = response.points
        results = [p.payload for p in points]

        return results
    except Exception as e:
        print(f"Query error: {e}")
        return []

# if __name__ == "__main__":
#     query = "Sifat-sifat dari teorema graf kuratowski"
#     query_twelve_labs(query, collection_name=os.getenv("TWELVE_LABS_COLLECTION"))