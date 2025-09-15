import os
from dotenv import load_dotenv
load_dotenv()
import qdrant_client
from twelvelabs import AsyncTwelveLabs, TwelveLabs
import asyncio
# Initialize an synchronous Qdrant client for database operations.
qdrant_client_async = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_URL2"),
    api_key=os.getenv("QDRANT_API_KEY2"),
    timeout=60.0,
)

# Initialize an synchronous Twelve Labs client
twelve_labs_client = TwelveLabs(api_key=os.getenv("TWELVE_LABS_API_KEY"))

def query_twelve_labs(query_text: str, collection_name: str, top_k: int = 3):
    """
    Asynchronously generates a text embedding with TwelveLabs and uses it to
    query a Qdrant collection for similar video segments.

    Args:
        query_text (str): The text to search for.
        collection_name (str): The name of the Qdrant collection to search in.
        top_k (int): The maximum number of results to return.

    Returns:
        A list of search results from Qdrant, or an empty list if an error occurs.
    """
    try:
        query_embedding = twelve_labs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            text=query_text,
        )

        # Validate that the API returned a valid embedding.
        if not query_embedding.text_embedding.segments:
            raise ValueError("No embedding segments returned from TwelveLabs")

        # Extract the numerical vector from the first segment of the response.
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
