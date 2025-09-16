from cognee import search, SearchType
import asyncio
import pathlib
from os import path
import os
from cognee import config
from cognee_community_vector_adapter_qdrant import register  # noqa: F401
from dotenv import load_dotenv
load_dotenv()

async def query_cognee(query: str, search_type = SearchType.RAG_COMPLETION):
    """
    Query the Cognee knowledge base and retrieve relevant text results.

    This function configures the Cognee system and databases (relational, vector, and graph),
    then performs a search for the input query using the specified search type.
    The results are filtered and combined into a single text output.

    Workflow:
    1. Configure Cognee system root and data storage directories.
    2. Set up database configurations:
       - Relational DB (SQLite by default).
       - Vector DB (Qdrant by default, configurable via environment variables).
       - Graph DB (provider configurable via environment variables).
    3. Execute a search on the knowledge base using the given query type.
    4. Extract valid text from the search results:
       - Append strings directly if non-empty.
       - If result is a dict, extract the value of the "text" field if present.
    5. Return the combined text results, or "text not found" if no results are available.

    Args:
        query (str): The user query string to search for.
        search_type (SearchType, optional): Type of search to perform (default is RAG_COMPLETION).

    Returns:
        str: Combined search results as a single string, or "text not found" if no results.
    """
    # Setup config Cognee
    system_path = pathlib.Path(__file__).parent
    config.system_root_directory(path.join(system_path, ".cognee_system"))
    config.data_root_directory(path.join(system_path, ".data_storage"))

    # Setup relational database
    config.set_relational_db_config(
        {
            "db_provider": "sqlite",
        }
    )
    # Setup vector database
    config.set_vector_db_config(
        {
            "vector_db_provider": os.getenv("VECTOR_DB_PROVIDER", "qdrant"),
            "vector_db_url": os.getenv("VECTOR_DB_URL"),
            "vector_db_key": os.getenv("VECTOR_DB_KEY"),
        }
    )
    # Setup graph database
    config.set_graph_db_config(
        {
            "graph_database_provider": os.getenv("GRAPH_DATABASE_PROVIDER"),
        }
    )
    try:
        # search knowledge
        results = await search(query_type=search_type, query_text=query)

        texts = []

        for result in results:
            if isinstance(result, str) and result.strip():
                # if resuls is string
                texts.append(result)
            elif isinstance(result, dict):
                # if result is dict, get value from 'text'
                text_val = result.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    texts.append(text_val)

        # concat results
        if texts:
            return "\n".join(texts)
        else:
            # fallback
            return "text not found"

    except Exception as e:
        # fallback
        return "text not found"


# if __name__ == "__main__":
#     query = "jelaskan System Interconnection"
#     # print("RAG_COMPLETION")
#     # result = asyncio.run(query_cognee(query, search_type=SearchType.RAG_COMPLETION))
#     # print("GRAPH_COMPLETION")
#     # result = asyncio.run(query_cognee(query, search_type=SearchType.GRAPH_COMPLETION))
#     # print("CHUNKS")
#     # result = asyncio.run(query_cognee(query, search_type=SearchType.CHUNKS))
#     print("SUMMARIES")
#     result = asyncio.run(query_cognee(query, search_type=SearchType.SUMMARIES))
#     print(result)