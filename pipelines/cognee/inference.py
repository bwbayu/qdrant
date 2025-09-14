from cognee import search, SearchType
import asyncio
import pathlib
from os import path
import os
from cognee import config
from cognee_community_vector_adapter_qdrant import register  # noqa: F401

async def query_cognee(query: str, search_type = SearchType.RAG_COMPLETION):
    # Setup config Cognee
    system_path = pathlib.Path(__file__).parent
    config.system_root_directory(path.join(system_path, ".cognee_system"))
    config.data_root_directory(path.join(system_path, ".data_storage"))

    # Setup database
    config.set_relational_db_config(
        {
            "db_provider": "sqlite",
        }
    )
    config.set_vector_db_config(
        {
            "vector_db_provider": os.getenv("VECTOR_DB_PROVIDER", "qdrant"),
            "vector_db_url": os.getenv("VECTOR_DB_URL"),
            "vector_db_key": os.getenv("VECTOR_DB_KEY"),
        }
    )
    config.set_graph_db_config(
        {
            "graph_database_provider": os.getenv("GRAPH_DATABASE_PROVIDER"),
        }
    )
    # search knowledge
    results = await search(query_type=search_type, query_text=query)

    print("\nSearch Results:")
    for result in results:
        print(result)

    return result

if __name__ == "__main__":
    query = "siapa dosen pengajar sistem informasi"
    asyncio.run(query_cognee(query))