import os
import asyncio
import pathlib
from os import path
from dotenv import load_dotenv
load_dotenv()
from cognee_community_vector_adapter_qdrant import register  # noqa: F401
from cognee import config, prune
from create_knowledge import create_knowledge

async def pipeline_cognee(reset_data=False):
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

    # clear data
    if reset_data == True:
        await prune.prune_data()
        await prune.prune_system(metadata=True)

    # create knowledge based on file at data/raw/logif
    await create_knowledge()

if __name__ == "__main__":
    # create knowledge
    asyncio.run(pipeline_cognee(reset_data=True))