import os
import asyncio
import pathlib
from os import path
from dotenv import load_dotenv
load_dotenv()
from cognee_community_vector_adapter_qdrant import register  # noqa: F401
from cognee import config, prune
from pipelines.cognee.create_knowledge import create_knowledge

async def pipeline_cognee(raw_dir, clean_dir, reset_data=False, upload_metadata=False ,job_id=None):
    """
    Run the Cognee pipeline to process documents and create knowledge.

    This function configures the Cognee system (system, data, and database paths),
    optionally resets existing data, and then calls `create_knowledge` to process
    PDF files into structured knowledge.

    Workflow:
    1. Configure Cognee system root and data storage directories.
    2. Set up database configurations:
       - Relational DB (SQLite by default).
       - Vector DB (Qdrant by default, configurable via environment variables).
       - Graph DB (provider configurable via environment variables).
    3. If `reset_data=True`, clear stored data and metadata.
    4. Call `create_knowledge` to process PDFs from `raw_dir` into `clean_dir`.
       - Optionally uploads metadata to GCS if `upload_metadata=True`.

    Args:
        raw_dir (str): Directory containing raw PDF documents.
        clean_dir (str): Directory where processed metadata will be stored.
        reset_data (bool, optional): If True, wipes previous data and metadata before processing.
        upload_metadata (bool, optional): If True, uploads generated metadata JSON to GCS.
        job_id (str, optional): Job identifier for organizing metadata uploads in GCS.

    Returns:
        None
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
            "vector_db_key": os.getenv("VECTOR_DB_KEY", ""),
        }
    )

    # Setup graph database
    config.set_graph_db_config(
        {
            "graph_database_provider": os.getenv("GRAPH_DATABASE_PROVIDER"),
        }
    )

    # clear data
    if reset_data == True:
        await prune.prune_data()
        await prune.prune_system(metadata=True)

    # create knowledge based on file at data/raw
    await create_knowledge(raw_dir, clean_dir, upload_metadata=upload_metadata, job_id=job_id)


# if __name__ == "__main__":
#     # create knowledge
#     # raw_dir = "data/raw/sisfor"
#     # clean_dir = "data/clean/sisfor"
#     asyncio.run(pipeline_cognee())