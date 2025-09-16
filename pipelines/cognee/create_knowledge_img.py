from dotenv import load_dotenv
load_dotenv()
import os
from os import path
import pathlib
from cognee_community_vector_adapter_qdrant import register  # noqa: F401
from cognee import config, add, cognify
from pipelines.cognee.utils.describe_image_llm import describe_image_llm

async def create_knowledge_from_image(gcs_url):
    """
    Create a Cognee knowledge dataset from an image URL.

    Steps:
    1. Setup Cognee system and data directories.
    2. Configure relational, vector, and graph databases.
    3. Generate a textual description of the image using an LLM.
    4. Combine the description with the image URL.
    5. Add the text to Cognee dataset and run Cognify to create embeddings and graphs.

    Args:
        gcs_url (str): The URL of the image to process.
    """
    # Setup Cognee system paths
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

    # generate description from image
    img_description = describe_image_llm(gcs_url=gcs_url)
    # Combine description and image URL for ingestion
    input_img = f"{img_description}\nURL = {gcs_url}"

    # Add text to Cognee dataset
    await add(data=[input_img])
    print(f"Add: Finish created dataset for {gcs_url}")

    # Run Cognify to create embeddings and knowledge graph
    await cognify()
    print(f"Cognify: Finish created knowledge for {gcs_url}")