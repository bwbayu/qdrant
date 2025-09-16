from dotenv import load_dotenv
load_dotenv()
import os
from os import path
import pathlib
from cognee_community_vector_adapter_qdrant import register  # noqa: F401
from cognee import config, add, cognify
from pipelines.cognee.utils.describe_image_llm import describe_image_llm

async def create_knowledge_from_image(gcs_url):
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
    input_img = f"{img_description}\nURL = {gcs_url}"
    # create dataset for cognee
    await add(data=[input_img])
    print(f"Add: Finish created dataset for {gcs_url}")
    await cognify()
    print(f"Cognify: Finish created knowledge for {gcs_url}")