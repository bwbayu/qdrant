import os
import json
from pipelines.twelve_labs.twelvelabs_utils import embed_and_store_video, extract_slides_from_url, query_video, url_to_id
from pipelines.qdrant.qdrant_utils import qdrant, create_collection_if_not_exists

# Define the path to the JSON file that acts as a registry for processed videos.
JSON_FILE = "data/registry/videos.json"

def pipeline_twelvelabs(video_url: str):
    """
    Orchestrates the entire video processing workflow. It checks if a video has
    already been processed, and if not, it sends it to Twelve Labs for embedding
    and extracts its slides.

    Args:
        video_url (str): The public URL of the video to be processed.
    """
    video_url = video_url
    # Define the Qdrant collection name and generate a unique ID for the video.
    collection = "hackaton-collection"
    external_id = url_to_id(video_url)

    # Ensure the target collection exists in Qdrant before proceeding.
    create_collection_if_not_exists(collection_name=collection)

    # Check if the registry file exists.
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            try:
                # Load the JSON data.
                data = json.load(f)
            except json.JSONDecodeError:
                # If the file is empty or corrupt, start with an empty list.
                data = []
    else:
        # If the file doesn't exist, start with an empty list.
        data = []

    # Ensure the loaded data is a list, in case the JSON file has an unexpected structure.
    if not isinstance(data, list):
        data = []

    # Check if the video's ID already exists in our registry to avoid reprocessing.
    exists = any(item.get("id") == external_id for item in data)

    # Process the video only if it's new
    if not exists:
        # If the video is new, add its ID (hashed url video) to registry list.
        data.append({"id": external_id})
        # Write the updated list back to the JSON file.
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("Video baru ditambahkan ke JSON")

         # Process the video with TwelveLabs and store embeddings in Qdrant.
        embed_and_store_video(video_url, collection, qdrant)
        # Download the video and extract its slides to a dedicated folder.
        # TODO: output_dir change to gcs dir
        extract_slides_from_url(video_url, output_dir=f"data/raw/{external_id}")
    else:
        print("Video sudah ada di JSON, tidak ditambahkan lagi")

import re
def convert_drive_link_regex(url: str) -> str:
    """
    Converts a Google Drive shareable link to a direct download link using regular expressions.

    Args:
        url: The Google Drive shareable URL.
             (e.g., https://drive.google.com/file/d/FILE_ID/view?usp=sharing)

    Returns:
        The direct download link.
        (e.g., https://drive.google.com/uc?export=download&id=FILE_ID)
        Returns an empty string if the URL pattern does not match.
    """
    # Regex to find and capture the FILE_ID from the shareable link
    match = re.search(r'/file/d/([^/]+)/', url)
    if match:
        file_id = match.group(1)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    
    # If the pattern doesn't match, return an informative message.
    return "Invalid or non-matching Google Drive URL format"


if __name__ == "__main__":
    original_link = "https://drive.google.com/file/d/1oEEZl7XVLZ7spNKxYcMPuEG4_6MCHlNL/view?usp=sharing"
    new_link_regex = convert_drive_link_regex(original_link)
    pipeline_twelvelabs(video_url=new_link_regex)
