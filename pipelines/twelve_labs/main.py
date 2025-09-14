import os
import json
from pipelines.twelve_labs.twelvelabs_utils import embed_and_store_video, extract_slides_from_url, query_video, url_to_id
from pipelines.qdrant.qdrant_utils import qdrant, create_collection_if_not_exists

import time

JSON_FILE = "data/registry/videos.json"


def pipeline_twelvelabs(video_url: str):
    video_url = video_url
    collection = "hackaton-collection"
    external_id = url_to_id(video_url)

    create_collection_if_not_exists(collection_name=collection)

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if not isinstance(data, list):
        data = []

    exists = any(item.get("id") == external_id for item in data)

    if not exists:
        data.append({"id": external_id})
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("Video baru ditambahkan ke JSON")

        embed_and_store_video(video_url, collection, qdrant)
        extract_slides_from_url(video_url, output_dir=f"data/raw/{external_id}")
    else:
        print("Video sudah ada di JSON, tidak ditambahkan lagi")

    
if __name__ == "__main__":
    start = time.time()
    pipeline_twelvelabs(video_url="https://drive.google.com/uc?export=download&id=11-DX3VtxpSQApTIosBXd0aSHURi9Auw_")
    end = time.time()
    elapsed = end - start
    print(f"\nTotal waktu eksekusi: {elapsed:.2f} detik")

    # collection = "hackaton-collection"

    # query_video("teknik yang dilakukan untuk melakukan klasifikasi pada gambar", collection, qdrant)