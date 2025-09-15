import os
import cv2
import numpy as np
import requests
import hashlib

from twelvelabs import TwelveLabs
from twelvelabs.indexes import IndexesCreateRequestModelsItem
from twelvelabs.tasks import TasksRetrieveResponse
from qdrant_client.models import PointStruct

from skimage.metrics import structural_similarity as ssim
from dotenv import load_dotenv
load_dotenv()

# Initialize the Twelve Labs client using an API key
client = TwelveLabs(api_key=os.getenv("TWELVE_LABS_API_KEY"))

def url_to_id(url: str):
    """
    Generates a unique and consistent MD5 hash from a URL string.

    Args:
        url (str): The URL to be hashed.
    """
    return hashlib.md5(url.encode()).hexdigest()


def embed_and_store_video(video_url: str, collection_name: str, qdrant_client):
    """
    Processes a video with TwelveLabs to get embeddings and transcriptions,
    then stores the resulting data into a Qdrant collection.

    Args:
        video_url (str): The public URL of the video to be processed.
        collection_name (str): The name of the Qdrant collection where the data will be stored.
        qdrant_client: An initialized Qdrant client instance.
    """
    # Retrieve an existing index from Twelve Labs
    index = client.indexes.list()
    index_id = None
    
    # Use the first available index found
    for idx in index.items:
        index_id = idx.id
        break

    # If no index exists, create a new one
    if index_id is None:
        print("Create index")
        index = client.indexes.create(
            index_name="my_index",
            models=[IndexesCreateRequestModelsItem(
                model_name="marengo2.7",
                model_options=["visual", "audio"] # Specify that we want both visual and audio features
            )]
        )
        index_id = index.id

    # Create a task in Twelve Labs to process the video from the given URL
    task = client.tasks.create(index_id=index_id, video_url=video_url)

    # Define a callback function to print the status of the task as it runs
    def on_task_update(task: TasksRetrieveResponse):
        print(f"  Status={task.status}")

    # Wait for the video processing task to complete, using the callback for updates
    task = client.tasks.wait_for_done(task_id=task.id, callback=on_task_update)

    # Retrieve the results: video embeddings and transcription
    result = client.indexes.videos.retrieve(
        index_id=index_id,
        video_id=task.video_id,
        embedding_option=["visual-text", "audio"],
        transcription=True
    )

    points = []
    # Iterate through each video segment (clip) that has an embedding
    for i, clip in enumerate(result.embedding.video_embedding.segments):
        # Get data to store to qdrant
        vector = clip.float_
        start = clip.start_offset_sec
        end = clip.end_offset_sec
        option = clip.embedding_option
        scope = clip.embedding_scope

        # Find all transcription words that fall within the current clip's timeframe
        clip_transcriptions = [
            t.value for t in result.transcription if t.start >= start and t.end <= end
        ]
        text = " ".join(clip_transcriptions)

        # Create a Qdrant PointStruct with the vector and payload metadata
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "url": video_url,
                    "start_offset_sec": start,
                    "end_offset_sec": end,
                    "embedding_option": option,
                    "embedding_scope": scope,
                    "transcription": text,
                }
            )
        )

    # Upsert (update or insert) all the created points into the specified Qdrant collection
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Sukses simpan {len(points)} clip ke Qdrant di koleksi '{collection_name}'")

    return index_id

def extract_slides(video_path, output_dir, checks_per_second=5, change_threshold=0.97, deduplication_threshold=0.98):
    """
    Extracts slides using sparse sampling, checking a set number of frames per second.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted slide images.
        checks_per_second (int): The target number of frames to check per second.
        change_threshold (float): SSIM threshold to detect a new slide.
        deduplication_threshold (float): SSIM threshold to avoid saving duplicate slides.
    """
    # TODO: change ? for gcs dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- UPDATED LOGIC for sparse sampling ---
    if fps < checks_per_second:
        # If video FPS is lower than our target, check every frame
        frames_to_skip = 1
    else:
        frames_to_skip = int(fps / checks_per_second)

    print(f"Processing video: {total_frames} frames at {fps:.2f} FPS.")
    print(f"🚀 Sampling: Checking 1 frame every {frames_to_skip} frames (approx. {checks_per_second} checks/sec).")

    saved_slides_gray = []
    previous_frame_gray = None
    previous_frame_color = None
    slide_count = 0
    frame_number = 0

    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # The core sparse sampling check
        if frame_number % frames_to_skip == 0:
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)

            if previous_frame_gray is None:
                print("✅ Found first slide! Saving as slide_0001.png")
                slide_count += 1
                filename = os.path.join(output_dir, f"slide_{slide_count:04d}.png")
                cv2.imwrite(filename, frame)
                saved_slides_gray.append(current_frame_gray)
            else:
                score, _ = ssim(previous_frame_gray, current_frame_gray, full=True)

                if score < change_threshold:
                    is_duplicate = False
                    for saved_slide in saved_slides_gray:
                        dup_score, _ = ssim(previous_frame_gray, saved_slide, full=True)
                        if dup_score > deduplication_threshold:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        slide_count += 1
                        filename = os.path.join(output_dir, f"slide_{slide_count:04d}.png")
                        cv2.imwrite(filename, previous_frame_color)
                        print(f"✅ New slide detected! Saved as {filename}")
                        saved_slides_gray.append(previous_frame_gray)

            previous_frame_gray = current_frame_gray
            previous_frame_color = frame

        frame_number += 1

    cap.release()
    print(f"\n✨ Done! Extracted {slide_count} unique slides.")

def extract_slides_from_url(video_url, output_dir="slides"):
    """
    Downloads a video from a URL and then extracts slides from it.
    
    Args:
        video_url (str): The public URL of the video to download.
        output_dir (str, optional): The directory where the extracted slide
            images will be saved. Defaults to "slides".
    """
    tmp_file = "temp_video.mp4"
    print("[*] Downloading video...")

    # Download the video file in chunks and save it to temporary file
    r = requests.get(video_url, stream=True)
    with open(tmp_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # Call the local slide extraction function 
    print("[*] Extracting slides...")
    extract_slides(tmp_file, output_dir=output_dir)

    # Clean up by removing the temporary video file
    os.remove(tmp_file)
    print("[*] Selesai, file sementara dihapus.")
