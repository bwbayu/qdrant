import os
import cv2
import numpy as np
import requests
import hashlib
import uuid
import json

from twelvelabs import TwelveLabs
from twelvelabs.indexes import IndexesCreateRequestModelsItem
from twelvelabs.tasks import TasksRetrieveResponse
from qdrant_client.models import PointStruct
from pipelines.cognee.utils.upload_to_gcs import upload_file
from pipelines.cognee.utils.describe_image_llm import describe_image_llm

from skimage.metrics import structural_similarity as ssim
from moviepy import VideoFileClip
from dotenv import load_dotenv
load_dotenv()

# Initialize the Twelve Labs client using an API key
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
client = TwelveLabs(api_key=os.getenv("TWELVE_LABS_API_KEY"))
TMP_DIR = "tmp"  # Temporary directory for file processing
CACHE_FILE = os.path.join(TMP_DIR, "extracted_image_cache.json")

def url_to_id(url: str):
    """
    Generates a unique and consistent MD5 hash from a URL string.

    Args:
        url (str): The URL to be hashed.
    """
    return hashlib.md5(url.encode()).hexdigest()

EXTRACTED_TEXT_FILE = os.path.join("data", f"text_extracted.json")

# contoh fungsi untuk simpan extracted_text ke JSON
def save_extracted_text(start_sec, end_sec, extracted_text, file_path=EXTRACTED_TEXT_FILE):
    """
    Save extracted texts with start and end times per clip to JSON file
    """
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

    # append new clip data
    data.append({
        "start": start_sec,
        "end": end_sec,
        "text": extracted_text.strip()
    })

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def embed_and_store_video(video_url: str, temp_file:str, collection_name: str, qdrant_client):
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

        # create temp file for clip
        temp_clip_file = os.path.join(TMP_DIR, "clip.mp4")
        
        # create temp dir for extracted image from clip
        img_clip_dir = os.path.join(TMP_DIR, "clips")
        os.makedirs(img_clip_dir, exist_ok=True)

        # Cut video to clip and save it locally
        cut_video(temp_file, temp_clip_file, start, end)
        # Extract slide for each clip
        extract_slides(temp_clip_file, img_clip_dir)
        # store extracted image to GCS and generate description using LLM
        extracted_texts = ""
        for img_file in os.listdir(img_clip_dir):
            # upload to gcs
            img_path = os.path.join(img_clip_dir, img_file)
            gcs_url = upload_file(img_path, BUCKET_NAME, "clips/extracted_image.png")

            # generate description
            text = describe_image_llm(gcs_url=gcs_url)
            extracted_texts += f"\n{text}"

        # (OPTIONAL) See output of extracted text 
        save_extracted_text(start, end, extracted_texts)

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
                    "transcription": text + "\n" + extracted_texts,
                }
            )
        )

    # Upsert (update or insert) all the created points into the specified Qdrant collection
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Sukses simpan {len(points)} clip ke Qdrant di koleksi '{collection_name}'")

    return index_id

def extract_slides(video_path, output_dir, checks_per_second=1, change_threshold=0.97, deduplication_threshold=0.98):
    """
    Extracts unique slides from a video presentation by sparsely sampling frames.
    
    This function works by:
    1.  Sampling frames at a specified rate (e.g., 1 frames per second).
    2.  Using the Structural Similarity Index (SSIM) to detect significant changes between consecutive sampled frames.
    3.  When a change is detected, it performs a deduplication check to ensure the new slide hasn't been saved before.
    4.  Saves the unique slides as PNG images in the specified directory.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted slide images.
        checks_per_second (int): The target number of frames to check per second.
        change_threshold (float): SSIM threshold to detect a new slide.
        deduplication_threshold (float): SSIM threshold to avoid saving duplicate slides.
    """

    # Ensure the output directory exists, creating it if necessary.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Open the video file for processing.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties (frame count and FPS).
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine how many frames to skip between checks to meet the `checks_per_second` target.
    # If video FPS is lower than our target, we check every frame (skip=1).
    if fps < checks_per_second:
        frames_to_skip = 1
    else:
        frames_to_skip = int(fps / checks_per_second)

    print(f"Processing video: {total_frames} frames at {fps:.2f} FPS.")
    print(f"🚀 Sampling: Checking 1 frame every {frames_to_skip} frames (approx. {checks_per_second} checks/sec).")

    # Variables to hold state between loops.
    saved_slides_gray = []        # Stores grayscale versions of saved slides for efficient comparison.
    previous_frame_gray = None    # The previously processed grayscale frame.
    previous_frame_color = None   # The previously processed color frame (the one we'll save).
    slide_count = 0               # Counter for saved slides.
    frame_number = 0              # Counter for total frames processed.

    
    while cap.isOpened():
        # Read the next frame from the video.
        ret, frame = cap.read()
        if not ret:
            break    # End of the video.

        # Only process the frame if it's on our sampling interval.
        if frame_number % frames_to_skip == 0:
            # Pre-process the frame for comparison: convert to grayscale and apply a blur to reduce noise and focus on structural changes.
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)

            # If this is the very first frame we're processing, save it as the first slide.
            if previous_frame_gray is None:
                print("✅ Found first slide! Saving as slide_0001.png")
                slide_count += 1
                filename = os.path.join(output_dir, f"slide_{slide_count:04d}.png")
                cv2.imwrite(filename, frame)
                saved_slides_gray.append(current_frame_gray)
            else:
                # Compare the current frame with the previous one. A low SSIM score means the frames are different, indicating a potential slide change.
                score, _ = ssim(previous_frame_gray, current_frame_gray, full=True)

                if score < change_threshold:
                    # A significant change was detected. Now check if this "new" slide is a duplicate.
                    # We check the `previous_frame`, as it's the last stable frame before the change.
                    is_duplicate = False
                    for saved_slide in saved_slides_gray:
                        dup_score, _ = ssim(previous_frame_gray, saved_slide, full=True)
                        if dup_score > deduplication_threshold:
                            is_duplicate = True
                            break    # Found a duplicate, no need to check further.
                    # If it's not a duplicate, save it.
                    if not is_duplicate:
                        slide_count += 1
                        filename = os.path.join(output_dir, f"slide_{slide_count:04d}.png")
                        cv2.imwrite(filename, previous_frame_color)
                        print(f"✅ New slide detected! Saved as {filename}")
                        saved_slides_gray.append(previous_frame_gray)
                        
            # Update the previous frame states for the next iteration.
            previous_frame_gray = current_frame_gray
            previous_frame_color = frame

        frame_number += 1
    # The loop saves a slide only when a change is detected. This means the very last slide is never saved. We handle that here.
    print("\nChecking for the final slide...")
    if previous_frame_gray is not None:
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
            print(f"✅ Saved final slide as {filename}")
        else:
            print("Final slide was a duplicate of a previously saved slide. Nothing to save.")
    cap.release()
    print(f"\n✨ Done! Extracted {slide_count} unique slides.")

def download_video_from_url(video_url, output_dir):
    """
    Downloads a file from a URL and saves it to a local path using a memory-efficient stream.

    Args:
        video_url (str): The direct download URL for the video file.
        output_path (str): The full local path (e.g., 'videos/my_video.mp4')
                           where the file will be saved.
    """
    tmp_file = output_dir
    print("[*] Downloading video...")

    # Send an HTTP GET
    r = requests.get(video_url, stream=True)
    # Open the target file path in binary write mode.
    with open(tmp_file, "wb") as f:
        # Iterate over the response content in chunks.
        for chunk in r.iter_content(chunk_size=8192):
            # Write each received chunk to the file.
            if chunk:
                f.write(chunk)

def extract_slides_from_url(tmp_file, output_dir="slides"):
    print("[*] Extracting slides...")
    extract_slides(tmp_file, output_dir=output_dir)

    os.remove(tmp_file)
    print("[*] Selesai, file sementara dihapus.")

def cut_video(input_path: str, output_path: str, start: float, end: float):
    """
    Cuts a video from a local file based on the start and end times.
    
    Args:
        input_path (str): Path to the original video file.
        output_path (str): Path to save the resulting video clip.
        start (float): The start time in seconds.
        end (float): The end time in seconds.
    """
    if start < end:
        # Load the original video file from the specified input path.
        video = VideoFileClip(input_path)
        
        # Create a new clip by cutting the original video from the start to the end time.
        clip = video.subclipped(start, end)
        
        # Save the newly created clip to the output path.
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Explicitly close the video and clip objects to release file handles and free up memory.
        video.close()
        clip.close()
    else:
        print(f"⚠️ Skip clip karena start >= end (start={start}, end={end})")
        return