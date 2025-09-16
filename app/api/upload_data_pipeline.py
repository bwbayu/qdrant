from dotenv import load_dotenv
load_dotenv(override=True)
import gradio as gr
import asyncio
import os
import uuid
import shutil

from pipelines.cognee.utils.upload_to_gcs import upload_file
from pipelines.twelve_labs.twelvelabs_utils import extract_slides
from pipelines.cognee.main import pipeline_cognee
from pipelines.twelve_labs.main import pipeline_twelvelabs
from pipelines.cognee.create_knowledge_img import create_knowledge_from_image

# Load GCS bucket name from environment
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
TMP_DIR = "tmp"  # Temporary directory for file processing

def make_dir(base_dir=TMP_DIR):
    """
    Create a unique directory inside the tmp folder using UUID.
    This directory is used to store temporary raw/clean files.
    """
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(base_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_id, job_dir

async def process_pdf(files, job_id, job_dir):
    """
    Handle PDF files:
    - Save uploaded PDFs locally
    - Upload them to GCS
    - Run Cognee pipeline (process + generate metadata)
    """
    # create dir
    raw_dir = os.path.join(job_dir, "raw")
    clean_dir = os.path.join(job_dir, "clean")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    # gcs url data
    gcs_urls = []

    for file in files:
        # save local
        filename = os.path.basename(file.name)
        local_path = os.path.join(raw_dir, filename)
        shutil.copy(file.name, local_path)

        # upload to gcs
        gcs_url = upload_file(local_path, BUCKET_NAME, f"pdfs/{job_id}/{filename}", content_type="application/pdf")
        gcs_urls.append(gcs_url)

    # cognee pipeline with upload metadata to gcs
    await pipeline_cognee(raw_dir, clean_dir, reset_data=False, upload_metadata=True, job_id=job_id)

    return gcs_urls

async def process_images(files, job_id, job_dir):
    """
    Handle image files:
    - Save uploaded images locally
    - Upload them to GCS
    - Run cognee pipeline to add knowledge based on image data
    """
    # create dir
    img_dir = os.path.join(job_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    gcs_urls = []
    for file in files:
        # save local
        filename = os.path.basename(file.name)
        local_path = os.path.join(img_dir, filename)
        shutil.copy(file.name, local_path)

        # upload to gcs
        gcs_url = upload_file(local_path, BUCKET_NAME, f"images/{job_id}/{filename}", content_type="image/jpeg")
        gcs_urls.append(gcs_url)

        # create knowledge base using cognee use the local path of the image
        await create_knowledge_from_image(gcs_url)

    return gcs_urls

def process_video(files, job_id, job_dir):
    """
    Handle video files:
    - Save uploaded videos locally
    - Upload them to GCS
    - Run TwelveLabs pipeline (analysis + slide extraction)
    - Upload extracted slide images to GCS
    """
    # create dir for video and extracted image data
    video_dir = os.path.join(job_dir, "video")
    img_dir = os.path.join(job_dir, "images")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    gcs_urls = []
    for file in files:
        # save video locally
        filename = os.path.basename(file.name)
        local_path = os.path.join(video_dir, filename)
        shutil.copy(file.name, local_path)

        # upload video to gcs and get gcs url
        video_url = upload_file(local_path, BUCKET_NAME, f"videos/{job_id}/{filename}", content_type='video/mp4')
        gcs_urls.append(video_url)

        # pipeline twelve labs to embed and store video to vector db
        pipeline_twelvelabs(video_url=video_url)

        # extract image from video for more context about video
        extract_slides(local_path, img_dir)

        for img_file in os.listdir(img_dir):
            # upload extracted image to gcs
            img_path = os.path.join(img_dir, img_file)
            gcs_url = upload_file(img_path, BUCKET_NAME, f"images/{job_id}/{img_file}")
            gcs_urls.append(gcs_url)

    return gcs_urls

async def pipeline_process_files(files):
    """
    Main handler:
    - Detect file type (PDF, video, image)
    - Process accordingly
    - Collect uploaded file URLs
    - Return them as clickable HTML links
    - Clean up tmp/ folder after completion
    """
    # create temporary directory
    job_id, job_dir = make_dir()
    results = []

    # store url from gcs for each type of data
    pdfs, videos, images = [], [], []

    try:
        for f in files:
            # detect file type from filename
            ext = os.path.splitext(f.name)[1].lower()
            if ext == ".pdf":
                pdfs.append(f)
            elif ext in [".mp4", ".mov", ".avi"]:
                videos.append(f)
            elif ext in [".png", ".jpg", ".jpeg"]:
                images.append(f)

        # process files by type
        if pdfs:
            results.extend(await process_pdf(pdfs, job_id, job_dir))
        if videos:
            results.extend(process_video(videos, job_id, job_dir))
        if images:
            results.extend(await process_images(images, job_id, job_dir))

        # return in html with clickable links
        messages = [f"Successfully uploaded: <a href='{url}' target='_blank'>{url}</a>" for url in results]
        return "<br>".join(messages)
    finally:
        # clean up the entire tmp directory after processing
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR, ignore_errors=True)

# # Gradio UI
# with gr.Blocks() as demo:
#     file_input = gr.File(file_types=[".pdf", ".png", ".jpg", ".jpeg", ".mp4", ".mov", ".avi"], file_count="multiple")
#     btn = gr.Button("Process Files")
#     output = gr.HTML(label="Result")
#     btn.click(process_files, inputs=[file_input], outputs=[output])

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860)