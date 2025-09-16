from dotenv import load_dotenv
load_dotenv(override=True)
import gradio as gr
import asyncio
import os
import uuid
import shutil
import io
from google.cloud import storage

from pipelines.cognee.utils.upload_to_gcs import upload_file
from pipelines.cognee.main import pipeline_cognee
from pipelines.twelve_labs.main import pipeline_twelvelabs
from pipelines.cognee.create_knowledge_img import create_knowledge_from_image
from pipelines.cognee.utils.describe_image_llm import describe_image_llm

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
        pipeline_twelvelabs(video_url=video_url, local_path=local_path)

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

def handle_uploaded_image(img, session_id):
    """
    Upload image for query search. Turn image to text description

    Flow:
    1. Save image to buffer
    2. Upload temporary image file to GCS
    3. Use GCS Url to get image description from LLM
    4. Return image description
    """
    # check img
    if img is None:
        return gr.update()

    # save image to buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    # upload image to gcs temporary file
    gcs_url = upload_file(buffer, BUCKET_NAME, f"chat_images/query_image.png", content_type="image/png")

    # query LLM to get image description
    desc = describe_image_llm(gcs_url)
    output = f"Explain about {desc}"

    # return description to fill the text input
    return gr.update(value=output)

def list_files_in_gcs():
    """List PDF & Video files from GCS folders and return HTML string"""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # header + grid container
    html = """
    <h3>Files in GCS</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
    """

    for prefix in ["videos/", "pdfs/"]:
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            # get url
            url = f"https://storage.googleapis.com/{BUCKET_NAME}/{blob.name}"
            filename = blob.name.split("/")[-1]

            # video preview
            if blob.name.endswith((".mp4", ".mov", ".avi")):
                html += f"""
                <div style="text-align:center;">
                    <div style="width:320px; height:180px; overflow:hidden; margin:0 auto;">
                        <video controls style="width:100%; height:100%; object-fit:cover;">
                            <source src="{url}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                    <p><a href="{url}" target="_blank">{filename}</a></p>
                </div>
                """

            # pdf preview
            elif blob.name.endswith(".pdf"):
                html += f"""
                <div style="text-align:center;">
                    <div style="width:320px; height:240px; overflow:hidden; margin:0 auto; border:1px solid #ddd;">
                        <iframe src="{url}" style="width:100%; height:100%;"></iframe>
                    </div>
                    <p><a href="{url}" target="_blank">{filename}</a></p>
                </div>
                """

    html += "</div>"
    return html

# # Gradio UI
# with gr.Blocks() as demo:
#     file_input = gr.File(file_types=[".pdf", ".png", ".jpg", ".jpeg", ".mp4", ".mov", ".avi"], file_count="multiple")
#     btn = gr.Button("Process Files")
#     output = gr.HTML(label="Result")
#     btn.click(process_files, inputs=[file_input], outputs=[output])

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860)