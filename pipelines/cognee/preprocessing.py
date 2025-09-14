import os
import json
import fitz
from utils.pdf_converter import convert_to_pdf
from utils.pipeline_utils import save_page_as_image, extract_text, clean_text, get_doc_id_from_filename
from utils.upload_to_gcs import upload_file
from utils.extract_image_llm import query_slide_llm

# Load environment variables for Google Cloud Storage configuration
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCS_DEST_IMAGES_PATH = os.environ.get("GCS_DEST_IMAGES_PATH")
# GCS_DEST_METADATA_PATH = os.environ.get("GCS_DEST_METADATA_PATH")

def process_document(input_file: str):
    """
    Process a document (PDF, DOCX, PPTX) into structured slide/page metadata.

    Steps:
        1. Convert the input file into a PDF (if not already PDF).
        2. Iterate through each page of the PDF.
        3. Extract raw text from the page.
        4. Render the page into an image and upload it directly to GCS.
        5. Query an LLM to refine text and optionally describe images/diagrams.
        6. Clean and normalize the final text.
        7. Store results as metadata for each slide/page.

    Args:
        input_file (str): Path to the input file (PDF, DOCX, PPTX).

    Returns:
        list[dict]: A list of metadata dictionaries, where each entry contains:
            - doc_id (str): Document identifier derived from filename.
            - slide_id (str): Unique slide/page identifier.
            - page_number (int): Page number (1-based index).
            - gcs_url (str): Public URL of the uploaded slide image in GCS.
            - text (str): Cleaned text (OCR/LLM-enhanced).
    """
    results = []

    # Ensure file is converted to PDF
    pdf_path = convert_to_pdf(input_file)

    # Open PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    doc_id = get_doc_id_from_filename(input_file)

    # Iterate through all pages
    for page_num, page in enumerate(doc):
        slide_id = f"{doc_id}_{page_num}"

        # Extract text from PDF page
        extracted_text = extract_text(page)

        # Render page as image (in-memory buffer)
        img_buf = save_page_as_image(page)

        # Upload image buffer to GCS
        gcs_blob_path = f"{GCS_DEST_IMAGES_PATH}/{slide_id}.jpg"
        gcs_url = upload_file(img_buf, GCS_BUCKET_NAME, gcs_blob_path, content_type="image/jpeg")

        # Use LLM to validate/augment extracted text and describe images
        llm_out = query_slide_llm(gcs_url, extracted_text)

        # Clean text (remove unwanted patterns, normalize spacing)
        cleaned_text = clean_text(llm_out.get("final_text", extracted_text))

        # Collect metadata for this slide/page
        metadata = {
            "doc_id": doc_id,
            "slide_id": slide_id,
            "page_number": page_num + 1,
            "gcs_url": gcs_url,
            "text": cleaned_text,
        }
        results.append(metadata)
        print(f"[{page_num+1}/{len(doc)}] slide {slide_id}")

    doc.close()
    return results