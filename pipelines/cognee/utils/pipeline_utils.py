import fitz
import uuid
from PIL import Image
from io import BytesIO
import os
import re
import hashlib

def gen_uuid() -> str:
    """
    Generate a unique identifier string using UUID4.

    Returns:
        str: A random UUID string.
    """
    return str(uuid.uuid4())

def save_page_as_image(page) -> BytesIO:
    """
    Render a PDF page into an image and return it as an in-memory JPEG buffer.

    Args:
        page (fitz.Page): A page object from PyMuPDF (fitz).

    Returns:
        BytesIO: In-memory JPEG image buffer of the rendered page.
    """
    # Render the page with higher resolution (2x zoom)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

    # Convert raw image bytes into a PIL Image
    img_bytes = pix.tobytes(output="png")
    img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")

    # Save the image into a buffer as JPEG
    buf = BytesIO()
    img_pil.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf

def extract_text(page) -> str:
    """
    Extract visible text from a PDF page.

    Args:
        page (fitz.Page): A page object from PyMuPDF (fitz).

    Returns:
        str: Extracted text, or an empty string if none is found.
    """
    return page.get_text("text") or ""

def clean_text(text: str) -> str:
    """
    Clean extracted text by removing unwanted patterns and formatting.

    Cleaning steps:
        - Remove lines starting with codes like "IK1234 - ..."
        - Remove lines containing only numbers
        - Collapse multiple blank lines into a single newline

    Args:
        text (str): Raw extracted text.

    Returns:
        str: Cleaned and normalized text.
    """
    # delete text that contain unnecessay information like Course Code like IK280, etc
    text = re.sub(r'(?m)^IK\d+\s*-\s*.*$', '', text, flags=re.IGNORECASE)
    # delete text that contain number of slide or page
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    text = re.sub(r'\n{2,}', '\n', text).strip()
    return text

def get_doc_id_from_filename(file_path: str) -> str:
    """
    Derive a document ID from the input filename + content hash.

    Rules:
        - Take the base filename (without extension).
        - If the filename contains " - ", only take the part before it.
        - Compute a short content hash (first 8 hex chars of SHA1).
        - doc_id = "{base}_{hash}"

    Args:
        file_path (str): Path to the file (PDF/DOCX/PPTX).

    Returns:
        str: Document ID string.
    """
    fname = os.path.basename(file_path)
    base, _ = os.path.splitext(fname)
    if " - " in base:
        base = base.split(" - ")[0].strip()
    else:
        base = base.strip()

    # Compute content hash
    h = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    content_hash = h.hexdigest()[:8]

    return f"{base}_{content_hash}"