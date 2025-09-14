import os
import subprocess
from docx2pdf import convert

def convert_to_pdf(input_file: str) -> str:
    """
    Convert a given document (DOCX, PPTX, or PDF) into a PDF file.

    Args:
        input_file (str): Path to the input file. Supported extensions are:
            - .pdf   : Returned directly without conversion
            - .docx  : Converted to PDF using docx2pdf
            - .pptx  : Converted to PDF using LibreOffice in headless mode

    Returns:
        str: Path to the resulting PDF file.

    Raises:
        ValueError: If the file extension is not supported.

    Notes:
        - Requires `docx2pdf` package for DOCX → PDF conversion.
        - Requires LibreOffice installed in the system for PPTX → PDF conversion.
        - Conversion output will be stored in the same directory as the input file.
    """

    ext = os.path.splitext(input_file)[1].lower()

    # Case 1: already a PDF → just return the file path
    if ext == ".pdf":
        return input_file

    # Case 2: DOCX → PDF (using docx2pdf library)
    elif ext == ".docx":
        out = input_file.replace(".docx", ".pdf")
        convert(input_file, out)
        return out

    # Case 3: PPTX → PDF (using LibreOffice in headless mode)
    elif ext == ".pptx":
        out = input_file.replace(".pptx", ".pdf")
        subprocess.run([
            "libreoffice", "--headless", "--convert-to", "pdf", input_file,
            "--outdir", os.path.dirname(out)
        ], check=True)
        return out

    # Unsupported file type
    else:
        raise ValueError(f"Unsupported file type: {ext}")
