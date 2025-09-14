import os
from dotenv import load_dotenv
import json
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("LLM_API_KEY"))

def query_slide_llm(image_url: str, extracted_text: str, model: str = "gpt-5-mini") -> dict:
    """
    Analyze a slide (page of a document) using an LLM with both image and text input.

    Args:
        image_url (str): Public URL of the slide image (uploaded to GCS).
        extracted_text (str): Raw text extracted from the slide using OCR or PDF text extraction.
        model (str, optional): The LLM model name. Defaults to "gpt-5-mini".

    Returns:
        dict: JSON object containing:
            - has_image (bool): Whether the slide contains an image/diagram/illustration.
            - image_description (str): Short description of the image/diagram if present (in Indonesian).
            - text_additions (str): Additional text identified by the LLM that was missing from the extracted text (in Indonesian).
            - final_text (str): Combined text = extracted_text + text_additions + image_description.
    """

    # System instruction for the model: 
    # - describe image if available
    # - check if extracted text is complete and add missing parts
    # - do not hallucinate or add unrelated content
    # - always return a JSON response with required fields
    system_msg = (
        "Kamu adalah asisten analisis slide.\n"
        "Tugasmu:\n"
        "1. Tentukan apakah ada gambar/diagram/ilustrasi yang bisa dideskripsikan. Jika ada, deskripsikan dalam bahasa indonesia.\n"
        "Jika Gambar berupa notasi/simbol matematika maka gunakan format latex misalnya seperti notasi OR, AND, dll.\n"
        "2. Periksa apakah teks hasil ekstraksi sudah lengkap. Jika ada teks yang hilang, tambahkan dalam bahasa indonesia.\n"
        "3. Jangan tambahkan informasi baru yang tidak ada di slide.\n"
        "4. Jika teks hasil ekstraksi jelek atau tidak sesuai atau tidak jelas maka hapus saja dan mungkin bisa diganti dengan teks hasil deskripsi dari gambar.\n"
        "5. Return hanya JSON dengan field: "
        "{has_image (bool), image_description (string), text_additions (string), final_text (string)}."
        "Field final_text = kombinasikan teks asli dan text_additions jika memang ada yang perlu diperbaiki dan image_description (jika relevan)."
        "Hapus kalimat/ide yang sama yang muncul lebih dari sekali"
    )

    # Call the LLM API with both extracted text and image
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_msg}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Teks hasil ekstraksi:\n{extracted_text}"},
                    {"type": "input_image", "image_url": image_url},
                ],
            },
        ],
    )

    # Extract the generated output as plain text
    content = response.output_text
    # Parse the output into JSON
    try:
        return json.loads(content)
    except Exception:
        # Fallback: return a default structure if parsing fails
        return {
            "has_image": False,
            "image_description": "",
            "text_additions": "",
            "final_text": extracted_text,
        }
