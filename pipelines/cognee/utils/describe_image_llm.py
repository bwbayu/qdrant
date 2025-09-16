import os
from dotenv import load_dotenv
import json
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("LLM_API_KEY"))

def describe_image_llm(gcs_url: str, model: str = "gpt-5-mini") -> str:
    """
    Generate a short description of an image using an LLM.

    Args:
        gcs_url (str): The URL of the image (e.g., stored in GCS).
        model (str): The LLM model to use for image analysis.

    Returns:
        str: A short description of the image in plain text.
    """

    # System instruction: describe the image briefly,
    # use the same language as the slide text (ID/EN),
    # and represent mathematical symbols in LaTeX format.
    system_msg_en = (
        "You are an assistant that describes images briefly. Don't add text that's not related to images.\n"
        "Describe them in the same language as the text in the slide if available (ID/EN).\n"
        "If the image contains mathematical notations/symbols, use LaTeX format (e.g., notation for OR, AND, etc.).\n"
    )
    
    # call the LLM API for image description
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_msg_en}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"What's in this image"},
                    {"type": "input_image", "image_url": gcs_url},
                ],
            },
        ],
    )

    # Extract the generated output as plain text
    content = response.output_text.strip()
    return content