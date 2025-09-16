import os
import json
from typing import Dict

# Path to store the registry of processed PDFs
# The registry keeps track of which PDFs have already been processed
REGISTRY_PATH = "data/registry/processed_pdf.json"

def load_registry() -> Dict[str, dict]:
    """
    Load the registry of processed PDFs from a JSON file.
    
    Returns:
        dict: A dictionary where keys are document IDs and values are metadata/status.
              Returns an empty dict if no registry file exists.
    """
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_registry(registry: Dict[str, dict]):
    """
    Save the registry of processed PDFs to a JSON file.
    
    Args:
        registry (dict): A dictionary of processed documents to be saved.
    """
    # Ensure the directory exists before writing the file
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)

    # Write registry data to JSON file 
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)