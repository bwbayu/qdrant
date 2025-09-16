# Project Title

## 1. Overview

*TODO: Add project description here*

---

## 2. Features

*TODO: Add list of features here*

---

## 3. Tech Stack

* **Python** – Core programming language
* **Gradio** – UI/Frontend interface
* **Mistral / OpenAI** – LLM providers
* **Twelve Labs** – Video embedding & clip extraction
* **CrewAI** – Agent framework that interacts with tools like vector databases and knowledge graphs
* **Qdrant** – Vector database (for video embeddings, text, slides, images)
* **Neo4j** – Knowledge graph database
* **Cognee** – Knowledge base creation & semantic similarity for documents (PDF, PPT, etc.)
* **Google Cloud Storage (GCS)** – Storage for video, document, and image data

---

## 4. Repository Layout

```
project-root/
│── app/                         # Application (frontend/backend)
│   ├── api/combine_output.py     # CrewAI tool calling Twelve Labs & Cognee inference
│   ├── api/upload_data_pipeline.py # Helper for GUI: handle upload/update KB (PDF, video, image)
│   ├── client/gui.py             # Gradio UI/frontend
│
│── agents/                       # CrewAI agent
│   ├── main.py                   # Entrypoint for CrewAI agent
│   ├── crew.py                   # Agent description (roles, goals)
│   ├── tools/custom_tools        # Tools calling Cognee & Twelve Labs inference
│   └── config/agent.yaml         # Agent role, goal, backstory
│   └── config/tasks.yaml         # Tasks & expected outputs
│
│── pipelines/
│   ├── twelve_labs/              # Twelve Labs pipeline
│   │   ├── inference.py          # Query Qdrant with Twelve Labs embeddings
│   │   ├── main.py               # Entrypoint: create collection, embed & store video
│   │   └── twelvelabs_utils.py   # Helpers: video embedding, clip/image extraction, LLM descriptions
│   │
│   ├── cognee/                   # Cognee pipeline
│   │   ├── utils/                # Helpers
│   │   │   ├── check_processed_docs.py  # Track processed docs
│   │   │   ├── describe_image_llm.py    # LLM-based image description
│   │   │   ├── text_ingest.py           # Extract text & enrich with LLM
│   │   │   ├── pdf_converter.py         # Convert docs to PDF
│   │   │   ├── pipeline_utils.py        # Common helpers (UUID, regex, hash, save page, etc.)
│   │   │   ├── upload_to_gcs.py         # Upload files to GCS
│   │   ├── create_knowledge_img.py      # Build KB from images
│   │   ├── create_knowledge.py          # Process PDFs → metadata + KB
│   │   ├── inference.py                 # Query Cognee KB
│   │   ├── main.py                      # Entrypoint for Cognee pipeline
│   │   └── preprocessing.py             # Extract, preprocess, metadata upload
│   │
│   └── qdrant/                  # Vector DB utilities
│       ├── qdrant_utils          # Create collection for video embeddings
│
│── data/                        # Storage
│   ├── raw/                     # Raw data (documents, videos, images)
│   ├── processed/               # Processed data (metadata, transcripts, image desc.)
│
│── requirements.txt             # Dependencies
│── chat_session.db              # Store chat history
```

---

## 5. How to Run

1. Create virtual environment

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```
2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
3. Fill environment variables in `.env` (see below)
4. Run GUI

   ```bash
   python -m app.client.gui
   ```

---

## 6. Environment Variables

```env
# Twelve Labs
TWELVE_LABS_API_KEY=
TWELVE_LABS_COLLECTION=

# LLM
MISTRAL_API_KEY=
OPENAI_API_KEY=
LLM_API_KEY=
LLM_MODEL="openai/gpt-5-mini"
LLM_PROVIDER="openai"

# Neo4j
GRAPH_DATABASE_PROVIDER="neo4j"
# GRAPH_DATABASE_URL=neo4j://localhost:7687  # Local
# GRAPH_DATABASE_URL=neo4j://127.0.0.1:7687  # Desktop
GRAPH_DATABASE_USERNAME=
GRAPH_DATABASE_PASSWORD=

# Qdrant (text/slide/image)
VECTOR_DB_PROVIDER=qdrant
# VECTOR_DB_URL=http://localhost:6333  # Local
VECTOR_DB_KEY=
VECTOR_DB_URL=

# Qdrant (video)
QDRANT_URL2=
QDRANT_API_KEY2=

# GCP (storage)
GOOGLE_APPLICATION_CREDENTIALS=
GOOGLE_CLOUD_PROJECT=
GCS_DEST_IMAGES_PATH=
GCS_DEST_METADATA_PATH=
GCS_DEST_RAW_PATH=
GCS_BUCKET_NAME=
```

---

## 7. Demo

*TODO: Add video demo here*
