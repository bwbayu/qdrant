# Qlassroom

https://youtu.be/kzP5Lp_Z_AI

## 1. Overview

This project extends the concept of a *Zoom summary* application into a **multi-modal knowledge assistant**.
Traditional Zoom summary tools focus only on conversation transcripts, but this project goes beyond by combining multiple modalities — including **speech, video, slides, images, and documents**.

It solves the challenge of managing and analyzing large collections of **educational or instructional videos** by providing:

* **Automatic Video Embedding**: Uses Twelve Labs to generate embeddings from both visual and audio content.
* **Semantic Search**: Stores embeddings in Qdrant to enable efficient similarity search with natural language queries.
* **Slide Extraction**: Detects and saves slides from lecture/classroom videos based on visual changes. Extracted slides become part of the knowledge base to enhance retrieval accuracy.
* **Detailed Explanations**: Generates explanations derived directly from learning videos, ensuring that answers are not only relevant but also accurate.

To achieve this, the system integrates multiple components:

* **Transcripts** of spoken content.
* **Video embeddings** capturing actions, objects, and visual context.
* **Slide/document analysis**, where text is extracted and images are described by LLMs.
* **Knowledge graphs** (Neo4j) to represent relationships between concepts.
* **Semantic similarity search** (Cognee + Qdrant) to connect extracted information across formats.

By combining these modalities, the project enables richer retrieval, reasoning, and summarization — effectively acting as a **multi-modal RAG engine** that understands not only what was said but also what was shown and presented.

---

## 2. Features

* **Multi-modal input support**

  * Text input.
  * Images → automatically converted to text descriptions using LLMs.
  * Video → transcribed and enriched with embeddings, OCR, object detection, and extracted slides.

* **Agent-driven retrieval**

  * Queries are handled by an **Agent (CrewAI)** that selects the right tools (Qdrant, Cognee, Knowledge Graph) based on intent.
  * Supports retrieval types such as RAG completion, graph reasoning, chunked retrieval, and summarization.

* **Rich video understanding**

  * Beyond transcripts: integrates **video embeddings** from Twelve Labs, slide extractions, and LLM-generated slide descriptions.
  * If slides are not available separately, they are extracted directly from video frames for description and indexing.

* **Knowledge base updates**

  * Accepts multiple formats (text, image, video).
  * Videos are processed into transcripts + features; images are described by LLMs; both are combined into embeddings stored in Qdrant.

* **Hybrid knowledge storage**

  * **Qdrant** for semantic similarity on video features, slides, and text.
  * **Cognee + Neo4j** for combining semantic similarity with structured knowledge graph reasoning.

* **Cloud storage integration**

  * All raw data (videos, PDFs, images) and processed metadata are stored in **Google Cloud Storage (GCS)**.


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
