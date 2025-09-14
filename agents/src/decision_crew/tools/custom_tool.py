from crewai.tools import BaseTool
from typing import Type, List, Union
from pydantic import BaseModel, Field
import json
from app.api.combine_output import RAGCompetionOutput, GraphCompletionOutput, ChunksOutput, SummariesOutput, VideoEmbedOutput

# =========================
# Input Schema
# =========================


class QueryItem(BaseModel):
    query: str = Field(..., description="A single query string")


class VideoEmbedInput(BaseModel):
    query: str = Field(...,
                       description="Teks query atau deskripsi untuk embedding video.")


class RAGCompletionInput(BaseModel):
    query: str = Field(...,
                       description="Teks query untuk RAG completion.")


class GraphCompletionInput(BaseModel):
    query: str = Field(...,
                       description="Teks query untuk graph completion.")


class ChunksInput(BaseModel):
    query: str = Field(...,
                       description="Teks query untuk mengambil chunks.")


class SummariesInput(BaseModel):
    query: str = Field(...,
                       description="Teks query untuk mengambil summaries.")


class CogneeInput(BaseModel):
    query: str = Field(...,
                       description="Teks query untuk memilih fungsi cognee yang dipakai. Pilihan: GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, SUMMARIES, GRAPH_SUMMARY_COMPLETION, GRAPH_COMPLETION_COT, GRAPH_COMPLETION_CONTEXT_EXTENSION, FEELING_LUCKY")

# =========================
# Tools
# =========================


class GraphCompletionTool(BaseTool):
    name: str = "Graph Completion Tool"
    description: str = """
    Graph-aware question answering.

    What it does: Finds relevant graph triplets using vector hints across indexed fields, resolves them into readable context, and asks an LLM to answer your question grounded in that context.
    Why it’s useful: Combines fuzzy matching (vectors) with precise structure (graph) so answers reflect relationships, not just nearby text.
    Typical output: A natural-language answer with references to the supporting graph context.

"""
    args_schema: Type[BaseModel] = GraphCompletionInput

    def _run(self, query: str) -> str:
        # TODO: implementasi embed video
        # misal call Qdrant / FAISS / service API buat embed video
        # Handle kalau "queries" masuk sebagai string
        # if isinstance(query, str):
        #     return "anjay ngajak ribut"
        return "hasil Graph Completion: perbedaan antara software engineer dan software engineering adalah sebagai berikut. Software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. Mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan. Sedangkan software engineering adalah disiplin ilmu yang mempelajari proses, metode, dan alat yang digunakan untuk mengembangkan perangkat lunak secara sistematis dan terstruktur."


class ChunksTool(BaseTool):
    name: str = "Chunks Tool"
    description: str = """
    Direct chunk retrieval.

    What it does: Returns the most similar text chunks to your query via vector search.
    When to use: You want raw passages/snippets to display or post-process.
    Output: Chunk objects with metadata.
"""
    args_schema: Type[BaseModel] = ChunksInput

    def _run(self, query: str) -> str:
        # TODO: implementasi embed video
        # misal call Qdrant / FAISS / service API buat embed video
        # Handle kalau "queries" masuk sebagai string
        # if isinstance(query, str):
        #     return "anjay ngajak ribut"
        return ChunksOutput(query)
        return "hasil Chunks: bagaimana perbedaan antara software engineer dan software engineering. Software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. Mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan. Sedangkan software engineering adalah disiplin ilmu yang mempelajari proses, metode, dan alat yang digunakan untuk mengembangkan perangkat lunak secara sistematis dan terstruktur."


class SummariesTool(BaseTool):
    name: str = "Summaries Tool"
    description: str = """
    Search over precomputed summaries.

    What it does: Vector search on TextSummary content for concise, high-signal hits.
    When to use: You prefer short summaries instead of full chunks.
    Output: Summary objects with provenance.
"""
    args_schema: Type[BaseModel] = SummariesInput

    def _run(self, query: str) -> str:
        # TODO: implementasi embed video
        # misal call Qdrant / FAISS / service API buat embed video
        # Handle kalau "queries" masuk sebagai string
        # if isinstance(query, str):
        #     return "anjay ngajak ribut"
        return SummariesOutput(query)
        return "hasil summaries: menurut saya, perbedaan antara software engineer dan software engineering adalah sebagai berikut. software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan. sedangkan software engineering adalah disiplin ilmu yang mempelajari proses, metode, dan alat yang digunakan untuk mengembangkan perangkat lunak secara sistematis dan terstruktur."


class RAGCompletionTool(BaseTool):
    name: str = "RAG Completion Tool"
    description: str = """
    Retrieve-then-generate over text chunks.

    What it does: Pulls top-k chunks via vector search, stitches a context window, then asks an LLM to answer.
    When to use: You want fast, text-only RAG without graph structure.
    Output: An LLM answer grounded in retrieved chunks.
"""
    args_schema: Type[BaseModel] = RAGCompletionInput

    def _run(self, query: str) -> str:
        # TODO: implementasi embed video
        # misal call Qdrant / FAISS / service API buat embed video
        # Handle kalau "queries" masuk sebagai string
        # if isinstance(query, str):
        #     return "anjay ngajak ribut"
        return RAGCompetionOutput(query)
        return "hasil RAG completion: software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. Mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan. Sedangkan software engineering adalah disiplin ilmu yang mempelajari proses, metode, dan alat yang digunakan untuk mengembangkan perangkat lunak secara sistematis dan terstruktur."


class VideoEmbedTool(BaseTool):
    name: str = "Video Embed Tool"
    description: str = "Gunakan tool ini untuk membuat embedding dari query video."
    args_schema: Type[BaseModel] = VideoEmbedInput

    def _run(self, query: str) -> str:
        # TODO: implementasi embed video
        # misal call Qdrant / FAISS / service API buat embed video
        # Handle kalau "queries" masuk sebagai string
        # if isinstance(query, str):
        #     return "anjay ngajak ribut"
        return VideoEmbedOutput(query)
        return "hasil transcribe video: ini ada perbedaan antara software engineer dan software engineering. Software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. Mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan. Sedangkan software engineering adalah disiplin ilmu yang mempelajari proses, metode, dan alat yang digunakan untuk mengembangkan perangkat lunak secara sistematis dan terstruktur."
        return f"[VideoEmbed] Hasil embedding untuk video query: {query}"


class CogneeTool(BaseTool):
    name: str = "Cognee Tool"
    description: str = "Gunakan tool ini untuk memilih fungsi cognee yang dipakai."
    args_schema: Type[BaseModel] = CogneeInput

    def _run(self, query: str) -> str:
        # TODO: implementasi embed video
        # misal call Qdrant / FAISS / service API buat embed video
        # Handle kalau "queries" masuk sebagai string
        # if isinstance(query, str):
        #     return "anjay ngajak ribut"
        if query == "GRAPH_COMPLETION":
            return "graph completion selected"
        elif query == "RAG_COMPLETION":
            return "rag completion selected"
        elif query == "CHUNKS":
            return "chunks selected"
        elif query == "SUMMARIES":
            return "summaries selected"
        elif query == "GRAPH_SUMMARY_COMPLETION":
            return "GRAPH SUMMARY COMPLETION completion selected"
        elif query == "GRAPH_COMPLETION_COT":
            return "GRAPH COMPLETION COT selected"
        elif query == "GRAPH_COMPLETION_CONTEXT_EXTENSION":
            return "GRAPH COMPLETION CONTEXT EXTENSION selected"
        elif query == "FEELING_LUCKY":
            return "FEELING LUCKY selected"
        else:
            return "fungsi cognee tidak ditemukan, silakan pilih dari: GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, SUMMARIES, GRAPH_SUMMARY_COMPLETION, GRAPH_COMPLETION_COT, GRAPH_COMPLETION_CONTEXT_EXTENSION, FEELING_LUCKY"
