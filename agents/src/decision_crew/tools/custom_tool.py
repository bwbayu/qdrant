from crewai.tools import BaseTool
from typing import Type, List, Union
from pydantic import BaseModel, Field
import json

# =========================
# Input Schema
# =========================


class QueryItem(BaseModel):
    query: str = Field(..., description="A single query string")


class VideoEmbedInput(BaseModel):
    query: str = Field(...,
                       description="Teks query atau deskripsi untuk embedding video.")


class TextEmbedInput(BaseModel):
    query: str = Field(...,
                       description="Teks yang ingin di-embed untuk pencarian text.")


class GraphEmbedInput(BaseModel):
    query: str = Field(...,
                       description="Node/relationship query untuk embedding graph.")


# =========================
# Tools
# =========================
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
        return "hasil transcribe: ini ada perbedaan antara software engineer dan software engineering. Software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. Mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan. Sedangkan software engineering adalah disiplin ilmu yang mempelajari proses, metode, dan alat yang digunakan untuk mengembangkan perangkat lunak secara sistematis dan terstruktur."
        return f"[VideoEmbed] Hasil embedding untuk video query: {query}"


class TextEmbedTool(BaseTool):
    name: str = "Text Embed Tool"
    description: str = "Gunakan tool ini untuk membuat embedding dari query text."
    args_schema: Type[BaseModel] = TextEmbedInput

    def _run(self, query: str) -> str:
        # TODO: implementasi embed text
        return "software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. Mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan."
        return f"[TextEmbed] Hasil embedding untuk text query: {query}"


class GraphEmbedTool(BaseTool):
    name: str = "Graph Embed Tool"
    description: str = "Gunakan tool ini untuk membuat embedding dari query graph (node/relationship)."
    args_schema: Type[BaseModel] = GraphEmbedInput

    def _run(self, query: str) -> str:
        # TODO: implementasi embed graph
        return f"pekerjaan -> software engineer, bidang -> software engineering"
        return f"[GraphEmbed] Hasil embedding untuk graph query: {query}"
