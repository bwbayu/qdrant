from crewai.tools import BaseTool
from typing import Type, List, Union
from pydantic import BaseModel, Field
import json
from app.api.combine_output import RAGCompetionOutput, GraphCompletionOutput, ChunksOutput, SummariesOutput, VideoEmbedOutput
import asyncio

# =========================
# Input Schemas (Pydantic models for validation)
# =========================

# Base input model for generic queries


class QueryItem(BaseModel):
    query: str = Field(..., description="A single query string")

# Input model for Video Embedding queries


class VideoEmbedInput(BaseModel):
    query: str = Field(...,
                       description="Text query or description for video embedding.")

# Input model for RAG Completion queries


class RAGCompletionInput(BaseModel):
    query: str = Field(..., description="Text query for RAG completion.")

# Input model for Graph Completion queries


class GraphCompletionInput(BaseModel):
    query: str = Field(..., description="Text query for graph completion.")

# Input model for retrieving text chunks


class ChunksInput(BaseModel):
    query: str = Field(..., description="Text query to retrieve chunks.")

# Input model for retrieving summaries


class SummariesInput(BaseModel):
    query: str = Field(..., description="Text query to retrieve summaries.")

# Input model for choosing Cognee function


class CogneeInput(BaseModel):
    query: str = Field(...,
                       description="Text query to select which Cognee function to use. "
                                   "Choices: GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, "
                                   "SUMMARIES, GRAPH_SUMMARY_COMPLETION, GRAPH_COMPLETION_COT, "
                                   "GRAPH_COMPLETION_CONTEXT_EXTENSION, FEELING_LUCKY")

# =========================
# Tools (Wrappers around different processing backends)
# =========================


class GraphCompletionTool(BaseTool):
    """Tool for graph-aware question answering."""
    name: str = "Graph Completion Tool"
    description: str = """
    Graph-aware question answering.
    Finds relevant graph triplets using vector hints across indexed fields,
    resolves them into readable context, and asks an LLM to answer your question grounded in that context.
    """
    args_schema: Type[BaseModel] = GraphCompletionInput

    def _run(self, query: str) -> str:
        # Run graph completion asynchronously
        return asyncio.run(GraphCompletionOutput(query))


class ChunksTool(BaseTool):
    """Tool to retrieve raw text chunks based on query similarity."""
    name: str = "Chunks Tool"
    description: str = """
    Direct chunk retrieval.
    Returns the most similar text chunks to your query via vector search.
    Useful for raw passages/snippets that you may want to display or post-process.
    """
    args_schema: Type[BaseModel] = ChunksInput

    def _run(self, query: str) -> str:
        return asyncio.run(ChunksOutput(query))


class SummariesTool(BaseTool):
    """Tool to retrieve concise precomputed summaries."""
    name: str = "Summaries Tool"
    description: str = """
    Search over precomputed summaries.
    Vector search on TextSummary content for concise, high-signal hits.
    Useful if you prefer short summaries instead of full text chunks.
    """
    args_schema: Type[BaseModel] = SummariesInput

    def _run(self, query: str) -> str:
        return asyncio.run(SummariesOutput(query))


class RAGCompletionTool(BaseTool):
    """Tool for retrieve-then-generate workflow over text chunks."""
    name: str = "RAG Completion Tool"
    description: str = """
    Retrieve-then-generate over text chunks.
    Pulls top-k chunks via vector search, stitches a context window,
    then asks an LLM to answer grounded in that retrieved context.
    """
    args_schema: Type[BaseModel] = RAGCompletionInput

    def _run(self, query: str) -> str:
        return asyncio.run(RAGCompetionOutput(query))


class VideoEmbedTool(BaseTool):
    """Tool for creating embeddings from video queries."""
    name: str = "Video Embed Tool"
    description: str = "Use this tool to create embeddings from a video query."
    args_schema: Type[BaseModel] = VideoEmbedInput

    def _run(self, query: str) -> str:
        # Returns list of dicts (JSON)
        output = VideoEmbedOutput(query)
        # Convert Python dict/list to JSON string (pretty printed)
        result = json.dumps(output, ensure_ascii=False, indent=2)
        return result


class CogneeTool(BaseTool):
    """Dispatcher tool for selecting different Cognee functions."""
    name: str = "Cognee Tool"
    description: str = "Use this tool to choose which Cognee function to run."
    args_schema: Type[BaseModel] = CogneeInput

    def _run(self, query: str) -> str:
        # Very simple function router based on query string
        if query == "GRAPH_COMPLETION":
            return "graph completion selected"
        elif query == "RAG_COMPLETION":
            return "rag completion selected"
        elif query == "CHUNKS":
            return "chunks selected"
        elif query == "SUMMARIES":
            return "summaries selected"
        elif query == "GRAPH_SUMMARY_COMPLETION":
            return "GRAPH SUMMARY COMPLETION selected"
        elif query == "GRAPH_COMPLETION_COT":
            return "GRAPH COMPLETION COT selected"
        elif query == "GRAPH_COMPLETION_CONTEXT_EXTENSION":
            return "GRAPH COMPLETION CONTEXT EXTENSION selected"
        elif query == "FEELING_LUCKY":
            return "FEELING LUCKY selected"
        else:
            # Default fallback if no valid option is found
            return (
                "Function not found. Please choose from: "
                "GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, SUMMARIES, "
                "GRAPH_SUMMARY_COMPLETION, GRAPH_COMPLETION_COT, "
                "GRAPH_COMPLETION_CONTEXT_EXTENSION, FEELING_LUCKY"
            )
