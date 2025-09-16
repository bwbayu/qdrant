import os
from pipelines.cognee.inference import query_cognee
from cognee import SearchType
from pipelines.twelve_labs.inference import query_twelve_labs

from typing import List


# Function to query Cognee with RAG (Retrieval-Augmented Generation) Completion
# Returns a text-based answer generated from knowledge retrieval + LLM completion
async def RAGCompetionOutput(query: str) -> str:
    results = await query_cognee(query, search_type=SearchType.RAG_COMPLETION)
    return results


# Function to query Cognee with Graph Completion
# Returns answers generated from graph-based reasoning or knowledge representation
async def GraphCompletionOutput(query: str) -> str:
    results = await query_cognee(query, search_type=SearchType.GRAPH_COMPLETION)
    return results


# Function to query Cognee and retrieve raw document chunks
# Useful for debugging or showing the context before summarization/completion
async def ChunksOutput(query: str) -> str:
    results = await query_cognee(query, search_type=SearchType.CHUNKS)
    return results


# Function to query Cognee and return document summaries
# Useful when only high-level insights are needed instead of full chunks
async def SummariesOutput(query: str) -> str:
    results = await query_cognee(query, search_type=SearchType.SUMMARIES)
    return results


# Function to query Twelve Labs API for video embeddings
# Takes a text query and retrieves relevant video segments as embeddings
# Returns a list of dictionaries containing video metadata and matches
def VideoEmbedOutput(query: str) -> List[dict]:
    results = query_twelve_labs(
        query_text=query, collection_name=os.getenv("TWELVE_LABS_COLLECTION"))
    return results
