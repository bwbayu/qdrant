import os
from pipelines.cognee.inference import query_cognee
from cognee import SearchType
from pipelines.twelve_labs.inference import query_twelve_labs

from typing import List

async def RAGCompetionOutput(query: str) -> str:
    results = await query_cognee(query, search_type=SearchType.RAG_COMPLETION)
    return results

async def GraphCompletionOutput(query: str) -> str:
    results = await query_cognee(query, search_type=SearchType.GRAPH_COMPLETION)
    return results

async def ChunksOutput(query: str) -> str:
    results = await query_cognee(query, search_type=SearchType.CHUNKS)
    return results

async def SummariesOutput(query: str) -> str:
    results = await query_cognee(query, search_type=SearchType.SUMMARIES)
    return results

async def VideoEmbedOutput(query: str) -> List[dict]:
    results = await query_twelve_labs(query_text=query, collection_name=os.getenv("TWELVE_LABS_COLLECTION"))
    return results
    # return [{"url": "https: // samplelib.com/lib/preview/mp4/sample-10s.mp4",
    #         "start_offset_sec": "0",
    #          "end_offset_sec": "5",
    #          "embedding_option": "visual-text",
    #          "embedding_scope": "clip",
    #          "transcription": "Hasil transcribe video:menurut saya, perbedaan antara software engineer dan software engineering adalah sebagai berikut. software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan. sedangkan software engineering adalah disiplin ilmu yang mempelajari proses, metode, dan alat yang digunakan untuk mengembangkan perangkat lunak secara sistematis dan terstruktur."
    #          }]
