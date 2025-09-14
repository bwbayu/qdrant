
from typing import List


def RAGCompetionOutput(query: str) -> str:
    return "hasil RAG completion: software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. Mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan. Sedangkan software engineering adalah disiplin ilmu yang mempelajari proses, metode, dan alat yang digunakan untuk mengembangkan perangkat lunak secara sistematis dan terstruktur."


def GraphCompletionOutput(query: str) -> str:
    return "hasil Graph completion: software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. Mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan. Sedangkan software engineering adalah disiplin ilmu yang mempelajari proses, metode, dan alat yang digunakan untuk mengembangkan perangkat lunak secara sistematis dan terstruktur."


def ChunksOutput(query: str) -> str:
    return "hasil Chunks: bagaimana perbedaan antara software engineer dan software engineering. Software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. Mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan. Sedangkan software engineering adalah disiplin ilmu yang mempelajari proses, metode, dan alat yang digunakan untuk mengembangkan perangkat lunak secara sistematis dan terstruktur."


def SummariesOutput(query: str) -> str:
    return "hasil summaries: menurut saya, perbedaan antara software engineer dan software engineering adalah sebagai berikut. software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan. sedangkan software engineering adalah disiplin ilmu yang mempelajari proses, metode, dan alat yang digunakan untuk mengembangkan perangkat lunak secara sistematis dan terstruktur."


def VideoEmbedOutput(query: str) -> List[dict]:
    return [{"url": "https: // samplelib.com/lib/preview/mp4/sample-10s.mp4",
            "start_offset_sec": "0",
             "end_offset_sec": "5",
             "embedding_option": "visual-text",
             "embedding_scope": "clip",
             "transcription": "Hasil transcribe video:menurut saya, perbedaan antara software engineer dan software engineering adalah sebagai berikut. software engineer adalah seseorang yang merancang, mengembangkan, menguji, dan memelihara perangkat lunak. mereka menggunakan prinsip-prinsip rekayasa untuk menciptakan solusi perangkat lunak yang efisien dan dapat diandalkan. sedangkan software engineering adalah disiplin ilmu yang mempelajari proses, metode, dan alat yang digunakan untuk mengembangkan perangkat lunak secara sistematis dan terstruktur."
             }]
