import os
from pipelines.cognee.inference import query_cognee
from cognee import SearchType
from pipelines.twelve_labs.inference import query_twelve_labs

from typing import List


async def RAGCompetionOutput(query: str) -> str:
    results = await query_cognee(query, search_type=SearchType.RAG_COMPLETION)
    return results
    # return "Components\n• The Control Unit and the Arithmetic and Logic Unit constitute the Central Processing Unit\n• Data and instructions need to get into the system and results out\n  ○ Input/output\n• Temporary storage of code and results is needed\n  ○ Main memory\nSlide description (Indonesian): Slide berjudul \"Components\" dengan daftar berpoin. Poin utama: 1) \"The Control Unit and the Arithmetic and Logic Unit constitute the Central Processing Unit\"; 2) \"Data and instructions need to get into the system and results out\" dengan subpoin \"Input/output\"; 3) \"Temporary storage of code and results is needed\" dengan subpoin \"Main memory\". Di pojok kiri bawah terdapat sebuah logo bundar berwarna merah/putih/hitam. Di tengah bawah tercantum kata \"Arskom\" dan di sudut kanan bawah nomor slide \"6\". URL: https://storage.googleapis.com/qdrant-hackathon/logif/docs/images/arsikom3_final_9c95a005_5.jpg"


async def GraphCompletionOutput(query: str) -> str:
    results = await query_cognee(query, search_type=SearchType.GRAPH_COMPLETION)
    return results
    # return "Function of Control Unit\nFor each operation a unique code is provided\ne.g. ADD, MOVE\nA hardware segment accepts the code and issues the control signals\nWe have a COMPUTER!\nFooter: Arskom (tengah bawah), slide number 5 (kanan bawah)\nLogo: lingkaran merah di pojok kiri bawah berisi gambar buku terbuka, lingkaran kuning di atas, dan pola hitam di bawah. URL: https://storage.googleapis.com/qdrant-hackathon/logif/docs/images/arsikom3_final_9c95a005_4.jpg"


async def ChunksOutput(query: str) -> str:
    results = await query_cognee(query, search_type=SearchType.CHUNKS)
    return results
    # return "Program Concept\n• Hardwired systems are inflexible\n• General purpose hardware can do different tasks, given correct control signals\n• Instead of re-wiring, supply a new set of control signals\nArskom URL: https://storage.googleapis.com/qdrant-hackathon/logif/docs/images/arsikom3_final_9c95a005_2.jpg"


async def SummariesOutput(query: str) -> str:
    results = await query_cognee(query, search_type=SearchType.SUMMARIES)
    return results
    # return "What is a program?\nA sequence of steps\nFor each step, an arithmetic or logical operation is done\nFor each operation, a different set of control signals is needed\nDeskripsi gambar: Slide menampilkan judul \"What is a program?\" di kiri atas; teks utama seperti di atas; ada logo kecil di pojok kiri bawah, footer \"Arskom\" di tengah bawah, dan nomor slide \"4\" di pojok kanan bawah. URL: https://storage.googleapis.com/qdrant-hackathon/logif/docs/images/arsikom3_final_9c95a005_3.jpg"


async def VideoEmbedOutput(query: str) -> List[dict]:
    results = await query_twelve_labs(query_text=query, collection_name=os.getenv("TWELVE_LABS_COLLECTION"))
    return results
    # return [{"url": "https: // samplelib.com/lib/preview/mp4/sample-10s.mp4",
    # "start_offset_sec": "0",
    #  "end_offset_sec": "5",
    #  "embedding_option": "visual-text",
    #  "embedding_scope": "clip",
    #  "transcription": "Example of Program Execution\nDiagram menampilkan enam langkah eksekusi program dengan keadaan memori dan register CPU.\nKondisi memori yang terlihat:\n- memori[300] = 1940\n- memori[301] = 5941\n- memori[302] = 2941\n- memori[940] = 0003\n- memori[941] = 0002\nLangkah-langkah (dilihat pada diagram):\nStep 1: Fetch instruksi dari memori[300] -> IR = 1940; PC menunjukkan 300.\nStep 2: Eksekusi instruksi 1940: muat isi memori[940] ke AC -> AC = 0003; PC = 301.\nStep 3: Fetch instruksi dari memori[301] -> IR = 5941; PC = 302.\nStep 4: Eksekusi instruksi 5941: AC = AC + memori[941] -> 0003 + 0002 = 0005.\nStep 5: Fetch instruksi dari memori[302] -> IR = 2941; PC = 303.\nStep 6: Eksekusi instruksi 2941: simpan AC ke memori[941] -> memori[941] = 0005.\nArskom URL: https://storage.googleapis.com/qdrant-hackathon/logif/docs/images/arsikom3_final_9c95a005_10.jpg"
    #  }]
