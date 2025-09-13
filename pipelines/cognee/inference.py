from cognee import search, SearchType
import asyncio

async def query_cognee(query: str, search_type = SearchType.RAG_COMPLETION):
    # search knowledge
    results = await search(query_type=search_type, query_text=query)

    print("\nSearch Results:")
    for result in results:
        print(result)

    return result

if __name__ == "__main__":
    query = "Berikan contoh komposisi majemuk"
    asyncio.run(query_cognee(query))