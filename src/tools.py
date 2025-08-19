from google.adk.tools import FunctionTool
from src.retriever import Retriever


def create_retrieval_tool(data_folder: str) -> FunctionTool:
    retriever = Retriever(data_folder)

    def retrieve_chunks(query: str, top_k: int) -> str:
        """Retrieve the most relevant document chunks for a given query"""
        results = retriever.get_chunks(query, top_k=top_k)

        formatted_chunks = []
        for i, (chunk, chunk_id, score) in enumerate(results, 1):
            formatted_chunks.append(
                f"Chunk {i} (ID: {chunk_id}, Score: {score:.3f}):\n{chunk}\n"
            )

        return "\n".join(formatted_chunks)

    return FunctionTool(retrieve_chunks)
