import os
from pathlib import Path
from typing import List, Tuple, Dict
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi


class Retriever:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.chunks: List[str] = []
        self.chunk_ids: List[str] = []
        self.bm25 = None
        self._load_and_chunk_pdfs()

    def _load_and_chunk_pdfs(self):
        pdf_files = list(self.folder_path.glob("*.pdf"))

        for pdf_file in pdf_files:
            try:
                doc = fitz.open(pdf_file)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()

                    if text.strip():
                        chunk_id = f"{pdf_file.stem}_page_{page_num + 1}"
                        self.chunks.append(text.strip())
                        self.chunk_ids.append(chunk_id)

                doc.close()
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

        if self.chunks:
            tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)

    def get_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        chunk_scores = list(zip(self.chunks, self.chunk_ids, scores))
        chunk_scores.sort(key=lambda x: x[2], reverse=True)

        return chunk_scores[:top_k]
