import unittest
from pathlib import Path
from src.retriever import Retriever


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).parent / "../data/"
        self.retriever = Retriever(str(self.data_dir))

    def test_initialization(self):
        self.assertIsNotNone(self.retriever.bm25)
        self.assertTrue(len(self.retriever.chunks) > 0)
        self.assertTrue(len(self.retriever.chunk_ids) > 0)
        self.assertEqual(len(self.retriever.chunks), len(self.retriever.chunk_ids))

    def test_chunk_ids_format(self):
        for chunk_id in self.retriever.chunk_ids:
            self.assertIn("_page_", chunk_id)

    def test_get_chunks_coca_cola_query(self):
        results = self.retriever.get_chunks("Coca Cola revenue")
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 5)

        for chunk, chunk_id, score in results:
            self.assertIsInstance(chunk, str)
            self.assertIsInstance(chunk_id, str)
            self.assertIsInstance(score, float)

    def test_get_chunks_quarterly_query(self):
        results = self.retriever.get_chunks("quarterly report")
        self.assertGreater(len(results), 0)

    def test_get_chunks_with_top_k(self):
        results = self.retriever.get_chunks("data", top_k=1)
        self.assertEqual(len(results), 1)

    def test_empty_query(self):
        results = self.retriever.get_chunks("")
        self.assertGreaterEqual(len(results), 0)

    def test_no_match_query(self):
        results = self.retriever.get_chunks("nonexistent irrelevant terms")
        self.assertGreaterEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
