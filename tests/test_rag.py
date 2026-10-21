import unittest

from rag import chunk_text, cosine_similarity, retrieve


class ChunkTests(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(chunk_text("abcdefghij", 5, 2), ["abcde", "defgh", "ghij"])

    def test_empty(self):
        self.assertEqual(chunk_text(" \n "), [])

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", 3, 3)


class SearchTests(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1], [1, 2])

    def test_top_k(self):
        chunks = [
            {"source": "B", "embedding": [0, 1]},
            {"source": "A", "embedding": [1, 0]},
        ]
        self.assertEqual(retrieve([1, 0], chunks, 1)[0]["source"], "A")

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            retrieve([1, 0], [], 0)


if __name__ == "__main__":
    unittest.main()
