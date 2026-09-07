"""
Local embeddings using scikit-learn's HashingVectorizer.

Why not Gemini embeddings?
- text-embedding-004 is unavailable on the AI Studio free tier.
- A local vectorizer has zero latency, zero API cost, and no rate limits.
- For finding *candidate pairs* (the only job of embeddings here),
  word-overlap similarity is plenty good enough — the LLM does the
  precise reasoning, not the embedder.

HashingVectorizer uses the hashing trick to produce a fixed-size
sparse vector from text without needing to fit on a corpus first.
We convert to dense and L2-normalize so cosine similarity == dot product.
"""

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

_vectorizer = HashingVectorizer(
    n_features=512,
    norm="l2",
    alternate_sign=False,
    analyzer="word",
    ngram_range=(1, 2),  # unigrams + bigrams for better semantic coverage
)


def embed(text: str) -> np.ndarray:
    """Return a normalized 512-dim vector for a piece of text."""
    sparse = _vectorizer.transform([text])
    dense = sparse.toarray()[0].astype(np.float32)
    norm = np.linalg.norm(dense)
    return dense / norm if norm > 0 else dense


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalized vectors.
    Equivalent to dot product after normalization.
    """
    return float(np.dot(a, b))


def embedding_to_list(embedding: np.ndarray) -> list:
    return embedding.tolist()


def embedding_from_list(lst: list) -> np.ndarray:
    return np.array(lst, dtype=np.float32)
