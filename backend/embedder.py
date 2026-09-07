"""
Embeddings via Gemini text-embedding-004 (google-genai SDK).

We use Gemini's free embedding model rather than sentence-transformers,
which requires PyTorch >= 2.4. This keeps the dependency footprint light
and works on any Python environment without a GPU.

text-embedding-004 returns 768-dim vectors. We normalize them so
cosine similarity reduces to a dot product.
"""

import os
import numpy as np
from google import genai
from google.genai import types

# text-embedding-004 requires v1 (not the SDK default v1beta)
_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options={"api_version": "v1"},
)
_EMBED_MODEL = "text-embedding-004"


def embed(text: str) -> np.ndarray:
    """Return a normalized embedding vector for a piece of text."""
    result = _client.models.embed_content(
        model=_EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    vec = np.array(result.embeddings[0].values, dtype=np.float32)
    # L2-normalize so cosine similarity == dot product
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


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
