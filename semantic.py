"""Dense retrieval with a pretrained bi-encoder and a corpus embedding cache."""

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from sentence_transformers import SentenceTransformer

from bm25 import BM25Result


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
BATCH_SIZE = 32


class SemanticIndex:
    def __init__(self, documents, cache_dir):
        self.doc_ids = sorted(documents)
        texts = [documents[doc_id] for doc_id in self.doc_ids]
        self.encoder = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)
        self.metadata = {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "sentence_transformers": version("sentence-transformers"),
            "transformers": version("transformers"),
            "torch": version("torch"),
            "max_input_tokens": self.encoder.max_seq_length,
            "dimensions": self.encoder.get_sentence_embedding_dimension(),
            "normalization": "L2",
            "similarity": "cosine",
            "batch_size": BATCH_SIZE,
        }
        corpus = json.dumps(list(zip(self.doc_ids, texts)), ensure_ascii=False).encode("utf-8")
        identity = json.dumps(self.metadata, sort_keys=True).encode("utf-8")
        cache_path = cache_dir / f"{sha256(identity + corpus).hexdigest()}.npz"
        self.metadata["cache_hit"] = cache_path.exists()
        if self.metadata["cache_hit"]:
            with np.load(cache_path, allow_pickle=False) as cached:
                self.embeddings = cached["embeddings"]
                self.metadata["embedding_seconds"] = float(cached["embedding_seconds"])
                self.metadata["truncated_documents"] = int(cached["truncated_documents"])
        else:
            self._build(texts, cache_path)

    def _build(self, texts, cache_path):
        tokenizer = self.encoder.tokenizer
        limit = self.encoder.max_seq_length
        truncated = 0
        for start in range(0, len(texts), BATCH_SIZE):
            tokenized = tokenizer(
                texts[start:start + BATCH_SIZE], truncation=False, return_length=True
            )
            truncated += sum(length > limit for length in tokenized["length"])

        start = perf_counter()
        self.embeddings = self.encoder.encode(
            texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True
        )
        elapsed = perf_counter() - start
        self.metadata.update(embedding_seconds=elapsed, truncated_documents=truncated)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=cache_path.parent, suffix=".npz", delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            np.savez(temp_path, embeddings=self.embeddings,
                     embedding_seconds=elapsed, truncated_documents=truncated)
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def rank(self, query, top_k=100):
        vector = self.encoder.encode(query, normalize_embeddings=True)
        scores = self.embeddings @ vector
        # Stable sorting breaks score ties by document ID.
        positions = np.argsort(-scores, kind="stable")[:top_k]
        return [BM25Result(self.doc_ids[i], float(scores[i])) for i in positions]
