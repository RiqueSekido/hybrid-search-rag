"""Reciprocal rank fusion for the lexical and semantic rankings."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import islice


@dataclass(frozen=True)
class HybridResult:
    document_id: str
    score: float


def reciprocal_rank_fusion(rankings, k):
    if k <= 0:
        raise ValueError("RRF k must be positive.")
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(islice(ranking, 100), start=1):
            scores[doc_id] += 1 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [HybridResult(doc_id, score) for doc_id, score in ordered]
