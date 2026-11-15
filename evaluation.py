"""Retrieval metrics shared by all ranking methods."""

from dataclasses import dataclass
from math import log2
from typing import Mapping, Sequence


@dataclass(frozen=True)
class QueryMetrics:
    P10: float
    R10: float
    AP10: float
    NDCG10: float


def evaluate_at_10(doc_ids: Sequence[str], judgments: Mapping[str, int]) -> QueryMetrics:
    total_relevant = sum(grade > 0 for grade in judgments.values())
    hits = 0
    precision_sum = 0.0
    dcg = 0.0
    for rank, doc_id in enumerate(doc_ids[:10], start=1):
        grade = judgments.get(doc_id, 0)
        if grade > 0:
            hits += 1
            precision_sum += hits / rank
            dcg += grade / log2(rank + 1)

    ideal_grades = sorted((grade for grade in judgments.values() if grade > 0), reverse=True)
    ideal_dcg = sum(grade / log2(rank + 1) for rank, grade in enumerate(ideal_grades[:10], 1))
    return QueryMetrics(
        P10=hits / 10,
        R10=hits / total_relevant if total_relevant else 0.0,
        AP10=precision_sum / total_relevant if total_relevant else 0.0,
        NDCG10=dcg / ideal_dcg if ideal_dcg else 0.0,
    )
