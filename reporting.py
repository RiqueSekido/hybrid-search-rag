"""Write the standardized ranking and per-query metrics CSVs."""

import csv
from dataclasses import asdict, fields

from evaluation import QueryMetrics


def export_rankings(output, model, rankings, qrels):
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"results_{model}.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("query_id", "doc_id", "rank", "score", "relevance"))
        for query_id, ranking in rankings.items():
            for rank, result in enumerate(ranking[:100], 1):
                doc_id = result.document_id
                writer.writerow((query_id, doc_id, rank, result.score, qrels[query_id].get(doc_id, 0)))



def export_metrics(output, metrics_by_model):
    columns = ["query_id", "model", *(field.name for field in fields(QueryMetrics))]
    with (output / "metrics_per_query.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for model, metrics in metrics_by_model.items():
            for query_id, values in metrics.items():
                writer.writerow({"query_id": query_id, "model": model, **asdict(values)})
