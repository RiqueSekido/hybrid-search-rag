"""Tune RRF on NFCorpus dev using the exported baseline rankings."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from evaluation import QueryMetrics, evaluate_at_10
from hybrid import reciprocal_rank_fusion
from nfcorpus import load_queries
from reporting import export_metrics, export_rankings


CANDIDATE_K = (10, 60, 100)


def load_ranking_ids(path, query_ids):
    ranked = {query_id: {} for query_id in query_ids}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            positions = ranked[row["query_id"]]
            position = int(row["rank"])
            if position in positions:
                raise ValueError(f"Duplicate rank for {row['query_id']} in {path}.")
            positions[position] = row["doc_id"]

    rankings = {}
    for query_id, positions in ranked.items():
        if sorted(positions) != list(range(1, len(positions) + 1)):
            raise ValueError(f"Nonconsecutive ranks for {query_id} in {path}.")
        docs = [positions[rank] for rank in sorted(positions)[:100]]
        if len(docs) != 100 or len(set(docs)) != 100:
            raise ValueError(f"Expected 100 distinct documents for {query_id} in {path}.")
        rankings[query_id] = docs
    return rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--results", type=Path, default=Path("results/dev"))
    args = parser.parse_args()

    try:
        _, qrels = load_queries(args.dataset, "dev")
        query_ids = sorted(qrels)
        lexical = load_ranking_ids(args.results / "results_lexical.csv", query_ids)
        semantic = load_ranking_ids(args.results / "results_semantic.csv", query_ids)
        baselines = {
            name: {q: evaluate_at_10(ranking[q], qrels[q]) for q in query_ids}
            for name, ranking in (("lexical", lexical), ("semantic", semantic))
        }

        ndcg_by_k = {}
        best_ndcg = -1.0
        for k in CANDIDATE_K:
            rankings = {q: reciprocal_rank_fusion((lexical[q], semantic[q]), k) for q in query_ids}
            metrics = {
                q: evaluate_at_10([result.document_id for result in rankings[q][:10]], qrels[q])
                for q in query_ids
            }
            ndcg = mean(values.NDCG10 for values in metrics.values())
            ndcg_by_k[k] = ndcg
            if ndcg > best_ndcg:
                selected_k, best_ndcg = k, ndcg
                best_rankings, best_metrics = rankings, metrics

        export_rankings(args.results, "hybrid", best_rankings, qrels)
        metrics_by_model = {**baselines, "hybrid": best_metrics}
        export_metrics(args.results, metrics_by_model)
        report = {"split": "dev", "selected_k": selected_k, "mean_ndcg10_by_k": ndcg_by_k}
        (args.results / "rrf_config.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    for k, ndcg in ndcg_by_k.items():
        print(f"k={k:>3} | dev NDCG@10: {ndcg:.6f}")
    print(f"Selected k: {selected_k}")
    print(f"{'Model':<10} {'P@10':>9} {'Recall@10':>9} {'MAP@10':>9} {'NDCG@10':>9}")
    for name, metrics in metrics_by_model.items():
        averages = [mean(getattr(row, field) for row in metrics.values())
                    for field in QueryMetrics.__dataclass_fields__]
        print(f"{name:<10} " + " ".join(f"{value:9.6f}" for value in averages))


if __name__ == "__main__":
    main()
