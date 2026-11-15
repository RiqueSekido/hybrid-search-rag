"""Evaluate BM25 over all NFCorpus dev queries."""

import argparse
import json
from itertools import islice
from pathlib import Path
from statistics import mean
from time import perf_counter

from bm25 import BM25Model, BM25Result
from evaluation import evaluate_at_10
from nfcorpus import load_document_texts, load_queries
from preprocessing import PREPROCESSING_CONFIGS, TextPreprocessor
from reporting import export_metrics, export_rankings


def rank_top_100(model, tokens, sorted_doc_ids):
    ranking = model.rank(tokens, top_k=100)
    missing = 100 - len(ranking)
    if missing:
        matched = {result.document_id for result in ranking}
        unmatched = (doc_id for doc_id in sorted_doc_ids if doc_id not in matched)
        ranking.extend(BM25Result(doc_id, 0.0) for doc_id in islice(unmatched, missing))
    return ranking


def evaluate_queries(queries, qrels, rank_query):
    rankings = {}
    metrics = {}
    query_times = []
    for query_id in sorted(queries):
        start = perf_counter()
        ranking = rank_query(queries[query_id])
        query_times.append(perf_counter() - start)
        rankings[query_id] = ranking
        metrics[query_id] = evaluate_at_10(
            [result.document_id for result in ranking], qrels[query_id]
        )
    return rankings, metrics, mean(query_times) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Extracted NFCorpus directory")
    parser.add_argument("--output", type=Path, default=Path("results/dev"))
    parser.add_argument("--semantic", action="store_true", help="Also evaluate the bi-encoder")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/embeddings"))
    args = parser.parse_args()

    try:
        queries, qrels = load_queries(args.dataset, "dev")
        preprocessor = TextPreprocessor(PREPROCESSING_CONFIGS[0])
        texts = load_document_texts(args.dataset)
        documents = {doc_id: preprocessor.process(text) for doc_id, text in texts.items()}
        start = perf_counter()
        model = BM25Model().fit(documents)
        sorted_doc_ids = sorted(documents)
        build_seconds = perf_counter() - start

        def lexical_rank(query):
            return rank_top_100(model, preprocessor.process(query), sorted_doc_ids)

        lexical_rankings, lexical_metrics, lexical_ms = evaluate_queries(queries, qrels, lexical_rank)
        metrics_by_model = {"lexical": lexical_metrics}
        rankings_by_model = {"lexical": lexical_rankings}

        if args.semantic:
            from semantic import SemanticIndex

            index = SemanticIndex(texts, args.cache_dir)
            rankings, metrics, semantic_ms = evaluate_queries(queries, qrels, index.rank)
            metrics_by_model["semantic"] = metrics
            rankings_by_model["semantic"] = rankings
            index.metadata.update(
                split="dev", mean_query_ms=semantic_ms, documents=len(texts)
            )

        for name, rankings in rankings_by_model.items():
            export_rankings(args.output, name, rankings, qrels)
        export_metrics(args.output, metrics_by_model)
        if args.semantic:
            metadata_path = args.output / "semantic_metadata.json"
            metadata_path.write_text(json.dumps(index.metadata, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.error(str(error))

    print(f"dev: {len(queries)} queries | {len(documents)} documents")
    print(f"Vocabulary: {model.vocabulary_size} | Postings: {model.posting_count}")
    print(f"Index build: {build_seconds:.3f}s | Lexical mean query: {lexical_ms:.2f}ms")
    if args.semantic:
        print(f"Semantic mean query: {semantic_ms:.2f}ms | Metadata: {metadata_path}")
    print(f"{'Model':<10} {'P@10':>9} {'Recall@10':>9} {'MAP@10':>9} {'NDCG@10':>9}")
    for name, metrics in metrics_by_model.items():
        averages = [mean(getattr(result, field) for result in metrics.values())
                    for field in ("P10", "R10", "AP10", "NDCG10")]
        print(f"{name:<10} " + " ".join(f"{value:9.6f}" for value in averages))
    print(f"CSVs: {args.output}")


if __name__ == "__main__":
    main()
