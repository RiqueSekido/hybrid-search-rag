"""Run a lexical search on the NFCorpus development split."""

import argparse
from pathlib import Path
from time import perf_counter

from bm25 import BM25Model
from nfcorpus import load_documents, load_queries
from preprocessing import PREPROCESSING_CONFIGS, TextPreprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Extracted NFCorpus directory")
    parser.add_argument("--query-id", help="Dev query ID (default: first in sorted order)")
    args = parser.parse_args()

    try:
        queries, qrels = load_queries(args.dataset, "dev")
        query_id = args.query_id or min(queries)
        if query_id not in queries:
            parser.error(f"{query_id!r} is not a dev query.")

        preprocessor = TextPreprocessor(PREPROCESSING_CONFIGS[0])
        documents = load_documents(args.dataset, preprocessor)
        start = perf_counter()
        model = BM25Model().fit(documents)
        build_seconds = perf_counter() - start
        start = perf_counter()
        ranking = model.rank(preprocessor.process(queries[query_id]), top_k=10)
        query_ms = (perf_counter() - start) * 1000
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    print(f"{query_id}: {queries[query_id]}")
    print(f"Documents: {len(documents)} | Vocabulary: {model.vocabulary_size} | "
          f"Postings: {model.posting_count} | Build: {build_seconds:.3f}s | Query: {query_ms:.2f}ms")
    print("rank\tdoc_id\tscore\trelevance")
    for position, result in enumerate(ranking, start=1):
        relevance = qrels[query_id].get(result.document_id, 0)
        print(f"{position}\t{result.document_id}\t{result.score:.6f}\t{relevance}")


if __name__ == "__main__":
    main()
