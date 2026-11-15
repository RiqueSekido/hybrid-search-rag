"""Read the standardized BEIR NFCorpus files."""

import csv
import json


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def load_queries(dataset, split):
    qrels = {}
    with (dataset / "qrels" / f"{split}.tsv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])

    queries = {
        item["_id"]: item["text"]
        for item in read_jsonl(dataset / "queries.jsonl")
        if item["_id"] in qrels
    }
    if not qrels or queries.keys() != qrels.keys():
        raise ValueError(f"Missing queries or judgments for split {split!r}.")
    return queries, qrels


def load_document_texts(dataset):
    return {
        item["_id"]: f"{item.get('title') or ''} {item['text']}"
        for item in read_jsonl(dataset / "corpus.jsonl")
    }


def load_documents(dataset, preprocessor):
    return {
        doc_id: preprocessor.process(text)
        for doc_id, text in load_document_texts(dataset).items()
    }
