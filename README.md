# Busca lexical no NFCorpus

Requer Python 3.10 ou superior. O baseline usa somente a biblioteca padrão.

1. Baixe o [NFCorpus oficial do BEIR](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip).
2. Extraia para `data/nfcorpus`. Esse diretório deve conter `corpus.jsonl`, `queries.jsonl` e `qrels/dev.tsv`.
3. Execute:

```powershell
python search.py data/nfcorpus
python search.py data/nfcorpus --query-id ID_DA_CONSULTA
```

O comando seleciona uma consulta do dev e mostra até dez documentos com correspondência lexical.
A saída inclui o grau de relevância (zero para documentos sem julgamento), as estatísticas
do índice e os tempos de indexação e consulta.

O baseline reaproveita o BM25 do T1 (k1=1.2, b=0.75). Usa título e texto, tokens
alfabéticos em minúsculas, sem remoção de stopwords ou stemming. O ranking percorre
as listas dos termos da consulta e ordena os documentos que receberam score.

## Avaliacao lexical em lote

```powershell
python evaluate_lexical.py data/nfcorpus --output results/dev
```

O comando avalia todas as consultas de `qrels/dev.tsv` e gera
`results_lexical.csv` e `metrics_per_query.csv`. Quando ha menos de 100
candidatos lexicais, completa o ranking com documentos de score zero, em ordem
de ID. O preenchimento nao usa os julgamentos de relevancia e examina no maximo
100 IDs da lista ordenada. As metricas usam exatamente o ranking exportado.

A AP@10 usa como denominador todos os relevantes nos qrels. O NDCG@10 usa
ganho linear e o ranking ideal de todos os julgamentos da consulta. O comando
tambem exibe as medias e os tempos de indexacao e consulta.

## Busca semantica e comparacao no dev

```powershell
python -m pip install -r requirements.txt
python evaluate_lexical.py data/nfcorpus --semantic --output results/dev
```

O comando avalia os dois modelos sobre as mesmas consultas e julgamentos.
Gera os dois rankings Top-100 e um `metrics_per_query.csv` com ambos os modelos.
O bi-encoder inicial e all-MiniLM-L6-v2. A busca usa cosseno por produto
escalar de vetores normalizados em L2. O primeiro uso baixa o modelo e
salva os embeddings em `data/embeddings`; as execucoes seguintes reutilizam
o cache quando o corpus e a configuracao sao os mesmos.
O arquivo `semantic_metadata.json` registra revisao do modelo, versoes,
dimensao, limite de tokens, truncamento e tempos.

## Busca hibrida com RRF

Depois de exportar os rankings lexical e semantico do dev, execute:

```powershell
python evaluate_hybrid.py data/nfcorpus --results results/dev
```

O script funde os Top-100, testa k=10, 60 e 100 pelo NDCG@10 medio do dev
e grava o vencedor em `rrf_config.json`. Empates favorecem o menor k.
A saida inclui `results_hybrid.csv` e atualiza `metrics_per_query.csv`
com lexical, semantico e hibrido.
