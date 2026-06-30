"""
# 04 — PPMI contextual diversity

This notebook builds the PPMI co-occurrence network used to approximate contextual diversity.
"""

!pip install -q scipy

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from tqdm.auto import tqdm

PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis")

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FILTERED_CORPORA_DIR = PROCESSED_DIR / "decade_corpora_top50000_min25"
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 4
SHIFT = np.log(5)
EPSILON = 1e-9

training_vocab = pd.read_csv(TABLES_DIR / "training_vocabulary_top50000.csv")
analysis_vocab = pd.read_csv(TABLES_DIR / "analysis_vocabulary_top10000_content_words_cleaned.csv")

training_words = set(training_vocab["word"])
analysis_words = set(analysis_vocab["word"])

print(f"Training vocab: {len(training_words)} words")
print(f"Analysis vocab: {len(analysis_words)} words")

decade_corpora = {}

for path in sorted(FILTERED_CORPORA_DIR.glob("*.txt")):
    decade = int(path.stem)
    with open(path, "r", encoding="utf-8") as f:
        decade_corpora[decade] = f.read().split()

def build_cooccurrence_matrix(tokens, vocabulary, window=4):
    word_to_id = {word: i for i, word in enumerate(sorted(vocabulary))}
    token_ids = [word_to_id[token] for token in tokens if token in word_to_id]
    pair_counts = Counter()
    for i, target_id in enumerate(token_ids):
        start = max(0, i - window)
        end = min(len(token_ids), i + window + 1)
        for j in range(start, end):
            if i == j:
                continue
            context_id = token_ids[j]
            pair_counts[(target_id, context_id)] += 1

    rows = []
    cols = []
    values = []

    for (row, col), count in pair_counts.items():
        rows.append(row)
        cols.append(col)
        values.append(count)
    matrix = coo_matrix((values, (rows, cols)),
                        shape=(len(word_to_id), len(word_to_id)),
                        dtype=np.float64).tocsr()

    return matrix, word_to_id

def build_ppmi_adjacency(cooc_matrix, shift=np.log(5)):
    cooc_matrix = cooc_matrix.tocsr()

    total = cooc_matrix.sum()
    row_sums = np.asarray(cooc_matrix.sum(axis=1)).ravel()
    col_sums = np.asarray(cooc_matrix.sum(axis=0)).ravel()

    coo = cooc_matrix.tocoo()

    rows = []
    cols = []
    values = []

    for row, col, count in zip(coo.row, coo.col, coo.data):
        denominator = row_sums[row] * col_sums[col]
        if denominator == 0:
            continue
        pmi = np.log((count * total) / denominator) - shift
        if pmi > 0:
            rows.append(row)
            cols.append(col)
            values.append(1)
    adjacency = coo_matrix((values, (rows, cols)),
                           shape=cooc_matrix.shape,
                           dtype=np.int8).tocsr()
    return adjacency

def measure_contextual_diversity(word, adjacency, word_to_id):
    if word not in word_to_id:
        return np.nan, 0, np.nan

    word_id = word_to_id[word]
    neighbours = adjacency[word_id].indices
    degree = len(neighbours)

    if degree < 2:
        return np.nan, degree, np.nan

    neighbour_graph = adjacency[neighbours][:, neighbours]
    observed_edges = neighbour_graph.nnz
    possible_edges = degree * (degree - 1)
    clustering = observed_edges / possible_edges
    contextual_diversity = 1 - clustering

    return contextual_diversity, degree, clustering

polysemy_rows = []

for decade, tokens in sorted(decade_corpora.items()):
    print(f"Processing {decade}")

    vocabulary = set(tokens) & training_words
    words_to_score = sorted(analysis_words & vocabulary)
    cooc_matrix, word_to_id = build_cooccurrence_matrix(tokens, vocabulary, window=WINDOW)
    adjacency = build_ppmi_adjacency(cooc_matrix, shift=SHIFT)

    print(f"  vocabulary: {len(vocabulary)} words")
    print(f"  PPMI edges: {adjacency.nnz}")

    for word in words_to_score:
        contextual_diversity, degree, clustering = measure_contextual_diversity(word, adjacency, word_to_id)
        polysemy_rows.append({"decade": decade,
                              "word": word,
                              "contextual_diversity": contextual_diversity,
                              "ppmi_degree": degree,
                              "local_clustering": clustering})

polysemy_scores = pd.DataFrame(polysemy_rows)
polysemy_scores.head()
