"""
# 05a — Train PPMI/SVD models

This notebook trains one count-based PPMI/SVD representation per decade.
The input files are the filtered decade corpora created in Notebook 02.
"""

!pip install -q scipy scikit-learn

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from tqdm.auto import tqdm

PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis")

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FILTERED_CORPORA_DIR = PROCESSED_DIR / "decade_corpora_top50000_min25"
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = PROJECT_DIR / "models"
SVD_MODELS_DIR = MODELS_DIR / "ppmi_svd"

SVD_MODELS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 4
VECTOR_SIZE = 300
ALPHA = 0.0
GAMMA = 0.0

training_vocab = pd.read_csv(TABLES_DIR / "training_vocabulary_top50000.csv")
training_words = set(training_vocab["word"])
print(f"Training vocabulary: {len(training_words)} words")

filtered_decade_corpora = {}

for path in sorted(FILTERED_CORPORA_DIR.glob("*.txt")):
    decade = int(path.stem)
    with open(path, "r", encoding="utf-8") as f:
        filtered_decade_corpora[decade] = f.read().split()
print(f"{len(filtered_decade_corpora)} filtered decade corpora")
for decade, tokens in sorted(filtered_decade_corpora.items()):
    print(decade, len(tokens))

def build_cooccurrence_matrix(tokens, vocabulary, window=4):
    word_to_id = {word: i for i, word in enumerate(sorted(vocabulary))}
    id_to_word = {i: word for word, i in word_to_id.items()}

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
    return matrix, word_to_id, id_to_word

def build_ppmi_matrix(cooc_matrix, alpha=0.0):
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
        pmi = np.log((count * total) / denominator) - alpha
        if pmi > 0:
            rows.append(row)
            cols.append(col)
            values.append(pmi)
    ppmi = coo_matrix((values, (rows, cols)),
                      shape=cooc_matrix.shape,
                      dtype=np.float64).tocsr()
    return ppmi

def train_svd_embeddings(ppmi_matrix, vector_size=300, gamma=0.0):
    k = min(vector_size, min(ppmi_matrix.shape) - 1)

    U, S, _ = svds(ppmi_matrix, k=k)
    order = np.argsort(S)[::-1]
    U = U[:, order]
    S = S[order]
    embeddings = U * (S ** gamma)
    return embeddings

RUN_TRAINING = True

if RUN_TRAINING:
    for decade, tokens in sorted(filtered_decade_corpora.items()):
        print(f"\nTraining PPMI/SVD model for {decade}")

        vocabulary = set(tokens) & training_words

        cooc_matrix, word_to_id, id_to_word = build_cooccurrence_matrix(tokens,
                                                                        vocabulary,
                                                                        window=WINDOW)

        ppmi_matrix = build_ppmi_matrix(cooc_matrix, alpha=ALPHA)
        embeddings = train_svd_embeddings(ppmi_matrix, vector_size=VECTOR_SIZE, gamma=GAMMA)

        words = np.array([id_to_word[i] for i in range(len(id_to_word))])
        output_path = SVD_MODELS_DIR / f"ppmi_svd_{decade}.npz"

        np.savez_compressed(output_path,
                            embeddings=embeddings,
                            words=words)

        print(f"Saved {output_path.name}")
        print(f"Vocabulary: {len(words)}")
        print(f"Embedding shape: {embeddings.shape}")

else:
    print("Training skipped")

saved_models = sorted(SVD_MODELS_DIR.glob("ppmi_svd_*.npz"))
print(f"Found {len(saved_models)} saved PPMI/SVD models")

for path in saved_models:
    data = np.load(path, allow_pickle=True)
    print(path.name, data["embeddings"].shape, len(data["words"]))
