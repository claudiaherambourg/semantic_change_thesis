# -*- coding: utf-8 -*-
"""07a-extract_contextual_embeddings.ipynb

# 07a — Extract contextual embeddings

This notebook extracts contextual word representations from CamemBERT.
"""

!pip install -q transformers sentencepiece accelerate

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from collections import defaultdict
import pickle
import re
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import CamembertTokenizerFast, CamembertModel

PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis")

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"
CAMEMBERT_DIR = MODELS_DIR / "camembert_contextual"
DECADE_CORPORA_DIR = PROCESSED_DIR / "decade_corpora"
ANNOTATIONS_DIR = PROCESSED_DIR / "annotations"
CAMEMBERT_OCCURRENCES_DIR = CAMEMBERT_DIR / "occurrence_embeddings"
CAMEMBERT_CENTROIDS_DIR = CAMEMBERT_DIR / "centroids"
CAMEMBERT_OCCURRENCES_DIR.mkdir(parents=True, exist_ok=True)
CAMEMBERT_CENTROIDS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

analysis_vocab = pd.read_csv(TABLES_DIR / "analysis_vocabulary_top10000_content_words_cleaned.csv")
TARGET_WORDS = set(analysis_vocab["word"])
print(f"Target vocabulary: {len(TARGET_WORDS)} words")
analysis_vocab.head()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

tokeniser = CamembertTokenizerFast.from_pretrained("camembert-base")
model = CamembertModel.from_pretrained("camembert-base")
model.to(device)
model.eval()

def make_word_chunks(tokens, chunk_size=180):
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        if len(chunk_tokens) >= 20:
            chunks.append(" ".join(chunk_tokens))
    return chunks

WORD_RE = re.compile(r"[a-zàâäéèêëîïôöùûüÿçæœ]+", re.IGNORECASE)

def find_target_word_spans(text, target_words):
    spans = []
    for match in WORD_RE.finditer(text):
        word = match.group(0).lower()
        if word in target_words:
            spans.append({"word": word,"start": match.start(),"end": match.end()})
    return spans

def extract_target_embeddings_from_chunk(chunk_text, target_words):
    target_spans = find_target_word_spans(chunk_text, target_words)
    if len(target_spans) == 0:
        return []
    inputs = tokeniser(chunk_text,return_tensors="pt",return_offsets_mapping=True,truncation=True,max_length=512)
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state[0].detach().cpu().numpy()
    extracted = []
    for span in target_spans:
        matching_token_indices = []
        for token_index, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == token_end:
                continue
            overlaps = (token_start < span["end"] and token_end > span["start"])
            if overlaps:
                matching_token_indices.append(token_index)
        if len(matching_token_indices) == 0:
            continue
        occurrence_embedding = hidden_states[matching_token_indices].mean(axis=0)
        extracted.append({"word": span["word"],"embedding": occurrence_embedding})
    return extracted

def extract_decade_occurrences(decade,target_words,chunk_size=180,max_chunks=None):
    path = DECADE_CORPORA_DIR / f"{decade}.txt"
    with open(path, "r", encoding="utf-8") as f:
        tokens = f.read().split()
    chunks = make_word_chunks(tokens, chunk_size=chunk_size)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    word_embeddings = defaultdict(list)
    total_extracted = 0
    for chunk in tqdm(chunks, desc=f"Extracting {decade}"):
        extracted = extract_target_embeddings_from_chunk(chunk,target_words)
        for item in extracted:
            word_embeddings[item["word"]].append(item["embedding"])
            total_extracted += 1

    print(f"Decade {decade}")
    print("Chunks processed:", len(chunks))
    print("Target occur. extracted:", total_extracted)
    print("Unique words found:", len(word_embeddings))

    return word_embeddings

def save_occurrences(word_embeddings, output_path):
    clean_embeddings = {}
    for word, vectors in word_embeddings.items():
        if len(vectors) > 0:
            clean_embeddings[word] = np.vstack(vectors).astype(np.float32)
    with open(output_path, "wb") as f:
        pickle.dump(clean_embeddings, f)

    print("Saved:", output_path)
    print("Words saved:", len(clean_embeddings))

RUN_EXTRACTION = True
SKIP_EXISTING = True

decades = sorted(int(path.stem) for path in DECADE_CORPORA_DIR.glob("*.txt"))

if RUN_EXTRACTION:
    for decade in decades:
        output_path = CAMEMBERT_OCCURRENCES_DIR / f"{decade}_contextual_occurrences.pkl"
        if SKIP_EXISTING and output_path.exists():
            print(f"Skipping {decade}, already exists.")
            continue
        word_embeddings = extract_decade_occurrences(
            decade=decade,
            target_words=TARGET_WORDS,
            chunk_size=180,
            max_chunks=None)
        save_occurrences(word_embeddings, output_path)
else:
    print("extraction  skipped")

saved_files = sorted(CAMEMBERT_OCCURRENCES_DIR.glob("*_contextual_occurrences.pkl"))
print("Saved files :", len(saved_files))
for path in saved_files:
    print(path.name)