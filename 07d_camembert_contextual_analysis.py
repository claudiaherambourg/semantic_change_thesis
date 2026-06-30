"""
# 07d - CamemBERT contextual analysis
It builds the contextual diagnostic tables used in Chapter 4.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from google.colab import drive

drive.mount("/content/drive")
PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis")

RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = PROJECT_DIR / "models"
CAMEMBERT_DIR = MODELS_DIR / "camembert_contextual"
CENTROIDS_DIR = CAMEMBERT_DIR / "centroids"
DIAGNOSTIC_DIR = TABLES_DIR / "chapter4_contextual_diagnostics"
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

MIN_OCCURRENCES = 100
MIN_DECADES = 5
CASE_STUDY_WORDS = ["mesure", "interrompit"]
DECADES = [1840, 1860, 1880, 1900, 1920]

dispersion = pd.read_csv(TABLES_DIR / "camembert_contextual_dispersion.csv")
occurrence_counts = pd.read_csv(TABLES_DIR / "camembert_contextual_occurrence_counts.csv")

with open(CENTROIDS_DIR / "camembert_contextual_centroids.pkl", "rb") as f:
    centroids = pickle.load(f)

filtered_dispersion = dispersion[dispersion["n_occurrences"] >= MIN_OCCURRENCES].copy()

dispersion_summary = (filtered_dispersion.groupby("word", as_index=False).agg(mean_dispersion=("mean_contextual_dispersion", "mean"),
                                                                              median_dispersion=("median_contextual_dispersion", "median"),
                                                                              mean_occurrences=("n_occurrences", "mean"),
                                                                              n_decades=("decade", "nunique")))

dispersion_summary = dispersion_summary[dispersion_summary["n_decades"] >= MIN_DECADES].copy()

top_words_dispersion = dispersion_summary.sort_values("mean_dispersion", ascending=False).head(10)
low_words_dispersion = dispersion_summary.sort_values("mean_dispersion", ascending=True).head(10)

top_words_dispersion.to_csv(DIAGNOSTIC_DIR / "top10_words_highest_mean_contextual_dispersion.csv",
                            index=False, encoding="utf-8")

low_words_dispersion.to_csv(DIAGNOSTIC_DIR / "top10_words_lowest_mean_contextual_dispersion.csv",
                            index=False, encoding="utf-8")

def cosine_dist(vector_a, vector_b):
    if np.linalg.norm(vector_a) == 0 or np.linalg.norm(vector_b) == 0:
        return np.nan
    return cosine(vector_a, vector_b)

def nearest_neighbours(word, decade, topn=8, min_occurrences=100):
    if decade not in centroids or word not in centroids[decade]:
        return ""

    target_vector = centroids[decade][word]

    valid_words = set(occurrence_counts[(occurrence_counts["decade"] == decade) & (occurrence_counts["n_occurrences"] >= min_occurrences)]["word"])
    neighbours = []

    for other_word, other_vector in centroids[decade].items():
        if other_word == word or other_word not in valid_words:
            continue

        distance = cosine_dist(target_vector, other_vector)
        if not np.isnan(distance):
            neighbours.append((other_word, distance))

    neighbours = sorted(neighbours, key=lambda item: item[1])
    return ", ".join([word for word, _ in neighbours[:topn]])


neighbour_records = []
for word in CASE_STUDY_WORDS:
    for decade in DECADES:
        current_row = filtered_dispersion[(filtered_dispersion["word"] == word) &
                                          (filtered_dispersion["decade"] == decade)]
        if len(current_row) == 0:
            continue
        neighbours = nearest_neighbours(word, decade, topn=8, min_occurrences=MIN_OCCURRENCES)
        if neighbours == "":
            continue
        neighbour_records.append({"word": word,
                                  "decade": decade,
                                  "n_occurrences": int(current_row.iloc[0]["n_occurrences"]),
                                  "mean_contextual_dispersion": current_row.iloc[0]["mean_contextual_dispersion"],
                                  "nearest_centroid_neighbours": neighbours})

neighbour_summary = pd.DataFrame(neighbour_records)
neighbour_summary.to_csv(DIAGNOSTIC_DIR / "compact_contextual_dispersion_examples_for_thesis.csv",
                         index=False, encoding="utf-8")