"""
# 07e - Analysis for historical contextual cases

It creates the tables, scores and graphs fro Chapter 4.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount("/content/drive")
PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis")

RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"
CAMEMBERT_DIR = MODELS_DIR / "camembert_contextual"
OCCURRENCES_DIR = CAMEMBERT_DIR / "occurrence_embeddings"
CENTROIDS_DIR = CAMEMBERT_DIR / "centroids"
OUT_TABLES = TABLES_DIR / "chapter4_historical_case_studies"
OUT_FIGURES = FIGURES_DIR / "chapter4_historical_case_studies"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

TARGET_WORDS = ["industrie", "bourgeoisie", "républicain"]
SELECTED_DECADES = [1840, 1860, 1880, 1900, 1920]
MIN_OCCURRENCES = 25
NEIGHBOUR_MIN_OCCURRENCES = 100
TOPN_NEIGHBOURS = 5

camembert_displacement = pd.read_csv(TABLES_DIR / "semantic_displacement_camembert_centroids.csv")
contextual_dispersion = pd.read_csv(TABLES_DIR / "camembert_contextual_dispersion.csv")
occurrence_counts = pd.read_csv(TABLES_DIR / "camembert_contextual_occurrence_counts.csv")

with open(CENTROIDS_DIR / "camembert_contextual_centroids.pkl", "rb") as f:
    centroids = pickle.load(f)

print("Displacement rows:", len(camembert_displacement))
print("Dispersion rows:", len(contextual_dispersion))
print("Occurrence rows:", len(occurrence_counts))
print("Centroid decades:", sorted(centroids.keys()))

historical_disp = camembert_displacement[camembert_displacement["word"].isin(TARGET_WORDS)].copy()

historical_disp_summary = (historical_disp.groupby("word", as_index=False).agg(mean_semantic_displacement=("semantic_displacement", "mean"),
                                                                               max_semantic_displacement=("semantic_displacement", "max"),
                                                                               n_transitions=("semantic_displacement", "count")))

historical_dispersion = contextual_dispersion[contextual_dispersion["word"].isin(TARGET_WORDS)].copy()

historical_dispersion_summary = (historical_dispersion.groupby("word", as_index=False).agg(mean_contextual_dispersion=("mean_contextual_dispersion", "mean"),
                                                                                           max_contextual_dispersion=("mean_contextual_dispersion", "max"),
                                                                                           mean_occurrences=("n_occurrences", "mean"),
                                                                                           n_decades=("decade", "nunique")))

historical_summary = historical_disp_summary.merge(historical_dispersion_summary,on="word",how="outer")

historical_summary = historical_summary.sort_values("mean_semantic_displacement",ascending=False)

historical_summary.to_csv(OUT_TABLES / "historical_words_contextual_summary.csv",index=False,encoding="utf-8")

display(historical_summary)

historical_dispersion_timeline = historical_dispersion[historical_dispersion["decade"].isin(SELECTED_DECADES)].copy()

historical_dispersion_timeline = historical_dispersion_timeline[["word", "decade", "n_occurrences", "mean_contextual_dispersion", "median_contextual_dispersion"]].sort_values(["word", "decade"])

historical_dispersion_timeline.to_csv(OUT_TABLES / "historical_words_contextual_dispersion_timeline.csv",index=False,encoding="utf-8")

display(historical_dispersion_timeline)

def cosine_distance(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return np.nan
    return cosine(a, b)

def nearest_centroid_neighbours(word, decade, topn=8, min_occurrences=100):
    if decade not in centroids:
        return ""
    if word not in centroids[decade]:
        return ""

    valid_words = set(occurrence_counts[(occurrence_counts["decade"] == decade) &(occurrence_counts["n_occurrences"] >= min_occurrences)]["word"])
    target_vector = centroids[decade][word]
    rows = []

    for other_word, other_vector in centroids[decade].items():
        if other_word == word:
            continue
        if other_word not in valid_words:
            continue

        distance = cosine_distance(target_vector, other_vector)
        if not np.isnan(distance):
            rows.append((other_word, distance))

    rows = sorted(rows, key=lambda x: x[1])
    return ", ".join([word for word, distance in rows[:topn]])

neighbour_rows = []

for word in TARGET_WORDS:
    for decade in SELECTED_DECADES:
        disp_row = historical_dispersion[(historical_dispersion["word"] == word) &(historical_dispersion["decade"] == decade)]
        if len(disp_row) == 0:
            continue

        neighbours = nearest_centroid_neighbours(word=word,decade=decade,topn=TOPN_NEIGHBOURS,min_occurrences=NEIGHBOUR_MIN_OCCURRENCES)
        if neighbours == "":
            continue

        neighbour_rows.append({ "word": word,"decade": decade,"n_occurrences": int(disp_row.iloc[0]["n_occurrences"]),
                               "mean_contextual_dispersion": float(disp_row.iloc[0]["mean_contextual_dispersion"]),
                                "nearest_centroid_neighbours": neighbours})

historical_neighbours = pd.DataFrame(neighbour_rows)

historical_neighbours.to_csv(OUT_TABLES / "historical_words_contextual_neighbours.csv",index=False,encoding="utf-8")

display(historical_neighbours)

def load_occurrences_for_decade(decade):
    path = OCCURRENCES_DIR / f"{decade}_contextual_occurrences.pkl"

    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)

def get_occurrence_vectors(word, decade, max_occurrences=500):
    data = load_occurrences_for_decade(decade)

    if word not in data:
        return None

    vectors = data[word]

    if len(vectors) > max_occurrences:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(vectors), size=max_occurrences, replace=False)
        vectors = vectors[indices]

    return vectors

def plot_contextual_clouds_for_word(word, decades, max_occurrences=500):
    all_vectors = []
    labels = []

    for decade in decades:
        vectors = get_occurrence_vectors(word=word,decade=decade,max_occurrences=max_occurrences)
        if vectors is None or len(vectors) < 10:
            continue

        all_vectors.append(vectors)
        labels.extend([decade] * len(vectors))

    if len(all_vectors) < 2:
        print(f"not enough data for {word}")
        return

    matrix = np.vstack(all_vectors)
    labels = np.array(labels)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)

    plot_data = pd.DataFrame({"x": coords[:, 0],"y": coords[:, 1],"decade": labels})

    plt.figure(figsize=(7, 5))

    for decade in sorted(plot_data["decade"].unique()):
        subset = plot_data[plot_data["decade"] == decade]
        plt.scatter(subset["x"],subset["y"],s=8,alpha=0.35,label=str(decade))

    plt.title(f"Contextual occurrence cloud for '{word}'")
    plt.xlabel("PCA dimension 1")
    plt.ylabel("PCA dimension 2")
    plt.legend(title="Decade", markerscale=2)
    plt.tight_layout()

    output_path = OUT_FIGURES / f"contextual_cloud_{word}.png"
    plt.savefig(output_path, dpi=200)
    plt.show()

for word in TARGET_WORDS:
    plot_contextual_clouds_for_word(word=word,decades=SELECTED_DECADES,max_occurrences=500)
