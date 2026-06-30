"""
# 7b - CamemBERT centroid displacement

Builds one centroid representation per word and decade from the contextual
CamemBERT embeddings extracted in Notebook 06a.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm
from google.colab import drive

drive.mount("/content/drive")
PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis")
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"

MODELS_DIR = PROJECT_DIR / "models"
CAMEMBERT_DIR = MODELS_DIR / "camembert_contextual"
OCCURRENCES_DIR = CAMEMBERT_DIR / "occurrence_embeddings"
CENTROIDS_DIR = CAMEMBERT_DIR / "centroids"
CENTROIDS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

occurrence_files = sorted(OCCURRENCES_DIR.glob("*_contextual_occurrences.pkl"))
print(f"{len(occurrence_files)} occurr. files")

def load_occurrences(path):
    with open(path, "rb") as f:
        return pickle.load(f)
def centroid(vectors):
    return vectors.mean(axis=0).astype(np.float32)
def cosine_dist(vector_a, vector_b):
    if np.linalg.norm(vector_a) == 0 or np.linalg.norm(vector_b) == 0:
        return np.nan
    return cosine(vector_a, vector_b)

centroids = {}
occurrence_records = []
dispersion_records = []

for path in tqdm(occurrence_files):
    decade = int(path.stem.split("_")[0])
    occurrences = load_occurrences(path)

    centroids[decade] = {}

    for word, vectors in occurrences.items():
        if len(vectors) == 0:
            continue

        centre = centroid(vectors)
        centroids[decade][word] = centre

        occurrence_records.append({"decade": decade,
                                   "word": word,
                                   "n_occurrences": vectors.shape[0]})

        distances = [cosine_dist(vector, centre) for vector in vectors]
        distances = [d for d in distances if not np.isnan(d)]

        if distances:
            dispersion_records.append({"decade": decade,
                                       "word": word,
                                       "n_occurrences": vectors.shape[0],
                                       "mean_contextual_dispersion": float(np.mean(distances)),
                                       "median_contextual_dispersion": float(np.median(distances))})

occurrence_counts = pd.DataFrame(occurrence_records)
contextual_dispersion = pd.DataFrame(dispersion_records)
occurrence_counts.to_csv(TABLES_DIR / "camembert_contextual_occurrence_counts.csv",
                         index=False, encoding="utf-8")
contextual_dispersion.to_csv(TABLES_DIR / "camembert_contextual_dispersion.csv",
                             index=False, encoding="utf-8")

with open(CENTROIDS_DIR / "camembert_contextual_centroids.pkl", "wb") as f:
    pickle.dump(centroids, f)

displacement_records = []
decades = sorted(centroids.keys())

for decade_1, decade_2 in zip(decades[:-1], decades[1:]):
    shared_words = sorted(set(centroids[decade_1]) & set(centroids[decade_2]))
    for word in shared_words:
        distance = cosine_dist(centroids[decade_1][word],
                               centroids[decade_2][word])
        if np.isnan(distance):
            continue
        displacement_records.append({"word": word,
                                     "decade": decade_1,
                                     "next_decade": decade_2,
                                     "semantic_displacement": distance,
                                     "model": "camembert_centroid"})

semantic_displacement = pd.DataFrame(displacement_records)
semantic_displacement.to_csv(TABLES_DIR / "semantic_displacement_camembert_centroids.csv",
                             index=False, encoding="utf-8")
