"""
# 03b — Analyse SGNS models

This notebook analyses the SGNS models trained in Notebook 03a.
"""

!pip install -q gensim statsmodels

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine

PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis")

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DECADE_CORPORA_DIR = PROCESSED_DIR / "decade_corpora"
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = PROJECT_DIR / "models"
SGNS_MODELS_DIR = MODELS_DIR / "sgns"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

decade_corpora = {}

for path in sorted(DECADE_CORPORA_DIR.glob("*.txt")):
    decade = int(path.stem)
    with open(path, "r", encoding="utf-8") as f:
        decade_corpora[decade] = f.read().split()
decade_counts = {decade: Counter(tokens) for decade, tokens in decade_corpora.items()}

def align_decade_pair(model_1, model_2):
    shared_words = sorted(set(model_1.wv.index_to_key) & set(model_2.wv.index_to_key))

    vectors_1 = np.array([model_1.wv[word] for word in shared_words])
    vectors_2 = np.array([model_2.wv[word] for word in shared_words])
    rotation, _ = orthogonal_procrustes(vectors_1, vectors_2)
    vectors_1_aligned = vectors_1 @ rotation

    return shared_words, vectors_1_aligned, vectors_2

displacement_rows = []
decades = sorted(sgns_models.keys())

for decade_1, decade_2 in zip(decades[:-1], decades[1:]):
    print(f"computing disp.: {decade_1} → {decade_2}")

    shared_words, vectors_1, vectors_2 = align_decade_pair(sgns_models[decade_1], sgns_models[decade_2])

    for word, vector_1, vector_2 in zip(shared_words, vectors_1, vectors_2):
        displacement_rows.append({"word": word,
                                  "decade_t": decade_1,
                                  "decade_t1": decade_2,
                                  "semantic_displacement": cosine(vector_1, vector_2),
                                  "frequency_t": decade_counts[decade_1].get(word, 0),
                                  "frequency_t1": decade_counts[decade_2].get(word, 0)})

semantic_displacement = pd.DataFrame(displacement_rows)
semantic_displacement.to_csv(TABLES_DIR / "semantic_displacement_sgns_consecutive_decades.csv",
                             index=False,
                             encoding="utf-8")

print(f"Rows: {len(semantic_displacement)}")
print(f"Unique words: {semantic_displacement['word'].nunique()}")

semantic_displacement.head()

analysis_vocab = pd.read_csv(TABLES_DIR / "analysis_vocabulary_top10000_content_words_cleaned.csv")
analysis_words = set(analysis_vocab["word"])

analysis_displacement = semantic_displacement[semantic_displacement["word"].isin(analysis_words)].copy()
analysis_displacement.to_csv(TABLES_DIR / "analysis_semantic_displacement_sgns_cleaned.csv",
                             index=False,
                             encoding="utf-8")

print(f"Rows after filtering: {len(analysis_displacement)}")
print(f"Unique analysis words: {analysis_displacement['word'].nunique()}")

analysis_displacement.head()

MIN_FREQUENCY = 500

conformity_data = analysis_displacement[(analysis_displacement["semantic_displacement"] > 0) &
 (analysis_displacement["frequency_t"] >= MIN_FREQUENCY) & (analysis_displacement["frequency_t1"] >= MIN_FREQUENCY)].copy()

conformity_data["log_frequency"] = np.log(conformity_data["frequency_t"])
conformity_data["log_displacement"] = np.log(conformity_data["semantic_displacement"])

conformity_data["norm_log_displacement"] = (conformity_data["log_displacement"] - conformity_data["log_displacement"].mean())
/ conformity_data["log_displacement"].std()

print(f"Rows: {len(conformity_data)}")
print(f"Unique words: {conformity_data['word'].nunique()}")

conformity_data.head()

conformity_model = smf.mixedlm("norm_log_displacement ~ log_frequency + C(decade_t)",
                               data=conformity_data,
                               groups=conformity_data["word"])

conformity_result = conformity_model.fit(reml=True,
                                         method="powell",
                                         maxiter=500)
print(conformity_result.summary())

conformity_results = pd.DataFrame([{"model": "SGNS_mixedlm_min500",
                                    "n_rows": len(conformity_data),
                                    "n_words": conformity_data["word"].nunique(),
                                    "frequency_coefficient": conformity_result.params["log_frequency"],
                                    "frequency_pvalue": conformity_result.pvalues["log_frequency"]}])

conformity_results.to_csv(TABLES_DIR / "sgns_conformity_results.csv",
                          index=False,
                          encoding="utf-8")
conformity_results

polysemy_scores = pd.read_csv(TABLES_DIR / "polysemy_scores_ppmi_network.csv")

innovation_data = conformity_data.merge(polysemy_scores[["word", "decade", "log_contextual_diversity_centered"]],
                                        left_on=["word", "decade_t"],
                                        right_on=["word", "decade"],
                                        how="inner")

innovation_data = innovation_data.dropna(subset=["log_contextual_diversity_centered"]).copy()

print(f"Rows: {len(innovation_data)}")
print(f"Unique words: {innovation_data['word'].nunique()}")

innovation_data.head()

innovation_model = smf.mixedlm("norm_log_displacement ~ log_frequency + log_contextual_diversity_centered + C(decade_t)",
                               data=innovation_data,
                               groups=innovation_data["word"])

innovation_result = innovation_model.fit(reml=True,
                                         method="powell",
                                         maxiter=500)

print(innovation_result.summary())

innovation_model = smf.mixedlm("norm_log_displacement ~ log_frequency + log_contextual_diversity_centered + C(decade_t)",
                               data=innovation_data,
                               groups=innovation_data["word"])

innovation_result = innovation_model.fit(reml=True,
                                         method="powell",
                                         maxiter=500)
print(innovation_result.summary())
