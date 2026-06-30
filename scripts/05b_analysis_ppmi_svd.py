"""
# 05b — Analysis PPMI/SVD models

This notebook analyses the PPMI/SVD models trained in Notebook 05a.
"""

!pip install -q scipy statsmodels

from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine

PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis")

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DECADE_CORPORA_DIR = PROCESSED_DIR / "decade_corpora"
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = PROJECT_DIR / "models"
SVD_MODELS_DIR = MODELS_DIR / "ppmi_svd"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

decade_corpora = {}

for path in sorted(DECADE_CORPORA_DIR.glob("*.txt")):
    decade = int(path.stem)
    with open(path, "r", encoding="utf-8") as f:
        decade_corpora[decade] = f.read().split()
decade_counts = {decade: Counter(tokens) for decade, tokens in decade_corpora.items()}

svd_models = {}

for path in sorted(SVD_MODELS_DIR.glob("ppmi_svd_*.npz")):
    decade = int(path.stem.split("_")[-1])

    data = np.load(path, allow_pickle=True)
    words = list(data["words"])
    embeddings = data["embeddings"]
    word_to_id = {word: i for i, word in enumerate(words)}
    svd_models[decade] = {"words": words,
                          "word_to_id": word_to_id,
                          "embeddings": embeddings}

def align_svd_pair(model_t, model_t1):
    shared_words = sorted(set(model_t["words"]) & set(model_t1["words"]))
    vectors_t = np.array([model_t["embeddings"][model_t["word_to_id"][word]] for word in shared_words])
    vectors_t1 = np.array([model_t1["embeddings"][model_t1["word_to_id"][word]] for word in shared_words])
    rotation, _ = orthogonal_procrustes(vectors_t, vectors_t1)
    vectors_t_aligned = vectors_t @ rotation
    return shared_words, vectors_t_aligned, vectors_t1

displacement_rows = []
decades = sorted(svd_models.keys())

for decade_t, decade_t1 in zip(decades[:-1], decades[1:]):
    print(f"Computing PPMI/SVD displacement: {decade_t} → {decade_t1}")

    shared_words, vectors_t, vectors_t1 = align_svd_pair(svd_models[decade_t], svd_models[decade_t1])

    for word, vector_t, vector_t1 in zip(shared_words, vectors_t, vectors_t1):
        displacement_rows.append({"word": word,
                                  "decade_t": decade_t,
                                  "decade_t1": decade_t1,
                                  "semantic_displacement": cosine(vector_t, vector_t1),
                                  "frequency_t": decade_counts[decade_t].get(word, 0),
                                  "frequency_t1": decade_counts[decade_t1].get(word, 0)})

semantic_displacement_svd = pd.DataFrame(displacement_rows)
semantic_displacement_svd.to_csv(TABLES_DIR / "semantic_displacement_ppmi_svd_consecutive_decades.csv",
                                 index=False,
                                 encoding="utf-8")

print(f"Rows : {len(semantic_displacement_svd)}")
print(f"Unique words: {semantic_displacement_svd['word'].nunique()}")
semantic_displacement_svd.head()

analysis_vocab = pd.read_csv(TABLES_DIR / "analysis_vocabulary_top10000_content_words_cleaned.csv")
analysis_words = set(analysis_vocab["word"])

analysis_displacement_svd = semantic_displacement_svd[semantic_displacement_svd["word"].isin(analysis_words)].copy()
analysis_displacement_svd.to_csv(TABLES_DIR / "analysis_semantic_displacement_ppmi_svd_cleaned.csv",
                                 index=False,
                                 encoding="utf-8")

print(f"Rows after filtering: {len(analysis_displacement_svd)}")
print(f"Unique analysis words: {analysis_displacement_svd['word'].nunique()}")

analysis_displacement_svd.head()

MIN_FREQUENCY = 500

conformity_svd = analysis_displacement_svd[(analysis_displacement_svd["semantic_displacement"] > 0) &
                                           (analysis_displacement_svd["frequency_t"] >= MIN_FREQUENCY) &
                                           (analysis_displacement_svd["frequency_t1"] >= MIN_FREQUENCY)].copy()

conformity_svd["log_frequency"] = np.log(conformity_svd["frequency_t"])
conformity_svd["log_displacement"] = np.log(conformity_svd["semantic_displacement"])

conformity_svd["norm_log_displacement"] = (conformity_svd["log_displacement"] - conformity_svd["log_displacement"].mean()) / conformity_svd["log_displacement"].std()

print(f"Rows: {len(conformity_svd)}")
print(f"Unique words: {conformity_svd['word'].nunique()}")

conformity_svd.head()

conformity_model_svd = smf.mixedlm("norm_log_displacement ~ log_frequency + C(decade_t)",
                                   data=conformity_svd,
                                   groups=conformity_svd["word"])

conformity_result_svd = conformity_model_svd.fit(reml=True,
                                                 method="powell",
                                                 maxiter=500)

print(conformity_result_svd.summary())

conformity_results_svd = pd.DataFrame([{"model": "PPMI_SVD_conformity_mixedlm_min500",
                                        "n_rows": len(conformity_svd),
                                        "n_words": conformity_svd["word"].nunique(),
                                        "frequency_coefficient": conformity_result_svd.params["log_frequency"],
                                        "frequency_pvalue": conformity_result_svd.pvalues["log_frequency"]}])

conformity_results_svd.to_csv(TABLES_DIR / "ppmi_svd_conformity_results.csv",
                              index=False,
                              encoding="utf-8")

conformity_results_svd

polysemy_scores = pd.read_csv(TABLES_DIR / "polysemy_scores_ppmi_network.csv")

innovation_svd = conformity_svd.merge(polysemy_scores[["word", "decade", "log_contextual_diversity_centered"]],
                                      left_on=["word", "decade_t"],
                                      right_on=["word", "decade"],
                                      how="inner")

innovation_svd = innovation_svd.dropna(subset=["log_contextual_diversity_centered"]).copy()

print(f"Rows: {len(innovation_svd)}")
print(f"Unique words: {innovation_svd['word'].nunique()}")
innovation_svd.head()

innovation_model_svd = smf.mixedlm("norm_log_displacement ~ log_frequency + log_contextual_diversity_centered + C(decade_t)",
                                   data=innovation_svd,
                                   groups=innovation_svd["word"])

innovation_result_svd = innovation_model_svd.fit(reml=True,
                                                 method="powell",
                                                 maxiter=500)

print(innovation_result_svd.summary())

innovation_results_svd = pd.DataFrame([{"model": "PPMI_SVD_innovation_mixedlm_min500",
                                        "n_rows": len(innovation_svd),
                                        "n_words": innovation_svd["word"].nunique(),
                                        "frequency_coefficient": innovation_result_svd.params["log_frequency"],
                                        "frequency_pvalue": innovation_result_svd.pvalues["log_frequency"],
                                        "contextual_diversity_coefficient": innovation_result_svd.params["log_contextual_diversity_centered"],
                                        "contextual_diversity_pvalue": innovation_result_svd.pvalues["log_contextual_diversity_centered"]}])

innovation_results_svd.to_csv(TABLES_DIR / "ppmi_svd_innovation_results.csv",
                              index=False,
                              encoding="utf-8")

innovation_results_svd
