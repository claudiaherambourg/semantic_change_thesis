"""

# 06 - Qualitative analysis SGNS and PPMI/SVD

It builds the tables and scores needed for Chapter 3.
"""

!pip install -q gensim

from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from google.colab import drive
drive.mount("/content/drive")

PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis")

RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = PROJECT_DIR / "models"
SGNS_MODELS_DIR = MODELS_DIR / "sgns"
SVD_MODELS_DIR = MODELS_DIR / "ppmi_svd"
QUAL_DIR = TABLES_DIR / "chapter3_final_tables"
QUAL_DIR.mkdir(parents=True, exist_ok=True)
STATIC_CASE_DIR = TABLES_DIR / "static_qualitative_case_studies"
QUAL_STATIC_DIR = TABLES_DIR / "qualitative_static"

TOP_N = 5

HIGH_WORDS = {"SGNS": "volant","PPMI/SVD": "assurant"}
LOW_WORDS = {"SGNS": "tuiles","PPMI/SVD": "yeux"}

DIVERSITY_WORDS = ["ailleurs", "orangers"]
CASE_STUDY_WORDS = ["industrie","bourgeoisie","républicain"]

CASE_STUDY_DECADES = [1840, 1860, 1880, 1900]
DIAGNOSTIC_DECADES = [1840, 1860, 1880, 1900]
LOW_DIAGNOSTIC_DECADES = [1840, 1860, 1880, 1900]

table_35 = pd.read_csv(STATIC_CASE_DIR / "table_top10_changing_words_static_models.csv")
table_35.to_csv(QUAL_DIR / "table_3_5_highest_semantic_displacement.csv",index=False,encoding="utf-8")
display(table_35)

table_36 = pd.read_csv(STATIC_CASE_DIR / "table_top10_least_changing_words_static_models.csv")
table_36.to_csv(QUAL_DIR / "table_3_6_lowest_semantic_displacement.csv",index=False,encoding="utf-8")
display(table_36)

sgns_models = {}

for path in sorted(SGNS_MODELS_DIR.glob("sgns_*.model")):
    decade = int(path.stem.split("_")[-1])
    sgns_models[decade] = Word2Vec.load(str(path))

svd_models = {}

for path in sorted(SVD_MODELS_DIR.glob("ppmi_svd_*.npz")):
    decade = int(path.stem.split("_")[-1])
    data = np.load(path, allow_pickle=True)
    words = data["words"]
    embeddings = data["embeddings"]
    svd_models[decade] = dict(zip(words, embeddings))

def nearest_neighbors_sgns(word, decade, topn=5):
    model = sgns_models.get(decade)

    if model is None:
        return []
    if word not in model.wv:
        return []
    return [w for w, score in model.wv.most_similar(word, topn=topn)]


def nearest_neighbors_svd(word, decade, topn=5):
    model = svd_models.get(decade)

    if model is None:
        return []
    if word not in model:
        return []

    words = list(model.keys())
    vectors = np.vstack([model[w] for w in words])
    target = model[word].reshape(1, -1)
    similarities = cosine_similarity(target, vectors)[0]
    order = np.argsort(-similarities)

    neighbours = []

    for idx in order:
        candidate = words[idx]
        if candidate == word:
            continue
        neighbours.append(candidate)
        if len(neighbours) == topn:
            break
    return neighbours

def nearest_neighbors(model_name, word, decade, topn=5):
    if model_name == "SGNS":
        return nearest_neighbors_sgns(word, decade, topn)

    if model_name == "PPMI/SVD":
        return nearest_neighbors_svd(word, decade, topn)
    raise ValueError("model_name must be 'SGNS' or 'PPMI/SVD'")

rows = []
for model_name, word in HIGH_WORDS.items():
    for decade in DIAGNOSTIC_DECADES:
        neighbours = nearest_neighbors(model_name, word, decade, TOP_N)

        rows.append({"Model": model_name,"Word": word,"Decade": decade,"Nearest neighbors": ", ".join(neighbours)})

table_37 = pd.DataFrame(rows)
table_37.to_csv(QUAL_DIR / "table_3_7_high_displacement_neighbour_histories.csv",index=False,encoding="utf-8")
display(table_37)

rows = []

for model_name, word in LOW_WORDS.items():
    for decade in LOW_DIAGNOSTIC_DECADES:
        neighbours = nearest_neighbors(model_name, word, decade, TOP_N)

        rows.append({"Model": model_name,"Word": word,"Decade": decade,"Nearest neighbors": ", ".join(neighbours)})

table_38 = pd.DataFrame(rows)
table_38.to_csv(QUAL_DIR / "table_3_8_low_displacement_neighbour_histories.csv",index=False,encoding="utf-8")
display(table_38)

top_div = pd.read_csv(STATIC_CASE_DIR / "table_top_polysemous_words.csv")
if {"Highest contextual diversity proxy", "Lowest contextual diversity proxy"}.issubset(top_div.columns):
    table_39 = top_div.copy()
else:
    high_div = pd.read_csv(QUAL_STATIC_DIR / "top10_contextual_diversity_proxy.csv")
    low_div = pd.read_csv(QUAL_STATIC_DIR / "bottom10_contextual_diversity_proxy.csv")

    table_39 = pd.DataFrame({"Rank": range(10),"Highest contextual diversity proxy": high_div["word"].head(10).values,"Lowest contextual diversity proxy": low_div["word"].head(10).values})

table_39.to_csv(QUAL_DIR / "table_3_9_contextual_diversity_extremes.csv",index=False,encoding="utf-8")
display(table_39)

rows = []

for word in DIVERSITY_WORDS:
    for model_name in ["SGNS", "PPMI/SVD"]:
        for decade in DIAGNOSTIC_DECADES:
            neighbours = nearest_neighbors(model_name, word, decade, TOP_N)
            rows.append({"Model": model_name,"Word": word,"Decade": decade,
                         "Nearest neighbors": ", ".join(neighbours)})

table_310 = pd.DataFrame(rows)
table_310.to_csv(QUAL_DIR / "table_3_10_contextual_diversity_neighbour_histories.csv",index=False,encoding="utf-8")
display(table_310)

rows = []

for word in CASE_STUDY_WORDS:
    for model_name in ["SGNS", "PPMI/SVD"]:
        for decade in CASE_STUDY_DECADES:
            neighbours = nearest_neighbors(model_name, word, decade, TOP_N)

            rows.append({"Model": model_name,"Word": word,"Decade": decade,"Core semantic neighbors": ", ".join(neighbours)})
table_312 = pd.DataFrame(rows)
table_312.to_csv(QUAL_DIR / "table_3_12_historical_case_words_neighbours.csv",index=False,encoding="utf-8")
display(table_312)

HISTORICAL_WORDS_TABLE_311 = ["industrie","bourgeoisie","citoyen","république","peuple"]

historical_scores = pd.read_csv(STATIC_CASE_DIR / "historical_case_words_displacement_sgns_ppmi_svd.csv")

table_311 = historical_scores[historical_scores["word"].isin(HISTORICAL_WORDS_TABLE_311)].copy()

table_311["word"] = pd.Categorical(table_311["word"],categories=HISTORICAL_WORDS_TABLE_311,ordered=True)

table_311 = table_311.sort_values("word")
table_311 = table_311.rename(columns={"word": "Target word","mean_sgns_displacement": "SGNS mean dist.",
                                      "mean_ppmi_svd_displacement": "PPMI/SVD mean dist."})

table_311 = table_311[["Target word","SGNS mean dist.","PPMI/SVD mean dist."]]

table_311["SGNS mean dist."] = table_311["SGNS mean dist."].round(3)
table_311["PPMI/SVD mean dist."] = table_311["PPMI/SVD mean dist."].round(3)
display(table_311)
