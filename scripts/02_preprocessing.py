# 02 - Preprocessing

This notebook prepares the *Corpus Chapitres* for the static embedding analyses.
"""

from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
from collections import Counter, defaultdict
import re
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis")

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"

CORPUS_DIR = RAW_DIR / "ANRChapitres-2000romans19e20e-ea770e4"
DECADE_CORPORA_DIR = PROCESSED_DIR / "decade_corpora"
FILTERED_CORPORA_DIR = PROCESSED_DIR / "decade_corpora_top50000_min25"
ANNOTATIONS_DIR = PROCESSED_DIR / "annotations"

for folder in [DECADE_CORPORA_DIR, FILTERED_CORPORA_DIR, ANNOTATIONS_DIR, TABLES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(METADATA_DIR / "metadata.csv")

metadata["publication_year"] = pd.to_numeric(metadata["publication_year"], errors="coerce").astype("Int64")
metadata = metadata.dropna(subset=["publication_year"]).copy()

print(f"Novels medadata length: {len(metadata)}")
metadata.head()

def read_body_text(xml_path):
    with open(xml_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")

    body = soup.find("body")
    if body is None:
        return ""
    return body.get_text(separator=" ", strip=True)

FRENCH_CLITIC_FRAGMENTS = {"j", "l", "d", "c", "m", "t", "s", "n",
                           "qu", "jusqu", "lorsqu", "puisqu", "aujourd", "hui"}

TOKEN_PATTERN = re.compile(r"[a-zA-ZàâäéèêëîïôöùûüÿçœæÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇŒÆ]+")

def tokenize_text(text):
    raw_tokens = TOKEN_PATTERN.findall(text)

    tokens = []
    case_rows = []

    for token in raw_tokens:
        token_lower = token.lower()

        if token_lower in FRENCH_CLITIC_FRAGMENTS:
            continue

        tokens.append(token_lower)
        case_rows.append({"token_lower": token_lower,
                          "token_original": token,
                          "is_initial_upper": token[:1].isupper(),
                          "is_all_upper": token.isupper()})

    return tokens, case_rows

decade_tokens = defaultdict(list)
document_rows = []

case_counter = Counter()
original_form_counter = Counter()

for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    filename = row["file"]
    publication_year = int(row["publication_year"])
    decade = int(publication_year // 10 * 10)

    xml_path = CORPUS_DIR / filename
    text = read_body_text(xml_path)
    tokens, case_rows = tokenize_text(text)

    decade_tokens[decade].extend(tokens)

    for item in case_rows:
        token_lower = item["token_lower"]
        case_counter[(token_lower, "total")] += 1
        original_form_counter[(token_lower, item["token_original"])] += 1

        if item["is_initial_upper"]:
            case_counter[(token_lower, "initial_upper")] += 1

        if item["is_all_upper"]:
            case_counter[(token_lower, "all_upper")] += 1

    document_rows.append({"file": filename,
                          "publication_year": publication_year,
                          "decade": decade,
                          "token_count": len(tokens),
                          "unique_token_count": len(set(tokens))})

document_stats = pd.DataFrame(document_rows)
print(f"Processed doc length: {len(document_stats)}")

for decade, tokens in sorted(decade_tokens.items()):
    with open(DECADE_CORPORA_DIR / f"{decade}.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(tokens))

print(f"{len(decade_tokens)} saved decade corpora")

document_stats.to_csv(TABLES_DIR / "processed_document_statistics.csv", index=False, encoding="utf-8")

decade_stats = (document_stats.groupby("decade").agg(number_of_documents=("file", "count"),
                                                     total_tokens=("token_count", "sum"),
                                                     mean_tokens_per_document=("token_count", "mean"),
                                                     median_tokens_per_document=("token_count", "median")).reset_index().sort_values("decade"))

decade_stats["vocabulary_size"] = decade_stats["decade"].map({decade: len(set(tokens)) for decade, tokens in decade_tokens.items()})
decade_stats.to_csv(TABLES_DIR / "processed_decade_statistics.csv", index=False, encoding="utf-8")

decade_stats.head()

case_rows = []

for (word, label), count in case_counter.items():
    case_rows.append({"token_lower": word,
                      "case_type": label,
                      "count": count})

token_case_counts = pd.DataFrame(case_rows)
token_case_summary = (token_case_counts.pivot_table(index="token_lower", columns="case_type", values="count", fill_value=0).reset_index())

for col in ["total", "initial_upper", "all_upper"]:
    if col not in token_case_summary.columns:
        token_case_summary[col] = 0

token_case_summary["initial_upper_ratio"] = token_case_summary["initial_upper"] / token_case_summary["total"]

original_form_rows = [{"token_lower": word, "token_original": form, "count": count}
                      for (word, form), count in original_form_counter.items()]

token_original_form_counts = pd.DataFrame(original_form_rows)
capitalised_candidates = token_case_summary[(token_case_summary["total"] >= 5) & (token_case_summary["initial_upper_ratio"]
                                                                                  >= 0.50)].sort_values(["initial_upper_ratio", "total"], ascending=False)

proper_noun_candidates = capitalised_candidates.copy()

token_case_summary.to_csv(ANNOTATIONS_DIR / "token_case_counts.csv", index=False, encoding="utf-8")
token_original_form_counts.to_csv(ANNOTATIONS_DIR / "token_original_form_counts.csv", index=False, encoding="utf-8")
capitalised_candidates.to_csv(ANNOTATIONS_DIR / "capitalised_candidates.csv", index=False, encoding="utf-8")
proper_noun_candidates.to_csv(ANNOTATIONS_DIR / "proper_noun_candidates.csv", index=False, encoding="utf-8")

print(f"Saved {len(proper_noun_candidates)} capitalisation-based proper noun candidates")

global_vocab = set()

for tokens in decade_tokens.values():
    global_vocab.update(tokens)

preprocessing_summary = pd.DataFrame({"statistic": ["Processed documents",
                                                    "Decade corpora",
                                                    "Total tokens",
                                                    "Global vocabulary size",
                                                    "Mean tokens per document",
                                                    "Median tokens per document"],
                                      "value": [len(document_stats),
                                                len(decade_stats),
                                                int(decade_stats["total_tokens"].sum()),
                                                len(global_vocab),
                                                round(document_stats["token_count"].mean(), 2),
                                                round(document_stats["token_count"].median(), 2)]})

preprocessing_summary.to_csv(TABLES_DIR / "preprocessing_summary.csv", index=False, encoding="utf-8")
preprocessing_summary

decade_corpora = {}

for path in sorted(DECADE_CORPORA_DIR.glob("*.txt")):
    decade = int(path.stem)
    with open(path, "r", encoding="utf-8") as f:
        decade_corpora[decade] = f.read().split()

global_counter = Counter()
decade_counters = {}

for decade, tokens in decade_corpora.items():
    counter = Counter(tokens)
    decade_counters[decade] = counter
    global_counter.update(counter)

global_vocab = pd.DataFrame(global_counter.items(), columns=["word", "global_frequency"])
global_vocab = global_vocab.sort_values("global_frequency", ascending=False).reset_index(drop=True)
global_vocab.head()

training_vocab_by_decade_rows = []
for decade, counter in sorted(decade_counters.items()):
    words_above_threshold = {word for word in training_vocab_set if counter.get(word, 0) >= MIN_COUNT_TRAINING}

    training_vocab_by_decade_rows.append({"decade": decade,
                                          "words_in_training_vocab": len(training_vocab_set),
                                          "words_above_min_count": len(words_above_threshold),
                                          "percent_surviving": round(len(words_above_threshold) / len(training_vocab_set) * 100, 2)})

training_vocab_by_decade = pd.DataFrame(training_vocab_by_decade_rows)
training_vocab_by_decade.to_csv(TABLES_DIR / "training_vocabulary_by_decade_min25.csv", index=False, encoding="utf-8")
training_vocab_by_decade

for decade, tokens in tqdm(decade_corpora.items()):
    counter = decade_counters[decade]
    allowed_words = {word for word in training_vocab_set if counter.get(word, 0) >= MIN_COUNT_TRAINING}
    filtered_tokens = [token for token in tokens if token in allowed_words]

    with open(FILTERED_CORPORA_DIR / f"{decade}.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(filtered_tokens))

!python -m spacy download fr_core_news_md -q

import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as FRENCH_STOPWORDS

nlp = spacy.load("fr_core_news_md", disable=["parser", "ner"])

proper_noun_path = ANNOTATIONS_DIR / "proper_noun_candidates.csv"
proper_nouns = set()

if proper_noun_path.exists():
    proper_noun_candidates = pd.read_csv(proper_noun_path)
    if "token_lower" in proper_noun_candidates.columns:
        proper_nouns = set(proper_noun_candidates["token_lower"].astype(str))
    elif "word" in proper_noun_candidates.columns:
        proper_nouns = set(proper_noun_candidates["word"].astype(str))

pos_rows = []
words_to_tag = list(training_vocab_set)

for doc in tqdm(nlp.pipe(words_to_tag, batch_size=1000), total=len(words_to_tag)):
    token = doc[0]
    pos_rows.append({"word": token.text,"pos": token.pos_,"lemma": token.lemma_})

pos_table = pd.DataFrame(pos_rows)
pos_table.to_csv(TABLES_DIR / "training_vocabulary_spacy_pos.csv", index=False, encoding="utf-8")
pos_table.head()

manual_exclusions = {"rosette", "alberte", "florine", "lia", "maximilien", "hilaire",
                     "hermine", "olivier", "alcide", "salvator", "cyprien", "amaury",
                     "éva", "flore", "clarisse", "eugène", "olympe", "gillette",
                     "prosper", "léonie", "vincent", "odette", "marion", "pierrot",
                     "hortense", "francine", "renée", "andrée", "hermann", "emma",
                     "mary", "clément", "dubreuil", "magloire", "irlande", "louvre",
                     "amazone", "b", "p", "xv", "mole", "dom", "gauthier",
                     "stanislas", "maria", "marianne", "dorothée", "germaine",
                     "bailli", "val", "charpentier", "recteur", "amiral", "pierre",
                     "jean", "jacques", "paul", "marie", "louise", "henri",
                     "charles", "georges", "joseph", "julien", "madeleine",
                     "marguerite", "thérèse", "antoine", "annette", "armande",
                     "clotilde", "honorine", "micheline", "pauline", "pierrette",
                     "ursule"}

proper_nouns = proper_nouns.union(manual_exclusions)

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
ANALYSIS_VOCAB_SIZE = 10_000

analysis_vocab = training_vocab.merge(pos_table, on="word", how="left")
analysis_vocab["is_stopword"] = analysis_vocab["word"].isin(FRENCH_STOPWORDS)
analysis_vocab["is_proper_noun_candidate"] = analysis_vocab["word"].isin(proper_nouns)
analysis_vocab["is_content_pos"] = analysis_vocab["pos"].isin(CONTENT_POS)

analysis_vocab_clean = analysis_vocab[(~analysis_vocab["is_stopword"]) & (~analysis_vocab["is_proper_noun_candidate"])
&(analysis_vocab["is_content_pos"])].copy()

analysis_vocab_clean = analysis_vocab_clean.sort_values("global_frequency", ascending=False).head(ANALYSIS_VOCAB_SIZE)
analysis_vocab_clean.to_csv(TABLES_DIR / "analysis_vocabulary_top10000_content_words_cleaned.csv", index=False, encoding="utf-8")

print(f"Analysis vocab size: {len(analysis_vocab_clean)}")
analysis_vocab_clean.head()