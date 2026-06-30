"""
# 01 — Corpus exploration

This notebook gives a first overview of the *Corpus Chapitres* before preprocessing.
"""

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis")
corpus_folder = RAW_DIR / "ANRChapitres-2000romans19e20e-ea770e4"

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"

for folder in [METADATA_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

xml_files = sorted(CORPUS_DIR.glob("*.xml"))
print(f"Found {len(xml_files)} XML files.")
for path in xml_files[:5]:
    print(path.name)

def read_tei_header(xml_path, max_chars=20000):
    with open(xml_path, "r", encoding="utf-8") as f:
        text = f.read(max_chars)
    if "</teiHeader>" in text:
        text = text.split("</teiHeader>")[0] + "</teiHeader>"
    return text

def find_first(pattern, text):
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else None

def extract_metadata(xml_path):
    header = read_tei_header(xml_path)
    genre_labels = re.findall(r"<term>(.*?)</term>", header, flags=re.DOTALL)
    genre_labels = [label.strip() for label in genre_labels if label.strip()]
    return {"file": xml_path.name,
            "title": find_first(r"<title>(.*?)</title>", header),
            "author": find_first(r'<author[^>]*name="([^"]*)"', header),
            "author_birth": find_first(r'<author[^>]*from="([^"]*)"', header),
            "author_death": find_first(r'<author[^>]*to="([^"]*)"', header),
            "author_sex": find_first(r'<author[^>]*sex="([^"]*)"', header),
            "publication_year": find_first(r'<date[^>]*type="created"[^>]*when="([^"]*)"',header),
            "genres": "; ".join(genre_labels) if genre_labels else None,}

metadata_rows = [extract_metadata(path) for path in tqdm(xml_files)]
metadata_raw = pd.DataFrame(metadata_rows)

metadata_raw.to_csv(METADATA_DIR / "metadata_raw.csv",
                    index=False,
                    encoding="utf-8")

metadata_raw.head()

metadata_quality = pd.DataFrame({"field": metadata_raw.columns,
                                 "missing_values": [metadata_raw[col].isna().sum() for col in metadata_raw.columns],
                                 "available_values": [metadata_raw[col].notna().sum() for col in metadata_raw.columns],})

metadata_quality.to_csv(TABLES_DIR / "metadata_quality.csv",
                        index=False,
                        encoding="utf-8")

metadata_quality

metadata_audit = metadata_raw.copy()

metadata_audit["publication_year_raw"] = metadata_audit["publication_year"]
metadata_audit["publication_year_corrected"] = metadata_audit["publication_year"]
metadata_audit["date_correction_note"] = ""

manual_date_corrections = {"1838_Gautier-Theophile_Fortunio.xml": 1838,
                           "1860_Vie-Anduze-Henri_Dernier-amour-de-Cinq-Mars-roman-historique.xml": 1860,
                           "1910_Ivoi-Paul-d-_Millionnaire-malgre-lui_tome-2.xml": 1910,
                           "1935_Delly_Contes.xml": 1935,}

for filename, corrected_year in manual_date_corrections.items():
    mask = metadata_audit["file"] == filename
    metadata_audit.loc[mask, "publication_year_corrected"] = corrected_year
    metadata_audit.loc[mask, "date_correction_note"] = ("Corrected manually after metadata validation")

metadata_audit.to_csv(METADATA_DIR / "metadata_audit.csv",
                      index=False,
                      encoding="utf-8")

metadata_audit[metadata_audit["date_correction_note"] != ""]

metadata = metadata_audit.copy()
metadata["publication_year"] = metadata["publication_year_corrected"]

metadata = metadata[["file","title","author","author_birth","author_death","author_sex","publication_year","genres",]]

metadata["publication_year"] = pd.to_numeric(metadata["publication_year"],errors="coerce").astype("Int64")

metadata.to_csv(METADATA_DIR / "metadata.csv",
                index=False,
                encoding="utf-8")

metadata.head()

genre_labels = (metadata["genres"].dropna().str.split(";").explode().astype(str).str.strip())

genre_labels = genre_labels[(genre_labels != "") & (genre_labels != '""') &(genre_labels != "''")]

corpus_stats = pd.DataFrame({"statistic": ["Number of XML files",
                                           "Number of unique authors",
                                           "Earliest publication year",
                                           "Latest publication year",
                                           "Number of unique genre labels",
                                           "Missing publication years",
                                           "Missing authors",
                                           "Manually corrected publication years",],
                             "value": [len(metadata),
                                       metadata["author"].nunique(dropna=True),
                                       int(metadata["publication_year"].min()),
                                       int(metadata["publication_year"].max()),
                                       genre_labels.nunique(),
                                       metadata["publication_year"].isna().sum(),
                                       metadata["author"].isna().sum(),
                                       (metadata_audit["date_correction_note"] != "").sum()],})

corpus_stats.to_csv(TABLES_DIR / "corpus_statistics.csv",
                    index=False,
                    encoding="utf-8")

corpus_stats

metadata["decade"] = (metadata["publication_year"] // 10 * 10).astype("Int64")

novels_per_decade = (metadata.dropna(subset=["decade"]).groupby("decade").size().reset_index(name="number_of_novels").sort_values("decade"))

authors_per_decade = (metadata.dropna(subset=["decade"]).groupby("decade")["author"].nunique().reset_index(name="number_of_authors").sort_values("decade"))

novels_per_decade.to_csv(TABLES_DIR / "novels_per_decade.csv",
                         index=False,
                         encoding="utf-8")

authors_per_decade.to_csv(TABLES_DIR / "authors_per_decade.csv",
                          index=False,
                          encoding="utf-8")

novels_per_decade.head()
