"""
# 03a — Train SGNS models

This notebook trains one skip-gram with negative sampling model per decade.
"""

!pip install -q gensim

from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
from gensim.models import Word2Vec
from tqdm.auto import tqdm

PROJECT_DIR = Path("/content/drive/MyDrive/SemanticChangeThesis")

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FILTERED_CORPORA_DIR = PROCESSED_DIR / "decade_corpora_top50000_min25"
MODELS_DIR = PROJECT_DIR / "models"
SGNS_MODELS_DIR = MODELS_DIR / "sgns"
SGNS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

decade_corpora = {}

for path in sorted(FILTERED_CORPORA_DIR.glob("*.txt")):
    decade = int(path.stem)
    with open(path, "r", encoding="utf-8") as f:
        decade_corpora[decade] = f.read().split()

for decade, tokens in sorted(decade_corpora.items()):
    print(decade, len(tokens))

VECTOR_SIZE = 300
WINDOW = 4
NEGATIVE = 5
SAMPLE = 1e-5
EPOCHS = 5
WORKERS = 2
SEED = 42

def make_token_chunks(tokens, chunk_size=1000):
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

RUN_TRAINING = False

if RUN_TRAINING:
    for decade, tokens in sorted(decade_corpora.items()):
        print(f"\nTraining SGNS model for {decade}...")

        sentences = make_token_chunks(tokens, chunk_size=1000)
        model = Word2Vec(sentences=sentences,
                         vector_size=VECTOR_SIZE,
                         window=WINDOW,
                         min_count=1,
                         sg=1,
                         negative=NEGATIVE,
                         sample=SAMPLE,
                         workers=WORKERS,
                         epochs=EPOCHS,
                         seed=SEED)

        model_path = SGNS_MODELS_DIR / f"sgns_{decade}.model"
        model.save(str(model_path))

        print(f"Saved {model_path.name} - {len(model.wv)} words")
else:
    print("training skipped")

sgns_models = {}

for model_path in sorted(SGNS_MODELS_DIR.glob("sgns_*.model")):
    decade = int(model_path.stem.split("_")[1])
    sgns_models[decade] = Word2Vec.load(str(model_path))

for decade, model in sorted(sgns_models.items()):
    print(decade, len(model.wv))

expected_decades = sorted(decade_corpora.keys())
available_decades = sorted(sgns_models.keys())

print("Expected decades:", expected_decades)
print("Available SGNS models:", available_decades)

if expected_decades == available_decades:
    print("All expected SGNS models are available")
else:
    missing_decades = sorted(set(expected_decades) - set(available_decades))
    print("Missing SGNS models:", missing_decades)

expected_decades = sorted(decade_corpora.keys())
available_decades = sorted(sgns_models.keys())

print("Expected decades:", expected_decades)
print("Available SGNS models:", available_decades)

if expected_decades == available_decades:
    print("All expected SGNS models are available.")
else:
    missing_decades = sorted(set(expected_decades) - set(available_decades))
    print("Missing SGNS models:", missing_decades)
