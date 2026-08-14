import os
import nltk


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# NLTK DATA DIRECTORY
# ============================================================

NLTK_DATA_DIR = os.path.join(
    BASE_DIR,
    "nltk_data"
)


os.makedirs(
    NLTK_DATA_DIR,
    exist_ok=True
)


print(
    "NLTK DATA DIRECTORY:",
    NLTK_DATA_DIR
)


# ============================================================
# DOWNLOAD NLTK RESOURCES
# ============================================================

print("Downloading NLTK resources...")


nltk.download(
    "punkt",
    download_dir=NLTK_DATA_DIR
)


nltk.download(
    "punkt_tab",
    download_dir=NLTK_DATA_DIR
)


nltk.download(
    "wordnet",
    download_dir=NLTK_DATA_DIR
)


# ============================================================
# VERIFY RESOURCES
# ============================================================

nltk.data.path.insert(
    0,
    NLTK_DATA_DIR
)


print("\nVerifying NLTK resources...")


nltk.data.find(
    "tokenizers/punkt"
)

nltk.data.find(
    "tokenizers/punkt_tab"
)

nltk.data.find(
    "corpora/wordnet"
)


print(
    "NLTK resources downloaded and verified successfully"
)