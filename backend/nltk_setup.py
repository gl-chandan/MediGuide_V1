import os
import nltk


# ============================================================
# NLTK DATA DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


NLTK_DATA_DIR = os.path.join(
    BASE_DIR,
    "nltk_data"
)


os.makedirs(
    NLTK_DATA_DIR,
    exist_ok=True
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


print(
    "NLTK resources downloaded successfully"
)