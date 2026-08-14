import os
import nltk

NLTK_DATA_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "nltk_data"
)

os.makedirs(
    NLTK_DATA_DIR,
    exist_ok=True
)

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

print("NLTK resources downloaded successfully")