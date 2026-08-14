import os
import zipfile
import urllib.request
import nltk


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

NLTK_DATA_DIR = os.path.join(
    BASE_DIR,
    "nltk_data"
)

CORPORA_DIR = os.path.join(
    NLTK_DATA_DIR,
    "corpora"
)


os.makedirs(
    CORPORA_DIR,
    exist_ok=True
)


# Tell NLTK where our resources are
nltk.data.path.insert(
    0,
    NLTK_DATA_DIR
)


print(
    "NLTK DATA DIRECTORY:",
    NLTK_DATA_DIR
)


# ============================================================
# DOWNLOAD NORMAL NLTK RESOURCES
# ============================================================

print("\nDownloading NLTK resources...")

nltk.download(
    "punkt",
    download_dir=NLTK_DATA_DIR
)

nltk.download(
    "punkt_tab",
    download_dir=NLTK_DATA_DIR
)


# ============================================================
# WORDNET
# ============================================================

WORDNET_DIR = os.path.join(
    CORPORA_DIR,
    "wordnet"
)

WORDNET_ZIP = os.path.join(
    CORPORA_DIR,
    "wordnet.zip"
)


# WordNet package from NLTK data repository
WORDNET_URL = (
    "https://raw.githubusercontent.com/"
    "nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
)


if not os.path.exists(
    WORDNET_DIR
):

    print(
        "\nDownloading WordNet manually..."
    )

    urllib.request.urlretrieve(
        WORDNET_URL,
        WORDNET_ZIP
    )

    print(
        "WordNet ZIP downloaded."
    )


    print(
        "Extracting WordNet..."
    )

    with zipfile.ZipFile(
        WORDNET_ZIP,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            CORPORA_DIR
        )


    print(
        "WordNet extracted."
    )


# ============================================================
# VERIFY
# ============================================================

print(
    "\nVerifying NLTK resources..."
)


resources = {

    "punkt":
        "tokenizers/punkt",

    "punkt_tab":
        "tokenizers/punkt_tab",

    "wordnet":
        "corpora/wordnet"
}


for name, resource in resources.items():

    try:

        nltk.data.find(
            resource
        )

        print(
            f"FOUND: {resource}"
        )

    except LookupError:

        print(
            f"NOT FOUND: {resource}"
        )

        raise


# ============================================================
# TEST WORDNET
# ============================================================

from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


test_word = lemmatizer.lemmatize(
    "running"
)


print(
    "\nWordNet test successful."
)

print(
    "running ->",
    test_word
)


print(
    "\nNLTK resources installed successfully."
)