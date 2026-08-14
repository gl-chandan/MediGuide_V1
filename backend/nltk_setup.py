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


nltk.data.path.insert(
    0,
    NLTK_DATA_DIR
)


print(
    "NLTK DATA DIRECTORY:",
    NLTK_DATA_DIR
)


# ============================================================
# DOWNLOAD NLTK RESOURCES
# ============================================================

print(
    "Downloading NLTK resources..."
)


resources = [
    "punkt",
    "punkt_tab",
    "wordnet",
]


for resource in resources:

    print(
        f"Downloading: {resource}"
    )

    success = nltk.download(
        resource,
        download_dir=NLTK_DATA_DIR,
        quiet=False
    )

    if not success:

        raise RuntimeError(
            f"Failed to download NLTK resource: {resource}"
        )


# ============================================================
# VERIFY RESOURCES
# ============================================================

print(
    "\nVerifying NLTK resources..."
)


checks = [
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
    "corpora/wordnet",
]


for resource in checks:

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


print(
    "\nNLTK resources downloaded and verified successfully"
)