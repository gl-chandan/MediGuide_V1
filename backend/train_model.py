import pandas as pd
import re
import nltk
import joblib

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# NLTK
# =========================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(
    "dataset/mtsamples.csv"
)

print("\nDataset loaded")
print("Original shape:", df.shape)


# =========================
# KEEP REQUIRED COLUMNS
# =========================

df = df[
    [
        "transcription",
        "medical_specialty"
    ]
]

df.dropna(
    inplace=True
)


# =========================
# CLEAN SPECIALTY NAMES
# =========================

df["medical_specialty"] = (
    df["medical_specialty"]
    .str.strip()
    .str.lower()
)


# =========================
# SELECT TOP 4 SPECIALTIES
# =========================

top_specialties = (
    df["medical_specialty"]
    .value_counts()
    .head(4)
    .index
)

df = df[
    df["medical_specialty"]
    .isin(top_specialties)
].copy()


print("\nSelected specialties:")

print(
    df["medical_specialty"]
    .value_counts()
)


# =========================
# TEXT PREPROCESSING
# =========================

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    tokens = word_tokenize(text)

    clean_tokens = []

    for word in tokens:

        if word.isalpha() and len(word) > 2:

            lemma = lemmatizer.lemmatize(
                word
            )

            clean_tokens.append(
                lemma
            )

    return " ".join(
        clean_tokens
    )


print("\nCleaning text...")

df["clean_text"] = (
    df["transcription"]
    .apply(clean_text)
)


# =========================
# FEATURES + TARGET
# =========================

X_text = df["clean_text"]

y = df["medical_specialty"]


# =========================
# TRAIN / TEST SPLIT
# =========================

X_train_text, X_test_text, y_train, y_test = train_test_split(

    X_text,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train_text))
print("Testing samples:", len(X_test_text))


# =========================
# TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
    min_df=2,
    max_df=0.95
)


X_train = vectorizer.fit_transform(
    X_train_text
)

X_test = vectorizer.transform(
    X_test_text
)


print("\nTF-IDF features:", X_train.shape[1])


# =========================
# LOGISTIC REGRESSION
# =========================

model = LogisticRegression(

    max_iter=2000,

    class_weight="balanced",
    C=2.0
)


# =========================
# TRAIN
# =========================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


print(
    "MODEL TRAINED SUCCESSFULLY"
)


# =========================
# PREDICTIONS
# =========================

predictions = model.predict(
    X_test
)


# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(

    y_test,

    predictions
)


print("\n=========================")
print("MODEL EVALUATION")
print("=========================")

print(
    f"\nAccuracy: {accuracy:.4f}"
)


# =========================
# CLASSIFICATION REPORT
# =========================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        predictions
    )
)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(

    y_test,

    predictions
)


print(
    "\nConfusion Matrix:"
)

print(cm)


# =========================
# SAVE MODEL
# =========================

joblib.dump(

    model,

    "models/medical_model.pkl"
)


joblib.dump(

    vectorizer,

    "models/vectorizer.pkl"
)


# =========================
# SAVE MODEL METADATA
# =========================

metadata = {

    "accuracy": accuracy,

    "classes":
        list(model.classes_),

    "num_features":
        X_train.shape[1],

    "ngram_range":
        "(1,2)",

    "max_features":
        10000
}


joblib.dump(

    metadata,

    "models/model_metadata.pkl"
)


print(
    "\nModel saved successfully."
)

print(
    "Files created:"
)

print(
    "models/medical_model.pkl"
)

print(
    "models/vectorizer.pkl"
)

print(
    "models/model_metadata.pkl"
)