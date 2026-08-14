from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

import os
import re
import joblib
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
    "NLTK DATA PATH:",
    NLTK_DATA_DIR
)

print(
    "NLTK SEARCH PATHS:",
    nltk.data.path
)

# ============================================================
# VERIFY NLTK RESOURCES
# ============================================================
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

try:

    nltk.data.find(
        "tokenizers/punkt_tab"
    )

    nltk.data.find(
        "corpora/wordnet"
    )

    print(
        "NLTK RESOURCES FOUND"
    )

except LookupError as e:

    print(
        "NLTK RESOURCE ERROR:",
        str(e)
    )

    raise
# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "medical_model.pkl"
)


VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "vectorizer.pkl"
)


METADATA_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_metadata.pkl"
)


model = joblib.load(
    MODEL_PATH
)


vectorizer = joblib.load(
    VECTORIZER_PATH
)


metadata = joblib.load(
    METADATA_PATH
)


print("MODEL LOADED SUCCESSFULLY")


# ============================================================
# TEXT PREPROCESSING
# ============================================================

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    # Lowercase
    text = text.lower()


    # Remove special characters and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )


    # Tokenization
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


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MediGuide ML API",
    description="Medical specialty prediction API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class SymptomRequest(BaseModel):

    symptoms: str = Field(
        ...,
        min_length=3,
        description="Patient symptoms or medical description"
    )


# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "MediGuide ML API Running",
        "model": "TF-IDF + Logistic Regression",
        "version": "1.0.0"
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return metadata


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.post("/predict")
def predict(data: SymptomRequest):

    try:

        # ====================================================
        # GET USER INPUT
        # ====================================================

        user_text = data.symptoms.strip()


        # ====================================================
        # VALIDATE INPUT
        # ====================================================

        if not user_text:

            raise HTTPException(
                status_code=400,
                detail="Symptoms cannot be empty."
            )


        # ====================================================
        # CLEAN TEXT
        # ====================================================

        cleaned = clean_text(
            user_text
        )


        if not cleaned:

            raise HTTPException(
                status_code=400,
                detail="Unable to process the provided text."
            )


        # ====================================================
        # TF-IDF TRANSFORMATION
        # ====================================================

        transformed = vectorizer.transform(
            [cleaned]
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(
            transformed
        )[0]


        # ====================================================
        # PREDICTION PROBABILITIES
        # ====================================================

        probabilities = model.predict_proba(
            transformed
        )[0]


        classes = model.classes_


        # ====================================================
        # PROBABILITY DISTRIBUTION
        # ====================================================

        probability_distribution = {

            str(label): round(
                float(probability),
                4
            )

            for label, probability
            in zip(
                classes,
                probabilities
            )
        }


        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = max(
            probabilities
        )


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        return {

            "success": True,

            "prediction": prediction,

            "confidence": round(
                float(confidence),
                4
            ),

            "probabilities":
                probability_distribution
        }


    # ========================================================
    # HTTP ERRORS
    # ========================================================

    except HTTPException:

        raise


    # ========================================================
    # UNEXPECTED ERRORS
    # ========================================================

    except Exception as e:

        import traceback

        print(
            "Prediction Error:",
            str(e)
        )

        traceback.print_exc()


        raise HTTPException(
            status_code=500,
            detail=str(e)
        )