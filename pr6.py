# =========================
# 📦 IMPORTS
# =========================

import re
import os
import pandas as pd
import joblib

import nltk

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =========================
# 🧠 NLP SETUP
# =========================

stop_words = set(stopwords.words("english"))
tokenizer = TreebankWordTokenizer()
stemmer = PorterStemmer()

# =========================
# 🔧 PREPROCESSING
# =========================


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def preprocess(text):
    text = clean_text(text)
    tokens = tokenizer.tokenize(text)

    return " ".join(stemmer.stem(word) for word in tokens if word not in stop_words)


# =========================
# 📊 LOAD DATASET (AUTO FIX)
# =========================


def load_data():
    df = pd.read_csv(
        "C:/Users/faiz/Desktop/Projectss/nlp-engineering/data/ai_ml_course_reviews.csv"
    )

    # 🔥 Auto-detect columns
    text_col = None
    label_col = None

    for col in df.columns:
        if "review" in col.lower():
            text_col = col
        if "sentiment" in col.lower() or "label" in col.lower():
            label_col = col

    if text_col is None or label_col is None:
        raise Exception("❌ Dataset must contain review_text & sentiment columns")

    df = df[[text_col, label_col]].dropna()
    df.columns = ["review_text", "sentiment"]

    df["clean_text"] = df["review_text"].apply(preprocess)

    return df


# =========================
# 🤖 TRAIN MODEL
# =========================


def train_model():
    df = load_data()

    X = df["clean_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LogisticRegression()),
        ]
    )

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "model.joblib")


# =========================
# 🔮 PREDICTION
# =========================


def predict(text, model):
    processed = preprocess(text)
    pred = model.predict([processed])[0]
    prob = model.predict_proba([processed])[0].max()
    return pred, prob, processed


# =========================
# 🌐 STREAMLIT UI (RESPONSIVE)
# =========================


def run_app():
    import streamlit as st

    st.set_page_config(page_title="SentimentScope", layout="wide")

    # 🔥 Custom CSS
    st.markdown(
        """
        <style>
        .main {background-color: #0e1117;}
        .title {text-align: center; font-size: 40px; color: #00ffd5;}
        .box {
            padding: 20px;
            border-radius: 10px;
            background-color: #1e2228;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='title'>🚀 SentimentScope AI</div>", unsafe_allow_html=True)

    if not os.path.exists("model.joblib"):
        with st.spinner("Training model..."):
            train_model()

    model = joblib.load("model.joblib")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Enter Review")
        user_input = st.text_area("", height=150)

        if st.button("Predict Sentiment"):
            if user_input.strip() == "":
                st.warning("Please enter text")
            else:
                pred, prob, processed = predict(user_input, model)

                st.markdown("### 🔍 Preprocessed Text")
                st.code(processed)

                st.markdown("### 🎯 Prediction")

                if pred == 1:
                    st.success(f"Positive 😊 | Confidence: {prob:.2f}")
                else:
                    st.error(f"Negative 😞 | Confidence: {prob:.2f}")

    with col2:
        st.markdown("### 📊 Model Info")
        st.info("""
        ✔ Algorithm: Logistic Regression  
        ✔ Features: TF-IDF (Unigram + Bigram)  
        ✔ Dataset: AI Course Reviews  
        ✔ Pipeline: End-to-End NLP  
        """)


# =========================
# 🚀 ENTRY POINT
# =========================

import sys

if "streamlit" in sys.modules:
    run_app()
else:
    train_model()
