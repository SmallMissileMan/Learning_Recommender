import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import streamlit as st

# ✅ Cached model loading
@st.cache_resource(show_spinner="🔁 Loading SentenceTransformer model...")
def load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Cached dataset loading
@st.cache_data(show_spinner="📂 Loading dataset...")
def load_dataset():
    dataset_path = os.path.join("dataset", "learning_resources.csv")
    df = pd.read_csv(dataset_path)

    # Rename columns for consistency
    df.rename(columns={
        "Channel": "Channel Name",
        "Resource URL": "Video Link",
        "Published At": "PublishedAt"
    }, inplace=True)

    # Drop rows with critical missing values
    df.dropna(subset=["Resource Name", "Description", "Channel Name", "Video Link"], inplace=True)

    return df

# ✅ Cached full dataset encoding
@st.cache_data(show_spinner="🧠 Encoding all descriptions...")
def encode_with_sentence_transformer(df):
    model = load_sentence_model()
    return model.encode(df["Description"].tolist(), show_progress_bar=False)

# 🔍 Recommender logic — uses precomputed embedding + live query
def recommend_resources(query, df, embeddings, top_n=10):
    model = load_sentence_model()
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]
    return df.iloc[top_indices]
