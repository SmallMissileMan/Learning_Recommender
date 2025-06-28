import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("dataset/learning_resources.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["Description"].tolist(), show_progress_bar=True)
np.save("dataset/embeddings.npy", embeddings)
