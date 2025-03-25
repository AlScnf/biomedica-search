import numpy as np
import pandas as pd
import faiss
from PIL import Image

# Load image metadata + embeddings
df = pd.read_pickle("data/pathmnist_subset.pkl")
embeddings = np.load("data/pathmnist_image_embeddings.npy")

# Normalize the embeddings (CLIP-style for cosine sim)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Create the FAISS index
dimension = embeddings.shape[1]  # 512
index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine Similarity when normalized
index.add(embeddings)

print(f"🔍 Indexed {len(embeddings)} images.")

# ---- TEST SEARCH ----

# Pick a random image from your dataset
query_idx = 10
query_vector = embeddings[query_idx].reshape(1, -1)

# Search top 5 similar
k = 5
D, I = index.search(query_vector, k)

print("\n🔎 Query image:")
df.iloc[query_idx]["image"].show()

print("\n🧠 Top similar images:")
for idx in I[0]:
    df.iloc[idx]["image"].show()
