import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

# Load the data
df = pd.read_pickle("data/pathmnist_subset.pkl")

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare output list
embeddings = []

print("🔍 Embedding images...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    image = row["image"]

    # Preprocess and move to device
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # Convert to numpy
    vector = image_features[0].cpu().numpy()
    embeddings.append(vector)

# Convert to array and save
embedding_matrix = np.vstack(embeddings)
np.save("data/pathmnist_image_embeddings.npy", embedding_matrix)

print(f"✅ Saved embeddings to 'data/pathmnist_image_embeddings.npy'")
