import os
import pandas as pd
from PIL import Image


# Percorso assoluto al Desktop (funziona su Mac)
desktop_path = os.path.expanduser("~/Desktop/biomedica_examples")
os.makedirs(desktop_path, exist_ok=True)

# Carica il DataFrame
df = pd.read_pickle("data/pathmnist_subset.pkl")

# Estrai un'immagine per ogni classe
unique_labels = df["label"].unique()

for label in unique_labels:
    img = df[df["label"] == label].iloc[0]["image"]
    img.save(os.path.join(desktop_path, f"class_{label}.png"))

print(f"✅ Immagini salvate sul Desktop in: {desktop_path}")
