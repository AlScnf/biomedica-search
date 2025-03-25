import pandas as pd
import os

# Load the dataset
df = pd.read_pickle("data/pathmnist_subset.pkl")

# Create a folder to store sample exports
os.makedirs("data/exported_images", exist_ok=True)

# Export a few sample images (change index to get others)
for idx in [0, 5, 17, 42, 88]:
    img = df.iloc[idx]["image"]
    label = df.iloc[idx]["label"]
    filename = f"data/exported_images/img_{idx}_label_{label}.png"
    img.save(filename)
    print(f"✅ Saved: {filename}")
