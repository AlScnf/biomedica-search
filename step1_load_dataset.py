from medmnist import INFO
from medmnist.dataset import PathMNIST
import pandas as pd
from tqdm import tqdm
import os

# --- CONFIG ---
MAX_SAMPLES = 1000
os.makedirs("data", exist_ok=True)

# Select dataset
data_flag = 'pathmnist'
download = True

# Get dataset info
info = INFO[data_flag]
DataClass = getattr(__import__('medmnist.dataset'), info['python_class'])

# Load training split (no transform needed)
dataset = DataClass(split='train', download=download)

# Convert to dataframe
samples = []
for i in tqdm(range(min(len(dataset), MAX_SAMPLES))):
    img, label = dataset[i]
    samples.append({
        "image": img,           # Already a PIL image
        "label": int(label)
    })

df = pd.DataFrame(samples)
df.to_pickle("data/pathmnist_subset.pkl")

print(f"✅ Saved {len(df)} samples to 'data/pathmnist_subset.pkl'")
