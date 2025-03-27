import gradio as gr
import numpy as np
import pandas as pd
import faiss
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from medmnist import INFO, PathMNIST

# ====================
# 📁 Setup and Download Dataset
# ====================
root = "./data"
os.makedirs(root, exist_ok=True)
info = INFO["pathmnist"]
data_class = PathMNIST

if not os.path.exists(f"{root}/pathmnist_subset.pkl"):
    print("📅 Downloading and processing PathMNIST dataset...")
    train_dataset = data_class(split="train", download=True, root=root)
    val_dataset = data_class(split="val", download=True, root=root)

    all_data = np.concatenate([train_dataset.imgs, val_dataset.imgs], axis=0)
    all_labels = np.concatenate([train_dataset.labels, val_dataset.labels], axis=0)
    all_images = []
for img in all_data:
    img = np.squeeze(img)  # get rid of any singleton dims (e.g. (1, 1, 28) -> (28,))
    if img.ndim == 2:  # grayscale
        img = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    elif img.ndim == 3:
        img = Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    all_images.append(img)


    df = pd.DataFrame({
        "image": all_images,
        "label": all_labels.flatten()
    })
    df = df.sample(500, random_state=42).reset_index(drop=True)
    df.to_pickle(f"{root}/pathmnist_subset.pkl")
    print("✅ Subset saved.")
else:
    df = pd.read_pickle(f"{root}/pathmnist_subset.pkl")

# ====================
# 🔗 CLIP Embedding
# ====================
print("🔗 Loading CLIP model and embedding images...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

embedding_path = f"{root}/pathmnist_image_embeddings.npy"
if not os.path.exists(embedding_path):
    embeddings = []
    for img in tqdm(df["image"], desc="🔍 Embedding images"):
        img_resized = img.resize((224, 224))
        inputs = processor(images=img_resized, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        embeddings.append(embedding.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    np.save(embedding_path, embeddings)
else:
    embeddings = np.load(embedding_path)

embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# ====================
# 🧠 FAISS Index
# ====================
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# ====================
# 🔍 Similarity Search
# ====================
def search_similar(input_image=None, input_text=""):
    try:
        if input_image is not None:
            input_image = input_image.resize((224, 224))
            inputs = processor(images=input_image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)

        elif input_text.strip() != "":
            inputs = processor(text=input_text, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                features = model.get_text_features(**inputs)
        else:
            raise ValueError("Provide either an image or text.")

        vector = features[0].cpu().numpy()
        vector /= np.linalg.norm(vector)

        D, I = index.search(vector.reshape(1, -1), 5)
        return [df.iloc[idx]["image"] for idx in I[0]]

    except Exception as e:
        print(f"Error: {e}")
        return [None] * 5

# ====================
# 🎛 Gradio UI + Suggestions
# ====================
sample_queries = [
    "pink stained epithelial cells",
    "tissue with dark nuclei",
    "inflamed area",
    "low density cancerous",
    "uniform pattern cells"
]

textbox = gr.Textbox(
    label="Or enter a biomedical description (optional)",
    placeholder="e.g. pink stained epithelial cells"
)

examples = gr.Examples(
    examples=[[None, q] for q in sample_queries],
    inputs=[gr.Image(type="pil"), textbox]
)

# ====================
# 🚀 Launch App
# ====================
demo = gr.Interface(
    fn=search_similar,
    inputs=[
        gr.Image(type="pil", label="Upload Image (optional)"),
        textbox
    ],
    outputs=[gr.Image(type="pil") for _ in range(5)],
    title="Biomedical Image Search Engine",
    description="Upload a biomedical image OR type a medical concept to retrieve the most visually similar scientific images.",
    allow_flagging="never",
    examples=examples
)

demo.launch()