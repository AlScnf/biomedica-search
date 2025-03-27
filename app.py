import gradio as gr
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from pathlib import Path
from medmnist import PathMNIST
from torchvision import transforms

# ----------------------
# STEP 1: Ensure data is available
# ----------------------
PKL_PATH = Path("data/pathmnist_subset.pkl")
NPY_PATH = Path("data/pathmnist_image_embeddings.npy")

if not PKL_PATH.exists() or not NPY_PATH.exists():
    print("🔄 Data not found — regenerating pickle and embeddings...")
    os.makedirs("data", exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = PathMNIST(split='train', transform=transform, download=True)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    images, labels, embs = [], [], []
    for i in range(1000):
        img, label = dataset[i]
        images.append(np.array(img))
        labels.append(int(label))
        inputs = processor(images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            feature = features[0].numpy()
            feature /= np.linalg.norm(feature)
            embs.append(feature)

    df = pd.DataFrame({"image": images, "label": labels})
    df.to_pickle(PKL_PATH)
    np.save(NPY_PATH, np.array(embs))
    print("✅ Regeneration complete!")

# ----------------------
# STEP 2: Load everything
# ----------------------
df = pd.read_pickle(PKL_PATH)
embeddings = np.load(NPY_PATH)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cpu"
model.to(device).eval()

# ----------------------
# STEP 3: Define interface logic
# ----------------------
def search_similar(input_image=None, input_text=""):
    try:
        print("🚀 Received input")
        if input_image is not None:
            print("🧠 Processing image input...")
            input_image = input_image.resize((224, 224))
            inputs = processor(images=input_image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
        elif input_text.strip() != "":
            print("🧠 Processing text input...")
            inputs = processor(text=input_text, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                features = model.get_text_features(**inputs)
        else:
            raise ValueError("No input provided.")

        vector = features[0].cpu().numpy()
        vector /= np.linalg.norm(vector)

        D, I = index.search(vector.reshape(1, -1), 5)
        print(f"✅ FAISS returned {len(I[0])} results")

        return [df.iloc[idx]["image"] for idx in I[0]]

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return [None] * 5

# ----------------------
# STEP 4: Launch Gradio app
# ----------------------
demo = gr.Interface(
    fn=search_similar,
    inputs=[
        gr.Image(type="pil", label="Upload Image (optional)"),
        gr.Textbox(label="Or enter a biomedical description (optional)")
    ],
    outputs=[gr.Image(type="pil") for _ in range(5)],
    title="Biomedical Image Search Engine",
    description="Upload a biomedical image OR type a medical concept to retrieve the most visually similar images.",
    allow_flagging="never"
)

demo.launch()
