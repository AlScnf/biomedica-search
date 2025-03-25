import gradio as gr
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Load data + embeddings
df = pd.read_pickle("data/pathmnist_subset.pkl")
embeddings = np.load("data/pathmnist_image_embeddings.npy")

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Set up FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Load CLIP model
device = "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Search function
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

        # Normalize vector
        vector = features[0].cpu().numpy()
        vector /= np.linalg.norm(vector)

        # Search
        D, I = index.search(vector.reshape(1, -1), 5)
        print(f"✅ FAISS returned {len(I[0])} results")

        return [df.iloc[idx]["image"] for idx in I[0]]

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return [None] * 5



# Launch the Gradio app
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
