---
title: Biomedical Image Search Engine
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.6.0
app_file: app.py
pinned: true
---

# Biomedical Image Search Engine 🧠🔬  
> Search biomedical images using image upload or natural language queries. Powered by CLIP, FAISS, and Gradio.

---

## 🚀 What it does

This app allows users to retrieve biomedical images based on:

- 🖼 **Visual similarity** — Upload an image and find others that look like it  
- 💬 **Text prompts** — Type descriptions like "lung carcinoma" or "pink-stained cells" to search by meaning

It uses **zero-shot learning** via OpenAI's CLIP model to embed both images and text into a common vector space, and **FAISS** to return the top-k most similar images from a curated dataset (PathMNIST, 9-class histopathology).

---

## 💡 Why it matters

Biomedical professionals often search large image datasets visually or semantically:

- A **clinician** might want to see cases similar to one they’re analyzing  
- A **student** might want examples of specific cell structures  
- A **researcher** might want to explore patterns across labeled images

This tool makes that kind of exploration **instant** and **multimodal**.

---

## 🧠 Powered by

- **CLIP** (Contrastive Language–Image Pre-training) for vision-language embeddings  
- **FAISS** for fast, scalable similarity search  
- **Gradio** for a beautiful and interactive UI  
- **PathMNIST** as the working dataset (upgradeable to BIOMEDICA or custom sets)

---

## ✍️ Example usage

> Upload: a microscopic slide of stained epithelial cells  
> Output: top-5 similar images from the dataset  

> Input: `"microscopic blood smear with nucleus"`  
> Output: 5 images that match the text semantically

---

## 👨‍💻 About the Author

**Alessandro Scanferla**  
Founder-minded | Tech Explorer  

This project was created to demonstrate engineering, AI, and scientific application skills for:
- 🧪 Professors and research collaborators  
- 💼 Potential partners for biomedical AI projects

---

## ⚖️ License

[MIT License](LICENSE) — open to use, remix, and learn from.

---

## 🔮 Roadmap

This is just the start. Future extensions may include:
- Caption + label display for retrieved images  
- Real clinical datasets (e.g., BIOMEDICA)  
- Hybrid text + image queries  
- Deployment to hospitals or research labs as internal tooling
